#!/usr/bin/env python


from dataclasses import dataclass
from typing import Unpack


import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from dataclasses import dataclass 

import copy
import math 
from collections import deque 
from typing import TypedDict, Unpack 

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE 
from lerobot.utils.import_utils import require_package 
from lerobot.policies.common.vla_utils import resize_with_pad, pad_vector

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues

from .configuration_smolvla import SmolVLAConfig 



from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForImageTextToText, 
    AutoProcessor, 
    DynamicCache, 
    SmolVLMForConditionalGeneration,
)

from .custom_utils import (
    sample_noise,
    sample_time,
    adapt_mask_and_position_length,
    add_padding_to_tensor,
    create_attention_mask,
    create_sinusodial_pos_embedding
)
from .smolvlm_with_expert import apply_rope, get_intermediate_size
from .modeling_smolvla_original import (
    aloha_gripper_from_angular,
    aloha_gripper_from_angular_inv,
    aloha_gripper_to_angular,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None 
    prev_chunk_left_over: int | None 
    execution_horizon: int | None

# @dataclass 
# class SmolVLAConfig():
#
#     # VLM configuration 
#     vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
#     load_vlm_weights = True
#     num_vlm_layers = 16
#     vlm_dim = 64
#
#
#     state_dim= 32 
#     resize_img_to = (512, 512)
#
#     # Action expert configuration 
#     ae_dim = int(vlm_dim * 0.75)
#     num_ae_layers = 16
#     denoising_steps = 10
#     action_dim = 32 
#     action_chunk_size = 50
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"




class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "custom_smolvla"

    """
        Real-Time Chunking is an inference-time method that allows large, flow-matching based robotic policies, such 
        as Pi0 and SmolVLA, to produce smooth, continuous, and reactive motion despite having high inference latency.
    """
    def supports_rtc(self) -> bool:
        return True

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = FlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    """
        Ingest a raw batch of env observations, process and align the visual, propriceptive
        and linguistic inputs, pass them to the neural network to sample a trajectory and 
        output a clean "chunk" of future actions. 
    
        self._queues:   FIFO queue storing observations from recent past timesteps. Stacking 
                        these past frames along the time dimension provides the model with     
                        temporal history so it can better predict future trajectory paths.
    """
    def _get_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    """
        This is the main inference entry point. Given the current robot observations, 
        this function predicts several future actions.

        @params:    -batch:  contains camera images, robot state, language tokens
                    -noise: 
                    -Tensor:
                    -kwargs: 
        @return: actions, with approximately the dimension (batch_size, n_action_steps, action_dim).
                 During inference the batch_size is 1.

    """
    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch) # adapt to aloha if necessary
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        actions = self._get_action_chunk(batch, noise, **kwargs)
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `_get_action_chunk()` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `_get_action_chunk()` when the
        queue is empty.


        Noise is always None when being called from lerobot_eval.py
        Only the preprocessed observations are passed as batch into this function. 
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch, noise)

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> dict[str, Tensor]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        """
        Conceptually: config.image_features = 
        [
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3"
        ]            
        """
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]


        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            # extract image and handle 2 cases:
            # case A: image has no temporal dimension -> [B, Channels, Height, Width]
            # case B: image has temporal/ observation dimension -> select the last ovservation
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0, # this padding is seen by SigLIP as black pixels, which is the expected behavior
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    """
        _pi_alpha... functions serve as hardware translation bridge inside the LeRobot toolchain.
        They are designed to align the coordinate systems, motor orientations, and gripper 
        ranged of the ALOHA bimanual robot with the native angular representation space that the 
        SmolVLA model was pretrained on.
    """
    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            ) 



class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        self.vlm = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 
            dtype="bfloat16",
            low_cpu_mem_usage=True
        )
        self.vlm_config = self.vlm.config
        self.text_config = copy.deepcopy(self.vlm_config.text_config)

        # Only keep the first num_vlm_layers layers of the VLM
        if config.num_vlm_layers > 0:
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:config.num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)

        # Smaller action expert: narrower hidden size, but the same number and size of attention heads
        # as the VLM, so their QKV can still be concatenated in the self-attention layers
        self.expert_config = copy.deepcopy(self.text_config)
        expert_hidden_size = int(self.text_config.hidden_size * config.expert_width_multiplier)
        self.expert_config.hidden_size = expert_hidden_size
        self.expert_config.intermediate_size = get_intermediate_size(expert_hidden_size)
        self.expert_config.num_hidden_layers = self.num_vlm_layers
        if config.num_expert_layers > 0:
            assert self.num_vlm_layers % config.num_expert_layers == 0, (
                f"Number of layers in the VLM {self.num_vlm_layers} are not multiple of num_expert_layers {config.num_expert_layers}"
            )
            self.expert_config.num_hidden_layers = config.num_expert_layers
        self.lm_expert = AutoModel.from_config(self.expert_config)
        self.num_expert_layers = len(self.lm_expert.layers)

        if "cross" in config.attention_mode:
            # In the cross-attention layers the expert projects the cached VLM keys/values,
            # so their k_proj/v_proj take the VLM KV width as input
            vlm_kv_dim = self.text_config.num_key_value_heads * self.text_config.head_dim
            expert_kv_dim = self.expert_config.num_key_value_heads * self.expert_config.head_dim
            for layer_idx in range(self.num_expert_layers):
                if config.self_attn_every_n_layers > 0 and layer_idx % config.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    vlm_kv_dim, expert_kv_dim, bias=self.expert_config.attention_bias
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    vlm_kv_dim, expert_kv_dim, bias=self.expert_config.attention_bias
                )
        # The expert only receives embeddings, it never embeds tokens itself
        self.lm_expert.embed_tokens = None

        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, config.max_action_dim)

        # The state token is part of the VLM prefix, so it is projected to the VLM hidden size
        self.state_proj = nn.Linear(
            config.max_state_dim, self.text_config.hidden_size
        )

        self.action_time_mlp_in = nn.Linear(
            expert_hidden_size * 2, expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            expert_hidden_size, expert_hidden_size
        )

        self.set_requires_grad()

    def set_requires_grad(self):
        # The whole VLM (vision encoder + text layers) stays frozen, only the action expert and the projections are trained
        if not self.config.train_expert_only:
            raise ValueError("CustomSmolVLA never trains the VLM, `train_expert_only=False` is not supported.")
        self.vlm.eval()
        self.vlm.requires_grad_(False)
        self.state_proj.requires_grad_(self.config.train_state_proj)

    def train(self, mode: bool = True):
        # Keep the frozen VLM in eval mode, even when lerobot-train calls policy.train()
        super().train(mode)
        self.vlm.eval()
        return self


    def get_vlm_model(self):
        return self.vlm.model

    def get_model_layers(self, models):
            vlm_layers = []
            expert_layers = []
            multiple_of = self.num_vlm_layers // self.num_expert_layers 
            for i in range(self.num_vlm_layers):
                if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                    expert_layer = None 
                else: 
                    expert_layer_index = i // multiple_of if multiple_of > 0 else i 
                    expert_layer = models[1].layers[expert_layer_index]
                vlm_layers.append(models[0].layers[i])
                expert_layers.append(expert_layer)
            return [vlm_layers, expert_layers]
                


    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)


    def embed_image(self, image):
        patch_attention_mask = None
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=None,
            )
            .last_hidden_state
        )
        return self.get_vlm_model().connector(image_hidden_states)


    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state = None):
        embs = []
        pad_masks = []
        att_masks = []

        for _, (img, img_mask) in enumerate(zip(images, img_masks, strict=False)):
            # image_start_token = self.embed_language_tokens(tokens)
            #     .unsequeeze(0)
            #     .expand(img.shape[0], -1, -1)
            #
            # image_start_mask = torch.ones_like(
            #     image_start_token[:, :, 0], dtype=torch.bool, device=self.config.device
            # )
            # att_masks = [0] * image_stark_mask.shape[-1]
            # embs.append(image_start_token)
            # pad_masks.append(image_start_mask)

            img_emb = self.embed_image(img)
            bsize, num_img_embs, img_emb_dim = *img_emb.shape[:2], img_emb.shape[-1]

            # Normalize image
            img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * num_img_embs

            # add image special tokens?
            
            # done with image embeddings 

        lang_emb = self.embed_language_tokens(lang_tokens)
        # normalize language embeddings 
        num_lang_emb, lang_emb_dim = lang_emb.shape[1], lang_emb.shape[-1]
        lang_emb = lang_emb * (lang_emb_dim ** 0.5)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * num_lang_emb

        # add state embeddings 
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)

        bsize = state_emb.shape[0]
        device = state_emb.device
        # states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(1, 1, dtype=torch.bool, device=device).expand(bsize, 1)
        pad_masks.append(state_mask)

        att_masks += [1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.config.prefix_length:
            embs = add_padding_to_tensor(embs, self.config.prefix_length, 0)
            pad_masks = add_padding_to_tensor(pad_masks, self.config.prefix_length, 0)
            att_masks = add_padding_to_tensor(att_masks, self.config.prefix_length, 0)
        att_masks = att_masks.expand(bsize, -1)
                                    
        return embs, pad_masks, att_masks



    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = [] 

        action_emb = self.action_in_proj(noisy_actions)
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        device = action_emb.device

        time_emb = create_sinusodial_pos_embedding(
            timestep,
            self.expert_config.hidden_size,
            self.config.min_period,
            self.config.max_period,
            o_dtype=dtype,
            device=device
        )
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu (sigmoid linear unit)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        embs.append(action_time_emb)

        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.ones(self.config.chunk_size, dtype=embs.dtype, device=device).expand(bsize, -1)
        return embs, pad_masks, att_masks 

        

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs: Unpack[ActionSelectKwargs],) -> Tensor:
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = create_attention_mask(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # fill up the K/V cache of the VLM: every layer performs self-attention on the prefix only.
        # The prefix does not change during denoising, so this is done once per action chunk.
        past_key_values = {}
        self.forward_layers(
            input_embs=[prefix_embs, None],
            position_ids=prefix_position_ids,
            attn_mask=prefix_att_2d_masks,
            past_key_values=past_key_values,
            fill_kv_cache=True,
        )

        return self.euler_integration(noise, prefix_pad_masks, past_key_values)

    def project_input_embs(self, layer, input_embs):
        hidden_states = layer.input_layernorm(input_embs)
        hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)

        # (B, L, hidden) -> (B, L, num_heads, head_dim)
        hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
        queries = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        keys = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        values = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

        return queries, keys, values



    def concat_vlm_ae_tokens(self, layer_idx, model_layers, input_embs):
        query_states = []
        key_states = []
        value_states = []
        seq_lens = []  # number of tokens each stream contributes, used to split the attention output again

        for i, hidden_states in enumerate(input_embs):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                seq_lens.append(None)
                continue

            queries, keys, values = self.project_input_embs(layer, hidden_states)
            query_states.append(queries)
            key_states.append(keys)
            value_states.append(values)
            seq_lens.append(hidden_states.shape[1])

        if not query_states:
            return None, None, None, seq_lens

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        return query_states, key_states, value_states, seq_lens



    def self_attention(self, model_layers, layer_idx, input_embs, position_ids, attn_mask, past_key_values=None, fill_kv_cache=False):
        """
            Joint attention over the concatenated VLM and AE tokens. Used for
            - training:        input_embs = [prefix, suffix], no cache
            - filling the KV cache: input_embs = [prefix, None]
            - denoising:       input_embs = [None, suffix], the AE tokens attend to the cached prefix and themselves
        """
        queries, keys, values, seq_lens = self.concat_vlm_ae_tokens(layer_idx, model_layers, input_embs)
        if queries is None:
            return [None] * len(input_embs)

        # If the AE has no layer here, only the prefix tokens were projected
        attn_mask, position_ids = adapt_mask_and_position_length(attn_mask, position_ids, queries.shape[1])

        # RoPE encodes the token positions into Q/K (where to look), not into V (what is retrieved)
        queries = apply_rope(queries, position_ids)
        keys = apply_rope(keys, position_ids)

        if fill_kv_cache:
            # Save the (post-RoPE) prefix KV into the cache
            past_key_values[layer_idx] = (keys, values)
        elif past_key_values is not None:
            # Retrieve the prefix KV from the cache and concat them with the AE keys and values
            vlm_keys, vlm_values = past_key_values[layer_idx]
            keys = torch.cat([vlm_keys, keys], dim=1)
            values = torch.cat([vlm_values, values], dim=1)

        att_output = self.grouped_query_attention(queries, keys, values, attention_mask=attn_mask)

        # Split the attention output back into one output per stream
        att_outputs = []
        start = 0
        for seq_len in seq_lens:
            if seq_len is None:
                att_outputs.append(None)
                continue
            att_outputs.append(att_output[:, start:start + seq_len])
            start += seq_len
        return att_outputs



    def cross_attention(self, model_layers, layer_idx, input_embs, position_ids, attn_mask, past_key_values=None, fill_kv_cache=False):
        """
            The AE queries attend to the VLM keys/values of the same layer, projected by the AE k_proj/v_proj.
            Used for
            - training:  input_embs = [prefix, suffix], the VLM performs self-attention on the prefix
            - denoising: input_embs = [None, suffix], the VLM keys/values come from the cache
        """
        prefix_embs, suffix_embs = input_embs
        att_outputs = [None, None]

        # ================== VLM ==============================
        if prefix_embs is not None:
            # TRAINING: Perform self-attention for the VLM
            vlm_seq_len = prefix_embs.shape[1]
            prefix_attn_mask = attn_mask[:, :vlm_seq_len, :vlm_seq_len]
            prefix_position_ids = position_ids[:, :vlm_seq_len]
            expert_position_ids = position_ids[:, vlm_seq_len:]

            layer = model_layers[0][layer_idx]
            queries, keys, values = self.project_input_embs(layer, prefix_embs)
            queries = apply_rope(queries, prefix_position_ids)
            keys = apply_rope(keys, prefix_position_ids)

            att_outputs[0] = self.grouped_query_attention(
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=prefix_attn_mask
            )
        else:
            # INFERENCE: Retrieve KV
            keys, values = past_key_values[layer_idx]
            expert_position_ids = position_ids

        # ############## Action Expert =========================
        e_layer = model_layers[1][layer_idx]
        if e_layer is None or suffix_embs is None:
            return att_outputs

        e_hidden_states = e_layer.input_layernorm(suffix_embs)
        e_hidden_states = e_hidden_states.to(dtype=e_layer.self_attn.q_proj.weight.dtype)
        e_hidden_shape = (*e_hidden_states.shape[:-1], -1, e_layer.self_attn.head_dim)
        e_query_len = suffix_embs.shape[1]

        e_queries = e_layer.self_attn.q_proj(e_hidden_states).view(e_hidden_shape)

        # (B, L, num_kv_heads, head_dim) -> (B, L, num_kv_heads * head_dim)
        vlm_keys = keys.to(dtype=e_layer.self_attn.k_proj.weight.dtype).reshape(*keys.shape[:2], -1)
        vlm_values = values.to(dtype=e_layer.self_attn.v_proj.weight.dtype).reshape(*values.shape[:2], -1)

        # project vlm KV tokens to action experts dimension utilizing a linear layer
        e_shape = (*vlm_keys.shape[:2], -1, e_layer.self_attn.head_dim)
        _vlm_keys = e_layer.self_attn.k_proj(vlm_keys).view(e_shape)
        _vlm_values = e_layer.self_attn.v_proj(vlm_values).view(e_shape)

        # The VLM keys already carry their RoPE positions, the AE query positions start from 0
        expert_position_ids = expert_position_ids - torch.min(expert_position_ids, dim=1, keepdim=True).values
        e_queries = apply_rope(e_queries, expert_position_ids)

        suffix_attn_mask = attn_mask[:, -e_query_len:, :_vlm_keys.shape[1]]
        att_outputs[1] = self.grouped_query_attention(
            queries=e_queries,
            keys=_vlm_keys,
            values=_vlm_values,
            attention_mask=suffix_attn_mask
        )

        return att_outputs


    def post_attention_processing(self, layer, first_residual, att_output):
        assert first_residual is not None, "Input embeds stream does not exist during post attention processing!"
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output)
        out_emb += first_residual
        second_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        out_emb = layer.mlp(out_emb)
        out_emb += second_residual
        return out_emb



    def grouped_query_attention(self, queries, keys, values, attention_mask):
        # The AE uses the same attention heads as the VLM
        head_dim = self.text_config.head_dim
        num_att_heads = self.text_config.num_attention_heads
        num_kv_heads = self.text_config.num_key_value_heads
        num_kv_groups = num_att_heads // num_kv_heads
        big_neg = torch.finfo(torch.float32).min

        print("Peforming GQA: ")
        print(f"Queries shape: {queries.shape}")
        print(f"Keys shape: {keys.shape}")
        print(f"Values shape: {values.shape}")
        print("=" * 50, "\n")


        batch_size, seq_len = keys.shape[:2]

        # Repeat every KV head for its group of query heads
        keys = keys.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        keys = keys.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)
        values = values.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        values = values.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)

        queries = queries.to(dtype=torch.float32)
        keys = keys.to(dtype=torch.float32)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        att_weights = torch.matmul(queries, keys.transpose(2, 3)) * (head_dim**-0.5)
        att_weights = att_weights.to(dtype=torch.float32)
        logits = torch.where(attention_mask.unsqueeze(1), att_weights, big_neg)
        probs = nn.functional.softmax(logits, dim=-1)
        probs = probs.to(dtype=values.dtype)

        att_output = torch.matmul(probs, values.permute(0, 2, 1, 3))
        att_output = att_output.permute(0, 2, 1, 3)
        return att_output.reshape(batch_size, -1, num_kv_heads * num_kv_groups * head_dim)



    def forward_layers(self, input_embs, position_ids, attn_mask, past_key_values=None, fill_kv_cache=False):
        """
            Pass the [prefix, suffix] streams through all layers. A stream is None if it is not computed
            (the suffix while filling the KV cache, the prefix while denoising).
        """
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)

        for layer_idx in range(self.num_vlm_layers):
            # While filling the KV cache every layer performs self-attention on the prefix
            use_self_attn = (
                fill_kv_cache
                or "cross" not in self.config.attention_mode
                or (self.config.self_attn_every_n_layers > 0 and layer_idx % self.config.self_attn_every_n_layers == 0)
            )
            print(f"Performing self-attention: {use_self_attn}")
            attention = self.self_attention if use_self_attn else self.cross_attention
            att_outputs = attention(
                model_layers,
                layer_idx,
                input_embs,
                position_ids,
                attn_mask,
                past_key_values=past_key_values,
                fill_kv_cache=fill_kv_cache,
            )

            output_embs = []
            for i, hidden_states in enumerate(input_embs):
                layer = model_layers[i][layer_idx]
                if hidden_states is None or layer is None or att_outputs[i] is None:
                    output_embs.append(hidden_states) # pass hidden_states to the next layer
                else:
                    output_embs.append(self.post_attention_processing(
                        layer=layer,
                        first_residual=hidden_states,
                        att_output=att_outputs[i]
                    ))
            input_embs = output_embs

        # final layer normalization
        return [
            models[i].norm(hidden_states) if hidden_states is not None else None
            for i, hidden_states in enumerate(input_embs)
        ]



    """
        Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)
    """
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):

        if noise is None:
            noise = sample_noise(actions.shape, actions.device)

        if time is None:
            time = sample_time(actions.shape[0], actions.device)


        # interpolation is inverted compared to standard flow matching, but mathematically
        # equivalent and implied by Black et al. (2024)
        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        v_t_target = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            noisy_actions=x_t, timestep=time
        )

        padding_mask = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        attention_mask = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = create_attention_mask(padding_mask, attention_mask)
        position_ids = torch.cumsum(padding_mask, dim=1) - 1 # basic position ids, which skip padding, not documented in the paper

        _, suffix_out = self.forward_layers(
            input_embs=[prefix_embs, suffix_embs],
            position_ids=position_ids,
            attn_mask=att_2d_masks,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t_pred = self.action_out_proj(suffix_out)
        return F.mse_loss(v_t_target, v_t_pred, reduction="none")


    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape[:2]

        # The AE tokens may attend to every (non-padding) prefix token ...
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        # ... and among themselves according to their attention mask
        suffix_att_2d_masks = create_attention_mask(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        _, suffix_out = self.forward_layers(
            input_embs=[None, suffix_embs],
            position_ids=position_ids,
            attn_mask=full_att_2d_masks,
            past_key_values=past_key_values,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)



    def euler_integration(self, noise, prefix_pad_masks, past_key_values):
        # Integrate from t = 1 (pure noise) to t = 0 (clean actions)
        bsize = noise.shape[0]
        dt = -1.0 / self.config.num_steps
        x_t = noise

        for step in range(self.config.num_steps):
            time = 1.0 + step * dt
            time_t = torch.full((bsize,), time, dtype=torch.float32, device=noise.device)
            velocity = self.denoise_step(prefix_pad_masks, past_key_values, x_t, time_t)
            x_t = x_t + velocity * dt

        return x_t
