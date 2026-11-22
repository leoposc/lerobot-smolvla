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
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True
        )
        self.vlm_config = self.vlm.config
        self.text_config = copy.deepcopy(self.vlm_config.text_config)
        self.expert = AutoModel.from_config(self.text_config)
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)

        self.action_in_proj = nn.Linear(config.max_action_dim, self.text_config.hidden_size)
        self.action_out_proj = nn.Linear(self.text_config.hidden_size, config.max_action_dim)

        # self.q_proj = nn.Linear(self.text_config.ae_hidden_dim, self.text_config.ae_hidden_dim)
        # self.v_proj = nn.Linear(condig.ae_hidden_dim, self.text_config.hidden_size)
        # self.k_proj = nn.Linear(condig.ae_hidden_dim, self.text_config.hidden_size)

        self.state_proj = nn.Linear(
            config.max_state_dim, self.text_config.hidden_size
        )

        self.action_time_mlp_in = nn.Linear(
            self.text_config.hidden_size * 2, self.text_config.hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.text_config.hidden_size, self.text_config.hidden_size
        )


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
            self.text_config.hidden_size,
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

        past_key_values = DynamicCache()

        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        past_key_values = DynamicCache()
        bsize = state.shape[0]
        

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1 

        # fill up the K/V cache of the VLM 
        for layer_idx in range(self.config.num_vlm_layers):
            layer = model_layers[0][layer_idx]
            prefix_embs, past_key_values = self.self_attention(
                model_layer=layer,
                layer_idx=layer_idx,
                input_embeds=prefix_embs,
                position_ids=prefix_position_ids,
                attention_mask=prefix_att_masks,
                past_key_values=past_key_values,
                do_training=False
            )
            

        return self.euler_integration(
            noise,
            past_key_values,

        )

    def project_input_embs(self, layer, input_embs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.config.self_attn.head_dim)

        hidden_states = input_embs.to(dtype=layer.self_attn.q_proj.weight.layer)
        hidden_states = layer.input_layernorm(hidden_states)
        queries = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
        keys = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
        values = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

        return queries, keys, values
                                                                   


    def concat_vlm_ae_tokens(self, layer_idx, model_layers, input_embs):
        query_states = []
        key_states = []
        value_states = []

        for i, hidden_states in enumerate(input_embs):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue 

            queries, keys, values = self.project_input_embs(layer, hidden_states)
            query_states.append(queries)
            key_states.append(keys)
            value_states.append(values)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        # Check if we need to control the sequence length with the position id length? 

        return query_states, key_states, value_states



    def self_attention(self, model_layers, layer_idx, input_embeds, positions_ids, attn_mask, past_key_values, do_training):

        if do_training:
            queries, keys, values = self.concat_vlm_ae_tokens(
            layer_idx, model_layers, input_embeds
            )

            # Save the KV into the cache
            past_key_values.update(
                keys.transpose(1, 2),
                values,transpoes(1, 2), 
                layer_idx
            )
        else:
            # retrieve KV from past_key_values 
            vlm_keys = past_key_values.layers[layer_idx].keys.transpose(1, 2) 
            vlm_values = past_key_values.layers[layer_idx].values.transpose(1, 2)

            # project ae tokens 
            layer = model_layers[1][layer_idx]
            queries, keys, values = self.project_input_embs(layer, input_embeds)

            # concat keys and values 
            keys = torch.cat([vlm_keys, keys], dim=1)
            values = torch.cat([vlm_vlaues, values], dim=1)

        # apply rope here? 
            
        att_ouput = self.grouped_query_attention(
            queries,
            keys,
            values,
            attention_mask=attn_mask
        )
        return [att_ouput], past_key_values



            

    def cross_attention(self, model_layers, layer_idx, input_embeds, position_ids, attn_mask, past_key_values):
        att_output = []

        # ================== VLM ==============================
        if past_key_values is None:
            # TRAINING: Perform self-attention for the VLM 
            vlm_seq_len = input_embeds[0].shape[1]
            prefix_attn_mask = attention_mask[:, :vlm_seq_len, :vlm_seq_len]
            layer = model_layers[0][layer_idx]

            queries, keys, values = project_input_embs(layer, input_embeds[0])

            # apply RoPE here? 

            att_output.append(self.grouped_query_attention(
                queries=queries,
                keys=keys,
                values=values,
                attention_mask=preifx_attn_mask
            ))
        else:
            # INFERENCE: Retrieve KV 
            keys = past_key_values.layers[layer_idx].keys.transpose(1, 2)
            values = past_key_values.layers[layer_idx].values.transpose(1, 2)

            

        # ############## Action Expert =========================
        e_layer = model_layers[1][layer_idx]
        e_hidden_states = e_layer.input_layernorm(input_embeds[1])
        dtype = layer.self_attn.k_proj.weight.dtype 
        e_hidden_states = e_hidden_states.to(dtype=dtype)
        e_input_shape = e_hidden_states.shape[:-1]
        e_hidden_shape = (*e_input_shape, -1, e_layer.self.self_attn.head_dim)
        e_query_len = input_embeds[1].shape[1]
        
        e_queries = e_layer.self_attn.q_proj(e_hidden_states).view(e_hidden_shape)

        vlm_keys = keys.to(dtype=dtype).reshape(*keys.shape[:2], -1)
        vlm_values = values.to(dtype=dtype).reshape(*values.shape[:2], -1)

        # project vlm KV tokens to action experts dimension utilizing a linear layer
        e_shape = (*vlm_keys.shape[:2], -1)
        _vlm_keys = e_layer.self_attn.k_proj(vlm_keys).view(e_shape)
        _vlm_values = e_layer.self_attn.v_proj(vlm_values).view(e_shape)

        # apply rope to ae queries? 


        suffix_attn_mask = attention_mask[:, -e_query_len:, -_vlm_keys.shape[1]:]
        att_output.append(grouped_query_attention(
            queries=e_queries,
            keys=_vlm_keys,
            values=_vlm_values,
            attention_mask=suffix_attn_mask                
        ))

        return att_output


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

        head_dim = self.config.head_dim 
        num_att_heads = self.config.num_att_heads
        num_kv_heads = self.config.num_kv_heads
        num_kv_groups = num_att_heads // num_kv_heads
        big_neg = torch.finfo(torch.float32).min 


        batch_size, seq_len = keys.shape[:2]

        keys = keys.unsqueeze(3).expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        keys = keys.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)
        values = values.unsequeeze(3).expand(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim)
        values = values.reshape(batch_size, seq_len, num_kv_heads * num_kv_groups, head_dim)

        queries = queries.to(dtype=torch.float32)
        keys = keys.to(dtype=torch.float32)
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)

        att_weights = torch.matmul(queries, keys,transpose(2,3)) * (head_dim**-0.5)
        att_weights = att_weights.to(dtype=torch.float32)
        logits = torch.where(attention_mask.unsqueeze(1), att_weights, big_neg)
        probs = nn.functional.softmax(logits, dim=-1)
        probs = probs.to(dtype=values.dtype)
        
        att_output = torch.matmul(probs, values.permute(0, 2, 1, 3))
        att_output = att_ouput.permute(0, 2, 1, 3)
        return att_output.reshape(batch_size, -1, num_kv_heads * num_kv_groups * head_dim)
        



    """
        Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)
    """
    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):


        if noise is None:
            noise = sample_noise(actions.shape, self.config.device)

        if time is None:
            time = sample_time(action.shape[0], self.config.device)


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

        att_pad_masks = create_attention_mask(padding_mask, attention_mask) 
        position_ids = torch.cumsum(pad_masks, dim=1) - 1 # basic position ids, which skip padding, not documented in the paper 

        # During training perform only self-attention by 
        # - projecting each tokens for VLM and AE separately
        # - concatenate the VLM QKV with the AE QKV 
        # - compute attention on concatenated QKV 
        models = [self.get_vlm_model().text_model, self.expert]
        model_layers = self.get_model_layers(models)
        input_embs = [prefix_embs, suffix_embs]

        for layer_idx in range(self.num_vlm_layers):
            
            output_embeds = []
            attention_mask_, position_ids_ = adapt_mask_and_position_length(attention_mask, position_ids, query_states.shape[1])

            if layer_idx % self.config.self_attn_every_n_layers == 0:

                att_outputs, _ = self.self_attention(
                    model_layers,
                    layer_idx, 
                    input_embs,
                    position_ids,
                    attention_mask,
                    past_key_values=None,
                    do_training=True
                )

                for i, hidden_states in enumerate(input_embs):
                    layer = model_layers[i][layer_idx]
                    if layer is None:
                        output_embeds.append(hidden_states) # pass hidden_states to the next layer 
                    else:
                        output_embeds.append(self.post_attention_processing(
                            layer,
                            hidden_states,
                            att_outputs[0]
                        ))

            else: 
                att_outputs = self.cross_attention(
                    model_layers, 
                    layer_idx,
                    input_embs,
                    positions_ids,
                    attention_mask,
                    past_key_values
                )

                for i, hidden_states in enumerate(input_embs):
                    layer = model_layers[i][layer_idx]
                    if layer is None:
                        output_embeds.append(hidden_states)
                    else:
                        output_embeds.append(self.post_attention_processing(
                            layer=layer,
                            first_residual=hidden_states,
                            att_output=att_outputs[i]
                        ))

            input_embs = output_embeds

        suffix_out = suffix_out[:, -self.config.chunk_size:] # remove the state tokens

        v_t_pred = self.action_out_proj(suffix_out)
        return F.mse_loss(v_t_target, v_t_pred, reduction="none")
        

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        
        suffix_embs, suffix_pad_masks, suffix_masks = self.embed_suffix(x_t, timestep)
        suffix_len =  suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape[:2]
        prefix_pad_2d_masks = prefix_pad_masks.expand(bsize, suffix_len, prefix_len)

        suffix_att_2d_masks = create_attention_mask(suffis_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dime=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)

        for layer_idx in range(self.config.num_vlm_layers):
            
            layer = model_layers[1][layer_idx]
            if layer_idx % self.config.every_n_layer_self_attn == 0:
                att_outputs, past_key_values = self.self_attention(
                    model_layer=layer,
                    layer_idx=layer_idx, 
                    input_embeds=suffix_embs,
                    position_ids=positions_ids,
                    attn_mask=full_att_2d_masks,
                    past_key_values=past_key_values,
                    do_training=False
                )
            else: 
                att_outputs, past_key_values = self.cross_attention(
                    model_layer=layer, 
                    layer_idx=layer_idx,
                    input_embeds=suffix_embs,
                    position_ids=position_ids,
                    attn_mask=full_att_2d_masks,
                    past_key_values=past_key_values
                )

            # cross_attention returns [vlm_att_out, ae_att_out] and self attention [att_out]
            # During inference only post process the last element inside the list
            att_out = att_outputs[-1]
            out_emb = self.post_attention_processing(
                layer=layer,
                first_residual=suffix_embs,
                att_output=att_out
            )
            suffix_embs = out_emb

        # Perform final layer normalization
        return self.lm_expert.norm(suffix_embs)
        




    def euler_integration(self):

        x_t = sample_noise(self.bsize, self.config.noise)     
        dt = -1.0 / config.denoising_steps

        for step in config.denoising_steps:
            time = 1.0 + step * dt
            time_t = torch.tensor(
                time,
                dtype=torch.float32,
                device=self.config.device
            ).expand(self.bsize)
            velocity = self.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
            x += velocity  * dt

        return x
