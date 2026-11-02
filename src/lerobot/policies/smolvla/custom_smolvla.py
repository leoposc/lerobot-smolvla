from dataclasses import dataclass

import torch 
from torch import nn
import torch.nn.Functional as F
from dataclasses import dataclass 


from transformer import (
    AutoConfig, 
    AutoModel, 
    AutoModelForImageTextToText, 
    AutoProcessor, 
    DynamicCache, 
    SmolVLMForConditionalGeneration,
)

from custom_utils import * 


@dataclass 
class SmolVLAConfig():

    # VLM configuration 
    vlm_model_name = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    load_vlm_weights = True
    num_vlm_layers = 16
    vlm_dim = ...


    state_dim= 32 
    resize_img_to = (512, 512)

    # Action expert configuration 
    ae_dim = int(vlm_dim * 0.75)
    num_ae_layers = 16
    denoising_steps = 10
    action_dim = 32 
    action_chunk_size = 50





class SmolVLA():
    def __init__(self, config) -> None 

        self.config = config
        self.vlm_model = load_vlm_model() 

        self.ae_model = FlowMatching(config)

        self.vlm_dim = config.vlm_dim
        self.action_expert = FlowMatching(config)


        self.state_vlm_proj = nn.Linear(config["state_dim"], vlm_dim)


    def load_vlm_model(self):
        ...

    def forward(self):
        
        for layer_idx in config.num_vlm_layers:
            pass

        self.ae_model.forward()


    def prepare_batch(self):
        ...


    def prepare_images(self): 
        ... 



class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 
            torch_dtype="bfloat16",
            low_cpu_mem_usage=True
        )

        self.action_in_proj = nn.Linear(config.action_dim, config.ae_hidden_dim)
        self.action_out_proj = nn.Linear(config.ae_hidden_dim, config.action_dim)

        self.q_proj = nn.Linear(config.ae_hidden_dim, condig.ae_hidden_dim)
        self.v_proj = nn.Linear(condig.ae_hidden_dim, condig.ae_hidden_dim)
        self.k_proj = nn.Linear(condig.ae_hidden_dim, condig.ae_hidden_dim)

        self.state_proj = nn.Linear(
            config.state_dim, config.vlm_hidden_size
        )

        self.action_time_mlp = nn.Sequential(
            nn.Linear(config.ae_hidden_dim * 2, config.ae_hidden_dim),
            F.silu(),
            nn.Linear(config.ae_hidden_dim, config.ae_hidden_dim)
        )


    def get_vlm_model(self):
        return self.vlm.model


    def embed_language_tokens(self, tokens: torch.Tensor):
        self.get_vlm_model().text_model.get_input_embeddings()(tokens)
        ...


    def embed_image(self, image):
        patch_attention_mask = None
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=None),
            )
            .last_hidden_state
        )
        return self.get_vlm_model().connector(image_hidden_states)


    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state = None):
        embs = [] 
        pad_masks = []

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
            print(img.shape)
            bsize, num_img_embs, img_emb_dim = img_emb.shape[:2, -1]

            # Normalize image 
            img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            img_mask = img_mask.expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * num_img_embs

            # add image special tokens?
            
            # done with image embeddings 

        lang_emb = self.embed_language_tokens(lang_tokens)
        # normalize language embeddings 
        num_lang_emb, lang_emb_dim = lang_emb.shape[1, -1]
        lang_emb = lang_emb * (lang_emb_dim ** 0.5)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * num_lang_emb

        # add state embeddings 
        state_emb = self.state_proj(state)
        state_emb = state[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)

        # states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(1, 1, dtype=torch.bool, device=device).expand(bsize, 1)
        pad_masks.append(state_mask)

        att_masks += 1
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)

        seq_len = pad_masks.shape[1]
        if seq_len < self.config.prefix_length:
            embs = add_padding_to_tensor(embs, self.prefix_length, 0)
            pad_masks = add_padding_to_tensor(pad_masks, self.prefix_length, 0)
            att_masks = add_padding_to_tensor(att_masks, self.prefix_length, 0)
        att_masks = att_masks.expand(bsize, -1)
                                    
        return embs, pad_masks, att_masks



    def embed_suffix(self, noisy_actions, timestep):
        embs = []
        pad_masks = [] 

        action_emb = self.action_in_proj(nosiy_actions)
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype 

        time_emb = create_sinusoidal_pos_embedding(
            timestep, 
            self.config.ae_hidden_dim, 
            self.config.min_period, 
            self.config.max_period, 
            dtype=dtype,
            device=device
        )
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp(action_time_emb)
        action_time_dim = action_time_emb.shape[1]
        embs.append(action_time_emb)

        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=self.config.device)
        pad_masks.append(action_time_mask)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.ones(config.action_chunk_size, dtype=embs.dtype, device=config.device).expand(bsize, -1)
        return embs, pad_masks, att_masks 

        

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None):
        bsize = state.shape[0]
        
        if noise is None:
            noise = sample_noise(bsize, self.config.noise)     


        # fill up the K/V cache of the VLM 
        for layer_idx in range(self.config.num_vlm_layers):
            pass


        # interleave self and cross attention 
        for layer_idx in range(self.config.num_ae_layers):
            input_embeds = None

            if layer_idx % 2 == 0:
                self.self_attention()
            else:
                self.cross_attention()

            output_embeds = None

            input_embeds = output_embeds


    def attention(self, queries, values, keys):
        B, T, C = queries.shape
        queries = queries.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # apply rope here 
        # ... 

        attn = (
                q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        attn = torch.softmax(attn, dim=1)
        out = attn @ v 
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


    def self_attention(self):
        ...

    def cross_attention(self):
        ...


    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        ...


    def euler_integration(self):

        dt = 1.0 / config.denoising_steps
        x = sample_noise()

        for step in config.denoising_steps:
            t = torch.full(
                (self.bsize,),
                step / config.denoising_steps, 
                device=config.device,
                dtype=x.dtype
            )


            velocity = self.forward()
            x += velocity  * dt
