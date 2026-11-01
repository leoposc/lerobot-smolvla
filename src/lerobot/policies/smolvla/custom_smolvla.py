from dataclasses import dataclass

import torch 
from torch import nn
from dataclasses import dataclass 

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
        ae_dim = config.action_dim
        vlm_dim = config.vlm_dim

        self.in_proj = nn.Linear(vlm_dim, ae_dim)
        self.q_proj = nn.Linear(ae_dim, ae_dim)
        self.v_proj = nn.Linear(ae_dim, ae_dim)
        self.k_proj = nn.Linear(ae_dim, ae_dim)


    def forward(self)

        for layer_idx in range(self.config.num_ae_layers):
        if layer_idx % 2 == 0:
            self.self_attention()
        else:
            self.cross_attention()


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


    def train(self):
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
