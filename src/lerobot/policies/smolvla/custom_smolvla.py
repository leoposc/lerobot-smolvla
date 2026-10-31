import torch 
from torch import nn


class SmolVLA():
    def __init__(self, vlm_model, config) -> None 

        self.config = config
        self.vlm_model = vlm_model 
        self.action_expert = FlowMatching(config)


    def forward(self):
        ...


    def prepare_batch(self):
        ...


    def prepare_images(self): 
        ... 



class FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


    def forward(self): 
        ... 

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


    def sample_time()
        return torch.rand(self.config["batch_size"], device=config["device"])
