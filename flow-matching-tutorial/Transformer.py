import torch
import math
from typing import override
from torch import nn

from utils import apply_rope

# ===============================================================
# Custom Attention Layer with RoPE
# ================================================================
class Attention(nn.Module):
    def __init__(self, embed_dim, n_heads):

        assert embed_dim & n_heads == 0, "embed_dim must be divisible by n_heads"
        
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @override
    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to q and k
        position_ids = torch.arange(T-2, device=x.device).unsqueeze(0).expand(B, T-2)  # Exclude the first two tokens (digit and time embeddings)
        q_content = apply_rope(q[:, :, 2:, :], position_ids)
        k_content = apply_rope(k[:, :, 2:, :], position_ids)

        q = torch.cat([q[:, :, :2, :], q_content], dim=2)
        k = torch.cat([k[:, :, :2, :], k_content], dim=2) 
        

        attn = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class Transformer(nn.Module):

    def __init__(self, embedding_dim=64, n_heads=4, ff_dim=128, config=None):
        super().__init__()
        self.config = config

        if config is not None and config["apply_AdaLN"] == True:
            do_elementwise_affine = False
        else:
            do_elementwise_affine = True

        self.norm1 = nn.LayerNorm(embedding_dim, elementwise_affine=do_elementwise_affine)
        self.attn = Attention(
            embedding_dim, n_heads
        )
        self.norm2 = nn.LayerNorm(embedding_dim, elementwise_affine=do_elementwise_affine)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.SiLU(),
            nn.Linear(ff_dim, embedding_dim)
        )

        if config is not None and config["apply_AdaLN"] == True:
            self.adaLN1 = nn.Linear(
                embedding_dim, embedding_dim * 2
            )
            self.adaLN2 = nn.Linear(
                embedding_dim, embedding_dim * 2
            )


    def forward(self, x, condition=None):
        if self.config is not None and self.config["apply_AdaLN"] == True:

            scale1, shift1 = self.adaLN1(condition).chunk(2, dim=-1)
            scale1 = scale1.unsqueeze(1)
            shift1 = shift1.unsqueeze(1)
            h = self.norm1(x)
            h = h * (1 + scale1) + shift1

            attn_out = self.attn(h)
            x = x + attn_out

            scale2, shift2 = self.adaLN2(condition).chunk(2, dim=-1)
            h = self.norm2(x)
            scale2 = scale2.unsqueeze(1)
            shift2 = shift2.unsqueeze(1)
            h = h * (1 + scale2) + shift2
            return x + self.ff(h)
            
        else:
            x = x + self.attn(self.norm1(x))
            return x + self.ff(self.norm2(x))

