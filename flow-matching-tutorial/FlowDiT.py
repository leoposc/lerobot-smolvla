"""
FlowDiT: a DiT-style flow matching transformer for class-conditional MNIST.

Design (and why):
- Conv stem + patches: the stem gives every token local edge features before attention.
  Default 4x4 patches (49 tokens) because they train ~5x faster than 2x2 patches (196 tokens)
  on a GTX 1060; the conv head below removes the patch seams. On a faster GPU, try patch_size=2.
- Fixed 2D sin-cos positions: tokens know their row and column, nothing to learn.
- Sinusoidal time embedding + class embedding, summed into one condition vector.
- adaLN-Zero blocks (DiT): the condition sets scale/shift/gate of every norm and residual.
  All gates start at zero, so every block starts as the identity -> very stable training.
- QK-norm in attention: keeps attention logits bounded at higher learning rates.
- Modulated final layer + a small residual conv head at full resolution that smooths
  patch seams. Output layers are zero-initialised, so the model starts predicting v = 0.
- Classifier-free guidance: labels are dropped 10% of the time during training
  (replaced by a "null" class) and mixed at sampling time.

Training: AdamW, warmup + cosine LR, grad clipping, EMA weights, logit-normal t sampling.
Sampling: Heun (2nd order) ODE solver with classifier-free guidance.

Convention (same as FlowMatching.py): x_t = (1 - t) * x0 + t * x1, x0 ~ N(0, I) at t = 0,
data at t = 1, target velocity v = x1 - x0.

Run from this folder:  python FlowDiT.py --steps 20000
"""

import argparse
import copy
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# Configs
# ================================================================
@dataclass
class ModelConfig:
    img_size: int = 28
    patch_size: int = 4  # 2 gives finer detail but is ~5x slower
    stem_channels: int = 32
    dim: int = 192
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    num_classes: int = 10
    class_dropout: float = 0.1  # probability of training without the label (for CFG)


@dataclass
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 128
    lr: float = 3e-4
    warmup_steps: int = 500
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.9995
    log_every: int = 500
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# Embeddings
# ================================================================
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal features of t in [0, 1]. Scaled by 1000 so the frequencies cover the range well."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sincos_pos_embedding_2d(dim: int, grid: int) -> torch.Tensor:
    """Fixed 2D sin-cos positions, shape (grid * grid, dim). Half the channels encode the row, half the column."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2D sin-cos positions"
    rows, cols = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")
    omega = 1.0 / 10_000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    parts = []
    for pos in (rows.flatten(), cols.flatten()):
        angles = pos[:, None] * omega[None]
        parts += [angles.sin(), angles.cos()]
    return torch.cat(parts, dim=1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


# ================================================================
# Transformer blocks
# ================================================================
class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    """Pre-norm transformer block with adaLN-Zero conditioning."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        # condition -> shift/scale/gate for attention and MLP
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate1[:, None] * self.attn(modulate(self.norm1(x), shift1, scale1))
        x = x + gate2[:, None] * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x


class FinalLayer(nn.Module):
    """Modulated LayerNorm + linear map from each token to its p x p pixel patch."""

    def __init__(self, dim: int, patch_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, patch_size * patch_size)
        for layer in (self.adaLN[-1], self.linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


# ================================================================
# Model
# ================================================================
class FlowDiT(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        assert cfg.img_size % cfg.patch_size == 0
        self.grid = cfg.img_size // cfg.patch_size
        self.null_class = cfg.num_classes  # extra label index meaning "no class"
        C = cfg.stem_channels

        # Local features at full resolution, then patchify
        self.stem = nn.Sequential(
            nn.Conv2d(1, C, 3, padding=1), nn.SiLU(),
            nn.Conv2d(C, C, 3, padding=1), nn.SiLU(),
        )
        self.patchify = nn.Conv2d(C, cfg.dim, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.register_buffer("pos", sincos_pos_embedding_2d(cfg.dim, self.grid)[None], persistent=False)

        # Condition = time + class
        self.t_freq_dim = 256
        self.t_embed = nn.Sequential(nn.Linear(self.t_freq_dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.y_embed = nn.Embedding(cfg.num_classes + 1, cfg.dim)

        self.blocks = nn.ModuleList([DiTBlock(cfg.dim, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.depth)])
        self.final = FinalLayer(cfg.dim, cfg.patch_size)

        # Residual conv head on the full-resolution output: blends neighbouring patches.
        # Sees the stem features, so it has pixel-level detail of x_t.
        self.refine = nn.Sequential(
            nn.Conv2d(1 + C, C, 3, padding=1), nn.SiLU(),
            nn.Conv2d(C, 1, 3, padding=1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight.abs().sum() > 0:  # skip zero-initialised layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.y_embed.weight, std=0.02)
        nn.init.normal_(self.t_embed[0].weight, std=0.02)
        nn.init.normal_(self.t_embed[2].weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 28, 28) noisy image, t: (B,) in [0, 1], y: (B,) labels (num_classes = no label)."""
        p = self.cfg.patch_size
        feats = self.stem(x)                                          # (B, C, 28, 28)
        tokens = self.patchify(feats).flatten(2).transpose(1, 2)      # (B, N, dim)
        tokens = tokens + self.pos

        c = self.t_embed(timestep_embedding(t, self.t_freq_dim)) + self.y_embed(y.long())  # (B, dim)

        for block in self.blocks:
            tokens = block(tokens, c)

        patches = self.final(tokens, c)                               # (B, N, p*p)
        v = F.fold(patches.transpose(1, 2), output_size=(self.cfg.img_size,) * 2, kernel_size=p, stride=p)
        return v + self.refine(torch.cat([v, feats], dim=1))          # (B, 1, 28, 28)


# ================================================================
# Data, loss, training
# ================================================================
def load_mnist(root: str = "./data", device: str = "cpu", train: bool = True):
    """Whole MNIST split as tensors in [-1, 1] on `device` (fast: no DataLoader needed)."""
    from torchvision import datasets
    ds = datasets.MNIST(root, train=train, download=True)
    X = (ds.data.float() / 255.0 * 2 - 1).unsqueeze(1).to(device)
    Y = ds.targets.to(device)
    return X, Y


def sample_t(batch_size: int, device) -> torch.Tensor:
    """Logit-normal t (SD3): more training signal at mid t, where the velocity is hardest to predict."""
    return torch.sigmoid(torch.randn(batch_size, device=device))


def flow_matching_loss(model: FlowDiT, x1: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = sample_t(B, x1.device)
    ti = t[:, None, None, None]
    xt = (1 - ti) * x0 + ti * x1

    # Classifier-free guidance training: sometimes hide the label
    drop = torch.rand(B, device=x1.device) < model.cfg.class_dropout
    y = torch.where(drop, torch.full_like(y, model.null_class), y)

    return F.mse_loss(model(xt, t, y), x1 - x0)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.lerp_(p, 1 - decay)


def lr_lambda(step: int, tcfg: TrainConfig) -> float:
    if step < tcfg.warmup_steps:
        return (step + 1) / tcfg.warmup_steps
    progress = (step - tcfg.warmup_steps) / max(1, tcfg.steps - tcfg.warmup_steps)
    return tcfg.min_lr_ratio + (1 - tcfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def train(model: FlowDiT, X: torch.Tensor, Y: torch.Tensor, tcfg: TrainConfig, on_log=None):
    """Train with random mini-batches from the in-memory dataset. Returns (ema_model, loss_history)."""
    torch.manual_seed(tcfg.seed)
    model.train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, betas=(0.9, 0.99), weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, tcfg))

    history, running, t0 = [], 0.0, time.time()
    for step in range(1, tcfg.steps + 1):
        idx = torch.randint(0, len(X), (tcfg.batch_size,), device=X.device)
        loss = flow_matching_loss(model, X[idx], Y[idx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        sched.step()
        # EMA warm-up: average over fewer steps early on, so short runs aren't dominated by the random init
        update_ema(ema, model, min(tcfg.ema_decay, (1 + step) / (10 + step)))

        running += loss.item()
        if step % tcfg.log_every == 0:
            avg = running / tcfg.log_every
            history.append(avg)
            running = 0.0
            print(f"step {step:6d} | loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
            if on_log is not None:
                on_log(step, model, ema)
    return ema, history


# ================================================================
# Sampling
# ================================================================
@torch.no_grad()
def sample(model: FlowDiT, labels: torch.Tensor, steps: int = 32, cfg_scale: float = 2.0,
           method: str = "heun", generator: torch.Generator | None = None) -> torch.Tensor:
    """Integrate dx/dt = v(x, t) from noise (t=0) to data (t=1). Returns images in [-1, 1]."""
    model.eval()
    B, device = len(labels), labels.device
    size = model.cfg.img_size
    x = torch.randn(B, 1, size, size, device=device, generator=generator)
    null = torch.full_like(labels, model.null_class)

    def velocity(x, t):
        tt = torch.full((B,), t, device=device)
        if cfg_scale == 1.0:
            return model(x, tt, labels)
        v_cond, v_uncond = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([labels, null])).chunk(2)
        return v_uncond + cfg_scale * (v_cond - v_uncond)

    ts = torch.linspace(0, 1, steps + 1).tolist()
    for t_now, t_next in zip(ts[:-1], ts[1:]):
        dt = t_next - t_now
        d = velocity(x, t_now)
        if method == "heun":
            d_next = velocity(x + dt * d, t_next)
            x = x + dt * 0.5 * (d + d_next)
        else:  # euler
            x = x + dt * d
    return x.clamp(-1, 1)


def save_grid(images: torch.Tensor, path: str, nrow: int = 10, title: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = math.ceil(len(images) / nrow)
    fig, axes = plt.subplots(rows, nrow, figsize=(nrow * 0.9, rows * 0.9))
    for ax, img in zip(axes.flat, images.cpu()):
        ax.imshow(img[0], cmap="gray", vmin=-1, vmax=1)
        ax.axis("off")
    for ax in list(axes.flat)[len(images):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def save_checkpoint(path: str, model: FlowDiT, ema: FlowDiT, mcfg: ModelConfig, tcfg: TrainConfig):
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "model_config": asdict(mcfg), "train_config": asdict(tcfg)}, path)


def load_checkpoint(path: str, device: str = "cpu", use_ema: bool = True) -> FlowDiT:
    ckpt = torch.load(path, map_location=device)
    model = FlowDiT(ModelConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    return model.eval()


# ================================================================
# Script
# ================================================================
if __name__ == "__main__":
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--out", default="./output/flowdit")
    parser.add_argument("--ckpt", default="./trained_models/flowdit.pt")
    args = parser.parse_args()

    mcfg = ModelConfig()
    tcfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                       log_every=min(TrainConfig.log_every, args.steps))
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    X, Y = load_mnist(device=tcfg.device)
    model = FlowDiT(mcfg).to(tcfg.device)
    print(f"FlowDiT: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, device {tcfg.device}")

    labels = torch.arange(10, device=tcfg.device).repeat(4)  # 4 rows of digits 0-9

    def on_log(step, model, ema):
        if step % (tcfg.log_every * 4) == 0 or step == tcfg.steps:
            g = torch.Generator(device=tcfg.device).manual_seed(0)
            save_grid(sample(ema, labels, generator=g), f"{args.out}/step_{step:06d}.png", title=f"EMA, step {step}")
            model.train()

    ema, history = train(model, X, Y, tcfg, on_log=on_log)
    save_checkpoint(args.ckpt, model, ema, mcfg, tcfg)

    g = torch.Generator(device=tcfg.device).manual_seed(0)
    for scale in (1.0, 2.0, 4.0):
        save_grid(sample(ema, labels, cfg_scale=scale, generator=g), f"{args.out}/final_cfg{scale}.png",
                  title=f"EMA, {tcfg.steps} steps, CFG {scale}")
    print(f"Saved checkpoint to {args.ckpt} and samples to {args.out}/")
