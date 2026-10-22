import torch 
import numpy as np 
from torch import nn, Tensor 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms
from typing import override 
import matplotlib.pyplot as plt 
from enum import Enum

from Transformer import Transformer
from Visualization import plot_loss


# ================================================================
# Version 1: CNN Architecture
# ================================================================
class FlowMatching(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.digit_embedding = nn.Embedding(10, 64)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )
        self.net = nn.Sequential(
            nn.Conv2d(1 + 64 + 64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )



    @override 
    def forward(self, x, t, digit):
        B, _, H, W = x.shape 
        
        digit_emb = self.digit_embedding(digit)

        # Embed digit: [B, 64] -> [B, 64, H, W]
        digit_emb = digit_emb[:, :, None, None]
        digit_emb = digit_emb.expand(-1, -1, H, W)

        # Embed time: [B] -> [B, 1] -> [B, 64]
        time_emb = self.time_embedding(t[:, None])

        # [B, 64] -> [B, 64, H, W]
        time_emb = time_emb[:, :, None, None].expand(-1, -1, H, W)

        # concatenate everything 
        x = torch.cat([x, digit_emb, time_emb], dim=1)

        return self.net(x)


# ================================================================
# version 2: transformer architecture
# ================================================================
class FlowMatching_with_Transformer(nn.Module):
    def __init__(self, config):

        self.config = config
        super().__init__() 
        self.num_layers = 8
        self.embedding_dim=64

        self.digit_embedding = nn.Embedding(
            num_embeddings=10,
            embedding_dim=self.embedding_dim
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=self.embedding_dim,
            kernel_size=4,
            stride=4,
        )

        if self.config["apply_rope"]:

           self.layers = nn.ModuleList([
                Transformer(
                    embedding_dim=self.embedding_dim,
                    n_heads=self.config["n_heads"],
                    ff_dim=self.config["ff_dim"],
                    config=self.config,
                )
                for _ in range(self.num_layers)
           ])

        else:
            self.pos_embedding = nn.Parameter(            
                torch.randn(1, 49, self.embedding_dim)
            )

            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.embedding_dim,
                    nhead=self.config["n_heads"],
                    dim_feedforward=self.config["ff_dim"],
                    batch_first=True,
                    dropout=0.0 
                )
                for _ in range(self.num_layers)
            ])

        self.output_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.SiLU(),
            nn.Linear(self.embedding_dim, 16),
        )


        if self.config["apply_AdaLN"]:
            self.adaLN1 = nn.Linear(
               self.embedding_dim, self.embedding_dim * 2 
            )
            self.adaLN2 = nn.Linear(
                self.embedding_dim, self.embedding_dim * 2
            )

    @override
    def forward(self, x, t, digit):

        digit_emb = self.digit_embedding(digit)
        digit_emb = digit_emb.unsqueeze(1)

        time_emb = self.time_embedding(t.unsqueeze(1)).unsqueeze(1)

        patch = self.patch_embedding(x)
        patch = patch.flatten(2)
        patch = patch.transpose(1, 2)

        if self.config["apply_rope"]:
            image_tokens = patch
        else:
            image_tokens = patch + self.pos_embedding


        if self.config["apply_AdaLN"]:
            condition = digit_emb + time_emb
            x = torch.cat([
                image_tokens,
                ], dim=1
            )
        else:
            condition = None
        
            x = torch.cat([
                digit_emb, 
                time_emb,
                image_tokens,
                ], dim=1
            )

        for layer in self.layers:
            if self.config["apply_AdaLN"]:
                x = layer(x, condition=condition)
            else:
                x = layer(x)

        if self.config is not None and not self.config["apply_AdaLN"]:
            # remove digit and time tokens again 
            image_tokens = x[:, 2:, :]

        # convert each token back into a 4x4 patch 
        patches = self.output_proj(image_tokens)
        patches = patches.transpose(1, 2) # back to [B, 16, 49]

        # reconstruct the velocity image 
        velocity = F.fold(
            patches, 
            output_size=(28, 28),
            kernel_size=4,
            stride=4,
        ) # [B, 1, 28, 28]

        return velocity

# ================================================================
# Training
# ================================================================

def create_loader(config) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    targets = dataset.targets
    samples_per_class = config["training_size"] // 10

    indices = []

    for i in range(10):
        class_indices = torch.where(targets == i)[0]

        # Randomly sample an equal number from each class
        perm = torch.randperm(len(class_indices))
        selected = class_indices[perm[:samples_per_class]]

        indices.append(selected)

    indices = torch.cat(indices)
    indices = indices[torch.randperm(len(indices))]

    dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )


@torch.no_grad()
def optimal_transport_match(x0, x1, digit, config):
    """
    OT-match Gaussian source samples x0 to target MNIST samples x1. 

    The corresponding digit labels are permuted together with the target images.
    """

    B = x0.shape[0]

    # flatten images
    x0_flat = x0.reshape(B, -1)
    x1_flat = x1.reshape(B, -1)

    # pairwise squared euclidean distances 
    cost = torch.cdist(x0_flat, x1_flat, p=2) ** 2 

    # solve assignment problem 
    _, col_indices = linear_sum_assignment(cost.cpu().numpy())

    col_indices = torch.as_tensor(col_indices, device=config["device"], dtype=torch.long)

    x1_matched = x1[col_indices]
    digit_matched = digit[col_indices]

    transport_cost = (x0 - x1_matched).pow(2).mean()

    return x0, x1_matched, digit_matched, transport_cost 


def train(model, loader, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    total_loss_list = []

    for _ in range(config["training_epochs"]):

        total_loss = 0 
        total_transport_cost = 0

        for x1, digit in loader:

            B = x1.shape[0]
            x1 = x1.to(config["device"])
            x0 = torch.randn_like(x1) # gausssian source distribution
            digit = digit.to(config["device"])

            if config["apply_optimal_transport"]:
                x0, x1, digit, transport_cost = optimal_transport_match(x0, x1, digit, config=config)
            else: 
                transport_cost = (x0 - x1).pow(2).mean()
            total_transport_cost += transport_cost.item()

            t = torch.rand(B, device=config["device"]) 

            t_img = t[:, None, None, None]
            xt = ((1 - t_img) * x0 + t_img * x1)

            target_velocity = x1 - x0
            predicted_velocity = model(xt, t, digit)

            loss = F.mse_loss(
                predicted_velocity,
                target_velocity,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            total_loss += loss.item() 

        total_loss_list.append(total_loss / len(loader))

    if config["plot_loss"]:
        plot_loss(total_loss_list)


@torch.no_grad()
def generate(model, digits, config):
    steps = config["denoising_steps"] if config is not None else 10
    do_print = config["print_generation_process"] if config is not None else False

    print(f"Generating digits with {config['denoising_steps']} denoising steps...")
    model.eval() 

    B = len(digits)
    x = torch.randn(
        B, 1, 28, 28,
        device=config["device"],
    )

    dt = 1.0 / steps

    if do_print:
        num_plots = min(10, steps)
        plot_steps = set(
            torch.linspace(0, steps - 1, num_plots)
            .round()
            .int()
            .tolist()
        )

        _, axes = plt.subplots(
            len(plot_steps), B,
            figsize=(B * 2, len(plot_steps) * 2),
            squeeze=False,
        )

        plot_idx = 0

    for i in range(steps):
        t = torch.full(
            (B,),
            i / steps,
            device=config["device"],
        )

        velocity = model(x, t, digits)
        x += dt * velocity

        if do_print and i in plot_steps:
            for j in range(B):
                axes[plot_idx, j].imshow(
                    x[j, 0].cpu(),
                    cmap="gray",
                )
                axes[plot_idx, j].axis("off")

                if plot_idx == 0:
                    axes[plot_idx, j].set_title(
                        str(digits[j].item())
                    )

            plot_idx += 1

    if do_print:
        plt.tight_layout()
        plt.show()

    return x