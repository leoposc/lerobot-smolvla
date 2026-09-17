import torch 
from torch import nn, Tensor 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import override 
import matplotlib.pyplot as plt 
from enum import Enum

class Mode(Enum):
    CNN = 1
    BasicTransformer = 2

MODE = Mode.CNN

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    def __init__(self):
        super().__init__() 

        self.embedding_dim=32

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

        self.pos_embedding = nn.Parameter(
            torch.randn(1, 49, self.embedding_dim)
        )

        self.num_layers = 4
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=4,
                dim_feedforward=128,
                batch_first=True
            )
            for _ in range(self.num_layers)
        ])

        self.output_proj = nn.Linear(
            self.embedding_dim, 
            4 * 4,
        )



    @override
    def forward(self, x, t, digit):

        digit_emb = self.digit_embedding(digit)
        digit_emb = digit_emb.unsqueeze(1)

        time_emb = self.time_embedding(t.unsqueeze(1)).unsqueeze(1)

        patch = self.patch_embedding(x)
        patch = patch.flatten(2)
        patch = patch.transpose(1, 2)

        image_tokens = patch + self.pos_embedding

        x = torch.cat([
            digit_emb, 
            time_emb,
            image_tokens,
            ], dim=1
        )

        for layer in self.layers:
            x = layer(x)


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

indices = torch.randperm(len(dataset))[:4096]
dataset = Subset(dataset, indices)

loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
)


match MODE:
    case Mode.CNN:
        model = FlowMatching().to(device)
    case Mode.BasicTransformer:
        model = FlowMatching_with_Transformer().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


for epoch in range(50):

    total_loss = 0 

    for x1, digit in loader:

        x1 = x1.to(device)
        digit = digit.to(device)

        B = x1.shape[0]
        x0 = torch.randn_like(x1) # gausssian source distribution

        t = torch.rand(B, device=device)

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

    print(
        f"Epoch {epoch + 1}: "
        f"loss = {total_loss / len(loader):.4f}"
    )

@torch.no_grad()
def generate(model, digits, steps=10):

    model.eval() 

    B = len(digits)

    x = torch.randn(
        B, 1, 28, 28,
        device=device,
    )

    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full(
            (B,),
            i / steps,
            device=device
        ) # Creates tensor of batch_size with the current time step

        velocity = model(
            x,
            t,
            digits,
        )

        x += dt * velocity

    return x 


digits = torch.arange(10, device=device)

samples = generate(
    model, digits, steps=50
)


fig, axes = plt.subplots(1, 10, figsize=(15, 2))

for i, ax in enumerate(axes):
    ax.imshow(
        samples[i, 0].cpu(),
        cmap="gray",
    )

    ax.set_title(str(i))
    ax.axis("off")

plt.show()
