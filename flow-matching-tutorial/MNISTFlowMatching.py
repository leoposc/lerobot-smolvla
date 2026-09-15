import torch 
from torch import nn, Tensor 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import override 
import matplotlib.pyplot as plt 

device = "cuda" if torch.cuda.is_available() else "cpu"


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

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)

model = FlowMatching().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


for epoch in range(10):

    total_loss = 0 

    for x1, digit in loader:

        x1 = x1.to(device)
        digit = digit.to(device)

        B = x1.shape[0]
        x0 = torch.rand_like(x1)

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
def generate(model, digit, steps=10):

    model.eval() 

    B = len(digit)

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
        )

        velocity = model(
            x,
            t,
            digit,
        )

        x += dt * velocity

    return x 


digits = torch.arange(10, device=device)

samples = generate(
    model, digits, steps=20
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
