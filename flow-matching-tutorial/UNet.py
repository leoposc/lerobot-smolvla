import torch 
import torch.nn as nn
import torch.nn.functional as F

class UNet(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        D = hidden_dim 

        self.conv1 = nn.Conv2d(1, D, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Conv2d(D, D, kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(D, 2 * D, kernel_size=2, stride=2, padding=0)

        self.mid = nn.Conv2d(2 * D, 2* D, kernel_size=3, stride=1, padding=1)

        # Decoder 
        self.up2 = nn.ConvTranspose2d(2 * D, D, kernel_size=2, stride=2, padding=0)
        self.fuse2 = nn.Conv2d(2 * D, D, 3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose2d(D, D, kernel_size=2, stride=2, padding=0)
        self.fuse1 = nn.Conv2d(2 * D, D, kernel_size=3, stride=1, padding=1)

        self.out = nn.Conv2d(D, 1, kernel_size=3, stride=1, padding=1) # predict one output value per pixel 


    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28) 

        skip1 = F.relu(self.conv1(x))
        x = self.down1(skip1)
        skip2 = F.relu(self.conv2(x))
        x = self.down2(skip2)

        x = F.relu(self.mid(x))

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.fuse2(x))

        x = self.up1(x)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.fuse1(x))

        return self.out(x)


class ConditionalUNet(torch.nn.Module):
    def __init__(self, hidden_dim: int): 
        super().__init__()
        D = hidden_dim 

        self.conv1 = nn.Conv2d(1, D, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Conv2d(D, D, kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1)
        self.down2 = nn.Conv2d(D, 2 * d, kernel_size=2, stride=2, padding=0)

        self.mid = nn.Conv2d(2 * D, 2 * D, kernel_size=3, stride=1, padding=1) 

        self.up2 = nn.ConvTranspose2d(2 * D, D, kernel_size=2, stride=2, padding=0)
        self.fuse2 = nn.Conv2d(2 * D, D, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.ConvTranspose2d(D, D, kernel_size=2, stride=2, padding=0)
        self.fuse1 = nn.Conv2d(2 * D, D, kernel_size=3, stride=1, padding=1)

        self.fcblock1_t = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(), 
            nn.Linear(D, D)
        )
        self.fcblock2_t = nn.Sequential(
            nn.Linear(D, D), 
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.fcblock1_c = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.fcblock2_c = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )

        self.out = nn.Conv2d(D, 1, kernel_size=3, stride=1, padding=1)

        


