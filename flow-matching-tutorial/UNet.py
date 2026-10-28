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


class ConditionalUNet(nn.Module):
    def __init__(self, config, num_classes: int = 10):
        super().__init__()
        D = config["embedding_dim"]

        self.conv1 = nn.Conv2d(1, D, 3, stride=1, padding=1)
        self.down1 = nn.Conv2d(D, D, 2, stride=2)

        self.conv2 = nn.Conv2d(D, D, 3, padding=1)
        self.down2 = nn.Conv2d(D, 2 * D, 2, stride=2)

        self.mid = nn.Conv2d(2 * D, 2 * D, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(2 * D, D, 2, stride=2)
        self.fuse2 = nn.Conv2d(2 * D, D, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse1 = nn.Conv2d(2 * D, D, 3, padding=1)

        # Embed scalar time into feature dimensions
        self.time_embed = nn.Sequential(
            nn.Linear(1, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )
        self.time_mid = nn.Linear(D, 2 * D)
        self.time_dec = nn.Linear(D, D)

        # Embed class labels into feature dimensions
        self.class_embed = nn.Embedding(num_classes, D)
        self.class_mid = nn.Linear(D, 2 * D)
        self.class_dec = nn.Linear(D, D)

        self.out = nn.Conv2d(D, 1, 3, padding=1)

    def forward(self, x, t, c):
        x = x.reshape(-1, 1, 28, 28)
        B = x.shape[0]

        # Make t shape (B, 1) and c shape (B,)
        t = t.reshape(B, 1).to(dtype=x.dtype)
        c = c.reshape(B).long()

        t_emb = self.time_embed(t)       # (B, D)
        c_emb = self.class_embed(c)      # (B, D)

        # Encoder
        skip1 = F.relu(self.conv1(x))    # (B, D, 28, 28)
        x = self.down1(skip1)            # (B, D, 14, 14)
        skip2 = F.relu(self.conv2(x))    # (B, D, 14, 14)
        x = self.down2(skip2)            # (B, 2D, 7, 7)
        x = F.relu(self.mid(x))          # (B, 2D, 7, 7)

        # Condition the bottleneck
        mid_cond = self.time_mid(t_emb) + self.class_mid(c_emb)
        x = x * (1 + mid_cond[:, :, None, None])

        # Decoder at 14 x 14
        x = self.up2(x)                  # (B, D, 14, 14)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.fuse2(x))        # (B, D, 14, 14)

        dec_cond = self.time_dec(t_emb) + self.class_dec(c_emb)
        x = x * (1 + dec_cond[:, :, None, None])

        # Decoder at 28 x 28
        x = self.up1(x)                  # (B, D, 28, 28)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.fuse1(x))        # (B, D, 28, 28)

        return self.out(x)               # (B, 1, 28, 28)



class ConditionalUNetV2(nn.Module):
    def __init__(self, config, num_classes: int = 10):
        super().__init__()
        D = config["embedding_dim"]

        self.conv1 = nn.Conv2d(1, D, 3, stride=1, padding=1)
        self.down1 = nn.Conv2d(D, D, 2, stride=2)
        self.down2 = nn.Conv2d(D, 2 * D, 2, stride=2)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (2 * D, 7, 7))

        self.up2 = nn.ConvTranspose2d(4 * D, D, 2, stride=2)
        self.up1 = nn.ConvTranpose2d(2 * D, D, 2, stride=2)

        # Embed sclar time into feature dimensions 
        self.time_embed = nn.Sequential(
            nn.Linear(1, D),
            nn.ReLU(),
            nn.Linear(D, D)
        )
        self.time_mid = nn.Linear(D, 2 * D)

        # Embed class labels into feature dimensions
        self.class_embed = nn.Embedding(num_classes, D)
        self.class_mid = nn.Linear(D, 2 * D)

        self.conv_dec = nn.Conv2d(2 * D, D, 3, stride=1, padding=1)
        self.out = nn.Conv2d(D, 1, 3, stride=1, padding=1)


    def forward(self, x, t, c):
        x = x.reshape(-1, 1, 28, 28)
        B = x.shape[0]

        # Make t shape (B, 1) and c shape (B,)
        t = t.reshape(B, 1).to(dtype=x.dtype)
        c = c.reshape(B).long() 

        t_emb = self.time_embed(t)
        c_emb = self.class_embed(c)

        # Encoder
        skip1 = F.relu(self.conv1(x))
        skip2 = F.relu(self.down1(skip1))
        skip3 = F.relu(self.down2(skip2))

        x = self.flatten(skip3)
        x = self.unflatten(x)

        # condition the bottleneck 
        mid_cond = self.time_mid(t_emb) + self.class_mid(c_emb)
        x = x * (1 + mid_cond[:, :, None, None])

        # concatenate skip connections
        x = torch.cat([x, skip3], dim=1)
        x = F.relu(self.up2(x))

        # condition the second decoder layer 
        dec_cond = self.time_embed(t_emb) + self.class_embed(c_emb)
        x = x * (1 + dec_cond[:, :, None, None])

        # concatenate skip connections
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.up1(x))

        # concate skip connections
        x = torch.cat([x, skip1], dim=1)

        

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


class ConditionalUNet(nn.Module):
    def __init__(self, config, num_classes: int = 10):
        super().__init__()
        D = config["embedding_dim"]

        self.conv1 = nn.Conv2d(1, D, 3, stride=1, padding=1)
        self.down1 = nn.Conv2d(D, D, 2, stride=2)

        self.conv2 = nn.Conv2d(D, D, 3, padding=1)
        self.down2 = nn.Conv2d(D, 2 * D, 2, stride=2)

        self.mid = nn.Conv2d(2 * D, 2 * D, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(2 * D, D, 2, stride=2)
        self.fuse2 = nn.Conv2d(2 * D, D, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(D, D, 2, stride=2)
        self.fuse1 = nn.Conv2d(2 * D, D, 3, padding=1)

        # Embed scalar time into feature dimensions
        self.time_embed = nn.Sequential(
            nn.Linear(1, D),
            nn.ReLU(),
            nn.Linear(D, D),
        )
        self.time_mid = nn.Linear(D, 2 * D)
        self.time_dec = nn.Linear(D, D)

        # Embed class labels into feature dimensions
        self.class_embed = nn.Embedding(num_classes, D)
        self.class_mid = nn.Linear(D, 2 * D)
        self.class_dec = nn.Linear(D, D)

        self.out = nn.Conv2d(D, 1, 3, padding=1)

    def forward(self, x, t, c):
        x = x.reshape(-1, 1, 28, 28)
        B = x.shape[0]

        # Make t shape (B, 1) and c shape (B,)
        t = t.reshape(B, 1).to(dtype=x.dtype)
        c = c.reshape(B).long()

        t_emb = self.time_embed(t)       # (B, D)
        c_emb = self.class_embed(c)      # (B, D)

        # Encoder
        skip1 = F.relu(self.conv1(x))    # (B, D, 28, 28)
        x = self.down1(skip1)            # (B, D, 14, 14)
        skip2 = F.relu(self.conv2(x))    # (B, D, 14, 14)
        x = self.down2(skip2)            # (B, 2D, 7, 7)
        x = F.relu(self.mid(x))          # (B, 2D, 7, 7)

        # Condition the bottleneck
        mid_cond = self.time_mid(t_emb) + self.class_mid(c_emb)
        x = x * (1 + mid_cond[:, :, None, None])

        # Decoder at 14 x 14
        x = self.up2(x)                  # (B, D, 14, 14)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.fuse2(x))        # (B, D, 14, 14)

        dec_cond = self.time_dec(t_emb) + self.class_dec(c_emb)
        x = x * (1 + dec_cond[:, :, None, None])

        # Decoder at 28 x 28
        x = self.up1(x)                  # (B, D, 28, 28)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.fuse1(x))        # (B, D, 28, 28)

        return self.out(x)               # (B, 1, 28, 28)





        

