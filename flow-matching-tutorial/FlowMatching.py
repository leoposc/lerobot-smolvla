import torch 
from torch import nn, Tensor 
from typing import override  
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        print("Initiallizing Flow class")
        super().__init__()
        # Interleaved Linear and Exponential Linear Unit layers
        # ELU is described as 
        # x, if x > 0 
        # alpha(=1.0) * (exp(x) - 1), if x <= 0
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))

    @override
    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        # Concatenate t and x_t along the last dimension and feed it inside the network
        return self.net(torch.cat((t, x_t), -1))

    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        # expand the shape of the time vector to the shape of x_t
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)

        # self(x_tm t_start): "Given that I'm at position/ state x_t at time t_start, what direction/ velocity should I move in?"
        x_mid = x_t + self(x_t, t_start) * (t_end + t_start) / 2 
        # This says: If I followed the velocity at the beginning for halb of the timestep, 
        # where would I approximately end up?

        # evaluate the network again at the midpoint
        return x_t + (t_end - t_start) * self(
            x_mid, (t_end - t_start) / 2
        )
        # Corresponds to: I'll estimate where I'll be halfway through,
        # calculate the velocity there, and use that velocity for the entire step.
        

# training 
flow = Flow()
optimizer = torch.optim.Adam(flow.parameters(), 1e-2)
loss_fn = nn.MSELoss()

for _ in range(5):
    x_1 = Tensor(make_moons(256, noise=0.05)[0])
    x_0 = torch.randn_like(x_1)
    t = torch.rand(len(x_1), 1)
    x_t = (1 - t) * x_0 + t * x_1
    dx_t = x_1 - x_0 
    optimizer.zero_grad()
    loss_fn(flow(x_t, t), dx_t).backward()
    optimizer.step() 
    print()


# sampling 
x = torch.randn(300, 2)
n_steps = 8
fig, axes = plt.subplots(1, n_steps + 1, figsize=(30, 4), sharex=True, sharey=True)
time_steps = torch.linspace(0, 1.0, n_steps + 1)

axes[0].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
axes[0].set_title(f't = {time_steps[0]:.2f}')
axes[0].set_xlim(-3.0, 3.0)
axes[0].set_ylim(-3.0, 3.0)

for i in range(n_steps):
    x = flow.step(x, time_steps[i], time_steps[i + 1])
    print("Done with step function")
    axes[i + 1].scatter(x.detach()[:, 0], x.detach()[:, 1], s=10)
    axes[i + 1].set_title(f't = {time_steps[i + 1]:.2f}')

plt.tight_layout()
plt.show()
