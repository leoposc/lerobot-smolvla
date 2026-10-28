import torch 
from torch import nn, Tensor 
from typing import override  
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


class Flow(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        print("Initiallizing Flow class")
        super().__init__()
        # I,nterleaved Linear and Exponential Linear Unit layers
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
        t_start = t_start.reshape(1, 1).expand(x_t.shape[0], 1)
        h = t_end - t_start 

        v_start = self(x_t, t_start)
        x_mid = x_t + 0.5 * h * v_start 
        t_mid = t_start + 0.5 * h 

        v_mid = self(x_mid, t_mid)
        return x_t + h * v_mid 

    def euler_integration(self, x_t: Tensor, t_start, t_end: Tensor) -> Tensor:
        t = t_start.reshape(1, 1).expand(x_t.shape[0], 1)
        dt = t_end - t_start
        return x_t + dt * self(x_t, t)        


    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def init_and_load_model(self, path: str) -> None:
        self.__init__()
        self.load_state_dict(torch.load(path))
        self.eval()


# training 
def train(): 
    optimizer = torch.optim.Adam(flow.parameters(), 2e-2)
    loss_fn = nn.MSELoss()

    for _ in range(10000):
        x_1 = Tensor(make_moons(256, noise=0.05)[0])
        x_0 = torch.randn_like(x_1)
        t = torch.rand(len(x_1), 1)
        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0 
        optimizer.zero_grad()
        loss_fn(flow(x_t, t), dx_t).backward()
        optimizer.step() 
        print()

    # Save the trained model
    flow.save_model("flow_model.pth")
    return flow

flow = Flow()
if True: 
    flow = train()
else: 
    flow.init_and_load_model("flow_model.pth")

# sampling 
x = torch.randn(300, 2)
n_steps = 10
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


