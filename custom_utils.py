import torch 


def sample_beta(bsize, device):
    alpha_t = torch.tensor(1.5, dtype=torch.float32)
    beta_t = torch.tensor(1, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)

def sample_time(bsize, device)
    time_beta = sample_beta(bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dype=torch.float32, device=device)


def sample_noise():
    ...
