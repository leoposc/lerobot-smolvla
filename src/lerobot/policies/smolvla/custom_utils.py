import torch 
import math 


def sample_beta(bsize, device):
    alpha_t = torch.tensor(1.5, dtype=torch.float32)
    beta_t = torch.tensor(1, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)

def sample_time(bsize, device):
    time_beta = sample_beta(bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dype=torch.float32, device=device)


def sample_noise(shape, device):
    """Standard-normal float32 nosie, the flow-matching x_1 sample."""
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device
    )



def create_sinusodial_pos_embedding(
        time: torch.Tensor, dimension: int, min_period: float, max_period: float, o_dtype, device="cpu"
) -> Tensor: 
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape '(batch_size, )'.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction 

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).to(dtype=o_dtype)


def add_padding_to_tensor(tensor, max_len, pad_value=0):
    bsize, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (bsize, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor 
    return padded_tensor

