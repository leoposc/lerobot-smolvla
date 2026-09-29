import torch


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Apply Rotary Positional Embeddings (RoPE) to the input tensor x.
    
    Args:
        x (torch.Tensor): Input Tensor of shape (batch_size, seq_len)
        positions (torch.Tensor): Positions Tensor of shape (seq_len)
        max_wavelength (int): Maximum wavelength for RoPE. Default is 10,000.
        
        Returns:
        torch.Tensor: Tensor after applying RoPE, same shape as input x.
    """
    d_half = x.shape[-1] // 2
    device = x.device 
    dtype = x.dtype 
    x = x.to(torch.float32) # Ensure x is in float32 for precision during RoPE computation 

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, device=device, dtype=dtype)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]

    sin = torch.sin(radians) 
    cos = torch.cos(radians)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x1 * sin + x2 * cos

    return res.to(dtype)