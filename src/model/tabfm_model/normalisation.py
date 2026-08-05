"""Normalization layers used by TabFM."""

import torch
from torch import nn

class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x):
    # Normalize entirely in float32 (x * rsqrt * weight), cast back at the end --
    # matches JAX/Flax, which keeps x*rsqrt in float32. Doing the multiply in bf16
    # (casting rsqrt down first) loses precision and accumulates across the ~36
    # RMSNorms per transformer stack.
    dt = x.dtype
    xf = x.float()
    v = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(v + self.eps)) * self.weight.float()).to(dt)

