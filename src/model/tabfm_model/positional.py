"""Rotary positional encoding helpers."""

import torch
from torch import nn

def rope_interleaved(x, base):
  """Interleaved RoPE over the T axis of [B, T, N, Dh] (lucidrains convention)."""
  dh, t = x.shape[-1], x.shape[1]
  inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
  f = torch.outer(torch.arange(t, device=x.device).float(), inv)
  cos = f.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
  sin = f.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
  x1, x2 = x[..., 0::2], x[..., 1::2]
  rot = torch.stack((-x2, x1), -1).reshape_as(x)
  return x * cos + rot * sin


class RoPE(nn.Module):
  """One RoPE per Encoder, holding the inverse-frequency buffer loaded FROM the
  checkpoint (JAX stores `rope.freqs`, computed in bf16 at train time -- recomputing
  it in fp32 differs by ~1e-3 and that error grows with sequence length)."""

  def __init__(self, dim, base):
    super().__init__()
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # init = formula; overwritten on load
    self.register_buffer("freqs", inv)

  def rotate(self, x):  # x: [B, T, N, Dh], rotate over the T axis
    t = x.shape[1]
    f = torch.outer(torch.arange(t, device=x.device).float(), self.freqs.float())
    cos = f.cos().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
    sin = f.sin().repeat_interleave(2, -1)[None, :, None, :].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), -1).reshape_as(x)
    return x * cos + rot * sin

