
"""Multi-head attention components used throughout TabFM."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .activations import get_activation
from .cache import QuantizedTensor
from .normalisation import RMSNorm
from .positional import rope_interleaved

class MultiheadAttention(nn.Module):
  def __init__(self, d_model, nhead, rope_base=None):
    super().__init__()
    if d_model <= 0:
      raise ValueError(f"d_model must be positive, got {d_model}.")
    if nhead <= 0:
      raise ValueError(f"nhead must be positive, got {nhead}.")
    if d_model % nhead != 0:
      raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
    head_dim = d_model // nhead
    if rope_base is not None and head_dim % 2 != 0:
      raise ValueError(f"RoPE requires an even head dimension, got {head_dim}.")
    self.nhead = nhead
    self.hd = head_dim
    self.rope_base = rope_base  # None => no RoPE
    self.q_proj = nn.Linear(d_model, d_model)
    self.k_proj = nn.Linear(d_model, d_model)
    self.v_proj = nn.Linear(d_model, d_model)
    self.out_proj = nn.Linear(d_model, d_model)
    self.query_ln = RMSNorm(self.hd)
    self.key_ln = RMSNorm(self.hd)
    self.per_dim_scale = nn.Parameter(torch.zeros(self.hd))

  def forward(self, query, key, value, attn_mask=None, rope=None,
              cached_kv=None, return_kv=False):
    """Computes multi-head attention, optionally with a K/V cache.

    At most one of cached_kv, return_kv is set. cached_kv is a (k, v) tuple of
    already-projected (and, where used, rotated and key-normalized) tensors of
    shape [B, T_src, N, D] from a prior call; key and value are then None and
    their projections are skipped. return_kv returns the freshly computed
    (k, v) in that layout for a later call to reuse.
    """
    if cached_kv is not None and return_kv:
      raise ValueError("Cannot both use cached_kv and return_kv.")
    b, tq, d = query.shape
    q = self.q_proj(query).view(b, tq, self.nhead, self.hd)

    if cached_kv is not None:
      if key is not None or value is not None:
        raise ValueError("key/value must be None when cached_kv is provided.")
      cached_k, cached_v = cached_kv
      # Cached K/V may be int8-quantized; dequantize to compute dtype before use.
      k = cached_k.dequantize(q.dtype) if isinstance(cached_k, QuantizedTensor) else cached_k
      v = cached_v.dequantize(q.dtype) if isinstance(cached_v, QuantizedTensor) else cached_v
    else:
      if key is None or value is None:
        raise ValueError("key/value must not be None when cached_kv is absent.")
      k = self.k_proj(key).view(b, key.shape[1], self.nhead, self.hd)
      v = self.v_proj(value).view(b, value.shape[1], self.nhead, self.hd)

    if self.rope_base is not None:
      # Cached K is already post-RoPE, so only rotate freshly-computed K.
      q = rope.rotate(q) if rope is not None else rope_interleaved(q, self.rope_base)
      if cached_kv is None:
        k = rope.rotate(k) if rope is not None else rope_interleaved(k, self.rope_base)

    q = self.query_ln(q)
    if cached_kv is None:
      k = self.key_ln(k)
    # per-dim scale in float32 (softplus), then cast to compute dtype -- matches JAX PerDimScale.
    scale = 1.442695041 / math.sqrt(self.hd) * F.softplus(self.per_dim_scale.float())
    q = q * scale.to(q.dtype)

    new_k, new_v = k, v  # cache format: [B, T_src, N, D], pre-transpose.

    q, k, v = (z.transpose(1, 2) for z in (q, k, v))  # [B,N,T,D]
    # bf16 SDPA (flash already does the softmax in float32 internally).
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
    out = self.out_proj(o.transpose(1, 2).reshape(b, tq, d))
    if return_kv:
      return out, (new_k, new_v)
    return out


class MultiheadAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, activation="swiglu", rope_base=None):
    super().__init__()
    self.attn = MultiheadAttention(d_model, nhead, rope_base)
    self.pre_attn_ln = RMSNorm(d_model)
    self.post_attn_ln = RMSNorm(d_model)
    self.pre_ff_ln = RMSNorm(d_model)
    self.post_ff_ln = RMSNorm(d_model)
    self.swiglu = activation == "swiglu"
    self.linear1 = nn.Linear(d_model, dim_ff)
    if self.swiglu:
      self.linear1_gate = nn.Linear(d_model, dim_ff)
      self.act = F.silu
    else:
      self.act = get_activation(activation)
    self.linear2 = nn.Linear(dim_ff, d_model)
    self.ffn_chunk_size = None  # set to an int to chunk the FFN over tokens

  def _ff_impl(self, x):
    xn = self.pre_ff_ln(x)
    if self.swiglu:
      x = self.act(self.linear1_gate(xn)) * self.linear1(xn)
    else:
      x = self.act(self.linear1(xn))
    return self.post_ff_ln(self.linear2(x))

  def _ff(self, x):
    # Eager FFN chunking: process tokens in slices so the expanded
    # [tokens, dim_feedforward] activation is never materialized in full.
    if self.ffn_chunk_size is None:
      return self._ff_impl(x)
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    out = torch.empty(flat.shape[0], self.linear2.out_features,
                      dtype=flat.dtype, device=flat.device)
    for s in range(0, flat.shape[0], self.ffn_chunk_size):
      out[s:s + self.ffn_chunk_size] = self._ff_impl(flat[s:s + self.ffn_chunk_size])
    return out.reshape(shape)

  def forward(self, q, k=None, v=None, attn_mask=None, rope=None,
              cached_kv=None, return_kv=False):
    q_n = self.pre_attn_ln(q)
    if cached_kv is not None:
      if k is not None or v is not None:
        raise ValueError("k/v must be None when cached_kv is provided.")
      k_n, v_n = None, None
    else:
      k = q if k is None else k
      v = q if v is None else v
      k_n = self.pre_attn_ln(k)
      v_n = self.pre_attn_ln(v)
    attn_res = self.attn(q_n, k_n, v_n, attn_mask, rope=rope,
                          cached_kv=cached_kv, return_kv=return_kv)
    if return_kv:
      attn_out, new_kv = attn_res
    else:
      attn_out = attn_res
    a = self.post_attn_ln(attn_out)
    x = q + a
    x = x + self._ff(x)
    if return_kv:
      return x, new_kv
    return x
