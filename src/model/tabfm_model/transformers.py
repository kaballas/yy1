
"""Transformer stacks used for row, column, and ICL processing."""

import torch
from torch import nn

from .attention import MultiheadAttentionBlock
from .positional import RoPE
from .validation import _validate_cache_depth

class InducedSelfAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, num_inds, activation="swiglu"):
    super().__init__()
    self.ind_vectors = nn.Parameter(torch.empty(num_inds, d_model))
    nn.init.normal_(self.ind_vectors, std=d_model ** -0.5)
    self.mab1 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)
    self.mab2 = MultiheadAttentionBlock(d_model, nhead, dim_ff, activation)

  def forward(self, src, attn_mask=None, cached_hidden=None, return_hidden=False):
    """Applies induced self-attention, optionally reusing a cached hidden.

    If cached_hidden (the mab1 output from a prior call) is given, mab1 is
    skipped and only mab2 runs; return_hidden returns the freshly computed
    hidden for a later call to reuse.
    """
    if cached_hidden is not None and return_hidden:
      raise ValueError("Cannot both use cached_hidden and return_hidden.")
    if cached_hidden is not None:
      hidden = cached_hidden
    else:
      ind = self.ind_vectors.unsqueeze(0).expand(src.shape[0], -1, -1)
      hidden = self.mab1(ind, src, src, attn_mask=attn_mask)
    out = self.mab2(src, hidden, hidden)
    if return_hidden:
      return out, hidden
    return out


class Encoder(nn.Module):
  def __init__(self, num_blocks, d_model, nhead, dim_ff, activation="swiglu",
               rope_base=100000.0):
    super().__init__()
    # One RoPE per Encoder (mirrors JAX `tf_row.rope.freqs`), shared by all blocks.
    self.rope = RoPE(d_model // nhead, rope_base) if rope_base is not None else None
    self.blocks = nn.ModuleList([
        MultiheadAttentionBlock(d_model, nhead, dim_ff, activation, rope_base)
        for _ in range(num_blocks)
    ])

  def forward(self, x, attn_mask=None, cached_kv=None, return_kv=False):
    """Runs the stacked attention blocks, optionally with a per-block K/V cache.

    cached_kv, if given, is a per-block list of (k, v) that each block uses in
    place of its k/v projections; return_kv collects the freshly computed
    per-block (k, v). At most one of the two is set.
    """
    if cached_kv is not None:
      if return_kv:
        raise ValueError("Cannot both use cached_kv and return_kv.")
      _validate_cache_depth(cached_kv, len(self.blocks), "cached_kv")
      for index, kv in enumerate(cached_kv):
        if not isinstance(kv, (tuple, list)) or len(kv) != 2:
          raise RuntimeError(f"cached_kv layer {index} must be a two-element (k, v) pair.")
      for blk, kv in zip(self.blocks, cached_kv):
        x = blk(x, attn_mask=attn_mask, rope=self.rope, cached_kv=kv)
      return x
    if return_kv:
      kvs = []
      for blk in self.blocks:
        x, kv = blk(x, attn_mask=attn_mask, rope=self.rope, return_kv=True)
        kvs.append(kv)
      return x, kvs
    for blk in self.blocks:
      x = blk(x, attn_mask=attn_mask, rope=self.rope)
    return x


class SetTransformer(nn.Module):
  def __init__(self, num_blocks, d_model, nhead, dim_ff, num_inds,
               activation="swiglu"):
    super().__init__()
    self.blocks = nn.ModuleList([
        InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, activation)
        for _ in range(num_blocks)
    ])

  def forward(self, src, attn_mask=None, cached_hidden=None, return_hidden=False):
    """Runs the stacked induced-attention blocks.

    cached_hidden, if given, is a per-block list of cached mab1 outputs (see
    InducedSelfAttentionBlock.forward()); return_hidden=True collects the
    freshly computed per-block hidden reprs for later reuse.
    """
    if cached_hidden is not None:
      if return_hidden:
        raise ValueError("Cannot both use cached_hidden and return_hidden.")
      _validate_cache_depth(cached_hidden, len(self.blocks), "cached_hidden")
      for blk, h in zip(self.blocks, cached_hidden):
        src = blk(src, cached_hidden=h)
      return src
    if return_hidden:
      hiddens = []
      for blk in self.blocks:
        src, h = blk(src, attn_mask=attn_mask, return_hidden=True)
        hiddens.append(h)
      return src, hiddens
    for blk in self.blocks:
      src = blk(src, attn_mask=attn_mask)
    return src
