

"""EXPERIMENTAL faithful-ish PyTorch port of the TabFM architecture (fwd path).

Module/param names mirror the JAX model (tabfm/src/model.py) so weight
conversion is mechanical. The transformer math (attention with RoPE +
PerDimScale + q/k RMSNorm + SDPA at scale=1.0, swiglu FFN, full MAB) has been
parity-verified against JAX to ~1e-6 (float32). The embedding/ICL paths mirror
the JAX code but are validated by the end-to-end converter parity test.

NOT yet wired to Orbax weights; see torch_parity_harness.py for the converter
and parity gates.
"""

import dataclasses
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _gelu_tanh(x):
  # jax.nn.gelu defaults to the tanh approximation -> match it.
  return F.gelu(x, approximate="tanh")


def get_activation(name):
  # Activations are stored as module attributes (e.g. MLP.act), so they must be
  # picklable for AutoGluon/TabArena's pickle-based save. A module-level
  # function pickles by reference; a lambda would not.
  return {"relu": F.relu,
          "gelu": _gelu_tanh,
          "silu": F.silu}[name]


_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint8)


def _validate_runtime_inputs(x, y, train_size, d=None, cat_mask=None):
  if x.ndim != 3:
    raise ValueError(f"x must have shape [batch, rows, features], got {tuple(x.shape)}.")
  batch_size, sequence_length, feature_count = x.shape
  if y.ndim != 2 or y.shape != x.shape[:2]:
    raise ValueError(f"y must have shape {tuple(x.shape[:2])}, got {tuple(y.shape)}.")
  if train_size.shape != (batch_size,):
    raise ValueError(f"train_size must have shape ({batch_size},), got {tuple(train_size.shape)}.")
  if train_size.dtype not in _INTEGER_DTYPES:
    raise ValueError(f"train_size must have an integer dtype, got {train_size.dtype}.")
  if train_size.numel() and (torch.any(train_size < 0) or
                             torch.any(train_size > sequence_length)):
    lo, hi = int(train_size.min()), int(train_size.max())
    raise ValueError(f"train_size values must be in [0, {sequence_length}], got range [{lo}, {hi}].")
  if d is not None:
    if d.shape != (batch_size,):
      raise ValueError(f"d must have shape ({batch_size},), got {tuple(d.shape)}.")
    if d.dtype not in _INTEGER_DTYPES:
      raise ValueError(f"d must have an integer dtype, got {d.dtype}.")
    if d.numel() and (torch.any(d < 0) or torch.any(d > feature_count)):
      lo, hi = int(d.min()), int(d.max())
      raise ValueError(f"d values must be in [0, {feature_count}], got range [{lo}, {hi}].")
  if cat_mask is not None:
    expected = (batch_size, feature_count)
    if cat_mask.shape != expected:
      raise ValueError(f"cat_mask must have shape {expected}, got {tuple(cat_mask.shape)}.")
    if cat_mask.dtype != torch.bool:
      raise ValueError(f"cat_mask must have dtype torch.bool, got {cat_mask.dtype}.")
  devices = {"x": x.device, "y": y.device, "train_size": train_size.device}
  if d is not None:
    devices["d"] = d.device
  if cat_mask is not None:
    devices["cat_mask"] = cat_mask.device
  if len(set(devices.values())) != 1:
    detail = ", ".join(f"{name}={device}" for name, device in devices.items())
    raise ValueError(f"Runtime input tensors must be on compatible devices; got {detail}.")


def _validate_classification_labels(y, train_size, max_classes, *,
                                    require_padding_after_context,
                                    padding_label=-100.0):
  if y.ndim != 2:
    raise ValueError(f"y must have shape [batch, rows], got {tuple(y.shape)}.")
  if not torch.isfinite(y).all():
    raise ValueError("y must contain only finite values.")
  rows = torch.arange(y.shape[1], device=y.device)[None, :]
  is_context = rows < train_size[:, None]
  is_integer = y == y.round()
  is_valid_class = is_integer & (y >= 0) & (y < max_classes)
  is_padding = y == padding_label
  if torch.any(is_context & ~is_valid_class):
    raise ValueError(f"Context labels must be integers in [0, {max_classes - 1}].")
  if require_padding_after_context:
    if torch.any((~is_context) & ~is_padding):
      raise ValueError(f"Rows after the context prefix must use {padding_label}.")
  elif torch.any((~is_context) & ~(is_valid_class | is_padding)):
    raise ValueError("Query labels must be valid class IDs or the padding sentinel.")


def _validate_contiguous_context_labels(y, padding_label=-100.0):
  if y.ndim != 2:
    raise ValueError(f"y must have shape [batch, rows], got {tuple(y.shape)}.")
  is_valid = y != padding_label
  has_seen_padding = (~is_valid).cumsum(dim=1) > 0
  valid_after_padding = is_valid & has_seen_padding
  if torch.any(valid_after_padding):
    bad_batch, bad_row = torch.nonzero(valid_after_padding, as_tuple=True)
    raise ValueError(
        "prefill requires context rows to be a contiguous valid prefix followed only by trailing padding. "
        f"First invalid ordering at batch={bad_batch[0].item()}, row={bad_row[0].item()}.")
  return is_valid


def _validate_cache_depth(cache, expected_depth, cache_name):
  if len(cache) != expected_depth:
    raise ValueError(f"{cache_name} contains {len(cache)} layers, but the model requires {expected_depth}.")


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


class InducedSelfAttentionBlock(nn.Module):
  def __init__(self, d_model, nhead, dim_ff, num_inds, activation="swiglu"):
    super().__init__()
    self.ind_vectors = nn.Parameter(torch.zeros(num_inds, d_model))
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


class MLP(nn.Module):
  def __init__(self, in_dim, hidden_dims: List[int], out_dim, activation="gelu"):
    super().__init__()
    self.act = get_activation(activation)
    dims = [in_dim] + list(hidden_dims)
    self.layers = nn.ModuleList()  # only Linears; activation applied between
    for i in range(len(hidden_dims)):
      self.layers.append(nn.Linear(dims[i], dims[i + 1]))
    self.layers.append(nn.Linear(dims[-1], out_dim))

  def forward(self, x):
    for i, lin in enumerate(self.layers):
      x = lin(x)
      if i < len(self.layers) - 1:
        x = self.act(x)
    return x


class OneHotAndLinear(nn.Module):
  def __init__(self, num_classes, embed_dim):
    super().__init__()
    self.num_classes = num_classes
    self.projection = nn.Linear(num_classes, embed_dim)

  def forward(self, y):  # y: [B, T] int
    y_long = y.long()
    y_mapped = torch.where(
        (y_long >= 0) & (y_long < self.num_classes),
        y_long,
        torch.tensor(self.num_classes, device=y.device)
    )
    oh = F.one_hot(y_mapped, self.num_classes + 1).to(self.projection.weight.dtype)
    oh_sliced = oh[..., :self.num_classes]
    return self.projection(oh_sliced)


class CellEmbedder(nn.Module):
  def __init__(self, embed_dim, max_classes, feature_group_size=3, num_freq=32,
               is_classifier=True):
    super().__init__()
    self.embed_dim = embed_dim
    self.fgs = feature_group_size
    self.is_classifier = is_classifier
    in_dim = feature_group_size
    self.register_buffer("fourier_frequencies", torch.zeros(in_dim, num_freq))
    self.register_buffer("fourier_frequencies_cat", torch.zeros(in_dim, num_freq))
    self.in_linear = nn.Linear(num_freq * 2, embed_dim)
    self.in_linear_cat = nn.Linear(num_freq * 2, embed_dim)
    if is_classifier:  # classification: embedding lookup over class ids
      self.y_embedder_lookup = nn.Embedding(max_classes, embed_dim)
    else:  # regression: MLP over the scalar target (y_col_embedder_encoder_nhid=6)
      self.y_embedder_lookup = MLP(1, [6], embed_dim, activation="gelu")
    self.row_chunk_size = None  # chunk the Fourier expansion over rows

  def _group(self, x, d=None):  # x: [B,T,H] -> [B,T,H,G]
    h = x.shape[-1]
    idxs = torch.arange(h, device=x.device)
    stacked = []
    if d is not None:
      # Per-batch wrap-around over each member's ACTIVE feature count d (not the
      # padded width h). Mirrors the JAX `% d_safe` path so zero-padded slots are
      # filled with wrapped real features rather than mixing padding into groups.
      d_safe = torch.clamp(d.to(torch.long), min=1)  # [B]
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        idx = (idxs[None, :] + offset) % d_safe[:, None]            # [B, H]
        idx = idx[:, None, :].expand(x.shape[0], x.shape[1], h)     # [B, T, H]
        stacked.append(torch.gather(x, -1, idx))
    else:
      for i in range(self.fgs):
        offset = (2 ** i) - 1
        stacked.append(x[..., (idxs + offset) % h])
    return torch.stack(stacked, dim=-1)

  def _cell(self, x, cat_mask, d=None):  # [B,t,H] -> [B,t,HC,E] (Fourier expansion + sum over G)
    g = self._group(x, d=d).unsqueeze(-1).float()  # float32 Fourier: args g*freq reach ~30,
    dt = x.dtype                                    # so sin/cos must run in fp32 (matches JAX,
    ff = self.fourier_frequencies.float()           # whose freq params stay float32). Cast the
    ffc = self.fourier_frequencies_cat.float()      # fourier features back to compute dtype before in_linear.
    num_out = self.in_linear(torch.cat([(g * ff).sin(), (g * ff).cos()], dim=-1).to(dt))
    if cat_mask is not None:
      cat_out = self.in_linear_cat(torch.cat([(g * ffc).sin(), (g * ffc).cos()], dim=-1).to(dt))
      cmg = self._group(cat_mask[:, None, :].float(), d=d).bool()[..., None]
      return torch.where(cmg, cat_out, num_out).sum(-2)
    return num_out.sum(-2)

  def forward(self, x, y, train_size, cat_mask=None, d=None,
              inject_context_labels=True):
    # The Fourier expansion materializes [B,T,HC,G,E]; chunk over rows so that
    # huge intermediate never exists in full (rows are independent here).
    if self.row_chunk_size is None:
      cell = self._cell(x, cat_mask, d=d)
    else:
      parts = [self._cell(x[:, s:s + self.row_chunk_size], cat_mask, d=d)
               for s in range(0, x.shape[1], self.row_chunk_size)]
      cell = torch.cat(parts, dim=1)
    if inject_context_labels:
      if self.is_classifier:
        y_clean = torch.clamp(y.long(), 0, self.y_embedder_lookup.num_embeddings - 1)
        y_emb = self.y_embedder_lookup(y_clean)  # [B,T,E]
      else:
        y_emb = self.y_embedder_lookup(y[..., None].to(cell.dtype))  # scalar -> [B,T,E]
      t = x.shape[1]
      tm = (torch.arange(t, device=x.device)[None, :] < train_size[:, None])[..., None, None]
      out = torch.where(tm, cell + y_emb[:, :, None, :], cell)
    else:
      out = cell
    if d is not None:
      # Zero the padded feature columns (cols >= d): the % d wrap above fills them
      # with real features for valid indexing, but they must not enter attention.
      hc = out.shape[2]
      colmask = (torch.arange(hc, device=out.device)[None, :]
                 < d[:, None])[:, None, :, None]  # [B, 1, HC, 1]
      out = torch.where(colmask, out, torch.zeros_like(out))
    return out


class ColEmbedding(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_inds):
    super().__init__()
    self.tf_col = SetTransformer(num_blocks, d_model, nhead, dim_ff, num_inds)
    self.out_w = nn.Linear(d_model, d_model)
    self.ln_w = RMSNorm(d_model)
    self.col_chunk_size = None  # chunk the independent column axis (B*HC)

  def _stage(self, src, mask=None, cached_hidden=None, return_hidden=False):
    out = self.tf_col(src, attn_mask=mask, cached_hidden=cached_hidden,
                       return_hidden=return_hidden)
    if return_hidden:
      out, hidden = out
      return self.ln_w(self.out_w(out)), hidden
    return self.ln_w(self.out_w(out))

  def forward(self, x, train_size, *, cached_repr=None, return_repr=False):
    """Transform input table into column-wise embeddings.

    train_size is None exactly when cached_repr is given (decode): cached_repr
    supplies the induced-point hidden, so mab1 and its mask are skipped.
    cached_repr and return_repr are mutually exclusive.
    """
    if cached_repr is not None and return_repr:
      raise ValueError("Cannot have both cached_repr not None and return_repr True.")
    if (cached_repr is not None) != (train_size is None):
      raise ValueError("train_size must be None iff cached_repr is given.")
    b, t, hc, e = x.shape
    src = x.permute(0, 2, 1, 3).reshape(b * hc, t, e)  # [B*HC, T, E]
    cc = self.col_chunk_size

    if cached_repr is not None:  # Decode: reuse cached induced-point hidden.
      if cc is None or src.shape[0] <= cc:
        out = self._stage(src, None, cached_hidden=cached_repr)
      else:
        parts = []
        for s in range(0, src.shape[0], cc):
          chunk_repr = [h[s:s + cc] for h in cached_repr]
          parts.append(self._stage(src[s:s + cc], None, cached_hidden=chunk_repr))
        out = torch.cat(parts, dim=0)
      return out.reshape(b, hc, t, e).permute(0, 2, 1, 3)

    ts = train_size.repeat_interleave(hc)  # [B*HC]
    mask = (torch.arange(t, device=x.device)[None, :] < ts[:, None])[:, None, None, :]

    if return_repr:  # Prefill: also return the induced-point hidden per block.
      if cc is None or src.shape[0] <= cc:
        out, hidden = self._stage(src, mask, return_hidden=True)
      else:
        out_parts = []
        hidden_parts = None
        for s in range(0, src.shape[0], cc):
          o, h = self._stage(src[s:s + cc], mask[s:s + cc], return_hidden=True)
          out_parts.append(o)
          if hidden_parts is None:
            hidden_parts = [[] for _ in h]
          for i, hh in enumerate(h):
            hidden_parts[i].append(hh)
        out = torch.cat(out_parts, dim=0)
        hidden = [torch.cat(parts, dim=0) for parts in hidden_parts]
      out = out.reshape(b, hc, t, e).permute(0, 2, 1, 3)
      return out, hidden

    if cc is None or src.shape[0] <= cc:
      out = self._stage(src, mask)
    else:
      out = torch.cat([self._stage(src[s:s + cc], mask[s:s + cc])
                       for s in range(0, src.shape[0], cc)], dim=0)
    return out.reshape(b, hc, t, e).permute(0, 2, 1, 3)


class RowInteraction(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, dim_ff, num_cls,
               rope_base=100000.0, output_full=True):
    super().__init__()
    self.tf_row = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=rope_base)
    self.out_ln = RMSNorm(d_model)
    self.num_cls = num_cls
    self.output_full = output_full
    self.row_chunk_size = None  # chunk the independent row axis (B*T)

  def _stage(self, src, mask=None):
    out = self.tf_row(src, attn_mask=mask)
    return self.out_ln(out if self.output_full else out[:, : self.num_cls, :])

  def forward(self, x, d=None):  # x: [B,T,HC,E]
    b, t, hc, e = x.shape
    src = x.reshape(b * t, hc, e)
    # Mask cross-column attention to the valid columns (CLS + d real features);
    # padded columns (>= d + num_cls) must not be attended to. Matches JAX.
    mask = None
    if d is not None:
      d_padded = d.to(torch.long) + self.num_cls  # [B]
      valid = torch.arange(hc, device=x.device)[None, :] < d_padded[:, None]  # [B, HC]
      mask = valid.repeat_interleave(t, dim=0)[:, None, None, :]  # [B*T, 1, 1, HC]
    rc = self.row_chunk_size
    if rc is None or src.shape[0] <= rc:
      out = self._stage(src, mask)
    else:
      out = torch.cat([self._stage(src[s:s + rc],
                                   None if mask is None else mask[s:s + rc])
                       for s in range(0, src.shape[0], rc)], dim=0)
    if self.output_full:
      return out.reshape(b, t, hc, e)
    return out.reshape(b, t, -1)


@dataclasses.dataclass
class QuantizedTensor:
  """Per-tensor symmetric integer quantization of a cached K or V tensor.

  data holds the quantized codes; scale is the per-tensor absmax / max_val
  factor, so the float value is data.to(dtype) * scale.
  """
  data: torch.Tensor
  scale: torch.Tensor

  def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
    """Dequantizes back to a floating-point tensor of the given dtype."""
    return self.data.to(dtype) * self.scale.to(dtype)


# Per dtype: (lo, hi, max_val) clamp range, one code below the dtype's full
# range so -max and +max map symmetrically and dequantization cannot exceed
# the original absmax in magnitude.
_QUANTIZATION_RANGES: Dict[torch.dtype, Tuple[int, int, int]] = {
    torch.int8: (-127, 127, 127),
}


def _quantize_tensor(t: torch.Tensor,
                      dtype: torch.dtype = torch.int8) -> QuantizedTensor:
  """Per-tensor symmetric integer quantization to dtype."""
  if dtype not in _QUANTIZATION_RANGES:
    raise ValueError(f"Unsupported quantization dtype {dtype}; supported: "
                      f"{list(_QUANTIZATION_RANGES.keys())}")
  lo, hi, max_val = _QUANTIZATION_RANGES[dtype]
  absmax = t.abs().amax()
  scale = absmax / max_val
  # Avoid division by zero for all-zero tensors.
  scale = torch.clamp(scale, min=torch.finfo(scale.dtype).tiny)
  data = (t / scale).round().clamp(lo, hi).to(dtype)
  return QuantizedTensor(data=data, scale=scale)


def move_cache_to_device(cache, device):
  """Recursively moves a TabFM.prefill() cache dict to device.

  Handles the nested structure of the cache returned by TabFM.prefill(): a
  dict (top-level col1/col2/icl keys), a list/tuple (per-block K/V pairs,
  per-block induced-point hidden reprs), the ICLearningCache/QuantizedTensor
  dataclasses, and plain tensor leaves. Needed for the
  keep_cache_on_device=False path (CPU-offload between predict calls).
  """
  if isinstance(cache, torch.Tensor):
    return cache.to(device)
  if isinstance(cache, dict):
    return {k: move_cache_to_device(v, device) for k, v in cache.items()}
  if isinstance(cache, list):
    return [move_cache_to_device(v, device) for v in cache]
  if isinstance(cache, tuple):
    return tuple(move_cache_to_device(v, device) for v in cache)
  if dataclasses.is_dataclass(cache):
    kwargs = {f.name: move_cache_to_device(getattr(cache, f.name), device)
              for f in dataclasses.fields(cache)}
    return type(cache)(**kwargs)
  return cache


def detach_cache(cache):
  """Recursively detaches every tensor in a prefill cache."""
  if isinstance(cache, torch.Tensor):
    return cache.detach()
  if isinstance(cache, QuantizedTensor):
    return QuantizedTensor(data=cache.data.detach(), scale=cache.scale.detach())
  if isinstance(cache, dict):
    return {k: detach_cache(v) for k, v in cache.items()}
  if isinstance(cache, list):
    return [detach_cache(v) for v in cache]
  if isinstance(cache, tuple):
    return tuple(detach_cache(v) for v in cache)
  if dataclasses.is_dataclass(cache):
    kwargs = {f.name: detach_cache(getattr(cache, f.name))
              for f in dataclasses.fields(cache)}
    return type(cache)(**kwargs)
  return cache


@dataclasses.dataclass
class ICLearningCache:
  """Per-block ICL K/V cache produced at prefill and reused at decode.

  layer_caches is a per-block list of (k, v) of shape [B, T_prefill, N, D],
  each a QuantizedTensor after quantize(). prefill_train_size is the [B] count
  of valid training rows used to build the decode attention mask; it stays
  full precision after quantize().
  """
  layer_caches: List[Tuple[Any, Any]]
  prefill_train_size: torch.Tensor

  @property
  def prefill_seq_len(self) -> int:
    """Sequence length of the prefill cache, derived from tensor shape."""
    k, _ = self.layer_caches[0]
    data = k.data if isinstance(k, QuantizedTensor) else k
    return data.shape[1]

  def quantize(self, dtype: torch.dtype = torch.int8) -> "ICLearningCache":
    """Returns a copy with the per-block ICL K/V quantized to dtype.

    Only the attention K/V is quantized; prefill_train_size stays full
    precision.
    """
    quantized = [(_quantize_tensor(k, dtype), _quantize_tensor(v, dtype))
                 for k, v in self.layer_caches]
    return ICLearningCache(layer_caches=quantized,
                            prefill_train_size=self.prefill_train_size)


class ICLearning(nn.Module):
  def __init__(self, d_model, num_blocks, nhead, max_classes, dim_ff,
               decoder_hidden, is_classifier=True):
    super().__init__()
    self.tf_icl = Encoder(num_blocks, d_model, nhead, dim_ff, rope_base=None)  # ICL has no RoPE
    self.ln = RMSNorm(d_model)
    self.is_classifier = is_classifier
    if is_classifier:  # one-hot y-encode; decode to per-class logits
      self.y_encoder = OneHotAndLinear(max_classes, d_model)
      self.decoder = MLP(d_model, [decoder_hidden], max_classes)
    else:  # MLP y-encode the scalar target; decode to a single value
      self.y_encoder = MLP(1, [decoder_hidden], d_model)
      self.decoder = MLP(d_model, [decoder_hidden], 1)

  def forward(self, reps, y, train_size, *,
              cache: Optional[ICLearningCache] = None,
              return_cache: bool = False,
              return_hidden: bool = False):
    """Forward pass for ICLearning. reps: [B, T, E] row representations.

    train_size is None exactly when cache is given (decode): the attention mask
    is then derived from the cached prefill's train size and sequence length
    rather than this call's, and y is unused. return_cache (prefill only) also
    returns the ICLearningCache for this call alongside the decoded output.
    """
    if (cache is not None) != (train_size is None):
      raise ValueError("train_size must be None iff cache is given.")
    b, t, _ = reps.shape

    if cache is not None:  # Decode.
      prefill_seq_len = cache.prefill_seq_len
      tm_ctx = (torch.arange(prefill_seq_len, device=reps.device)[None, :]
                < cache.prefill_train_size[:, None])
      mask = tm_ctx[:, None, None, :]
      out = self.tf_icl(reps, attn_mask=mask, cached_kv=cache.layer_caches)
      hidden = self.ln(out)
      logits = self.decoder(hidden)
      return (logits, hidden) if return_hidden else logits

    tm = (torch.arange(t, device=reps.device)[None, :] < train_size[:, None])
    if self.is_classifier:
      y_enc = self.y_encoder(y)
    else:
      y_enc = self.y_encoder(y[..., None].to(reps.dtype))
    r = reps + y_enc * tm[..., None]
    mask = tm[:, None, None, :]
    if return_cache:  # Prefill.
      out, kvs = self.tf_icl(r, attn_mask=mask, return_kv=True)
      new_cache = ICLearningCache(layer_caches=kvs, prefill_train_size=train_size)
      hidden = self.ln(out)
      logits = self.decoder(hidden)
      if return_hidden:
        return logits, new_cache, hidden
      return logits, new_cache
    out = self.tf_icl(r, attn_mask=mask)
    hidden = self.ln(out)
    logits = self.decoder(hidden)
    return (logits, hidden) if return_hidden else logits


class RaceSetEncoder(nn.Module):
  """Race self-attention producing representation-level runner corrections.

  All runners belonging to one race must be present in the same call; splitting
  a race across decode calls changes race-conditioned predictions.
  """

  def __init__(self, input_dim, race_dim=32, attention_heads=2,
               attention_layers=1, feedforward_dim=64, residual=True):
    super().__init__()
    if race_dim % attention_heads != 0:
      raise ValueError("race_dim must be divisible by attention_heads")
    if attention_layers < 1 or attention_heads < 1:
      raise ValueError("race encoder layers and heads must be positive")
    self.residual = residual
    self.input_projection = nn.Linear(input_dim, race_dim)
    layer = nn.TransformerEncoderLayer(
        d_model=race_dim, nhead=attention_heads,
        dim_feedforward=feedforward_dim, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True)
    self.encoder = nn.TransformerEncoder(layer, num_layers=attention_layers)
    self.output_norm = nn.LayerNorm(race_dim)
    self.output_projection = nn.Linear(race_dim, input_dim)
    nn.init.zeros_(self.output_projection.weight)
    nn.init.zeros_(self.output_projection.bias)

  def forward(self, hidden, race_group_ids):
    if hidden.shape[:2] != race_group_ids.shape:
      raise ValueError("race_group_ids must match hidden batch and sequence dimensions")
    result = torch.zeros_like(hidden)
    grouped_rows = []
    max_runners = 0
    for batch_index in range(hidden.shape[0]):
      for group_id in torch.unique(race_group_ids[batch_index]):
        if int(group_id.item()) < 0:
          continue
        rows = torch.nonzero(race_group_ids[batch_index] == group_id, as_tuple=False).flatten()
        grouped_rows.append((batch_index, rows))
        max_runners = max(max_runners, int(rows.numel()))
    if not grouped_rows:
      return hidden if self.residual else result
    padded = hidden.new_zeros((len(grouped_rows), max_runners, hidden.shape[-1]))
    padding_mask = torch.ones((len(grouped_rows), max_runners), dtype=torch.bool, device=hidden.device)
    for index, (batch_index, rows) in enumerate(grouped_rows):
      count = rows.numel()
      padded[index, :count] = hidden[batch_index, rows]
      padding_mask[index, :count] = False
    encoded = self.encoder(self.input_projection(padded), src_key_padding_mask=padding_mask)
    delta = self.output_projection(self.output_norm(encoded))
    for index, (batch_index, rows) in enumerate(grouped_rows):
      result[batch_index, rows] = delta[index, :rows.numel()]
    return hidden + result if self.residual else result


class RaceSetHead(nn.Module):
  """Permutation-equivariant correction over runners in each target race.

  All runners belonging to one race must be present in the same decode call;
  splitting a race across calls changes race-conditioned predictions.
  """

  def __init__(self, input_dim, race_dim=32, attention_heads=2,
               attention_layers=1, feedforward_dim=64, output_classes=2):
    super().__init__()
    if race_dim % attention_heads != 0:
      raise ValueError("race_context_dim must be divisible by race_context_heads")
    if attention_layers < 1 or attention_heads < 1:
      raise ValueError("race context layers and heads must be positive")
    self.input_projection = nn.Linear(input_dim, race_dim)
    layer = nn.TransformerEncoderLayer(
        d_model=race_dim,
        nhead=attention_heads,
        dim_feedforward=feedforward_dim,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    self.encoder = nn.TransformerEncoder(layer, num_layers=attention_layers)
    self.output_norm = nn.LayerNorm(race_dim)
    self.output_projection = nn.Linear(race_dim, output_classes)
    nn.init.zeros_(self.output_projection.weight)
    nn.init.zeros_(self.output_projection.bias)

  def forward(self, hidden, race_group_ids):
    if hidden.shape[:2] != race_group_ids.shape:
      raise ValueError("race_group_ids must match hidden batch and sequence dimensions")
    result = hidden.new_zeros((*hidden.shape[:2], self.output_projection.out_features))
    grouped_rows = []
    max_runners = 0
    for batch_index in range(hidden.shape[0]):
      for group_id in torch.unique(race_group_ids[batch_index]):
        if int(group_id) < 0:
          continue
        row_indices = torch.nonzero(
            race_group_ids[batch_index] == group_id, as_tuple=False
        ).flatten()
        grouped_rows.append((batch_index, row_indices))
        max_runners = max(max_runners, int(row_indices.numel()))
    if not grouped_rows:
      return result

    padded = hidden.new_zeros((len(grouped_rows), max_runners, hidden.shape[-1]))
    padding_mask = torch.ones(
        (len(grouped_rows), max_runners), dtype=torch.bool, device=hidden.device
    )
    for race_index, (batch_index, row_indices) in enumerate(grouped_rows):
      length = row_indices.numel()
      padded[race_index, :length] = hidden[batch_index, row_indices]
      padding_mask[race_index, :length] = False
    encoded = self.input_projection(padded)
    encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
    delta = self.output_projection(self.output_norm(encoded))
    for race_index, (batch_index, row_indices) in enumerate(grouped_rows):
      result[batch_index, row_indices] = delta[race_index, :row_indices.numel()]
    return result


# Always-on activation-chunk sizes. TabFM runs the whole training fold as one
# in-context sequence, so a single forward materialises activations that grow
# with rows * features and OOM the GPU on large tasks. These sizes split each
# stage's largest activation along an independent axis, bounding peak memory.
# Chunking is exact (identical outputs) and a no-op when an input is smaller
# than the chunk size, and it costs <1% runtime otherwise, so it is always on.
# Sizes are chosen for memory safety on a 40 GB GPU across TabArena-scale tasks.
_ROW_CHUNK_SIZE = 4096   # rows per chunk (Fourier cell embedding + row interaction)
_COL_CHUNK_SIZE = 16     # feature-instances per chunk (column set-transformer)
_FFN_CHUNK_SIZE = 8192   # tokens per chunk (feed-forward expansion in every block)
_CACHE_FORMAT_VERSION = 2


class TabFM(nn.Module):
  def __init__(self, *, embed_dim=8, max_classes=10, col_num_blocks=2,
               col_nhead=2, col_num_inds=4, row_num_blocks=2, row_nhead=2,
               row_num_cls=2, icl_num_blocks=2, icl_nhead=2, ff_factor=2,
               feature_group_size=3, num_freq=32, decoder_hidden=None,
               is_classifier=True, race_context_mode="none", race_context_dim=32,
               race_context_layers=1, race_context_heads=2,
               race_context_ff_dim=64, race_context_residual=True,
               encode_races_before_icl=False,
               strict_input_validation=False):
    positive_integer_args = {
        "embed_dim": embed_dim, "max_classes": max_classes,
        "col_num_blocks": col_num_blocks, "col_nhead": col_nhead,
        "col_num_inds": col_num_inds, "row_num_blocks": row_num_blocks,
        "row_nhead": row_nhead, "row_num_cls": row_num_cls,
        "icl_num_blocks": icl_num_blocks, "icl_nhead": icl_nhead,
        "ff_factor": ff_factor, "feature_group_size": feature_group_size,
        "num_freq": num_freq, "race_context_dim": race_context_dim,
        "race_context_layers": race_context_layers,
        "race_context_heads": race_context_heads,
        "race_context_ff_dim": race_context_ff_dim,
    }
    for name, value in positive_integer_args.items():
      if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    divisibility = (
        ("embed_dim", embed_dim, "col_nhead", col_nhead),
        ("embed_dim", embed_dim, "row_nhead", row_nhead),
        ("embed_dim * row_num_cls", embed_dim * row_num_cls,
         "icl_nhead", icl_nhead),
        ("race_context_dim", race_context_dim,
         "race_context_heads", race_context_heads),
    )
    for left_name, left, right_name, right in divisibility:
      if left % right != 0:
        raise ValueError(f"{left_name} must be divisible by {right_name}; got {left} and {right}.")
    super().__init__()
    self.max_classes = max_classes
    self.is_classifier = is_classifier
    self.strict_input_validation = strict_input_validation
    if race_context_mode not in {"none", "self_attention"}:
      raise ValueError("race_context_mode must be 'none' or 'self_attention'")
    if race_context_mode == "self_attention" and not is_classifier:
      raise ValueError("race self-attention is currently classification-only")
    if race_context_dim % race_context_heads != 0:
      raise ValueError("race_context_dim must be divisible by race_context_heads")
    if race_context_layers < 1 or race_context_heads < 1:
      raise ValueError("race context layers and heads must be positive")
    self.race_context_mode = race_context_mode
    self.race_context_residual = race_context_residual
    self.encode_races_before_icl = bool(encode_races_before_icl)
    ff = embed_dim * ff_factor
    icl_dim = embed_dim * row_num_cls
    self.cell_embedder = CellEmbedder(embed_dim, max_classes, feature_group_size,
                                      num_freq, is_classifier)
    self.col_embedder = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff, col_num_inds)
    self.col_embedder_2 = ColEmbedding(embed_dim, col_num_blocks, col_nhead, ff, col_num_inds)
    self.row_interactor = RowInteraction(embed_dim, row_num_blocks, row_nhead, ff,
                                         row_num_cls, output_full=True)
    self.row_interactor_2 = RowInteraction(embed_dim, row_num_blocks, row_nhead, ff,
                                           row_num_cls, output_full=False)
    self.cls_tokens = nn.Parameter(torch.zeros(row_num_cls, embed_dim))
    self.icl_predictor = ICLearning(icl_dim, icl_num_blocks, icl_nhead, max_classes,
                                    icl_dim * ff_factor,
                                    decoder_hidden or icl_dim * 2, is_classifier)
    self.pre_icl_race_encoder = (
        RaceSetEncoder(icl_dim, race_context_dim, race_context_heads,
                       race_context_layers, race_context_ff_dim,
                       residual=race_context_residual)
        if self.encode_races_before_icl else None
    )
    self.race_set_head = (
        RaceSetHead(icl_dim, race_context_dim, race_context_heads,
                    race_context_layers, race_context_ff_dim, max_classes)
        if race_context_mode == "self_attention" else None
    )

    # Enable activation chunking by default (see the module constants above).
    # The per-block knobs stay settable (e.g. to None) for benchmarking.
    for module in self.modules():
      if hasattr(module, "row_chunk_size"):
        module.row_chunk_size = _ROW_CHUNK_SIZE
      if hasattr(module, "col_chunk_size"):
        module.col_chunk_size = _COL_CHUNK_SIZE
      if hasattr(module, "ffn_chunk_size"):
        module.ffn_chunk_size = _FFN_CHUNK_SIZE

  def _validate_race_groups(self, x, train_size, race_group_ids,
                            valid_row_mask=None):
    if race_group_ids is None:
      raise ValueError("race_group_ids are required when race context is enabled")
    if race_group_ids.shape != x.shape[:2]:
      raise ValueError("race_group_ids must have shape [batch_size, sequence_rows]")
    if race_group_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
      raise ValueError("race_group_ids must be an integer tensor")
    if race_group_ids.device != x.device:
      raise ValueError("race_group_ids must be on the same device as x")
    if valid_row_mask is None:
      raise ValueError("valid_row_mask is required when race context is enabled")
    if valid_row_mask.shape != x.shape[:2] or valid_row_mask.dtype != torch.bool:
      raise ValueError("valid_row_mask must be a bool tensor matching x batch and sequence dimensions")
    if valid_row_mask.device != x.device:
      raise ValueError("valid_row_mask must be on the same device as x")
    if torch.any((~valid_row_mask) & (race_group_ids != -1)):
      raise ValueError("padding rows must have race group -1")
    if self.encode_races_before_icl:
      if torch.any(valid_row_mask & (race_group_ids < 0)):
        raise ValueError("every valid historical and query row must have a non-negative race group")
      for batch_index in range(x.shape[0]):
        context_end = int(train_size[batch_index].item())
        context_ids = set(race_group_ids[batch_index, :context_end][valid_row_mask[batch_index, :context_end]].detach().cpu().tolist())
        query_ids = set(race_group_ids[batch_index, context_end:][valid_row_mask[batch_index, context_end:]].detach().cpu().tolist())
        overlap = context_ids.intersection(query_ids)
        if overlap:
          raise ValueError(f"historical and query rows share race groups: {sorted(overlap)}")
      return
    for batch_index in range(x.shape[0]):
      context_rows = int(train_size[batch_index])
      context_mask = valid_row_mask[batch_index, :context_rows]
      if torch.any(context_mask & (race_group_ids[batch_index, :context_rows] != -1)):
        raise ValueError("historical context rows must have race group -1")
      query_mask = valid_row_mask[batch_index, context_rows:]
      query_groups = race_group_ids[batch_index, context_rows:]
      if torch.any(query_mask & (query_groups < 0)):
        raise ValueError("every valid query row must have a non-negative race group")

  def _race_condition(self, base_logits, hidden, race_group_ids,
                      return_race_delta=False):
    delta = self.race_set_head(hidden, race_group_ids)
    conditioned = base_logits + delta if self.race_context_residual else delta
    return (conditioned, delta) if return_race_delta else conditioned

  def _validate_targets(self, y, train_size, *, require_padding_after_context):
    if not torch.isfinite(y).all():
      raise ValueError("y must contain only finite values.")
    if self.is_classifier:
      _validate_classification_labels(
          y, train_size, self.max_classes,
          require_padding_after_context=require_padding_after_context)

  @staticmethod
  def _cache_tensor(tensor, cache_name):
    if isinstance(tensor, QuantizedTensor):
      tensor = tensor.data
    if not isinstance(tensor, torch.Tensor):
      raise RuntimeError(f"{cache_name} must be a tensor or QuantizedTensor.")
    return tensor

  def _cache_architecture_metadata(self):
    race_module = self.race_set_head or self.pre_icl_race_encoder
    return {
        "embed_dim": self.cls_tokens.shape[1],
        "max_classes": self.max_classes,
        "is_classifier": self.is_classifier,
        "num_cls_tokens": self.cls_tokens.shape[0],
        "col_num_blocks": len(self.col_embedder.tf_col.blocks),
        "col_nhead": self.col_embedder.tf_col.blocks[0].mab1.attn.nhead,
        "row_num_blocks": len(self.row_interactor.tf_row.blocks),
        "row_nhead": self.row_interactor.tf_row.blocks[0].attn.nhead,
        "icl_num_blocks": len(self.icl_predictor.tf_icl.blocks),
        "icl_nhead": self.icl_predictor.tf_icl.blocks[0].attn.nhead,
        "race_context_mode": self.race_context_mode,
        "encode_races_before_icl": self.pre_icl_race_encoder is not None,
        "race_context_dim": (None if race_module is None else
                             race_module.encoder.layers[0].linear1.in_features),
        "race_context_layers": (None if race_module is None else
                                 len(race_module.encoder.layers)),
        "race_context_heads": (None if race_module is None else
                                race_module.encoder.layers[0].self_attn.num_heads),
        "race_context_ff_dim": (None if race_module is None else
                                 race_module.encoder.layers[0].linear1.out_features),
        "race_context_residual": self.race_context_residual,
    }

  def _validate_decode_cache(self, cache, x, d, cat_mask):
    if not isinstance(cache, dict):
      raise ValueError("cache must be the dictionary returned by prefill().")
    required = {"col1", "col2", "icl", "metadata"}
    if set(cache) != required:
      raise ValueError(f"cache must contain exactly {sorted(required)}, got {sorted(cache)}.")
    metadata = cache["metadata"]
    if not isinstance(metadata, dict) or metadata.get("format_version") != _CACHE_FORMAT_VERSION:
      raise ValueError("cache metadata has an unsupported or missing format_version.")
    expected_race_context = self.pre_icl_race_encoder is not None
    if metadata.get("pre_icl_race_context") != expected_race_context:
      raise ValueError("cache pre_icl_race_context does not match the current model configuration.")
    expected_label_mode = "icl_only" if expected_race_context else "cell_and_icl"
    if metadata.get("label_injection_mode") != expected_label_mode:
      raise ValueError("cache label_injection_mode does not match the current model configuration.")
    expected_architecture = self._cache_architecture_metadata()
    cached_architecture = metadata.get("architecture")
    if not isinstance(cached_architecture, dict):
      raise ValueError("cache is missing architecture compatibility metadata.")
    for name, expected in expected_architecture.items():
      if cached_architecture.get(name) != expected:
        raise ValueError(
            f"cache architecture field {name}={cached_architecture.get(name)!r} "
            f"does not match current model value {expected!r}.")
    batch_size, _, feature_count = x.shape
    if metadata.get("batch_size") != batch_size:
      raise ValueError(f"Cache batch size {metadata.get('batch_size')} does not match decode batch size {batch_size}.")
    if metadata.get("feature_count") != feature_count:
      raise ValueError(f"Cache feature count {metadata.get('feature_count')} does not match decode feature count {feature_count}.")
    supplied_d = None if d is None else tuple(int(v) for v in d.detach().cpu())
    if metadata.get("d") != supplied_d:
      raise ValueError(f"decode d {supplied_d} does not match prefill d {metadata.get('d')}.")
    cached_cat = metadata.get("cat_mask")
    supplied_cat = None if cat_mask is None else cat_mask.detach().to(device="cpu", dtype=torch.bool)
    if (cached_cat is None) != (supplied_cat is None) or (
        cached_cat is not None and not torch.equal(cached_cat.detach().cpu(), supplied_cat)):
      raise ValueError("decode cat_mask does not match the schema used during prefill().")

    col_specs = (
        ("col1", cache["col1"], len(self.col_embedder.tf_col.blocks),
         batch_size * feature_count),
        ("col2", cache["col2"], len(self.col_embedder_2.tf_col.blocks),
         batch_size * (feature_count + self.cls_tokens.shape[0])),
    )
    for name, layers, depth, expected_first_dim in col_specs:
      if not isinstance(layers, (list, tuple)):
        raise RuntimeError(f"{name} cache must be a per-layer sequence.")
      _validate_cache_depth(layers, depth, name)
      for index, hidden in enumerate(layers):
        block = (self.col_embedder.tf_col.blocks[index] if name == "col1" else
                 self.col_embedder_2.tf_col.blocks[index])
        expected_shape = (expected_first_dim, block.ind_vectors.shape[0],
                          block.ind_vectors.shape[1])
        if not isinstance(hidden, torch.Tensor):
          raise RuntimeError(f"{name} layer {index} must be a tensor.")
        if hidden.shape != expected_shape:
          raise ValueError(f"{name} layer {index} has shape {tuple(hidden.shape)}, expected {expected_shape} for batch times feature-column structure.")
        if hidden.device != x.device:
          raise ValueError(f"{name} layer {index} is on {hidden.device}, but decode x is on {x.device}.")

    icl = cache["icl"]
    if not isinstance(icl, ICLearningCache):
      raise RuntimeError("icl cache must be an ICLearningCache.")
    layers = icl.layer_caches
    depth = len(self.icl_predictor.tf_icl.blocks)
    _validate_cache_depth(layers, depth, "icl")
    if icl.prefill_train_size.shape != (batch_size,):
      raise ValueError(f"Cached train_size has shape {tuple(icl.prefill_train_size.shape)}, expected ({batch_size},).")
    if icl.prefill_train_size.device != x.device:
      raise ValueError(f"Cached train_size is on {icl.prefill_train_size.device}, but decode x is on {x.device}.")
    for index, pair in enumerate(layers):
      if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise RuntimeError(f"ICL cache layer {index} must be a two-element (k, v) pair.")
      k = self._cache_tensor(pair[0], f"ICL cache layer {index} K")
      v = self._cache_tensor(pair[1], f"ICL cache layer {index} V")
      if k.shape != v.shape:
        raise ValueError(f"ICL cache layer {index} K shape {tuple(k.shape)} does not match V shape {tuple(v.shape)}.")
      attn = self.icl_predictor.tf_icl.blocks[index].attn
      if k.ndim != 4 or k.shape[0] != batch_size:
        raise ValueError(f"ICL cache layer {index} must have shape [batch, rows, heads, head_dim], got {tuple(k.shape)}.")
      if k.shape[2:] != (attn.nhead, attn.hd):
        raise ValueError(f"ICL cache layer {index} has heads/head_dim {tuple(k.shape[2:])}, expected {(attn.nhead, attn.hd)}.")
      if k.device != x.device or v.device != x.device:
        raise ValueError(f"ICL cache layer {index} tensors must be on decode device {x.device}.")

    # A cache captures activations under the weights present at prefill time.
    # Rebuild it after changing or loading model weights; cache compatibility
    # metadata cannot detect arbitrary weight changes.

  def forward(self, x, y, train_size, cat_mask=None, d=None, race_group_ids=None,
              return_race_delta=False, valid_row_mask=None):
    # Mirror the JAX model's entry: replace NaN with the -100 sentinel and cast
    # to the compute dtype (JAX: `jnp.nan_to_num(X, nan=-100.0).astype(self.dtype)`).
    # NaN is already imputed in the shared preprocessing, so nan_to_num is a
    # no-op in the normal flow, but it keeps the model robust + JAX-faithful.
    _validate_runtime_inputs(x, y, train_size, d=d, cat_mask=cat_mask)
    if self.strict_input_validation and not torch.isfinite(x).all():
      raise ValueError("x must contain only finite values when strict_input_validation is enabled.")
    self._validate_targets(y, train_size, require_padding_after_context=False)
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    requires_race_groups = (
        self.pre_icl_race_encoder is not None
        or self.race_context_mode == "self_attention"
    )
    if requires_race_groups:
      if valid_row_mask is None:
        positions = torch.arange(x.shape[1], device=x.device)[None, :]
        has_label_padding = torch.any(
            (positions >= train_size[:, None]) & (y == -100.0)
        )
        if has_label_padding:
          raise ValueError(
              "valid_row_mask is required for padded full forward inputs")
        valid_row_mask = torch.ones(
            x.shape[:2], dtype=torch.bool, device=x.device)
      self._validate_race_groups(x, train_size, race_group_ids,
                                 valid_row_mask=valid_row_mask)
    emb = self.cell_embedder(
        x, y, train_size, cat_mask, d=d,
        inject_context_labels=not self.encode_races_before_icl,
    )
    emb = self.col_embedder(emb, train_size)
    b, t, _, e = emb.shape
    cls = self.cls_tokens.expand(b, t, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, train_size)
    reps = self.row_interactor_2(emb, d=d)
    if self.pre_icl_race_encoder is not None:
      reps = self.pre_icl_race_encoder(reps, race_group_ids)
    if self.race_context_mode == "none":
      if return_race_delta:
        raise ValueError("return_race_delta requires race_context_mode=self_attention")
      return self.icl_predictor(reps, y, train_size)
    base_logits, hidden = self.icl_predictor(
        reps, y, train_size, return_hidden=True
    )
    post_groups = race_group_ids
    if self.encode_races_before_icl:
      positions = torch.arange(x.shape[1], device=x.device)[None, :]
      post_groups = torch.where(positions < train_size[:, None], -1, race_group_ids)
    return self._race_condition(base_logits, hidden, post_groups,
                                return_race_delta=return_race_delta)

  def prefill(self, x, y, cat_mask=None, d=None, race_group_ids=None):
    """Encodes context (training) rows once and returns (logits, cache).

    x is [B, T, H] context rows and y is [B, T] context labels; cat_mask is an
    optional [B, H] categorical-feature mask and d an optional [B] active-
    feature count. Pads the sequence to a multiple of 128 with the -100
    sentinel, derives train_size from the non-sentinel y entries, runs the full
    pipeline while collecting the col-embedder induced-point reprs and the ICL
    encoder per-layer K/V, and unpads the logits to the original length. cache
    is a dict with keys 'col1', 'col2' (per-block induced-point reprs) and
    'icl' (an ICLearningCache).
    """
    if x.ndim != 3:
      raise ValueError(f"x must have shape [batch, rows, features], got {tuple(x.shape)}.")
    if y.ndim != 2 or y.shape != x.shape[:2]:
      raise ValueError(f"y must have shape {tuple(x.shape[:2])}, got {tuple(y.shape)}.")
    is_valid = _validate_contiguous_context_labels(y)
    train_size = is_valid.sum(dim=-1).to(torch.long)
    _validate_runtime_inputs(x, y, train_size, d=d, cat_mask=cat_mask)
    if self.strict_input_validation and not torch.isfinite(x).all():
      raise ValueError("x must contain only finite values when strict_input_validation is enabled.")
    self._validate_targets(y, train_size, require_padding_after_context=True)
    if self.pre_icl_race_encoder is not None:
      self._validate_race_groups(x, train_size, race_group_ids,
                                 valid_row_mask=is_valid)
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    y = y.to(self.cls_tokens.dtype)

    t_orig = x.shape[1]
    block_size = 128
    pad_len = ((t_orig - 1) // block_size + 1) * block_size - t_orig
    if pad_len > 0:
      x = F.pad(x, (0, 0, 0, pad_len), value=-100.0)
      y = F.pad(y, (0, pad_len), value=-100.0)
      is_valid = F.pad(is_valid, (0, pad_len), value=False)
      if race_group_ids is not None:
        race_group_ids = F.pad(race_group_ids, (0, pad_len), value=-1)

    cell = self.cell_embedder(
        x, y, train_size, cat_mask, d=d,
        inject_context_labels=not self.encode_races_before_icl,
    )
    emb, cache_col1 = self.col_embedder(cell, train_size, return_repr=True)
    b1, t1, _, e = emb.shape
    cls = self.cls_tokens.expand(b1, t1, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb, cache_col2 = self.col_embedder_2(emb, train_size, return_repr=True)
    reps = self.row_interactor_2(emb, d=d)
    if self.pre_icl_race_encoder is not None:
      reps = self.pre_icl_race_encoder(reps, race_group_ids)
    logits, cache_icl = self.icl_predictor(reps, y, train_size, return_cache=True)

    cache = {
        "col1": cache_col1,
        "col2": cache_col2,
        "icl": cache_icl,
        "metadata": {
            "format_version": _CACHE_FORMAT_VERSION,
            "batch_size": x.shape[0],
            "feature_count": x.shape[2],
            "d": None if d is None else tuple(int(v) for v in d.detach().cpu()),
            "cat_mask": (None if cat_mask is None else
                         cat_mask.detach().to(device="cpu", dtype=torch.bool).clone()),
            "pre_icl_race_context": self.pre_icl_race_encoder is not None,
            "label_injection_mode": "icl_only" if self.pre_icl_race_encoder is not None else "cell_and_icl",
            "architecture": self._cache_architecture_metadata(),
        },
    }
    return logits[:, :t_orig, :], detach_cache(cache)

  def decode(self, x, cache, cat_mask=None, d=None, race_group_ids=None):
    """Generates predictions for test rows using a cache from prefill.

    x is [B, T, H] test rows and cache is the dict returned by prefill;
    cat_mask and d are as in prefill. Pads to a multiple of 128, re-runs the
    row-independent cell embedder and row interactors on the test rows, reuses
    the cached col-embedder reprs and ICL per-layer K/V instead of recomputing
    them from the context, and unpads to the original test length. Returns
    [B, T, K_or_1] logits.
    """
    if x.ndim != 3:
      raise ValueError(f"x must have shape [batch, rows, features], got {tuple(x.shape)}.")
    dummy_y = torch.empty(x.shape[:2], device=x.device)
    zero_train_size = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    _validate_runtime_inputs(x, dummy_y, zero_train_size, d=d, cat_mask=cat_mask)
    if self.strict_input_validation and not torch.isfinite(x).all():
      raise ValueError("x must contain only finite values when strict_input_validation is enabled.")
    self._validate_decode_cache(cache, x, d, cat_mask)
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    b, t_orig, _ = x.shape
    requires_race_groups = (
        self.pre_icl_race_encoder is not None
        or self.race_context_mode == "self_attention"
    )
    valid_row_mask = torch.ones((b, t_orig), dtype=torch.bool, device=x.device)
    if requires_race_groups:
      decode_train_size = torch.zeros(b, dtype=torch.long, device=x.device)
      self._validate_race_groups(x, decode_train_size, race_group_ids,
                                 valid_row_mask=valid_row_mask)

    block_size = 128
    pad_len = ((t_orig - 1) // block_size + 1) * block_size - t_orig
    if pad_len > 0:
      x = F.pad(x, (0, 0, 0, pad_len), value=-100.0)
      valid_row_mask = F.pad(valid_row_mask, (0, pad_len), value=False)
      if race_group_ids is not None:
        race_group_ids = F.pad(race_group_ids, (0, pad_len), value=-1)

    # y carries no information for test rows in decode (unlabeled); use the
    # -100 sentinel throughout, matching JAX.
    y = torch.full((b, x.shape[1]), -100.0, dtype=self.cls_tokens.dtype, device=x.device)
    train_size_zero = torch.zeros(b, dtype=torch.long, device=x.device)

    cell = self.cell_embedder(
        x, y, train_size_zero, cat_mask, d=d,
        inject_context_labels=not self.encode_races_before_icl,
    )
    emb = self.col_embedder(cell, None, cached_repr=cache["col1"])
    b1, t1, _, e = emb.shape
    cls = self.cls_tokens.expand(b1, t1, -1, -1)
    emb = torch.cat([cls, emb], dim=2)
    emb = self.row_interactor(emb, d=d)
    emb = self.col_embedder_2(emb, None, cached_repr=cache["col2"])
    reps = self.row_interactor_2(emb, d=d)
    if self.pre_icl_race_encoder is not None:
      reps = self.pre_icl_race_encoder(reps, race_group_ids)
    if self.race_context_mode == "none":
      out = self.icl_predictor(reps, y, None, cache=cache["icl"])
    else:
      base_logits, hidden = self.icl_predictor(
          reps, y, None, cache=cache["icl"], return_hidden=True
      )
      out = self._race_condition(base_logits, hidden, race_group_ids)

    return out[:, :t_orig, :]
