"""Cell, column, and row embedding modules."""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .activations import get_activation
from .normalisation import RMSNorm
from .transformers import Encoder, SetTransformer

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

