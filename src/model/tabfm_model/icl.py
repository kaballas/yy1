
"""In-context learning prediction module."""

from typing import Optional

import torch
from torch import nn

from .cache import ICLearningCache
from .embeddings import MLP, OneHotAndLinear
from .normalisation import RMSNorm
from .transformers import Encoder

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
