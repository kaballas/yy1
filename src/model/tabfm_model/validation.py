"""Runtime and cache validation helpers for TabFM."""

import torch

_INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64,
                   torch.uint8)


def _validate_runtime_inputs(x, y, train_size, d=None, cat_mask=None):
  if x.ndim != 3:
    raise ValueError(f"x must have shape [batch, rows, features], got {tuple(x.shape)}.")
  batch_size, sequence_length, feature_count = x.shape
  if x.is_complex():
    raise ValueError(f"x must be real-valued, got {x.dtype}")
  if torch.isinf(x).any():
    raise ValueError("x must not contain positive or negative infinity")
  if sequence_length == 0:
    raise ValueError("x must contain at least one row")
  if feature_count == 0:
    raise ValueError("x must contain at least one feature")
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


def _validate_contiguous_context_mask(valid_context_mask):
  """Validate and return a contiguous context-prefix mask."""
  if valid_context_mask.ndim != 2 or valid_context_mask.dtype != torch.bool:
    raise ValueError("valid_context_mask must be a bool tensor with shape [batch, rows].")
  has_seen_padding = (~valid_context_mask).cumsum(dim=1) > 0
  invalid_ordering = valid_context_mask & has_seen_padding
  if torch.any(invalid_ordering):
    bad_batch, bad_row = torch.nonzero(invalid_ordering, as_tuple=True)
    raise ValueError(
        "prefill requires context rows to be a contiguous valid prefix followed only by trailing padding. "
        f"First invalid ordering at batch={bad_batch[0].item()}, row={bad_row[0].item()}.")
  return valid_context_mask


def _validate_cache_depth(cache, expected_depth, cache_name):
  if len(cache) != expected_depth:
    raise ValueError(f"{cache_name} contains {len(cache)} layers, but the model requires {expected_depth}.")
