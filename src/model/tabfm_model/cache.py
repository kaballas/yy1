"""Cache dataclasses and recursive cache utilities."""

import dataclasses
from typing import Any, Dict, List, Tuple

import torch

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

