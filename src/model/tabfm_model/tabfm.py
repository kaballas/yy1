"""Top-level TabFM orchestration module."""

import hashlib

import torch
import torch.nn.functional as F
from torch import nn

from .cache import ICLearningCache, QuantizedTensor, detach_cache
from .embeddings import CellEmbedder, ColEmbedding, RowInteraction
from .icl import ICLearning
from .race_context import RaceSetEncoder, RaceSetHead
from .validation import (
    _validate_cache_depth,
    _validate_classification_labels,
    _validate_contiguous_context_mask,
    _validate_contiguous_context_labels,
    _validate_runtime_inputs,
)

_ROW_CHUNK_SIZE = 4096   # rows per chunk (Fourier cell embedding + row interaction)
_COL_CHUNK_SIZE = 16     # feature-instances per chunk (column set-transformer)
_FFN_CHUNK_SIZE = 8192   # tokens per chunk (feed-forward expansion in every block)
_CACHE_FORMAT_VERSION = 3


class TabFM(nn.Module):
  def __init__(self, *, embed_dim=8, max_classes=10, col_num_blocks=2,
               col_nhead=2, col_num_inds=4, row_num_blocks=2, row_nhead=2,
               row_num_cls=2, icl_num_blocks=2, icl_nhead=2, ff_factor=2,
               feature_group_size=3, num_freq=32, decoder_hidden=None,
               is_classifier=True, race_context_mode="none", race_context_dim=32,
               race_context_layers=1, race_context_heads=2,
               race_context_ff_dim=64, race_context_residual=True,
               encode_races_before_icl=False,
               strict_input_validation=False, max_runners_per_race=256,
               checkpoint_id=None, feature_schema_hash=None,
               preprocessing_version=None):
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
        "max_runners_per_race": max_runners_per_race,
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
    if max_runners_per_race < 1:
      raise ValueError("max_runners_per_race must be positive")
    self.max_runners_per_race = max_runners_per_race
    self.checkpoint_id = checkpoint_id
    self.feature_schema_hash = feature_schema_hash
    self.preprocessing_version = preprocessing_version
    if race_context_mode not in {"none", "self_attention"}:
      raise ValueError("race_context_mode must be 'none' or 'self_attention'")
    if race_context_mode == "self_attention" and not is_classifier:
      raise ValueError("race self-attention is currently classification-only")
    if (encode_races_before_icl or race_context_mode == "self_attention") and not race_context_residual:
      raise ValueError(
          "race_context_residual=False is incompatible with zero-initialised "
          "race projections")
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
                       residual=race_context_residual,
                       max_runners_per_race=max_runners_per_race)
        if self.encode_races_before_icl else None
    )
    self.race_set_head = (
        RaceSetHead(icl_dim, race_context_dim, race_context_heads,
                    race_context_layers, race_context_ff_dim, max_classes,
                    max_runners_per_race=max_runners_per_race)
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

  def _model_state_id(self):
    digest = hashlib.sha256()
    for name, tensor in self.state_dict().items():
      digest.update(name.encode("utf-8"))
      digest.update(str(tensor.dtype).encode("ascii"))
      digest.update(str(tuple(tensor.shape)).encode("ascii"))
      digest.update(tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

  def _cache_compatibility_metadata(self, feature_schema_hash,
                                    preprocessing_version):
    if feature_schema_hash is None:
      feature_schema_hash = self.feature_schema_hash
    if preprocessing_version is None:
      preprocessing_version = self.preprocessing_version
    if feature_schema_hash is None:
      raise ValueError("feature_schema_hash is required for prefill/decode cache compatibility")
    if preprocessing_version is None:
      raise ValueError("preprocessing_version is required for prefill/decode cache compatibility")
    return {
        "model_state_id": self._model_state_id(),
        "checkpoint_id": self.checkpoint_id,
        "model_dtype": str(self.cls_tokens.dtype),
        "feature_schema_hash": str(feature_schema_hash),
        "preprocessing_version": str(preprocessing_version),
    }

  def _validate_decode_cache(self, cache, x, d, cat_mask,
                             feature_schema_hash, preprocessing_version):
    if not isinstance(cache, dict):
      raise ValueError("cache must be the dictionary returned by prefill().")
    required = {"col1", "col2", "icl", "metadata"}
    if set(cache) != required:
      raise ValueError(f"cache must contain exactly {sorted(required)}, got {sorted(cache)}.")
    metadata = cache["metadata"]
    if not isinstance(metadata, dict) or metadata.get("format_version") != _CACHE_FORMAT_VERSION:
      raise ValueError("cache metadata has an unsupported or missing format_version.")
    expected_compatibility = self._cache_compatibility_metadata(
        feature_schema_hash, preprocessing_version)
    for name, expected in expected_compatibility.items():
      if metadata.get(name) != expected:
        raise ValueError(
            f"cache compatibility field {name}={metadata.get(name)!r} "
            f"does not match current model value {expected!r}.")
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
        if hidden.dtype != self.cls_tokens.dtype:
          raise ValueError(f"{name} layer {index} has dtype {hidden.dtype}, expected {self.cls_tokens.dtype}.")

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
      for name, tensor in (("K", pair[0]), ("V", pair[1])):
        if isinstance(tensor, QuantizedTensor):
          if tensor.data.dtype != torch.int8 or tensor.scale.dtype != self.cls_tokens.dtype:
            raise ValueError(
                f"ICL cache layer {index} {name} quantized dtype does not match the model dtype.")
        elif tensor.dtype != self.cls_tokens.dtype:
          raise ValueError(
              f"ICL cache layer {index} {name} has dtype {tensor.dtype}, expected {self.cls_tokens.dtype}.")

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

  def prefill(self, x, y, cat_mask=None, d=None, race_group_ids=None,
              train_size=None, valid_context_mask=None,
              feature_schema_hash=None, preprocessing_version=None):
    """Encodes context (training) rows once and returns (logits, cache).

    x is [B, T, H] context rows and y is [B, T] context labels; cat_mask is an
    optional [B, H] categorical-feature mask and d an optional [B] active-
    feature count. Pads the sequence to a multiple of 128 with the -100
    sentinel. Classification may derive train_size from y; regression requires
    explicit train_size or valid_context_mask. The method runs the full
    pipeline while collecting the col-embedder induced-point reprs and the ICL
    encoder per-layer K/V, and unpads the logits to the original length. cache
    is a dict with keys 'col1', 'col2' (per-block induced-point reprs) and
    'icl' (an ICLearningCache).
    """
    if x.ndim != 3:
      raise ValueError(f"x must have shape [batch, rows, features], got {tuple(x.shape)}.")
    if y.ndim != 2 or y.shape != x.shape[:2]:
      raise ValueError(f"y must have shape {tuple(x.shape[:2])}, got {tuple(y.shape)}.")
    if train_size is not None and valid_context_mask is not None:
      raise ValueError("prefill accepts either train_size or valid_context_mask, not both")
    if train_size is not None:
      if train_size.shape != (x.shape[0],) or train_size.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"train_size must have shape ({x.shape[0]},) and an integer dtype.")
      if train_size.device != x.device:
        raise ValueError("train_size must be on the same device as x")
      is_valid = (torch.arange(x.shape[1], device=x.device)[None, :] < train_size[:, None])
    elif valid_context_mask is not None:
      if valid_context_mask.shape != x.shape[:2] or valid_context_mask.device != x.device:
        raise ValueError("valid_context_mask must match x batch and row dimensions and device")
      is_valid = _validate_contiguous_context_mask(valid_context_mask)
      train_size = is_valid.sum(dim=-1).to(torch.long)
    elif self.is_classifier:
      is_valid = _validate_contiguous_context_labels(y)
      train_size = is_valid.sum(dim=-1).to(torch.long)
    else:
      raise ValueError(
          "regression prefill requires explicit train_size or valid_context_mask; "
          "target value -100 is a valid regression target")
    _validate_runtime_inputs(x, y, train_size, d=d, cat_mask=cat_mask)
    if torch.any(train_size < 0) or torch.any(train_size > x.shape[1]):
      raise ValueError(f"train_size values must be in [0, {x.shape[1]}].")
    compatibility = self._cache_compatibility_metadata(
        feature_schema_hash, preprocessing_version)
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
            **compatibility,
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

  def decode(self, x, cache, cat_mask=None, d=None, race_group_ids=None,
             valid_row_mask=None, feature_schema_hash=None,
             preprocessing_version=None):
    """Generates predictions for test rows using a cache from prefill.

    x is [B, T, H] test rows and cache is the dict returned by prefill;
    cat_mask and d are as in prefill. Pads to a multiple of 128, re-runs the
    row-independent cell embedder and row interactors on the test rows, reuses
    the cached col-embedder reprs and ICL per-layer K/V instead of recomputing
    them from the context, and unpads to the original test length. Optional
    valid_row_mask marks padded batch rows; those output rows are zeroed.
    Returns [B, T, K_or_1] logits.
    """
    if x.ndim != 3:
      raise ValueError(f"x must have shape [batch, rows, features], got {tuple(x.shape)}.")
    dummy_y = torch.empty(x.shape[:2], device=x.device)
    zero_train_size = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    _validate_runtime_inputs(x, dummy_y, zero_train_size, d=d, cat_mask=cat_mask)
    if self.strict_input_validation and not torch.isfinite(x).all():
      raise ValueError("x must contain only finite values when strict_input_validation is enabled.")
    self._validate_decode_cache(
        cache, x, d, cat_mask, feature_schema_hash, preprocessing_version)
    x = torch.nan_to_num(x, nan=-100.0).to(self.cls_tokens.dtype)
    b, t_orig, _ = x.shape
    requires_race_groups = (
        self.pre_icl_race_encoder is not None
        or self.race_context_mode == "self_attention"
    )
    if valid_row_mask is None:
      valid_row_mask = torch.ones((b, t_orig), dtype=torch.bool, device=x.device)
    elif (valid_row_mask.shape != (b, t_orig)
          or valid_row_mask.dtype != torch.bool
          or valid_row_mask.device != x.device):
      raise ValueError(
          "valid_row_mask must be a bool tensor matching x batch and row dimensions")
    original_valid_row_mask = valid_row_mask
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

    out = out[:, :t_orig, :]
    return torch.where(original_valid_row_mask[..., None], out, torch.zeros_like(out))
