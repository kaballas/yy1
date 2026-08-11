"""Race-group representation and prediction context modules."""

import math

import torch
from torch import nn


def _pack_race_rows(hidden, race_group_ids, max_runners_per_race):
  """Pack non-negative race rows into bounded, sorted dense race batches."""
  batch_size, sequence_length = race_group_ids.shape
  flat_groups = race_group_ids.reshape(-1)
  flat_batch = torch.arange(batch_size, device=race_group_ids.device)
  flat_batch = flat_batch[:, None].expand(batch_size, sequence_length).reshape(-1)
  valid = flat_groups >= 0
  if not torch.any(valid):
    return None

  source_positions = torch.nonzero(valid, as_tuple=False).flatten()
  groups = flat_groups[source_positions]
  batches = flat_batch[source_positions]
  order = torch.argsort(groups, stable=True)
  order = order[torch.argsort(batches[order], stable=True)]
  source_positions = source_positions[order]
  groups = groups[order]
  batches = batches[order]

  starts = torch.cat([
      torch.zeros(1, dtype=torch.long, device=groups.device),
      torch.nonzero(
          (batches[1:] != batches[:-1]) | (groups[1:] != groups[:-1]),
          as_tuple=False,
      ).flatten() + 1,
  ])
  ends = torch.cat([starts[1:], torch.tensor([groups.numel()], device=groups.device)])
  counts = ends - starts
  max_count = int(counts.max().item())
  if max_count > max_runners_per_race:
    raise ValueError(
        f"race contains {max_count} runners, exceeding the configured maximum "
        f"of {max_runners_per_race}.")
  segment_ids = torch.repeat_interleave(torch.arange(
      counts.numel(), device=groups.device), counts)
  offsets = torch.arange(groups.numel(), device=groups.device) - torch.repeat_interleave(starts, counts)
  packed = hidden.new_zeros((counts.numel(), max_count, hidden.shape[-1]))
  packed[segment_ids, offsets] = hidden.reshape(-1, hidden.shape[-1])[source_positions]
  padding_mask = torch.ones(
      (counts.numel(), max_count), dtype=torch.bool, device=hidden.device)
  padding_mask[segment_ids, offsets] = False
  return packed, padding_mask, source_positions, segment_ids, offsets


class RaceSetEncoder(nn.Module):
  """Race self-attention producing representation-level runner corrections.

  All runners belonging to one race must be present in the same call; splitting
  a race across decode calls changes race-conditioned predictions.
  """

  def __init__(self, input_dim, race_dim=32, attention_heads=2,
               attention_layers=1, feedforward_dim=64, residual=True,
               max_runners_per_race=256):
    super().__init__()
    if race_dim % attention_heads != 0:
      raise ValueError("race_dim must be divisible by attention_heads")
    if attention_layers < 1 or attention_heads < 1:
      raise ValueError("race encoder layers and heads must be positive")
    if not residual:
      raise ValueError(
          "race_context_residual=False is incompatible with zero-initialised "
          "race projections")
    self.residual = residual
    if max_runners_per_race < 1:
      raise ValueError("max_runners_per_race must be positive")
    self.max_runners_per_race = max_runners_per_race
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
    packed_info = _pack_race_rows(
        hidden, race_group_ids, self.max_runners_per_race)
    if packed_info is None:
      return hidden if self.residual else torch.zeros_like(hidden)
    padded, padding_mask, source_positions, segment_ids, offsets = packed_info
    encoded = self.encoder(self.input_projection(padded), src_key_padding_mask=padding_mask)
    delta = self.output_projection(self.output_norm(encoded))
    result = torch.zeros_like(hidden).reshape(-1, hidden.shape[-1])
    result[source_positions] = delta[segment_ids, offsets]
    result = result.reshape_as(hidden)
    return hidden + result if self.residual else result


class RaceSetHead(nn.Module):
  """Permutation-equivariant correction over runners in each target race.

  All runners belonging to one race must be present in the same decode call;
  splitting a race across calls changes race-conditioned predictions.
  """

  def __init__(self, input_dim, race_dim=32, attention_heads=2,
               attention_layers=1, feedforward_dim=64, output_classes=2,
               max_runners_per_race=256):
    super().__init__()
    if race_dim % attention_heads != 0:
      raise ValueError("race_context_dim must be divisible by race_context_heads")
    if attention_layers < 1 or attention_heads < 1:
      raise ValueError("race context layers and heads must be positive")
    self.input_projection = nn.Linear(input_dim, race_dim)
    if max_runners_per_race < 1:
      raise ValueError("max_runners_per_race must be positive")
    self.max_runners_per_race = max_runners_per_race
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
    packed_info = _pack_race_rows(
        hidden, race_group_ids, self.max_runners_per_race)
    if packed_info is None:
      return hidden.new_zeros((*hidden.shape[:2], self.output_projection.out_features))
    padded, padding_mask, source_positions, segment_ids, offsets = packed_info
    result = hidden.new_zeros(
        (hidden.shape[0] * hidden.shape[1], self.output_projection.out_features))
    encoded = self.input_projection(padded)
    encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
    delta = self.output_projection(self.output_norm(encoded))
    result[source_positions] = delta[segment_ids, offsets]
    return result.reshape(hidden.shape[0], hidden.shape[1], -1)


class ContextPrototypeHead(nn.Module):
  """Bounded query-logit correction from labelled context prototypes.

  Context rows are split by their binary label, projected into a shared metric
  space, and averaged into positive and negative prototypes.  Each query row is
  compared with both prototypes.  The head is deliberately small: it gives the
  historical label a direct connection to its runner representation while the
  normal ICL and race-set paths remain responsible for the base prediction.
  """

  def __init__(self, input_dim, prototype_dim=16, output_classes=2,
               max_correction=0.5):
    super().__init__()
    if prototype_dim < 1:
      raise ValueError("context_prototype_dim must be positive")
    if output_classes != 2:
      raise ValueError("context prototypes currently require binary classification")
    if not isinstance(max_correction, (int, float)) or max_correction <= 0:
      raise ValueError("context_prototype_max_correction must be positive")
    self.prototype_dim = prototype_dim
    self.output_classes = output_classes
    self.max_correction = float(max_correction)
    self.projection = nn.Linear(input_dim, prototype_dim)
    self.projection_norm = nn.LayerNorm(prototype_dim)
    # A small non-zero initial gain makes label permutation observable from the
    # first probe without allowing this new branch to dominate base logits.
    self.raw_logit_gain = nn.Parameter(torch.tensor(-2.2521685))

  def project(self, hidden):
    return torch.nn.functional.normalize(
        self.projection_norm(self.projection(hidden)), dim=-1, eps=1e-6)

  def build_prototypes(self, hidden, labels, train_size, valid_row_mask=None):
    if hidden.shape[:2] != labels.shape:
      raise ValueError("labels must match hidden batch and sequence dimensions")
    if train_size.shape != (hidden.shape[0],):
      raise ValueError("train_size must have one value per batch sequence")
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    context_mask = positions < train_size[:, None]
    if valid_row_mask is not None:
      if valid_row_mask.shape != labels.shape or valid_row_mask.dtype != torch.bool:
        raise ValueError("valid_row_mask must be bool and match context labels")
      context_mask &= valid_row_mask
    if torch.any(context_mask & ~((labels == 0) | (labels == 1))):
      raise ValueError("context prototype labels must be binary")

    projected = self.project(hidden)
    prototypes = []
    class_valid = []
    for class_index in (0, 1):
      class_mask = context_mask & (labels == class_index)
      count = class_mask.sum(dim=1)
      summed = (projected * class_mask[..., None]).sum(dim=1)
      prototype = summed / count.clamp_min(1).to(projected.dtype)[:, None]
      prototypes.append(torch.nn.functional.normalize(
          prototype, dim=-1, eps=1e-6))
      class_valid.append(count > 0)
    return torch.stack(prototypes, dim=1), torch.stack(class_valid, dim=1).all(dim=1)

  def correction_from_prototypes(self, hidden, prototypes, prototype_valid,
                                 query_mask):
    expected = (hidden.shape[0], 2, self.prototype_dim)
    if prototypes.shape != expected:
      raise ValueError(
          f"context prototypes have shape {tuple(prototypes.shape)}, expected {expected}")
    if prototype_valid.shape != (hidden.shape[0],):
      raise ValueError("prototype_valid must have one value per batch sequence")
    if query_mask.shape != hidden.shape[:2] or query_mask.dtype != torch.bool:
      raise ValueError("query_mask must be bool and match hidden rows")
    projected = self.project(hidden)
    negative_similarity = (projected * prototypes[:, 0, None, :]).sum(dim=-1)
    positive_similarity = (projected * prototypes[:, 1, None, :]).sum(dim=-1)
    gain = torch.nn.functional.softplus(self.raw_logit_gain)
    score = gain * (positive_similarity - negative_similarity)
    score = self.max_correction * torch.tanh(score)
    active = query_mask & prototype_valid[:, None]
    score = torch.where(active, score, torch.zeros_like(score))
    return torch.stack((-score, score), dim=-1)

  def forward(self, hidden, labels, train_size, valid_row_mask=None):
    prototypes, prototype_valid = self.build_prototypes(
        hidden, labels, train_size, valid_row_mask=valid_row_mask)
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    query_mask = positions >= train_size[:, None]
    if valid_row_mask is not None:
      query_mask &= valid_row_mask
    return self.correction_from_prototypes(
        hidden, prototypes, prototype_valid, query_mask)


class LabelAwareContextHead(nn.Module):
  """Cross-attend query runners to labelled historical runner representations.

  In the preferred mode, historical runner representations alone form the
  attention keys, while their representations plus learned outcome embeddings
  form the values. Thus runner similarity chooses the evidence and labels
  determine what is retrieved. Query labels are never read, and corrections
  are emitted only for valid query rows.
  """

  def __init__(self, input_dim, attention_heads=2, output_classes=2,
               max_correction=0.5, labels_in_values_only=False,
               temperature=1.0, top_k=0):
    super().__init__()
    if input_dim % attention_heads:
      raise ValueError("label context input dimension must be divisible by its heads")
    if attention_heads < 1:
      raise ValueError("label_context_heads must be positive")
    if output_classes != 2:
      raise ValueError("label-aware context currently requires binary classification")
    if not isinstance(max_correction, (int, float)) or max_correction <= 0:
      raise ValueError("label_context_max_correction must be positive")
    if (isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature <= 0):
      raise ValueError("label_context_temperature must be finite and positive")
    if top_k is None:
      top_k = 0
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
      raise ValueError("label_context_top_k must be a non-negative integer")
    self.input_dim = input_dim
    self.attention_heads = attention_heads
    self.output_classes = output_classes
    self.max_correction = float(max_correction)
    self.labels_in_values_only = bool(labels_in_values_only)
    self.temperature = float(temperature)
    self.top_k = int(top_k)
    self.label_embedding = nn.Embedding(2, input_dim)
    self.query_norm = nn.LayerNorm(input_dim)
    self.memory_norm = nn.LayerNorm(input_dim)
    self.cross_attention = nn.MultiheadAttention(
        input_dim, attention_heads, dropout=0.0, batch_first=True)
    self.output_norm = nn.LayerNorm(input_dim)
    self.output_projection = nn.Linear(input_dim, output_classes)
    nn.init.zeros_(self.output_projection.weight)
    nn.init.zeros_(self.output_projection.bias)

  def __setstate__(self, state):
    """Preserve the original keys-and-values behavior for old module checkpoints."""
    super().__setstate__(state)
    if not hasattr(self, "labels_in_values_only"):
      self.labels_in_values_only = False
    if not hasattr(self, "temperature"):
      self.temperature = 1.0
    if not hasattr(self, "top_k"):
      self.top_k = 0

  @staticmethod
  def _masks(labels, train_size, valid_row_mask):
    positions = torch.arange(labels.shape[1], device=labels.device)[None, :]
    context_mask = positions < train_size[:, None]
    query_mask = positions >= train_size[:, None]
    if valid_row_mask is not None:
      if valid_row_mask.shape != labels.shape or valid_row_mask.dtype != torch.bool:
        raise ValueError("valid_row_mask must be bool and match label context rows")
      context_mask &= valid_row_mask
      query_mask &= valid_row_mask
    if torch.any(context_mask & ~((labels == 0) | (labels == 1))):
      raise ValueError("label-aware context labels must be binary")
    if torch.any(context_mask.sum(dim=1) == 0):
      raise ValueError("label-aware context requires at least one historical row")
    return context_mask, query_mask

  def correction_from_context(self, query_representations,
                              context_representations, context_labels,
                              context_valid_mask=None,
                              query_valid_mask=None,
                              return_attention=False,
                              return_attention_diagnostics=False,
                              temperature=None,
                              top_k=None,
                              force_uniform_attention=False,
                              zero_label_embedding=False):
    if query_representations.ndim != 3 or context_representations.ndim != 3:
      raise ValueError("label context representations must have shape [B, T, E]")
    if query_representations.shape[0] != context_representations.shape[0]:
      raise ValueError("label context query and history batch sizes must match")
    if query_representations.shape[-1] != self.input_dim or context_representations.shape[-1] != self.input_dim:
      raise ValueError("label context representation dimensions do not match the head")
    if context_labels.shape != context_representations.shape[:2]:
      raise ValueError("context_labels must match historical representations")
    if context_valid_mask is None:
      context_valid_mask = torch.ones_like(context_labels, dtype=torch.bool)
    if (context_valid_mask.shape != context_labels.shape
        or context_valid_mask.dtype != torch.bool):
      raise ValueError("context_valid_mask must be bool and match context labels")
    if query_valid_mask is None:
      query_valid_mask = torch.ones(
          query_representations.shape[:2], dtype=torch.bool,
          device=query_representations.device)
    if (query_valid_mask.shape != query_representations.shape[:2]
        or query_valid_mask.dtype != torch.bool):
      raise ValueError("query_valid_mask must be bool and match query representations")
    if torch.any(context_valid_mask.sum(dim=1) == 0):
      raise ValueError("every label context sequence needs historical rows")
    if torch.any(context_valid_mask & ~((context_labels == 0) | (context_labels == 1))):
      raise ValueError("valid label context rows must have binary labels")

    effective_temperature = self.temperature if temperature is None else temperature
    if (isinstance(effective_temperature, bool)
        or not isinstance(effective_temperature, (int, float))
        or not math.isfinite(float(effective_temperature))
        or effective_temperature <= 0):
      raise ValueError("label-context attention temperature must be finite and positive")
    effective_top_k = self.top_k if top_k is None else top_k
    if effective_top_k is None:
      effective_top_k = 0
    if (isinstance(effective_top_k, bool)
        or not isinstance(effective_top_k, int)
        or effective_top_k < 0):
      raise ValueError("label-context attention top_k must be a non-negative integer")

    safe_labels = torch.where(
        context_valid_mask, context_labels, torch.zeros_like(context_labels)
    ).long()
    label_values = self.label_embedding(safe_labels)
    if zero_label_embedding:
      label_values = torch.zeros_like(label_values)
    labelled_memory = self.memory_norm(context_representations + label_values)
    attention_keys = (
        self.memory_norm(context_representations)
        if self.labels_in_values_only else labelled_memory
    )
    normalized_query = self.query_norm(query_representations)
    projection_weight = self.cross_attention.in_proj_weight
    projection_bias = self.cross_attention.in_proj_bias
    query_weight, key_weight, value_weight = projection_weight.chunk(3, dim=0)
    if projection_bias is None:
      query_bias = key_bias = value_bias = None
    else:
      query_bias, key_bias, value_bias = projection_bias.chunk(3, dim=0)
    projected_query = torch.nn.functional.linear(
        normalized_query, query_weight, query_bias)
    projected_key = torch.nn.functional.linear(
        attention_keys, key_weight, key_bias)
    projected_value = torch.nn.functional.linear(
        labelled_memory, value_weight, value_bias)
    head_dim = self.input_dim // self.attention_heads
    query_heads = projected_query.reshape(
        *projected_query.shape[:2], self.attention_heads, head_dim
    ).transpose(1, 2)
    key_heads = projected_key.reshape(
        *projected_key.shape[:2], self.attention_heads, head_dim
    ).transpose(1, 2)
    value_heads = projected_value.reshape(
        *projected_value.shape[:2], self.attention_heads, head_dim
    ).transpose(1, 2)
    base_attention_logits = torch.matmul(
        query_heads, key_heads.transpose(-2, -1)
    ) / math.sqrt(head_dim)
    if force_uniform_attention:
      attention_logits = torch.zeros_like(base_attention_logits)
    else:
      attention_logits = base_attention_logits / float(effective_temperature)
    valid_attention = context_valid_mask[:, None, None, :].expand_as(
        attention_logits)
    attention_logits = attention_logits.masked_fill(
        ~valid_attention, float("-inf"))
    if not force_uniform_attention and effective_top_k > 0:
      selection_count = min(effective_top_k, attention_logits.shape[-1])
      selected_indices = torch.topk(
          attention_logits, k=selection_count, dim=-1).indices
      selected_mask = torch.zeros_like(valid_attention)
      selected_mask.scatter_(-1, selected_indices, True)
      attention_logits = attention_logits.masked_fill(
          ~(selected_mask & valid_attention), float("-inf"))
    attention = torch.softmax(attention_logits, dim=-1)
    if self.cross_attention.dropout > 0:
      attention = torch.nn.functional.dropout(
          attention, p=self.cross_attention.dropout, training=self.training)
    attended_heads = torch.matmul(attention, value_heads)
    attended = attended_heads.transpose(1, 2).contiguous().reshape(
        query_representations.shape[0], query_representations.shape[1], self.input_dim)
    attended = self.cross_attention.out_proj(attended)
    raw = self.output_projection(self.output_norm(attended))
    correction = self.max_correction * torch.tanh(raw)
    correction = torch.where(
        query_valid_mask[..., None], correction, torch.zeros_like(correction))
    if return_attention_diagnostics:
      return correction, {
          "attention": attention.mean(dim=1),
          "attention_by_head": attention,
          "attention_logits": attention_logits,
          "base_attention_logits": base_attention_logits.masked_fill(
              ~valid_attention, float("-inf")),
          "projected_query": projected_query,
          "projected_key": projected_key,
          "projected_value": projected_value,
          "query_heads": query_heads,
          "key_heads": key_heads,
          "context_before_norm": context_representations,
          "attention_keys": attention_keys,
          "temperature_scale": 1.0 / math.sqrt(head_dim),
          "temperature": float(effective_temperature),
          "top_k": int(effective_top_k),
          "force_uniform_attention": bool(force_uniform_attention),
      }
    if return_attention:
      # [batch, heads, query, context] -> mean attention across heads.
      return correction, attention.mean(dim=1)
    return correction

  def forward(self, representations, labels, train_size, valid_row_mask=None):
    if representations.shape[:2] != labels.shape:
      raise ValueError("labels must match label-context representations")
    if train_size.shape != (representations.shape[0],):
      raise ValueError("train_size must have one value per label-context sequence")
    context_mask, query_mask = self._masks(
        labels, train_size, valid_row_mask)
    # The batch layout has a contiguous context prefix but variable train sizes.
    # Keeping the full sequence as memory lets the mask handle every batch item.
    correction = self.correction_from_context(
        representations,
        representations,
        labels,
        context_valid_mask=context_mask,
        query_valid_mask=query_mask,
    )
    return correction


# Always-on activation-chunk sizes. TabFM runs the whole training fold as one
# in-context sequence, so a single forward materialises activations that grow
# with rows * features and OOM the GPU on large tasks. These sizes split each
# stage's largest activation along an independent axis, bounding peak memory.
# Chunking is exact (identical outputs) and a no-op when an input is smaller
# than the chunk size, and it costs <1% runtime otherwise, so it is always on.
# Sizes are chosen for memory safety on a 40 GB GPU across TabArena-scale tasks.
