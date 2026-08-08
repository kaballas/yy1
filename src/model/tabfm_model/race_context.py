"""Race-group representation and prediction context modules."""

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


# Always-on activation-chunk sizes. TabFM runs the whole training fold as one
# in-context sequence, so a single forward materialises activations that grow
# with rows * features and OOM the GPU on large tasks. These sizes split each
# stage's largest activation along an independent axis, bounding peak memory.
# Chunking is exact (identical outputs) and a no-op when an input is smaller
# than the chunk size, and it costs <1% runtime otherwise, so it is always on.
# Sizes are chosen for memory safety on a 40 GB GPU across TabArena-scale tasks.
