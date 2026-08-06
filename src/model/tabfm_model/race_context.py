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


# Always-on activation-chunk sizes. TabFM runs the whole training fold as one
# in-context sequence, so a single forward materialises activations that grow
# with rows * features and OOM the GPU on large tasks. These sizes split each
# stage's largest activation along an independent axis, bounding peak memory.
# Chunking is exact (identical outputs) and a no-op when an input is smaller
# than the chunk size, and it costs <1% runtime otherwise, so it is always on.
# Sizes are chosen for memory safety on a 40 GB GPU across TabArena-scale tasks.
