"""TabFM training sampling helpers."""

from __future__ import annotations

import numpy as np
import torch


def build_query_race_schedule(
    race_ids: list[int], steps: int, query_races_per_step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Maximize distinct query-race coverage before repeating any race."""
    available = np.asarray(sorted(race_ids), dtype=np.int64)
    required = steps * query_races_per_step
    scheduled: list[np.ndarray] = []
    remaining = required
    while remaining:
        cycle = rng.permutation(available)
        take = min(remaining, len(cycle))
        scheduled.append(cycle[:take])
        remaining -= take
    return np.concatenate(scheduled).reshape(steps, query_races_per_step)


def sample_race_batch(
    x: np.ndarray,
    y: np.ndarray,
    race_indices: dict[int, np.ndarray],
    context_races_per_step: int,
    query_races_per_step: int,
    rng: np.random.Generator,
    forced_query_race_ids: np.ndarray | None = None,
    group_context_races: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
]:
    """Sample disjoint complete context and query races for one optimiser step."""
    available_races = np.asarray(sorted(race_indices), dtype=np.int64)
    required = context_races_per_step + query_races_per_step

    if len(available_races) < required:
        raise ValueError(
            f"Need at least {required} training races but only "
            f"{len(available_races)} are available"
        )

    if forced_query_race_ids is None:
        sampled = rng.choice(available_races, size=required, replace=False)
        context_race_ids = sampled[:context_races_per_step]
        query_race_ids = sampled[context_races_per_step:]
    else:
        query_race_ids = np.asarray(forced_query_race_ids, dtype=np.int64)
        if len(query_race_ids) != query_races_per_step:
            raise ValueError("Forced query race count differs from query_races_per_step")
        #if len(set(map(int, query_race_ids))) != len(query_race_ids):
        #    raise ValueError("Forced query races contain duplicates")
        if not set(map(int, query_race_ids)) <= set(map(int, available_races)):
            raise ValueError("Forced query race is not in the training pool")
        context_candidates = available_races[
            ~np.isin(available_races, query_race_ids)
        ]
        context_race_ids = rng.choice(
            context_candidates, size=context_races_per_step, replace=False
        )

    context_indices = np.concatenate(
        [race_indices[int(race_id)] for race_id in context_race_ids]
    )
    query_indices = np.concatenate(
        [race_indices[int(race_id)] for race_id in query_race_ids]
    )
    query_row_race_ids = np.concatenate(
        [
            np.full(
                len(race_indices[int(race_id)]), int(race_id), dtype=np.int64
            )
            for race_id in query_race_ids
        ]
    )
    combined_indices = np.concatenate([context_indices, query_indices])

    race_group_ids = np.full(len(combined_indices), -1, dtype=np.int64)
    offset = len(context_indices)
    if group_context_races:
        for group_id, race_id in enumerate(context_race_ids):
            rows = race_indices[int(race_id)]
            race_group_ids[np.arange(len(rows)) + sum(len(race_indices[int(r)]) for r in context_race_ids[:group_id])] = group_id
        query_group_offset = len(context_race_ids)
    else:
        query_group_offset = 0
    for group_id, race_id in enumerate(query_race_ids):
        race_rows = race_indices[int(race_id)]
        end = offset + len(race_rows)
        race_group_ids[offset:end] = query_group_offset + group_id
        offset = end
    validate_sampled_race_groups(
        race_group_ids, len(context_indices), query_row_race_ids, query_race_ids
    )

    batch_x = torch.from_numpy(x[combined_indices][None, ...])
    batch_y = torch.from_numpy(y[combined_indices][None, ...])
    context_rows = len(context_indices)
    return (
        batch_x,
        batch_y,
        context_rows,
        context_race_ids,
        query_race_ids,
        query_row_race_ids,
        torch.from_numpy(race_group_ids[None, ...]),
    )


def validate_sampled_race_groups(
    race_group_ids: np.ndarray,
    context_rows: int,
    query_row_race_ids: np.ndarray,
    sampled_query_race_ids: np.ndarray,
) -> None:
    """Reject missing, merged, split, or context-contaminated query groups."""
    groups = np.asarray(race_group_ids)
    row_races = np.asarray(query_row_race_ids)
    sampled_races = np.asarray(sampled_query_race_ids)
    if groups.ndim != 1 or len(groups) != context_rows + len(row_races):
        raise ValueError("Race group tensor is not aligned with the sampled sequence")
    context_groups = groups[:context_rows]
    if not (np.all(context_groups == -1) or np.all(context_groups >= 0)):
      raise ValueError("Context rows must use either all -1 or non-negative race groups")
    query_groups = groups[context_rows:]
    if np.any(query_groups < 0):
        raise ValueError("Every query row must have a non-negative group")
    unique_groups = list(dict.fromkeys(map(int, query_groups)))
    if len(unique_groups) != len(sampled_races):
        raise ValueError("Each sampled query race must map to exactly one group")
    seen_races = set()
    for group_id in unique_groups:
        grouped_races = set(map(int, row_races[query_groups == group_id]))
        if len(grouped_races) != 1:
            raise ValueError("A query group contains rows from multiple races")
        seen_races.update(grouped_races)
    if seen_races != set(map(int, sampled_races)):
        raise ValueError("Race grouping does not cover every sampled query race")


def build_race_group_ids(
    query_race_ids: np.ndarray, context_rows: int,
    context_race_ids: np.ndarray | None = None,
) -> torch.Tensor:
    """Build [1, context + query] local group IDs without merging races."""
    query_race_ids = np.asarray(query_race_ids)
    if query_race_ids.ndim != 1:
        raise ValueError("query_race_ids must be one-dimensional")
    groups = np.full(context_rows + len(query_race_ids), -1, dtype=np.int64)
    mapping: dict[int, int] = {}
    next_group = 0
    if context_race_ids is not None:
      context_race_ids = np.asarray(context_race_ids)
      if len(context_race_ids) != context_rows:
        raise ValueError("context_race_ids must align with context_rows")
      for row_index, race_id_value in enumerate(context_race_ids):
        race_id = int(race_id_value)
        if race_id not in mapping:
          mapping[race_id] = next_group
          next_group += 1
        groups[row_index] = mapping[race_id]
    for row_index, race_id_value in enumerate(query_race_ids, start=context_rows):
        race_id = int(race_id_value)
        if race_id not in mapping:
          mapping[race_id] = next_group
          next_group += 1
        groups[row_index] = mapping[race_id]
    return torch.from_numpy(groups[None, ...])
