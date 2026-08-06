"""TabFM training sampling helpers."""

from __future__ import annotations

import numpy as np
import torch


def build_query_race_schedule(
    race_ids: list[int], steps: int, query_races_per_step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build query steps with no duplicate race inside a step."""
    available = np.asarray(sorted(set(map(int, race_ids))), dtype=np.int64)
    if steps < 1:
        raise ValueError("steps must be positive")
    if query_races_per_step < 1:
        raise ValueError("query_races_per_step must be positive")
    if query_races_per_step > len(available):
        raise ValueError(
            "query_races_per_step cannot exceed the number of available races"
        )
    schedule = np.empty((steps, query_races_per_step), dtype=np.int64)
    cycle: list[int] = []
    for step_index in range(steps):
        selected: list[int] = []
        while len(selected) < query_races_per_step:
            if not cycle:
                cycle = list(map(int, rng.permutation(available)))
            candidate = cycle.pop()
            if candidate not in selected:
                selected.append(candidate)
        schedule[step_index] = selected
    return schedule


def eligible_query_race_ids(
    race_ids: list[int],
    race_time_by_id: dict[int, object],
    required_context_races: int,
) -> list[int]:
    """Return races with enough strictly earlier races for context."""
    if required_context_races < 1:
        raise ValueError("required_context_races must be positive")
    available = sorted(
        map(int, race_ids),
        key=lambda race_id: (race_time_by_id[race_id], race_id),
    )
    return [
        race_id
        for race_id in available
        if sum(
            race_time_by_id[other_id] < race_time_by_id[race_id]
            for other_id in available
        ) >= required_context_races
    ]


def sample_independent_race_batch(
    x: np.ndarray,
    y: np.ndarray,
    race_indices: dict[int, np.ndarray],
    context_races_per_query: int,
    query_race_ids: np.ndarray,
    race_time_by_id: dict[int, object],
    group_context_races: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    torch.Tensor,
    torch.Tensor,
]:
    """Build one independently contextualised sequence per query race.

    Validation predicts each race with its own most-recent strictly earlier
    training races.  Training must use the same contract: combining unrelated
    query races behind one shared random context creates a different task and
    makes a one-race query batch the only way to match inference.  This helper
    instead uses the tensor batch dimension, padding only after each complete
    query race, so several independently contextualised races contribute to one
    optimizer update.
    """
    if context_races_per_query < 1:
        raise ValueError("context_races_per_query must be positive")
    query_race_ids = np.asarray(query_race_ids, dtype=np.int64)
    if query_race_ids.ndim != 1 or len(query_race_ids) < 1:
        raise ValueError("query_race_ids must be a non-empty one-dimensional array")
    if len(set(map(int, query_race_ids))) != len(query_race_ids):
        raise ValueError("Query races contain duplicates")
    missing = sorted(set(map(int, query_race_ids)) - set(race_indices))
    if missing:
        raise ValueError(f"Query races are not in the training pool: {missing}")

    ordered_context_races = sorted(
        race_indices,
        key=lambda race_id: (race_time_by_id[int(race_id)], int(race_id)),
    )
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    canonical_ids: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    train_sizes: list[int] = []
    selected_context_ids: list[int] = []

    for query_race_id_value in query_race_ids:
        query_race_id = int(query_race_id_value)
        query_time = race_time_by_id[query_race_id]
        eligible_context = [
            int(race_id)
            for race_id in ordered_context_races
            if int(race_id) != query_race_id
            and race_time_by_id[int(race_id)] < query_time
        ]
        if len(eligible_context) < context_races_per_query:
            raise ValueError(
                f"Query race {query_race_id} has only {len(eligible_context)} "
                f"earlier context races; {context_races_per_query} are required"
            )
        context_race_ids = eligible_context[-context_races_per_query:]
        selected_context_ids.extend(context_race_ids)
        context_indices = np.concatenate(
            [race_indices[race_id] for race_id in context_race_ids]
        )
        query_indices = race_indices[query_race_id]
        combined_indices = np.concatenate((context_indices, query_indices))
        context_rows = len(context_indices)

        sequence_canonical_ids = np.concatenate(
            (
                np.concatenate(
                    [
                        np.full(
                            len(race_indices[race_id]), race_id, dtype=np.int64
                        )
                        for race_id in context_race_ids
                    ]
                ),
                np.full(len(query_indices), query_race_id, dtype=np.int64),
            )
        )
        sequence_group_ids = np.full(len(combined_indices), -1, dtype=np.int64)
        if group_context_races:
            sequence_group_ids[:context_rows] = sequence_canonical_ids[:context_rows]
        sequence_group_ids[context_rows:] = query_race_id

        sequences.append(x[combined_indices])
        targets.append(y[combined_indices])
        canonical_ids.append(sequence_canonical_ids)
        group_ids.append(sequence_group_ids)
        train_sizes.append(context_rows)

    max_rows = max(len(sequence) for sequence in sequences)
    batch_shape = (len(sequences), max_rows)
    batch_x = np.zeros((*batch_shape, x.shape[1]), dtype=x.dtype)
    batch_y = np.full(batch_shape, -100, dtype=y.dtype)
    batch_canonical_ids = np.full(batch_shape, -1, dtype=np.int64)
    batch_group_ids = np.full(batch_shape, -1, dtype=np.int64)
    valid_row_mask = np.zeros(batch_shape, dtype=bool)
    for batch_index, sequence in enumerate(sequences):
        row_count = len(sequence)
        batch_x[batch_index, :row_count] = sequence
        batch_y[batch_index, :row_count] = targets[batch_index]
        batch_canonical_ids[batch_index, :row_count] = canonical_ids[batch_index]
        batch_group_ids[batch_index, :row_count] = group_ids[batch_index]
        valid_row_mask[batch_index, :row_count] = True

    expected_query_rows = {
        int(race_id): len(race_indices[int(race_id)]) for race_id in query_race_ids
    }
    validate_complete_race_batch(
        batch_canonical_ids,
        batch_group_ids,
        np.asarray(train_sizes, dtype=np.int64),
        valid_row_mask,
        batch_y,
        expected_race_row_counts=expected_query_rows,
        require_one_winner=False,
    )
    return (
        torch.from_numpy(batch_x),
        torch.from_numpy(batch_y),
        torch.tensor(train_sizes, dtype=torch.long),
        np.asarray(selected_context_ids, dtype=np.int64),
        query_race_ids,
        torch.from_numpy(batch_group_ids),
        torch.from_numpy(valid_row_mask),
    )


def sample_race_batch(
    x: np.ndarray,
    y: np.ndarray,
    race_indices: dict[int, np.ndarray],
    context_races_per_step: int,
    query_races_per_step: int,
    rng: np.random.Generator,
    forced_query_race_ids: np.ndarray | None = None,
    group_context_races: bool = False,
    race_time_by_id: dict[int, object] | None = None,
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
    if race_time_by_id is None:
        raise ValueError("race_time_by_id is required for chronological training")
    available_races = np.asarray(sorted(race_indices), dtype=np.int64)
    if len(available_races) < query_races_per_step:
        raise ValueError(
            f"Need at least {query_races_per_step} query races but only "
            f"{len(available_races)} are available"
        )

    if forced_query_race_ids is None:
        eligible = eligible_query_race_ids(
            list(map(int, available_races)), race_time_by_id, context_races_per_step
        )
        if not eligible:
            raise ValueError("No query race has enough earlier context races")
        query_race_ids = rng.choice(
            np.asarray(eligible, dtype=np.int64),
            size=query_races_per_step,
            replace=False,
        )
    else:
        query_race_ids = np.asarray(forced_query_race_ids, dtype=np.int64)
        if len(query_race_ids) != query_races_per_step:
            raise ValueError("Forced query race count differs from query_races_per_step")
        if len(set(map(int, query_race_ids))) != len(query_race_ids):
            raise ValueError("Forced query races contain duplicates")
        if not set(map(int, query_race_ids)) <= set(map(int, available_races)):
            raise ValueError("Forced query race is not in the training pool")
    earliest_query_time = min(
        race_time_by_id[int(race_id)] for race_id in query_race_ids
    )
    query_id_set = set(map(int, query_race_ids))
    context_candidates = np.asarray(
        [
            race_id
            for race_id in available_races
            if int(race_id) not in query_id_set
            and race_time_by_id[int(race_id)] < earliest_query_time
        ],
        dtype=np.int64,
    )
    if len(context_candidates) < context_races_per_step:
        raise ValueError(
            "Insufficient earlier context races for query step: "
            f"requested={context_races_per_step} "
            f"available={len(context_candidates)} "
            f"earliest_query_time={earliest_query_time}"
        )
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
    canonical_race_ids = np.concatenate(
        [
            np.concatenate([
                np.full(len(race_indices[int(race_id)]), int(race_id), dtype=np.int64)
                for race_id in context_race_ids
            ]),
            query_row_race_ids,
        ]
    )
    validate_complete_race_batch(
        canonical_race_ids,
        race_group_ids,
        np.asarray([len(context_indices)]),
        np.ones((1, len(combined_indices)), dtype=bool),
        y[combined_indices][None, ...],
        expected_race_row_counts={
            int(race_id): len(race_indices[int(race_id)])
            for race_id in np.concatenate([context_race_ids, query_race_ids])
        },
        # ``y`` is top3_mask for this model, so a complete race has exactly
        # three positive targets.  The canonical one-winner validator is for
        # winner-labelled models and must not be applied to top-three labels.
        require_one_winner=False,
    )
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
    if len(set(map(int, sampled_races))) != len(sampled_races):
        raise ValueError("Sampled query races contain duplicate canonical race IDs")
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


def validate_complete_race_batch(
    canonical_race_ids,
    race_group_ids,
    train_size,
    valid_row_mask,
    targets,
    expected_race_row_counts: dict[int, int] | None = None,
    require_one_winner: bool = True,
) -> None:
    """Validate complete, non-overlapping query races before model execution."""
    canonical = np.asarray(canonical_race_ids)
    groups = np.asarray(race_group_ids)
    target_array = np.asarray(targets)
    if canonical.ndim == 1:
        canonical = canonical[None, :]
    if groups.ndim == 1:
        groups = groups[None, :]
    if target_array.ndim == 1:
        target_array = target_array[None, :]
    if canonical.shape != groups.shape or target_array.shape != groups.shape:
        raise ValueError("canonical race IDs, groups, and targets must have matching [B, T] shapes")
    sizes = np.asarray(train_size).reshape(-1)
    if sizes.shape != (groups.shape[0],):
        raise ValueError("train_size must contain one value per batch")
    valid = np.ones(groups.shape, dtype=bool) if valid_row_mask is None else np.asarray(valid_row_mask)
    if valid.shape != groups.shape or valid.dtype != bool:
        raise ValueError("valid_row_mask must be a bool array matching race rows")
    if np.any((~valid) & (groups != -1)):
        raise ValueError("padding rows must have race group -1")
    if np.any(valid & (canonical < 0)):
        raise ValueError("every valid row must preserve a non-negative canonical race ID")

    for batch_index, context_size in enumerate(sizes.astype(int)):
        context_mask = valid[batch_index] & (np.arange(groups.shape[1]) < context_size)
        query_mask = valid[batch_index] & (np.arange(groups.shape[1]) >= context_size)
        context_races = set(map(int, canonical[batch_index][context_mask]))
        query_races = set(map(int, canonical[batch_index][query_mask]))
        overlap = context_races.intersection(query_races)
        if overlap:
            raise ValueError(
                f"canonical race IDs appear in both context and query: {sorted(overlap)}")
        if np.any(query_mask & (groups[batch_index] < 0)):
            raise ValueError("every valid query row must have a non-negative race_group_id")

        for group_id in np.unique(groups[batch_index][query_mask]):
            group_mask = query_mask & (groups[batch_index] == group_id)
            race_ids = set(map(int, canonical[batch_index][group_mask]))
            if len(race_ids) != 1:
                raise ValueError(
                    f"query race group {int(group_id)} maps to multiple canonical races")
            race_id = next(iter(race_ids))
            count = int(group_mask.sum())
            if expected_race_row_counts is not None and count != expected_race_row_counts.get(race_id):
                raise ValueError(
                    f"query race {race_id} is incomplete: found {count} rows, "
                    f"expected {expected_race_row_counts.get(race_id)}")
            if require_one_winner:
                winner_count = int((target_array[batch_index][group_mask] == 1).sum())
                if winner_count != 1:
                    raise ValueError(
                        f"Race group {int(group_id)} in batch {batch_index} must contain exactly one winner, "
                        f"found {winner_count}.")


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
