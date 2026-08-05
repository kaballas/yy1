"""TabFM training validation helpers."""

from __future__ import annotations

import numpy as np


def validation_flag_masks(
    race_ids: np.ndarray, validation_flags: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint train/validation masks and reject split race fields."""
    if len(race_ids) != len(validation_flags):
        raise ValueError("race_ids and validation_flags have different lengths")
    invalid_flags = sorted(set(map(int, validation_flags)) - {0, 1})
    if invalid_flags:
        raise ValueError(f"is_validation must contain only 0 or 1, found {invalid_flags}")
    race_flag: dict[int, int] = {}
    for race_id, flag in zip(race_ids, validation_flags):
        race_id_int = int(race_id)
        flag_int = int(flag)
        previous = race_flag.setdefault(race_id_int, flag_int)
        if previous != flag_int:
            raise ValueError(
                f"race_id {race_id_int} has inconsistent is_validation values"
            )
    valid_mask = validation_flags == 1
    return validation_flags == 0, valid_mask


def build_race_indices(
    race_ids: np.ndarray, mask: np.ndarray
) -> dict[int, np.ndarray]:
    """Map selected race IDs to their complete row indices."""
    result: dict[int, np.ndarray] = {}
    for race_id in np.unique(race_ids[mask]):
        result[int(race_id)] = np.flatnonzero(mask & (race_ids == race_id))
    return result


def invalid_race_targets(
    y: np.ndarray,
    race_ids: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Return races violating runner-count or three-positive contracts."""
    invalid: list[tuple[int, int, int]] = []

    for race_id in np.unique(race_ids[mask]):
        indices = np.flatnonzero(mask & (race_ids == race_id))
        runner_count = len(indices)
        top3_count = int(y[indices].sum())

        if runner_count < 3 or top3_count != 3:
            invalid.append((int(race_id), runner_count, top3_count))

    return invalid


def format_invalid_races(invalid: list[tuple[int, int, int]]) -> str:
    """Format invalid-race diagnostics for logs and exceptions."""
    return ", ".join(
        f"race_id={race_id} runners={runner_count} top3={top3_count}"
        for race_id, runner_count, top3_count in invalid[:10]
    )


def validate_race_targets(
    y: np.ndarray,
    race_ids: np.ndarray,
    mask: np.ndarray,
    partition_name: str,
) -> None:
    """Reject selected races with invalid whole-race targets."""
    invalid = invalid_race_targets(y, race_ids, mask)

    if invalid:
        raise ValueError(
            f"{partition_name} contains {len(invalid)} invalid races. "
            f"{format_invalid_races(invalid)}"
        )


def exclude_invalid_races(
    y: np.ndarray,
    race_ids: np.ndarray,
    mask: np.ndarray,
    partition_name: str,
) -> tuple[np.ndarray, list[int]]:
    """Exclude invalid races while reporting every skipped race ID."""
    invalid = invalid_race_targets(y, race_ids, mask)
    invalid_race_ids = [race_id for race_id, _, _ in invalid]
    if invalid:
        print(
            f"WARNING skipped_invalid_{partition_name.lower().replace(' ', '_')}_races="
            f"{len(invalid)} preview=[{format_invalid_races(invalid)}]",
            flush=True,
        )
        mask = mask & ~np.isin(race_ids, invalid_race_ids)
    return mask, invalid_race_ids
