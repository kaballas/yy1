"""Leakage-safe chronological partitions for RaceFormer CSV snapshots."""

from __future__ import annotations

import numpy as np

from src.validation import invalid_race_targets


def combine_disjoint_snapshots(
    train_x: np.ndarray, train_y: np.ndarray, train_ids: np.ndarray,
    train_times: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
    valid_ids: np.ndarray, valid_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    overlap = np.intersect1d(np.unique(train_ids), np.unique(valid_ids))
    if len(overlap):
        raise ValueError(
            f"Training and validation snapshots overlap on {len(overlap)} race IDs"
        )
    x = np.concatenate((train_x, valid_x))
    y = np.concatenate((train_y, valid_y))
    race_ids = np.concatenate((train_ids, valid_ids))
    times = np.concatenate((train_times, valid_times))
    order = np.asarray(
        sorted(range(len(race_ids)), key=lambda i: (times[i], int(race_ids[i]), i)),
        dtype=np.int64,
    )
    return x[order], y[order], race_ids[order], times[order]


def chronological_validation_ids(
    y: np.ndarray, race_ids: np.ndarray, count: int
) -> np.ndarray:
    """Select the latest N complete races from chronologically ordered rows."""
    if count < 1:
        raise ValueError("chronological validation race count must be positive")
    invalid = {
        race_id for race_id, _, _ in invalid_race_targets(
            y, race_ids, np.ones(len(race_ids), dtype=bool)
        )
    }
    ordered = [
        race_id for race_id in dict.fromkeys(map(int, race_ids))
        if race_id not in invalid
    ]
    if len(ordered) <= count:
        raise ValueError(
            f"Need more than {count} eligible races for a chronological split; "
            f"found {len(ordered)}"
        )
    return np.asarray(ordered[-count:], dtype=np.int64)


def partition_by_validation_ids(
    x: np.ndarray, y: np.ndarray, race_ids: np.ndarray, times: np.ndarray,
    validation_ids: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    known = np.unique(race_ids)
    missing = np.setdiff1d(validation_ids, known)
    if len(missing):
        raise ValueError(
            f"Saved validation cohort is missing {len(missing)} races from snapshots"
        )
    valid = np.isin(race_ids, validation_ids)
    if not valid.any() or valid.all():
        raise ValueError("Chronological split must leave both training and validation rows")
    return (
        x[~valid], y[~valid], race_ids[~valid], times[~valid],
        x[valid], y[valid], race_ids[valid], times[valid],
    )
