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
    validation_ids, _ = chronological_holdout_ids(y, race_ids, count, 0)
    return validation_ids


def chronological_holdout_ids(
    y: np.ndarray,
    race_ids: np.ndarray,
    validation_count: int,
    test_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select consecutive validation and sealed-test cohorts chronologically."""
    if validation_count < 1:
        raise ValueError("chronological validation race count must be positive")
    if test_count < 0:
        raise ValueError("chronological test race count must be zero or positive")
    invalid = {
        race_id for race_id, _, _ in invalid_race_targets(
            y, race_ids, np.ones(len(race_ids), dtype=bool)
        )
    }
    ordered = [
        race_id for race_id in dict.fromkeys(map(int, race_ids))
        if race_id not in invalid
    ]
    holdout_count = validation_count + test_count
    if len(ordered) <= holdout_count:
        raise ValueError(
            f"Need more than {holdout_count} eligible races for a chronological split; "
            f"found {len(ordered)}"
        )
    if test_count:
        validation = ordered[-holdout_count:-test_count]
        test = ordered[-test_count:]
    else:
        validation = ordered[-validation_count:]
        test = []
    return (
        np.asarray(validation, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


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


def partition_by_validation_and_test_ids(
    x: np.ndarray,
    y: np.ndarray,
    race_ids: np.ndarray,
    times: np.ndarray,
    validation_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Partition rows without exposing the sealed test cohort to training."""
    overlap = np.intersect1d(validation_ids, test_ids)
    if len(overlap):
        raise ValueError("Chronological validation and test race IDs overlap")
    known = np.unique(race_ids)
    missing = np.setdiff1d(np.concatenate((validation_ids, test_ids)), known)
    if len(missing):
        raise ValueError(
            f"Saved chronological cohorts are missing {len(missing)} races from snapshots"
        )
    validation = np.isin(race_ids, validation_ids)
    test = np.isin(race_ids, test_ids)
    training = ~(validation | test)
    if not training.any() or not validation.any():
        raise ValueError("Chronological split must leave training and validation rows")
    if len(test_ids) and not test.any():
        raise ValueError("Chronological split requested a test cohort but found no rows")
    return (
        x[training], y[training], race_ids[training], times[training],
        x[validation], y[validation], race_ids[validation], times[validation],
        x[test], y[test], race_ids[test], times[test],
    )
