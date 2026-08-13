"""Tests for RaceFormer's chronological whole-race validation split."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.raceformer_partition import (
    chronological_holdout_ids,
    chronological_validation_ids,
    combine_disjoint_snapshots,
    partition_by_validation_and_test_ids,
    partition_by_validation_ids,
)


def _rows(race_ids):
    ids = np.repeat(np.asarray(race_ids, dtype=np.int64), 4)
    x = ids[:, None].astype(np.float32)
    y = np.tile(np.array([1, 1, 1, 0]), len(race_ids))
    times = np.asarray([
        datetime(2026, 1, int(race_id), tzinfo=timezone.utc)
        for race_id in ids
    ], dtype=object)
    return x, y, ids, times


def test_combined_snapshots_are_sorted_before_latest_complete_races_are_held_out():
    late = _rows([4, 2])
    early = _rows([3, 1])
    x, y, ids, times = combine_disjoint_snapshots(*late, *early)
    validation_ids = chronological_validation_ids(y, ids, 2)
    split = partition_by_validation_ids(x, y, ids, times, validation_ids)
    assert validation_ids.tolist() == [3, 4]
    assert np.unique(split[2]).tolist() == [1, 2]
    assert np.unique(split[6]).tolist() == [3, 4]


def test_invalid_latest_race_does_not_consume_validation_slot():
    x, y, ids, times = _rows([1, 2, 3, 4])
    y[ids == 4] = [1, 1, 0, 0]
    assert chronological_validation_ids(y, ids, 2).tolist() == [2, 3]


def test_three_way_chronological_split_seals_newest_complete_races():
    x, y, ids, times = _rows([1, 2, 3, 4, 5, 6])
    validation_ids, test_ids = chronological_holdout_ids(y, ids, 2, 2)
    split = partition_by_validation_and_test_ids(
        x, y, ids, times, validation_ids, test_ids
    )

    assert validation_ids.tolist() == [3, 4]
    assert test_ids.tolist() == [5, 6]
    assert np.unique(split[2]).tolist() == [1, 2]
    assert np.unique(split[6]).tolist() == [3, 4]
    assert np.unique(split[10]).tolist() == [5, 6]


def test_invalid_latest_race_does_not_consume_sealed_test_slot():
    x, y, ids, times = _rows([1, 2, 3, 4, 5, 6])
    y[ids == 6] = [1, 1, 0, 0]

    validation_ids, test_ids = chronological_holdout_ids(y, ids, 2, 2)

    assert validation_ids.tolist() == [2, 3]
    assert test_ids.tolist() == [4, 5]


def test_overlapping_snapshot_races_are_rejected():
    first = _rows([1, 2])
    second = _rows([2, 3])
    with pytest.raises(ValueError, match="overlap"):
        combine_disjoint_snapshots(*first, *second)
