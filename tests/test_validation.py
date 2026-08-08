import numpy as np

from src.validation import exclude_invalid_races, invalid_race_targets


def test_three_runner_all_positive_race_is_not_rankable():
    race_ids = np.asarray([10, 10, 10, 20, 20, 20, 20])
    target = np.asarray([1, 1, 1, 1, 1, 1, 0])
    mask = np.ones(len(target), dtype=bool)

    assert invalid_race_targets(target, race_ids, mask) == [(10, 3, 3)]
    filtered, skipped = exclude_invalid_races(
        target, race_ids, mask, "Training pool"
    )

    assert skipped == [10]
    np.testing.assert_array_equal(filtered, race_ids == 20)
