"""Contracts for robust, versioned RaceFormer preprocessing."""

import numpy as np

from src.raceformer_preprocessing import (
    fit_raceformer_preprocessor,
    model_feature_columns,
    raceformer_base_diagnostics,
    race_percentiles,
    transform_raceformer,
)


def test_v3_preprocessing_logs_clips_and_adds_relative_features():
    features = ["career_starts", "win_percentage", "active_field_size"]
    raw = np.array([
        [0.0, 0.1, 4.0],
        [10.0, 0.3, 4.0],
        [1_000_000.0, 0.2, 4.0],
        [2.0, 0.4, 5.0],
        [4.0, 0.5, 5.0],
    ], dtype=np.float32)
    race_ids = np.array([1, 1, 1, 2, 2])
    contract = fit_raceformer_preprocessor(raw, features, clip=2.0)

    result = transform_raceformer(raw, race_ids, features, [], contract)

    assert result.shape == (5, 5)
    assert np.max(np.abs(result[:, :3])) <= 2.0
    assert model_feature_columns(features, contract) == [
        "career_starts", "win_percentage", "active_field_size",
        "career_starts__race_percentile", "win_percentage__race_percentile",
    ]
    assert np.allclose(result[:3, 3], [-1.0, 0.0, 1.0])


def test_v3_current_prices_are_logged_clipped_and_race_relative():
    features = ["open_price", "fluc1", "fluc2"]
    raw = np.array([
        [2.0, 3.0, 4.0],
        [10.0, 20.0, 30.0],
        [1_000_000.0, 1_000_000.0, 1_000_000.0],
    ], dtype=np.float32)
    race_ids = np.ones(3, dtype=np.int64)

    contract = fit_raceformer_preprocessor(raw, features, clip=2.0)
    result = transform_raceformer(raw, race_ids, features, [], contract)

    assert contract["version"] == 3
    assert contract["log1p_features"] == features
    assert contract["relative_features"] == features
    assert np.max(np.abs(result[:, :3])) <= 2.0
    assert model_feature_columns(features, contract) == [
        "open_price", "fluc1", "fluc2",
        "open_price__race_percentile",
        "fluc1__race_percentile",
        "fluc2__race_percentile",
    ]
    assert np.allclose(result[:, 3:], [[-1.0] * 3, [0.0] * 3, [1.0] * 3])


def test_promoted_residual_bundle_has_audited_relative_representations():
    features = [
        "sectional_last600_best_6",
        "recent_3_total_runners",
        "trainer_recent_top3_excess",
    ]
    raw = np.array([
        [34.0, 8.0, -0.10],
        [36.0, 10.0, 0.00],
        [38.0, 12.0, 0.10],
    ], dtype=np.float32)
    race_ids = np.ones(3, dtype=np.int64)

    contract = fit_raceformer_preprocessor(raw, features)
    result = transform_raceformer(raw, race_ids, features, [], contract)

    assert contract["log1p_features"] == ["recent_3_total_runners"]
    assert contract["relative_features"] == features
    assert model_feature_columns(features, contract) == [
        *features,
        "sectional_last600_best_6__race_percentile",
        "recent_3_total_runners__race_percentile",
        "trainer_recent_top3_excess__race_percentile",
    ]
    assert np.allclose(result[:, 3:], [[-1.0] * 3, [0.0] * 3, [1.0] * 3])


def test_race_percentiles_ties_and_missing_are_neutral():
    raw = np.array([[2.0], [2.0], [np.nan]], dtype=np.float32)
    result = race_percentiles(raw, np.array([7, 7, 7]), ["career_wins"], ["career_wins"])
    assert np.allclose(result[:, 0], [0.0, 0.0, 0.0])


def test_legacy_preprocessing_remains_available():
    raw = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    result = transform_raceformer(
        raw, np.array([1, 1]), ["a", "b"], [], None,
        legacy_median=np.array([1.0, 2.0], dtype=np.float32),
        legacy_scale=np.array([2.0, 2.0], dtype=np.float32),
    )
    assert np.allclose(result, [[0.0, 0.0], [1.0, 1.0]])


def test_diagnostics_preserve_unclipped_and_missing_values():
    features = ["career_starts"]
    training = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    contract = fit_raceformer_preprocessor(training, features, clip=1.0)
    raw = np.array([[1_000_000.0], [np.nan]], dtype=np.float32)
    result = raceformer_base_diagnostics(raw, features, contract)
    assert result["unclipped_standardized"][0, 0] > 1.0
    assert result["clipped_standardized"][0, 0] == 1.0
    assert result["was_clipped"][0, 0]
    assert result["was_missing"][1, 0]
    assert result["unclipped_standardized"][1, 0] == 0.0


def test_layoff_buckets_are_cumulative_and_missing_is_separate():
    features = ["recent_days_since_last_run"]
    raw = np.array([[17.0], [90.0], [392.0], [np.nan]], dtype=np.float32)
    contract = fit_raceformer_preprocessor(
        raw, features, layoff_buckets=True
    )
    result = transform_raceformer(
        raw, np.array([1, 1, 1, 1]), features, [], contract
    )
    buckets = result[:, 2:]  # continuous base, race percentile, then six indicators
    assert np.array_equal(buckets[0], [0, 0, 0, 0, 0, 0])
    assert np.array_equal(buckets[1], [1, 1, 1, 0, 0, 0])
    assert np.array_equal(buckets[2], [1, 1, 1, 1, 1, 0])
    assert np.array_equal(buckets[3], [0, 0, 0, 0, 0, 1])


def test_exclusive_layoff_bands_use_zero_to_29_as_reference():
    features = ["recent_days_since_last_run"]
    raw = np.array(
        [[17.0], [30.0], [60.0], [90.0], [180.0], [365.0], [np.nan]],
        dtype=np.float32,
    )
    contract = fit_raceformer_preprocessor(
        raw, features, layoff_bucket_mode="exclusive"
    )
    result = transform_raceformer(
        raw, np.ones(len(raw), dtype=np.int64), features, [], contract
    )
    buckets = result[:, 2:]
    assert np.array_equal(buckets[0], [0, 0, 0, 0, 0, 0])
    for row in range(1, 6):
        expected = np.zeros(6)
        expected[row - 1] = 1
        assert np.array_equal(buckets[row], expected)
    assert np.array_equal(buckets[6], [0, 0, 0, 0, 0, 1])
