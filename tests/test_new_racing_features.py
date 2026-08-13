import numpy as np
import pandas as pd

from src.advanced_racing_features import (
    ADVANCED_FEATURE_NAMES,
    derive_context_features,
    derive_entity_history_features,
    derive_sectional_class_features,
)
from src.derived_racing_features import DERIVED_FEATURE_NAMES, derive_racing_features
from update_derived_racing_features import FEATURES_TO_STORE


def history_frame() -> pd.DataFrame:
    row = {
        "race_id": 10, "distance_m": 1200., "draw_number": 2,
        "active_field_size": 10, "field_size": 10, "weight_kg": 57.,
        "career_starts": 20, "career_wins": 3, "career_seconds": 4,
        "career_thirds": 2, "place_percentage": 45.,
        "race_name": "Example BM78", "grade": None,
    }
    for run in range(1, 7):
        row.update({
            f"recent_{run}_place": run,
            f"recent_{run}_margin": run - 1.,
            f"recent_{run}_total_runners": 10,
            f"recent_{run}_barrier": run,
            f"recent_{run}_starting_price": 4. + run,
            f"recent_{run}_distance_m": 1200 if run != 2 else 1500,
            f"recent_{run}_weight_kg": 56. + run / 2,
            f"recent_{run}_last600": f"0:{34 + run:.2f}",
            f"recent_{run}_time": "1:10.00",
            f"recent_{run}_class": "BM70",
        })
    return pd.DataFrame([row])


def entity_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "start_time_iso": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
                           "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
        "status": ["finished"] * 4, "runner_mask": [1] * 4,
        "top3_mask": [1, 0, 1, 0], "finish_place": [1, 5, 2, 8],
        "active_field_size": [10] * 4, "field_size": [10] * 4,
        "jockey": ["J"] * 4, "trainer": ["T"] * 4,
    })


def test_finish_percentile_margin_quality_market_and_missing_weighting():
    frame = history_frame()
    frame.loc[0, "recent_1_place"] = np.nan
    result = derive_racing_features(frame)
    finish2, finish3 = 1 - 1 / 9, 1 - 2 / 9
    assert np.isclose(result.loc[0, "recent_finish_percentile_weighted_3"],
                      (2 * finish2 + finish3) / 3)
    assert np.isclose(result.loc[0, "recent_margin_quality_last1"], 1.)
    assert np.isclose(result.loc[0, "historical_market_overperformance_weighted_3"],
                      (2 * (finish2 - 1 / 6) + (finish3 - 1 / 7)) / 3)


def test_positive_form_slope_means_newer_runs_are_better():
    result = derive_racing_features(history_frame())
    assert result.loc[0, "recent_finish_percentile_slope_3"] > 0
    assert result.loc[0, "recent_margin_quality_slope_3"] > 0


def test_similar_distance_filter_and_null_rules():
    frame = history_frame()
    result = derive_racing_features(frame)
    assert result.loc[0, "similar_distance_runs"] == 5
    frame.loc[0, "distance_m"] = 0
    result = derive_racing_features(frame)
    assert np.isnan(result.loc[0, "similar_distance_runs"])
    assert np.isnan(result.loc[0, "similar_distance_finish_percentile_weighted"])


def test_entity_target_future_and_same_race_results_are_excluded():
    frame = entity_frame()
    original = derive_entity_history_features(frame)
    changed = frame.copy()
    changed.loc[2:, ["finish_place", "top3_mask"]] = [[1, 1], [1, 1]]
    altered = derive_entity_history_features(changed)
    # A target result cannot affect itself; future changes cannot affect earlier rows.
    pd.testing.assert_series_equal(original.loc[2, :], altered.loc[2, :])
    assert np.isnan(original.loc[0, "jockey_trainer_history_runs"])
    assert np.isnan(original.loc[1, "jockey_trainer_history_runs"])
    assert original.loc[2, "jockey_trainer_history_runs"] == 2


def test_bayesian_pair_smoothing_and_determinism():
    frame = entity_frame()
    first = derive_entity_history_features(frame)
    second = derive_entity_history_features(frame)
    pd.testing.assert_frame_equal(first, second)
    raw = first.loc[2, "jockey_trainer_history_top3_rate"]
    smooth = first.loc[2, "jockey_trainer_history_smoothed_top3_rate"]
    assert .3 < smooth < raw


def test_race_relative_rank_and_gap_direction():
    frame = pd.concat([history_frame(), history_frame()], ignore_index=True)
    base = pd.concat([derive_racing_features(frame),
                      derive_sectional_class_features(frame)], axis=1)
    # Entity output is not needed by this unit: supply a controlled supported source.
    base["jockey_trainer_history_smoothed_top3_rate"] = [.2, .4]
    base.loc[1, "recent_finish_percentile_weighted_3"] = .1
    context = derive_context_features(frame, base)
    assert context.loc[0, "recent_finish_percentile_weighted_3_rank_in_race"] == 1
    assert context.loc[0, "recent_finish_percentile_weighted_3_gap_to_best"] == 0
    assert context.loc[1, "recent_finish_percentile_weighted_3_gap_to_best"] > 0


def test_feature_registry_exactly_matches_generated_outputs():
    frame = history_frame()
    base = pd.concat([derive_racing_features(frame),
                      derive_sectional_class_features(frame)], axis=1)
    base["jockey_trainer_history_smoothed_top3_rate"] = .3
    context = derive_context_features(frame, base)
    generated = set(base) | set(context) | set(ADVANCED_FEATURE_NAMES)
    assert set(FEATURES_TO_STORE) == set(DERIVED_FEATURE_NAMES) | set(ADVANCED_FEATURE_NAMES)
    assert set(FEATURES_TO_STORE) <= generated
