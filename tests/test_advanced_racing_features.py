import numpy as np
import pandas as pd

from src.advanced_racing_features import (
    derive_entity_history_features,
    derive_sectional_class_features,
    parse_class_level,
)


def _history_frame():
    row = {"race_name": "R4 Example (Bm78)", "grade": "UNKNOWN_GROUP", "distance_m": 1200}
    for run in range(1, 7):
        row.update({
            f"recent_{run}_last600": f"0:{34 + run:.2f}",
            f"recent_{run}_time": "01.10.00",
            f"recent_{run}_distance_m": 1200,
            f"recent_{run}_class": "BM70",
        })
    return pd.DataFrame([row])


def test_class_parser_handles_race_names_and_historical_labels():
    assert parse_class_level("R2 Example (Bm78)", "UNKNOWN_GROUP") == 78
    assert parse_class_level("anything", "THREE") == 115
    assert parse_class_level("Cls 2") == 60
    assert parse_class_level("Barrier Trial-Open") != parse_class_level(
        "Barrier Trial-Open"
    )


def test_sectional_and_class_features_use_only_prior_run_fields():
    result = derive_sectional_class_features(_history_frame())

    assert result.loc[0, "current_class_level"] == 78
    assert result.loc[0, "form_class_level_weighted_3"] == 70
    assert result.loc[0, "class_change_vs_recent_3"] == 8
    assert result.loc[0, "sectional_last600_best_6"] == 35
    ratio = result.loc[0, "sectional_closing_speed_ratio_weighted_6"]
    assert 0.5 < ratio < 2.0


def test_sectional_parser_rejects_full_race_time_in_last600_field():
    frame = _history_frame()
    frame.loc[0, "recent_1_last600"] = "2:00.00"

    result = derive_sectional_class_features(frame)

    assert result.loc[0, "sectional_last600_best_6"] == 36
    assert result.loc[0, "sectional_last600_seconds_weighted_6"] < 45


def test_entity_history_is_strictly_earlier_than_current_start_time():
    frame = pd.DataFrame({
        "start_time_iso": [
            "2026-01-01T01:00:00+00:00",
            "2026-01-01T01:00:00+00:00",
            "2026-01-02T01:00:00+00:00",
        ],
        "status": ["finished", "finished", "finished"],
        "runner_mask": [1, 1, 1],
        "top3_mask": [1, 0, 1],
        "active_field_size": [10, 10, 10],
        "field_size": [10, 10, 10],
        "jockey": [" A Rider ", "a rider", "A RIDER"],
        "trainer": ["Trainer X", "trainer x", "TRAINER X"],
    })

    result = derive_entity_history_features(frame)

    assert np.isnan(result.loc[0, "jockey_history_starts"])
    assert np.isnan(result.loc[1, "jockey_history_starts"])
    assert result.loc[2, "jockey_history_starts"] == 2
    assert np.isclose(result.loc[2, "jockey_history_top3_excess"], 0.4 / 22)
    assert result.loc[2, "jockey_recent_top3_excess"] > 0
    assert result.loc[2, "jockey_trainer_history_starts"] == 2
