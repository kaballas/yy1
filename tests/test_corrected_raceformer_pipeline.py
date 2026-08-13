import json
import sqlite3

import pandas as pd
import pytest

pytest.importorskip("torch")

from rank_raceformer_models import combine_rankings, load_active_race
from train_corrected_raceformer_pipeline import (
    CORRECTED_CANDIDATE_FEATURES,
    build_manifests,
    is_current_market_feature,
)


def test_corrected_manifests_add_candidates_and_separate_current_market(tmp_path):
    base = tmp_path / "base.json"
    residual = tmp_path / "residual.json"
    unanchored = tmp_path / "unanchored.json"
    base.write_text(json.dumps({
        "schema_version": 1,
        "label": "top3_mask",
        "features": ["fluc2", "old_form", "form_class_level_weighted_6"],
        "zeroed_features": ["old_form", "form_class_level_weighted_6"],
    }))
    columns = {"fluc2", "old_form", *CORRECTED_CANDIDATE_FEATURES}

    features, residual_active, unanchored_active = build_manifests(
        base, residual, unanchored, columns
    )

    assert all(feature in features for feature in CORRECTED_CANDIDATE_FEATURES)
    assert all(feature in residual_active for feature in CORRECTED_CANDIDATE_FEATURES)
    assert "fluc2" in residual_active
    assert "fluc2" not in unanchored_active
    assert json.loads(residual.read_text())["features"] == json.loads(
        unanchored.read_text()
    )["features"]


def test_current_market_detection_does_not_remove_historical_market_form():
    assert is_current_market_feature("fluc2")
    assert is_current_market_feature("market_open_to_fluc2_move")
    assert not is_current_market_feature(
        "historical_market_overperformance_weighted_3_zscore_in_race"
    )


def _scored(ranks, probabilities, *, residual):
    frame = pd.DataFrame({
        "runner_number": [1, 2, 3],
        "runner_name": ["A", "B", "C"],
        "fluc2": [2., 4., 8.],
        "market_rank": [1, 2, 3],
        "probability": probabilities,
        "model_rank": ranks,
    })
    if residual:
        frame["anchor_logit"] = [1., 0., -1.]
        frame["residual_logit"] = [0., 0., 0.]
    return frame


def test_consensus_uses_rank_percentiles_and_is_deterministic():
    market = _scored([1, 2, 3], [.8, .5, .2], residual=True)
    unanchored = _scored([3, 1, 2], [.2, .8, .5], residual=False)

    result = combine_rankings(market, unanchored, .5, .5)

    # Runner 2 is second in the residual model and first unanchored, so wins.
    assert result.iloc[0]["runner_number"] == 2
    assert result.iloc[0]["consensus_rank"] == 1
    assert result["consensus_rank"].tolist() == [1, 2, 3]


def test_consensus_rejects_invalid_weights():
    market = _scored([1, 2, 3], [.8, .5, .2], residual=True)
    unanchored = _scored([3, 1, 2], [.2, .8, .5], residual=False)

    with pytest.raises(ValueError, match="weights"):
        combine_rankings(market, unanchored, 0, 0)


def test_race_loader_excludes_inactive_runners(tmp_path):
    database = tmp_path / "race.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE race_runners (race_id INTEGER, start_time_iso TEXT, "
            "competition_id INTEGER, race_number INTEGER, race_name TEXT, "
            "runner_number INTEGER, runner_name TEXT, runner_mask INTEGER, "
            "fluc2 REAL, status TEXT, source_betting_status TEXT, "
            "active_field_size INTEGER, derived_racing_features_version TEXT, "
            "feature REAL)"
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (1, '2026-01-01Z', 999, 1, 'R1', "
            "?, ?, ?, 5.0, 'finished', 'RESULTED', 2, 'v1', ?)",
            [(1, "Active", 1, 2.), (2, "Scratched", 0, 9.)],
        )

    frame = load_active_race(database, 1, [["feature"]])

    assert frame["runner_number"].tolist() == [1]
    assert frame["runner_name"].tolist() == ["Active"]
