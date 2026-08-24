import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import pytest

from train_market_mover_tests import (
    best_improving_result,
    forward_feature_pool,
    forward_selection_model_parameters,
    infer_ablation_features,
    load_feature_sets,
    parse_competition_ids,
    recommended_validation_races,
    top3_capture,
    validate_production_selection_scope,
)


def test_forward_selection_uses_full_feature_and_row_sampling():
    parameters = forward_selection_model_parameters(
        SimpleNamespace(jobs=1), seed=42, max_estimators=100
    )

    assert parameters["colsample_bytree"] == 1.0
    assert parameters["subsample"] == 1.0
    assert parameters["random_state"] == 42
    assert parameters["n_estimators"] == 100


def test_winner_selection_early_stops_on_smoother_map_metric():
    parameters = forward_selection_model_parameters(
        SimpleNamespace(jobs=1),
        seed=42,
        max_estimators=100,
        selection_objective="winner",
    )

    assert parameters["eval_metric"][-1] == "map"


def test_top3_capture_is_aggregated_race_locally():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "top3_mask": [1, 1, 1, 0, 1, 1, 1, 0],
        "is_winner": [1, 0, 0, 0, 0, 1, 0, 0],
    })
    scores = np.array([4, 3, 2, 1, 4, 3, 1, 2], dtype=float)

    result = top3_capture(frame, scores)

    assert result["top3_hits"] == 5
    assert result["possible_top3_hits"] == 6
    assert result["top3_capture_rate"] == 5 / 6
    assert result["races_with_3_of_3"] == 1
    assert result["races_with_2plus_of_3"] == 2
    assert result["winner_hits"] == 1
    assert result["winner_hit_rate"] == 0.5


def test_tied_scores_receive_expected_not_row_order_credit():
    frame = pd.DataFrame({
        "race_id": [1] * 10,
        "top3_mask": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "is_winner": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    })

    result = top3_capture(frame, np.zeros(10))

    assert result["top3_hits"] == pytest.approx(0.9)
    assert result["top3_capture_rate"] == pytest.approx(0.3)
    assert result["winner_hits"] == pytest.approx(0.1)
    assert result["winner_hit_rate"] == pytest.approx(0.1)
    assert result["races_with_score_ties"] == 1


def test_infers_all_shared_base_features_and_one_addition():
    base, additions = infer_ablation_features({
        "t1": ["open_price", "fluc1", "fluc2", "form", "speed"],
        "t2": ["open_price", "fluc1", "fluc2", "form", "weight"],
    })

    assert base == ["open_price", "fluc1", "fluc2", "form"]
    assert additions == {"t1": "speed", "t2": "weight"}


def test_rejects_more_than_one_varying_feature():
    with pytest.raises(ValueError, match="exactly one tested feature"):
        infer_ablation_features({
            "t1": ["market", "speed", "weight"],
            "t2": ["market", "draw", "age"],
        })


def test_parses_one_or_multiple_competition_ids():
    assert parse_competition_ids("330") == [330]
    assert parse_competition_ids("330, 580,330") == [330, 580]


def test_rejects_invalid_competition_ids():
    with pytest.raises(Exception, match="positive integers"):
        parse_competition_ids("330,nope")


def test_recommends_twenty_percent_validation_cohort():
    assert recommended_validation_races(1000) == 200
    assert recommended_validation_races(27) == 5
    assert recommended_validation_races(10_000) == 1000


def test_selects_best_strict_forward_improvement():
    results = [
        {"top3_capture_rate": 0.50, "winner_hit_rate": 0.30, "candidate_order": 0},
        {"top3_capture_rate": 0.54, "winner_hit_rate": 0.29, "candidate_order": 1},
        {"top3_capture_rate": 0.54, "winner_hit_rate": 0.31, "candidate_order": 2},
    ]
    assert best_improving_result(results, 0.50) is results[2]
    assert best_improving_result(results, 0.54) is None


def test_forward_selection_requires_configured_material_uplift():
    results = [
        {"top3_capture_rate": 0.505, "winner_hit_rate": 0.30, "candidate_order": 0},
        {"top3_capture_rate": 0.511, "winner_hit_rate": 0.29, "candidate_order": 1},
    ]

    assert best_improving_result(
        results, 0.50, minimum_uplift=0.01
    ) is results[1]


def test_reverse_selection_chooses_lowest_material_result():
    results = [
        {"top3_capture_rate": 0.48, "winner_hit_rate": 0.30, "candidate_order": 0},
        {"top3_capture_rate": 0.43, "winner_hit_rate": 0.25, "candidate_order": 1},
        {"top3_capture_rate": 0.44, "winner_hit_rate": 0.20, "candidate_order": 2},
    ]

    assert best_improving_result(
        results, 0.50, minimum_uplift=0.01, reverse=True
    ) is results[1]
    assert best_improving_result(
        results, 0.43, minimum_uplift=0.01, reverse=True
    ) is None


def test_forward_pool_accepts_new_base_missing_from_old_models(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "base_features": ["market", "new_base"],
        "excluded_features": ["do_not_test"],
        "models": {
            "t1": {"features": ["market", "speed"]},
            "t2": {"features": ["market", "weight", "do_not_test"]},
            "t3": {"features": ["market", "new_base"]},
        },
    }))
    sets = load_feature_sets(manifest, allow_forward_pool=True)

    base, candidates, additions = forward_feature_pool(manifest, sets)

    assert base == ["market", "new_base"]
    assert list(additions.values()) == ["speed", "weight"]
    assert all(features[:2] == base for features in candidates.values())


def test_forward_pool_excludes_current_market_by_default(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "base_features": ["form"],
        "models": {
            "t1": {"features": ["form", "fluc2"]},
            "t2": {"features": ["form", "speed_rating"]},
        },
    }))
    sets = load_feature_sets(manifest, allow_forward_pool=True)

    _, _, additions = forward_feature_pool(manifest, sets)

    assert list(additions.values()) == ["speed_rating"]


def test_rejects_outcome_conditioned_competition_999():
    with pytest.raises(ValueError, match="assigned after results"):
        validate_production_selection_scope([999])

    validate_production_selection_scope([6])
    validate_production_selection_scope(None)
