import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from run_market_blind_ranker_audit import release_test3
from src.market_blind_ranker_audit import (
    aggregate_multiseed_bootstrap, assert_market_blind_contract,
    assign_feature_groups, build_neural_model, load_development_snapshot,
    redundant_feature_clusters, validate_final_selection,
    xgboost_group_contract,
)
from src.race_moe_snapshot import create_split_snapshot


def _rows(race_ids=(1, 2, 3)):
    rows = []
    for race_id in race_ids:
        for runner in range(1, 5):
            rows.append({
                "race_id": race_id, "runner_number": runner,
                "start_time_iso": f"2026-01-{race_id:02d}T00:00:00+00:00",
                "competition_id": 7, "is_winner": int(runner == 1),
                "finish_place": runner, "distance_m": 1200,
                "class_name": "BM", "field_size": 4, "active_field_size": 4,
                "track_status": "Good", "career_starts": runner,
                "runner_name": f"R{runner}", "age": float(runner),
            })
    return pd.DataFrame(rows)


def test_development_loader_uses_immutable_snapshot_without_opening_old_test(tmp_path):
    frame = _rows()
    manifest = create_split_snapshot(
        tmp_path / "snapshot",
        {
            "training": frame[frame.race_id == 1],
            "validation": frame[frame.race_id == 2],
            "test": frame[frame.race_id == 3],
        },
        ["age"], database=tmp_path / "db.sqlite", excluded_features=[],
    )
    metadata = json.loads(manifest.read_text())
    test_path = manifest.parent / metadata["splits"]["test"]["path"]
    test_path.write_bytes(test_path.read_bytes() + b"intentionally-corrupt-unopened-test")
    frames, _ = load_development_snapshot(manifest)
    assert set(frames) == {"training", "validation"}


def test_test3_cannot_be_scored_before_final_lock(tmp_path):
    with pytest.raises(PermissionError, match="TEST-3 IS SEALED"):
        release_test3(SimpleNamespace(), tmp_path)


def test_multiseed_comparison_requires_identical_seed_pairing_and_races():
    seeds = (11, 29, 42, 73, 101)
    base = {
        seed: pd.DataFrame({"race_id": [1, 2], "winner_rank": [1, 2]})
        for seed in seeds
    }
    challenge = {
        seed: pd.DataFrame({"race_id": [1, 2], "winner_rank": [1, 1]})
        for seed in seeds
    }
    result = aggregate_multiseed_bootstrap(base, challenge, seeds, samples=100, bootstrap_seed=3)
    assert result["races"] == 2
    assert result["mean_top1_difference"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="identical seeds"):
        aggregate_multiseed_bootstrap(base, {11: challenge[11]}, seeds, samples=10)
    broken = dict(challenge)
    broken[42] = pd.DataFrame({"race_id": [1, 9], "winner_rank": [1, 1]})
    with pytest.raises(ValueError, match="race IDs differ"):
        aggregate_multiseed_bootstrap(base, broken, seeds, samples=10)


def test_market_and_identifier_features_are_rejected():
    assert_market_blind_contract(["age", "career_starts"])
    for feature in ("competition_id", "runner_number", "fluc2", "race_overlay_score"):
        with pytest.raises(ValueError, match="MARKET-BLIND CONTRACT VIOLATION"):
            assert_market_blind_contract(["age", feature])


def test_every_feature_has_exactly_one_group():
    features = ["recent_margin_avg_3", "distance_m", "jockey_history_starts", "age"]
    groups = assign_feature_groups(features)
    assigned = [feature for members in groups.values() for feature in members]
    assert sorted(assigned) == sorted(features)
    assert len(assigned) == len(set(assigned))


def test_duplicate_and_constant_feature_detection():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0],
        "c": [7.0, 7.0, 7.0, 7.0], "d": [4.0, 3.0, 2.0, 1.0],
    })
    result = redundant_feature_clusters(frame, ["a", "b", "c", "d"])
    assert ["a", "b"] in result["exact_duplicate_columns"]
    assert "c" in result["constant_columns"]


def test_same_seed_builds_deterministic_model():
    x = torch.randn(2, 4, 3); valid = torch.ones((2, 4), dtype=torch.bool)
    torch.manual_seed(29); first = build_neural_model("current_mlp", 3).eval()
    torch.manual_seed(29); second = build_neural_model("current_mlp", 3).eval()
    assert torch.equal(first(x, valid), second(x, valid))


def test_xgboost_groups_are_contiguous_and_validation_never_enters_training():
    training = _rows((10, 20)).reset_index(drop=True)
    validation = _rows((30,)).reset_index(drop=True)
    contract = xgboost_group_contract(training, validation)
    assert contract["training_group_sizes"] == [4, 4]
    assert contract["validation_group_sizes"] == [4]
    with pytest.raises(ValueError, match="entered XGBoost training"):
        xgboost_group_contract(training, training.copy())
    interleaved = training.sort_values("runner_number").reset_index(drop=True)
    with pytest.raises(ValueError, match="not contiguous"):
        xgboost_group_contract(interleaved, validation)


def test_final_selection_cannot_contain_multiple_challengers():
    with pytest.raises(ValueError, match="more than one challenger"):
        validate_final_selection({"selected_challengers": ["a", "b"]})

