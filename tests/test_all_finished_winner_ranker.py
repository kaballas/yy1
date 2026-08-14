import json

import pandas as pd
import pytest

pytest.importorskip("xgboost")

from train_tune_all_finished_winner_ranker import (
    crossfit_fold_ids,
    load_model_feature_sets,
    merge_reused_oof_scores,
    print_model_feature_report,
    tree_counts,
    tune_dynamic_model_blend,
)


def test_crossfit_assigns_every_whole_race_once():
    race_ids = list(range(1, 12))

    folds = crossfit_fold_ids(race_ids, folds=4)

    flattened = [race_id for fold in folds for race_id in fold]
    assert sorted(flattened) == race_ids
    assert len(flattened) == len(set(flattened))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


def test_crossfit_requires_at_least_two_folds():
    with pytest.raises(ValueError, match="at least two"):
        crossfit_fold_ids([1, 2], folds=1)


def test_tree_counts_reuses_validated_bundle_counts(tmp_path):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(json.dumps({
        "best_tree_counts": {"form": [10, 20, 30]},
    }))

    counts = tree_counts(bundle, "form", ensemble_size=5, fallback=99)

    assert counts == [10, 20, 30, 10, 20]


def test_tree_counts_has_deterministic_fallback(tmp_path):
    counts = tree_counts(
        tmp_path / "missing.json", "market_aware", ensemble_size=3, fallback=44
    )

    assert counts == [44, 44, 44]


def test_model_feature_sets_show_exact_inputs_for_each_model(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "models": {
            "form": {"features": ["speed"]},
            "market_aware": {"features": [
                "weight", "current_market_log_price",
            ]},
            "fun": {"features": ["speed", "weight"]},
        },
    }))

    feature_sets = load_model_feature_sets(manifest, ["speed", "weight"])

    assert feature_sets == {
        "form": ["speed"],
        "market_aware": [
            "weight",
            "current_market_log_price",
        ],
        "fun": ["speed", "weight"],
    }


def test_model_feature_report_prints_features_and_absolute_file(
    tmp_path, capsys
):
    feature_file = tmp_path / "bundle.json"

    print_model_feature_report(feature_file, {"form": ["speed", "weight"]})

    output = capsys.readouterr().out
    assert "MODEL FEATURES" in output
    assert f"feature_file={feature_file.resolve()}" in output
    assert 'model=form feature_count=2 features=["speed", "weight"]' in output


def test_model_feature_manifest_rejects_unavailable_features(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "models": {
            "form": {"features": ["missing"]},
            "market_aware": {"features": [
                "speed", "current_market_log_price", "current_market_rank_pct",
            ]},
        },
    }))

    with pytest.raises(ValueError, match="unavailable form features: missing"):
        load_model_feature_sets(manifest, ["speed"])


def test_dynamic_blend_tunes_every_model_group():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "is_winner": [1, 0, 0, 1],
        "form_score": [1.0, 0.0, 0.0, 1.0],
        "market_aware_score": [0.0, 1.0, 1.0, 0.0],
        "fun_score": [0.8, 0.2, 0.2, 0.8],
    })

    weights, sweep = tune_dynamic_model_blend(
        frame, ["form", "market_aware", "fun"], 0.5, "top1"
    )

    assert set(weights) == {"form", "market_aware", "fun", "market"}
    assert weights["market"] == 0.0
    assert weights["form"] + weights["market_aware"] + weights["fun"] == pytest.approx(1.0)
    assert all(
        weight / 0.5 == pytest.approx(round(weight / 0.5))
        for label, weight in weights.items()
        if label != "market"
    )
    assert {"form_weight", "market_aware_weight", "fun_weight"} <= set(sweep)
    assert "global_simplex" in set(sweep["phase"])


def test_selective_retraining_merges_reused_model_oof_scores():
    fresh = pd.DataFrame({
        "race_id": [1, 1],
        "runner_number": [1, 2],
        "form_score": [0.8, 0.2],
    })
    existing = pd.DataFrame({
        "race_id": [1, 1],
        "runner_number": [1, 2],
        "fun_score": [0.3, 0.7],
        "fun_rank": [2, 1],
    })

    merged = merge_reused_oof_scores(fresh, existing, ["fun"])

    assert merged["form_score"].tolist() == [0.8, 0.2]
    assert merged["fun_score"].tolist() == [0.3, 0.7]
    assert merged["fun_rank"].tolist() == [2, 1]


def test_selective_retraining_rejects_a_different_oof_cohort():
    fresh = pd.DataFrame({"race_id": [1], "runner_number": [1]})
    existing = pd.DataFrame({
        "race_id": [2], "runner_number": [1],
        "fun_score": [0.5], "fun_rank": [1],
    })

    with pytest.raises(ValueError, match="does not match"):
        merge_reused_oof_scores(fresh, existing, ["fun"])
