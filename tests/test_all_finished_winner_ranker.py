import json

import pandas as pd
import pytest

pytest.importorskip("xgboost")

from train_tune_all_finished_winner_ranker import (
    aggregate_tree_counts,
    crossfit_fold_ids,
    filter_races_by_utc_weekday,
    inner_tree_count_split,
    load_model_feature_sets,
    load_race_model_feature_sets,
    normalize_requested_models,
    select_requested_model_groups,
    merge_reused_oof_scores,
    print_model_feature_report,
    tree_count_eval_metrics,
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


def test_training_weekday_filter_keeps_utc_saturdays():
    races = pd.DataFrame({
        "race_id": [1, 2, 3],
        "start_time": [
            "2026-08-21T23:59:59Z",
            "2026-08-22T00:00:00Z",
            "2026-08-23T00:00:00Z",
        ],
    })

    filtered = filter_races_by_utc_weekday(races, "Saturday")

    assert filtered["race_id"].tolist() == [2]


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


def test_inner_tree_count_split_uses_chronological_tail_and_twenty_percent_cap():
    inner_train, inner_validation = inner_tree_count_split(
        list(range(1, 101)), maximum_validation_races=50
    )

    assert inner_train == list(range(1, 81))
    assert inner_validation == list(range(81, 101))
    assert set(inner_train).isdisjoint(inner_validation)


def test_inner_tree_count_split_respects_smaller_configured_maximum():
    inner_train, inner_validation = inner_tree_count_split(
        list(range(1, 101)), maximum_validation_races=7
    )

    assert inner_train == list(range(1, 94))
    assert inner_validation == list(range(94, 101))


def test_aggregate_tree_counts_uses_per_member_fold_medians():
    counts = aggregate_tree_counts([
        [10, 100, 30],
        [20, 80, 35],
        [30, 120, 25],
        [40, 90, 40],
        [50, 110, 20],
    ])

    assert counts == [30, 100, 30]


@pytest.mark.parametrize(("objective", "metric"), [
    ("top1", "ndcg@1"),
    ("top3", "ndcg@3"),
    ("mrr", "map"),
    ("composite", "map"),
])
def test_tree_count_selection_metric_matches_objective(objective, metric):
    assert tree_count_eval_metrics(objective)[-1] == metric


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


def test_race_model_manifest_imports_every_embedded_feature_set(tmp_path):
    manifest = tmp_path / "per_race_models_manifest.json"
    manifest.write_text(json.dumps({
        "models": [
            {"name": "race_10", "details": {"input_features": ["speed"]}},
            {"name": "race_11", "details": {
                "input_features": ["weight", "current_market_log_price"],
            }},
        ],
    }))

    feature_sets = load_race_model_feature_sets(
        manifest, ["speed", "weight"]
    )

    assert feature_sets == {
        "race_10": ["speed"],
        "race_11": ["weight", "current_market_log_price"],
    }


def test_race_model_manifest_removes_duplicate_feature_definitions(
    tmp_path, capsys
):
    manifest = tmp_path / "per_race_models_manifest.json"
    manifest.write_text(json.dumps({
        "models": [
            {"name": "race_10", "details": {
                "input_features": ["speed", "weight"],
            }},
            {"name": "race_11", "details": {
                "input_features": ["speed", "weight"],
            }},
        ],
    }))

    feature_sets = load_race_model_feature_sets(
        manifest, ["speed", "weight"]
    )

    assert feature_sets == {"race_10": ["speed", "weight"]}
    output = capsys.readouterr().out
    assert "duplicate_race_model_feature_sets_removed=1" in output
    assert '"race_11": "race_10"' in output


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


def test_models_option_creates_an_exclusive_single_model_run():
    feature_sets = {"f": ["speed"], "m1": ["weight"], "x1": ["barrier"]}

    selected, training, reused = select_requested_model_groups(
        feature_sets, ["f"], False
    )

    assert selected == {"f": ["speed"]}
    assert training == ["f"]
    assert reused == []


def test_models_option_accepts_comma_and_space_separated_names():
    assert normalize_requested_models(["f,x1"]) == ["f", "x1"]
    assert normalize_requested_models(["f", "x1,m1"]) == ["f", "x1", "m1"]


def test_reusing_unselected_models_must_be_explicit():
    feature_sets = {"f": ["speed"], "m1": ["weight"]}

    selected, training, reused = select_requested_model_groups(
        feature_sets, ["f"], True
    )

    assert selected == feature_sets
    assert training == ["f"]
    assert reused == ["m1"]


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


def test_selective_retraining_warns_and_keeps_only_complete_matching_races():
    fresh = pd.DataFrame({"race_id": [1, 1, 2], "runner_number": [1, 2, 1]})
    existing = pd.DataFrame({
        "race_id": [1, 1, 2], "runner_number": [1, 2, 2],
        "fun_score": [0.5, 0.4, 0.3], "fun_rank": [1, 2, 1],
    })

    with pytest.warns(RuntimeWarning, match="only complete races"):
        merged = merge_reused_oof_scores(fresh, existing, ["fun"])

    assert merged["race_id"].tolist() == [1, 1]
    assert merged["runner_number"].tolist() == [1, 2]


def test_selective_retraining_rejects_when_no_complete_races_match():
    fresh = pd.DataFrame({"race_id": [1], "runner_number": [1]})
    existing = pd.DataFrame({
        "race_id": [2], "runner_number": [1],
        "fun_score": [0.5], "fun_rank": [1],
    })

    with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match="no complete"):
        merge_reused_oof_scores(fresh, existing, ["fun"])
