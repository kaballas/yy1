import json

import numpy as np
import pandas as pd
import pytest

from analyze_winner_features import (
    activate_top_manifest_features,
    eligible_race_table,
    feature_permutation_scope,
    permute_feature,
    select_features,
    summarize_permutations,
    winner_metrics,
)


def test_feature_selection_excludes_leakage_market_and_sparse_columns():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "is_winner": [1, 0, 0, 1],
        "is_trainable": [1, 1, 1, 1],
        "fluc2": [2.0, 3.0, 4.0, 5.0],
        "market_move": [0.1, 0.2, 0.3, 0.4],
        "recent_1_starting_price": [3.0, 4.0, 5.0, 6.0],
        "form_signal": [1.0, 2.0, 3.0, 4.0],
        "sparse": [1.0, np.nan, np.nan, np.nan],
        "constant": [1.0, 1.0, 1.0, 1.0],
    })

    features = select_features(frame, minimum_observations=3)

    assert features == ["recent_1_starting_price", "form_signal"]


def test_within_race_permutation_preserves_race_constant_features():
    values = np.asarray([1200, 1200, 1400, 1400, 1400])
    race_ids = np.asarray([1, 1, 2, 2, 2])

    shuffled = permute_feature(
        values, race_ids, np.random.default_rng(42), "within-race"
    )

    assert np.array_equal(shuffled, values)


def test_auto_uses_race_block_for_race_constant_feature():
    values = np.asarray([1200, 1200, 1400, 1400, 1600, 1600])
    race_ids = np.asarray([1, 1, 2, 2, 3, 3])

    scope = feature_permutation_scope(values, race_ids, "auto")
    shuffled = permute_feature(values, race_ids, np.random.default_rng(3), scope)

    assert scope == "race-block"
    assert all(len(set(shuffled[race_ids == race_id])) == 1 for race_id in (1, 2, 3))
    assert not np.array_equal(shuffled, values)


def test_auto_uses_within_race_for_runner_varying_feature():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    race_ids = np.asarray([1, 1, 2, 2])

    assert feature_permutation_scope(values, race_ids, "auto") == "within-race"


def test_winner_metrics_are_equal_per_race_and_rank_winners():
    targets = np.asarray([1, 0, 0, 0, 1])
    scores = np.asarray([0.8, 0.2, 0.9, 0.7, 0.8])
    race_ids = np.asarray([1, 1, 2, 2, 2])

    metrics = winner_metrics(targets, scores, race_ids)

    assert metrics["top1_hit_rate"] == pytest.approx(0.5)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert metrics["mean_winner_rank"] == pytest.approx(1.5)
    assert metrics["race_logloss"] > 0


def test_winner_metrics_softmax_raw_margins():
    metrics = winner_metrics(
        np.asarray([1, 0]), np.asarray([0.0, 1.0]), np.asarray([7, 7])
    )

    assert metrics["race_logloss"] == pytest.approx(np.log1p(np.e))


def test_permutation_summary_uses_positive_values_for_worse_metrics():
    baseline = {
        "auc": 0.7,
        "top1_hit_rate": 0.5,
        "mrr": 0.6,
        "race_logloss": 1.2,
        "mean_winner_rank": 2.0,
    }
    shuffled = [{
        "auc": 0.65,
        "top1_hit_rate": 0.4,
        "mrr": 0.55,
        "race_logloss": 1.3,
        "mean_winner_rank": 2.2,
    }]

    summary = summarize_permutations(baseline, shuffled)

    assert summary["auc_drop_mean"] == pytest.approx(0.05)
    assert summary["top1_drop_mean"] == pytest.approx(0.1)
    assert summary["mrr_drop_mean"] == pytest.approx(0.05)
    assert summary["race_logloss_increase_mean"] == pytest.approx(0.1)
    assert summary["winner_rank_increase_mean"] == pytest.approx(0.2)


def test_eligible_races_skip_missing_or_multiple_winners():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3],
        "start_time_iso": [
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z",
        ],
        "is_winner": [1, 0, 0, 0, 1, 1],
    })

    races, skipped = eligible_race_table(frame)

    assert races["race_id"].tolist() == [1]
    assert skipped == 2


def test_activate_top_manifest_features_removes_them_from_zero_bucket(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "features": ["speed", "distance", "weight"],
        "zeroed_features": ["speed", "distance", "weight"],
    }))

    activated = activate_top_manifest_features(manifest, ["speed", "missing"])
    payload = json.loads(manifest.read_text())

    assert activated == ["speed"]
    assert payload["zeroed_features"] == ["distance", "weight"]
