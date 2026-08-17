import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from analyze_winner_features import (
    activate_top_manifest_features,
    aggregate_fold_metrics,
    eligible_race_table,
    feature_permutation_scope,
    load_finished_runners,
    numeric_heuristic_scores,
    outcome_conditioned_market_cohort,
    parse_competition_ids,
    permute_feature,
    random_top3_metrics,
    resolve_validation_races,
    select_features,
    summarize_permutations,
    temporal_validation_folds,
    top_features_select_sql,
    top3_metrics,
)


def test_feature_selection_excludes_leakage_market_and_sparse_columns():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "is_winner": [1, 0, 0, 1],
        "top3_mask": [1, 1, 0, 0],
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


def test_feature_selection_excludes_market_disagreement_columns():
    frame = pd.DataFrame({
        "form_signal": [0.1, 0.2, 0.3, 0.4],
        "finish_rank_minus_market_rank": [-2.0, 1.0, 0.0, 3.0],
        "finish_market_rank_abs_gap": [2.0, 1.0, 0.0, 3.0],
    })

    features = select_features(frame, minimum_observations=2)

    assert features == ["form_signal"]


def test_feature_selection_combines_observation_and_coverage_thresholds():
    frame = pd.DataFrame({
        "well_covered": [1.0, 2.0, 3.0, 4.0],
        "half_covered": [1.0, 2.0, np.nan, np.nan],
    })

    features = select_features(
        frame, minimum_observations=1, minimum_coverage=0.75
    )

    assert features == ["well_covered"]


def test_load_finished_runners_keeps_positive_and_negative_top3_rows(tmp_path):
    database = tmp_path / "runners.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                race_id INTEGER,
                runner_number INTEGER,
                top3_mask INTEGER,
                runner_mask INTEGER,
                status TEXT,
                competition_id INTEGER,
                start_time_iso TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 1, 1, "finished", 580, "2026-01-01T00:00:00Z"),
                (1, 2, 0, 1, "finished", 580, "2026-01-01T00:00:00Z"),
                (1, 3, 0, 0, "finished", 580, "2026-01-01T00:00:00Z"),
                (1, 4, 0, 1, "scratched", 580, "2026-01-01T00:00:00Z"),
            ],
        )

    loaded = load_finished_runners(database, competition_id=580)

    assert loaded["runner_number"].tolist() == [1, 2]
    assert loaded["top3_mask"].tolist() == [1, 0]


def test_competition_ids_accept_comma_separated_values_without_duplicates():
    assert parse_competition_ids("570, 580,570,335") == [570, 580, 335]


def test_load_finished_runners_accepts_multiple_competitions(tmp_path):
    database = tmp_path / "runners.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                race_id INTEGER,
                runner_number INTEGER,
                top3_mask INTEGER,
                runner_mask INTEGER,
                status TEXT,
                competition_id INTEGER,
                start_time_iso TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 1, 1, 1, "finished", 570, "2026-01-01T00:00:00Z"),
                (2, 1, 1, 1, "finished", 580, "2026-01-02T00:00:00Z"),
                (3, 1, 1, 1, "finished", 999, "2026-01-03T00:00:00Z"),
            ],
        )

    loaded = load_finished_runners(database, competition_id=[570, 580])

    assert loaded["competition_id"].tolist() == [570, 580]


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


def test_top3_metrics_are_equal_per_race_and_rank_top3_runners():
    targets = np.asarray([1, 1, 1, 0, 1, 1, 1, 0])
    scores = np.asarray([0.8, 0.7, 0.1, 0.6, 0.9, 0.8, 0.7, 0.1])
    race_ids = np.asarray([1, 1, 1, 1, 2, 2, 2, 2])

    metrics = top3_metrics(targets, scores, race_ids)

    assert metrics["top3_hit_rate"] == pytest.approx(5 / 6)
    assert metrics["top3_mrr"] == pytest.approx(
        (1 + 1 / 2 + 1 / 4 + 1 + 1 / 2 + 1 / 3) / 6
    )


def test_random_top3_metrics_use_each_race_field_size():
    metrics = random_top3_metrics(
        np.asarray([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    )

    assert metrics["auc"] == 0.5
    assert metrics["top3_hit_rate"] == pytest.approx((3 / 6 + 3 / 4) / 2)
    assert metrics["top3_mrr"] == pytest.approx(
        ((1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) / 6
         + (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
        / 2
    )


def test_fold_metrics_weight_auc_by_rows_and_rank_metrics_by_races():
    metrics = aggregate_fold_metrics(
        [
            {"auc": 0.5, "top3_hit_rate": 0.2, "top3_mrr": 0.3},
            {"auc": 0.8, "top3_hit_rate": 0.6, "top3_mrr": 0.7},
        ],
        row_counts=[100, 200],
        race_counts=[10, 30],
    )

    assert metrics["auc"] == pytest.approx(0.7)
    assert metrics["top3_hit_rate"] == pytest.approx(0.5)
    assert metrics["top3_mrr"] == pytest.approx(0.6)


def test_numeric_heuristic_uses_training_median_for_missing_validation():
    scores = numeric_heuristic_scores(
        pd.DataFrame({"win_percentage": [0.1, 0.3, np.nan]}),
        pd.DataFrame({"win_percentage": [0.5, np.nan]}),
        "win_percentage",
    )

    assert scores == pytest.approx([0.5, 0.2])


def test_adaptive_validation_and_expanding_temporal_folds():
    assert resolve_validation_races(1305, None) == 300
    assert resolve_validation_races(92, None) == 23
    assert resolve_validation_races(92, 20) == 20

    races = pd.DataFrame({"race_id": np.arange(1, 13)})
    folds = temporal_validation_folds(races, validation_races=6, fold_count=3)

    assert [(train.tolist(), validation.tolist()) for train, validation in folds] == [
        ([1, 2, 3, 4, 5, 6], [7, 8]),
        ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]),
    ]


def test_permutation_summary_uses_positive_values_for_worse_metrics():
    baseline = {
        "auc": 0.7,
        "top3_hit_rate": 0.7,
        "top3_mrr": 0.6,
    }
    shuffled = [{
        "auc": 0.65,
        "top3_hit_rate": 0.6,
        "top3_mrr": 0.55,
    }]

    summary = summarize_permutations(baseline, shuffled)

    assert summary["auc_drop_mean"] == pytest.approx(0.05)
    assert summary["top3_hit_drop_mean"] == pytest.approx(0.1)
    assert summary["top3_mrr_drop_mean"] == pytest.approx(0.05)


def test_eligible_races_require_four_runners_and_exactly_three_top3():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
        "start_time_iso": [
            *["2026-01-01T00:00:00Z"] * 4,
            *["2026-01-02T00:00:00Z"] * 4,
            *["2026-01-03T00:00:00Z"] * 3,
        ],
        "top3_mask": [1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1],
    })

    races, skipped = eligible_race_table(frame)

    assert races["race_id"].tolist() == [1]
    assert skipped == 2


def test_detects_outcome_conditioned_market_miss_cohort():
    rows = []
    for race_id in range(100):
        rows.extend([
            {"race_id": race_id, "fluc2": 2.0, "is_winner": 0},
            {"race_id": race_id, "fluc2": 10.0, "is_winner": 1},
        ])

    conditioned, races, wins = outcome_conditioned_market_cohort(pd.DataFrame(rows))

    assert conditioned
    assert races == 100
    assert wins == 0


def test_normal_market_cohort_is_not_flagged():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "fluc2": [2.0, 5.0, 3.0, 6.0],
        "is_winner": [1, 0, 0, 1],
    })

    conditioned, races, wins = outcome_conditioned_market_cohort(
        frame, minimum_races=2
    )

    assert not conditioned
    assert races == 2
    assert wins == 1


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


def test_top_features_select_sql_is_copy_pasteable_and_quotes_columns():
    sql = top_features_select_sql(["speed", 'odd"name'], competition_id=580)

    assert '"race_id",\n    "selection_id",\n    "runner_number"' in sql
    assert '"speed",\n    "odd""name"' in sql
    assert 'AND "competition_id" = 580' in sql
    assert 'WHERE "top3_mask" IN (0, 1)' in sql
    assert sql.endswith(
        'ORDER BY "start_time_iso", "race_id", "runner_number";'
    )


def test_top_features_select_sql_accepts_multiple_competitions():
    sql = top_features_select_sql(["speed"], competition_id=[570, 580, 335])

    assert 'AND "competition_id" IN (570, 580, 335)' in sql
