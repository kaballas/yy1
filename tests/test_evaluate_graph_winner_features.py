import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from build_racing_graph_features import GRAPH_FEATURE_NAMES
from evaluate_graph_winner_features import (
    chronological_split,
    debutant_presence_metrics,
    graph_experiment_feature_sets,
    load_baseline_features,
    load_joined_rows,
    paired_bootstrap_table,
    rank_metrics,
    selection_eval_metrics,
    winner_start_bands,
)


def test_feature_sets_are_nested_abcd_and_use_clean_baseline(tmp_path):
    manifest = tmp_path / "features.json"
    manifest.write_text(
        json.dumps(
            {
                "models": {
                    "clean": {"features": ["distance_m", "field_size"]},
                    "market": {"features": ["distance_m", "fluc2"]},
                }
            }
        ),
        encoding="utf-8",
    )

    baseline = load_baseline_features(manifest, "clean")
    groups = graph_experiment_feature_sets(baseline)

    assert baseline == ["distance_m", "field_size"]
    assert list(groups) == [
        "graph_a", "graph_b", "graph_c", "graph_d", "graph_e", "graph_only",
    ]
    assert [len(groups[label]) for label in groups] == [2, 8, 18, 29, 77, 87]
    assert set(groups["graph_a"]) < set(groups["graph_b"])
    assert set(groups["graph_b"]) < set(groups["graph_c"])
    assert set(groups["graph_c"]) < set(groups["graph_d"])
    assert set(groups["graph_d"]) < set(groups["graph_e"])
    assert "graph_sire_trainer_similarity" not in groups["graph_c"]
    assert "graph_sire_trainer_similarity" in groups["graph_d"]
    assert "graph_horse_recent_run_similarity" in groups["graph_e"]
    assert (
        "graph_horse_recent_run_similarity_rank_in_race" in groups["graph_e"]
    )
    assert "graph_recent_class_embedding_available" in groups["graph_e"]
    assert groups["graph_only"] == list(GRAPH_FEATURE_NAMES)
    assert all(feature.startswith("graph_") for feature in groups["graph_only"])
    assert not set(baseline) & set(groups["graph_only"])

    with pytest.raises(ValueError, match="not market-blind"):
        load_baseline_features(manifest, "market")


def test_chronological_split_keeps_latest_races_sealed():
    races = pd.DataFrame({"race_id": list(range(1, 11))})

    train, validation, test = chronological_split(races, 2, 3)

    assert train == [1, 2, 3, 4, 5]
    assert validation == [6, 7]
    assert test == [8, 9, 10]


def test_rank_metrics_include_top2_and_winner_start_bands():
    frame = pd.DataFrame(
        {
            "race_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "is_winner": [1, 0, 0, 1, 1, 0, 0, 1],
            "career_starts": [0, 4, 8, 2, 4, 7, 10, 9],
            "runner_mask": [1] * 8,
            "status": ["finished"] * 8,
            "start_time_iso": ["2026-01-01"] * 8,
            "competition_id": [6] * 8,
            "competition_name": ["Venue"] * 8,
            "race_number": [1, 1, 2, 2, 3, 3, 4, 4],
            "race_name": ["R1", "R1", "R2", "R2", "R3", "R3", "R4", "R4"],
        }
    )
    targets = frame["is_winner"].to_numpy()
    scores = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)

    report, bands = winner_start_bands(frame, targets, scores)
    metrics = rank_metrics(report)

    assert metrics["top1_hit_rate"] == 0.5
    assert metrics["top2_hit_rate"] == 1.0
    assert metrics["top3_hit_rate"] == 1.0
    assert set(bands) == {"0", "1-2", "3-5", "6+"}
    assert bands["0"]["top1_hit_rate"] == 1.0
    assert bands["1-2"]["top1_hit_rate"] == 0.0

    presence = debutant_presence_metrics(frame, report)
    assert presence["races_with_debutant"]["races"] == 1.0
    assert presence["races_without_debutant"]["races"] == 3.0


def test_debutant_presence_detects_losing_debutants():
    frame = pd.DataFrame(
        {
            "race_id": [1, 1, 2, 2],
            "career_starts": [8, 0, 5, 3],
        }
    )
    report = pd.DataFrame({"race_id": [1, 2], "winner_rank": [1.0, 2.0]})

    presence = debutant_presence_metrics(frame, report)

    assert presence["races_with_debutant"]["top1_hit_rate"] == 1.0
    assert presence["races_without_debutant"]["top1_hit_rate"] == 0.0


def test_paired_bootstrap_uses_per_race_differences_and_is_deterministic():
    scored = pd.DataFrame(
        {
            "race_id": [1, 2, 3, 4],
            "is_winner": [1, 1, 1, 1],
            "graph_a_rank": [2.0, 2.0, 2.0, 2.0],
            "graph_b_rank": [1.0, 1.0, 1.0, 1.0],
        }
    )

    first = paired_bootstrap_table(scored, "graph_a", 100, 0.95, 42)
    second = paired_bootstrap_table(scored, "graph_a", 100, 0.95, 42)
    top1 = first.loc[first["metric"].eq("top1_hit_rate")].iloc[0]

    pd.testing.assert_frame_equal(first, second)
    assert top1["delta"] == 1.0
    assert top1["ci_lower"] == 1.0
    assert top1["ci_upper"] == 1.0
    assert top1["display_delta"] == 100.0


def _joined_database(path, invalid_snapshot=False):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                race_id INTEGER, start_time_iso TEXT, competition_id INTEGER,
                competition_name TEXT, race_number INTEGER, race_name TEXT,
                runner_number INTEGER, runner_name TEXT, career_starts INTEGER,
                status TEXT, runner_mask INTEGER, is_winner INTEGER,
                distance_m REAL
            )
            """
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "2026-02-10T00:00:00Z", 6, "Venue", 1, "R1", 1, "A", 0, "finished", 1, 1, 1200),
                (1, "2026-02-10T00:00:00Z", 6, "Venue", 1, "R1", 2, "B", 3, "finished", 1, 0, 1200),
            ],
        )
        connection.execute(
            """
            CREATE TABLE graph_features (
                source_rowid INTEGER PRIMARY KEY,
                snapshot_date TEXT,
                graph_horse_trainer_similarity REAL
            )
            """
        )
        snapshot = "2026-02-10T00:00:00Z" if invalid_snapshot else "2026-02-01T00:00:00Z"
        connection.executemany(
            "INSERT INTO graph_features VALUES (?, ?, ?)",
            [(1, snapshot, 0.5), (2, snapshot, 0.2)],
        )


def test_join_uses_source_rowid_and_rejects_noncausal_snapshot(tmp_path):
    valid = tmp_path / "valid.sqlite"
    invalid = tmp_path / "invalid.sqlite"
    _joined_database(valid)
    _joined_database(invalid, invalid_snapshot=True)

    frame = load_joined_rows(
        valid,
        "graph_features",
        ["distance_m", "graph_horse_trainer_similarity"],
    )

    assert frame["source_rowid"].tolist() == [1, 2]
    assert frame["graph_horse_trainer_similarity"].tolist() == [0.5, 0.2]
    with pytest.raises(ValueError, match="chronology violation"):
        load_joined_rows(
            invalid,
            "graph_features",
            ["distance_m", "graph_horse_trainer_similarity"],
        )


def test_selection_metric_is_last_for_xgboost_early_stopping():
    assert selection_eval_metrics("top1")[-1] == "ndcg@1"
    assert selection_eval_metrics("top3")[-1] == "ndcg@3"
    assert selection_eval_metrics("map")[-1] == "map"
    with pytest.raises(ValueError, match="Unknown selection objective"):
        selection_eval_metrics("mrr")
