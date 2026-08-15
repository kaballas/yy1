import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from inspect_winner_ranker import (
    attach_oof_scores,
    comparison_indices,
    contribution_delta_table,
    load_finished_race,
    model_features_from_bundle,
)


def test_model_features_use_exact_configured_schema():
    bundle = {
        "form_features": ["legacy"],
        "model_features": {"x1": ["speed", "weight"]},
    }

    assert model_features_from_bundle(bundle, "x1") == ["speed", "weight"]


def test_comparison_uses_winner_for_wrong_pick_and_runner_up_for_correct_pick():
    target = np.asarray([0, 1, 0])

    assert comparison_indices(target, np.asarray([0.9, 0.8, 0.1])) == (
        0, 1, "selected_vs_actual_winner"
    )
    assert comparison_indices(target, np.asarray([0.8, 0.9, 0.1])) == (
        1, 0, "correct_winner_vs_model_runner_up"
    )


def test_contribution_delta_reports_direction_and_member_agreement():
    matrix = pd.DataFrame({"speed": [8.0, 5.0], "weight": [55.0, 57.0]})
    # Two members, two runners, two features plus bias.
    contributions = np.asarray([
        [[2.0, -0.5, 0.1], [0.5, 0.5, 0.1]],
        [[1.0, -0.2, 0.2], [0.0, 0.2, 0.2]],
    ])

    result = contribution_delta_table(matrix, contributions, 0, 1)

    speed = result.set_index("feature").loc["speed"]
    assert speed["shap_delta_mean"] == pytest.approx(1.25)
    assert speed["member_sign_agreement"] == 1.0
    assert result.iloc[0]["feature"] == "speed"


def test_oof_scores_attach_by_runner_identity(tmp_path):
    path = tmp_path / "oof.csv"
    pd.DataFrame({
        "race_id": [7, 7],
        "runner_number": [2, 1],
        "x1_score": [0.2, 0.8],
    }).to_csv(path, index=False)
    output = pd.DataFrame({"runner_number": [1, 2], "runner_name": ["A", "B"]})

    result = attach_oof_scores(output, path, 7, "x1")

    assert result["x1_oof_score"].tolist() == [0.8, 0.2]
    assert result["x1_oof_rank"].tolist() == [1, 2]


def test_finished_race_loader_keeps_only_active_runners(tmp_path):
    path = tmp_path / "races.sqlite"
    metadata_types = {
        "race_id": "INTEGER", "start_time_iso": "TEXT",
        "competition_id": "INTEGER", "competition_name": "TEXT",
        "race_number": "INTEGER", "race_name": "TEXT",
        "runner_number": "INTEGER", "runner_name": "TEXT",
        "runner_mask": "INTEGER", "status": "TEXT",
        "source_betting_status": "TEXT", "active_field_size": "INTEGER",
        "fluc2": "REAL", "derived_racing_features_version": "TEXT",
        "is_winner": "INTEGER", "speed": "REAL",
    }
    with sqlite3.connect(path) as connection:
        columns = ", ".join(
            f'"{name}" {kind}' for name, kind in metadata_types.items()
        )
        connection.execute(f"CREATE TABLE race_runners ({columns})")
        keys = list(metadata_types)
        placeholders = ", ".join("?" for _ in keys)
        rows = []
        for runner, active, winner in ((1, 1, 1), (2, 1, 0), (3, 0, 0)):
            values = {
                "race_id": 7, "start_time_iso": "2026-01-01Z",
                "competition_id": 580, "competition_name": "Track",
                "race_number": 1, "race_name": "Race", "runner_number": runner,
                "runner_name": chr(64 + runner), "runner_mask": active,
                "status": "finished", "source_betting_status": "RESULTED",
                "active_field_size": 2, "fluc2": float(runner + 1),
                "derived_racing_features_version": "v3",
                "is_winner": winner, "speed": float(10 - runner),
            }
            rows.append(tuple(values[key] for key in keys))
        connection.executemany(
            f"INSERT INTO race_runners VALUES ({placeholders})", rows
        )

    result = load_finished_race(path, 7, ["speed"])

    assert result["runner_number"].tolist() == [1, 2]
    assert result["is_winner"].tolist() == [1, 0]
