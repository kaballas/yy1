import json

import pandas as pd

from backtest_model_wars import (
    feature_wars_summary,
    load_model_entries,
    model_wars_summary,
    parse_date,
    update_feature_manifest_model,
)


def test_model_wars_loads_saved_ensemble_members(tmp_path):
    first = tmp_path / "member_1.json"
    second = tmp_path / "member_2.json"
    first.write_text("{}")
    second.write_text("{}")
    (tmp_path / "per_race_models_manifest.json").write_text(json.dumps({
        "models": [{
            "name": "race_10",
            "model": str(first),
            "models": [str(first), str(second)],
            "trained_on_race_id": 10,
            "details": {"input_features": ["speed"]},
        }],
    }))

    entries = load_model_entries(tmp_path)

    assert entries[0]["models"] == [first, second]


def test_model_wars_ranks_top3_then_winner_rate():
    results = pd.DataFrame({
        "model": ["a", "a", "b", "b"],
        "trained_on_race_id": [1, 1, 2, 2],
        "race_id": [10, 11, 10, 11],
        "winner_rank": [1, 4, 2, 3],
        "is_winner_1": [1, 0, 0, 0],
        "winner_top3": [1, 0, 1, 1],
    })

    summary = model_wars_summary(results)

    assert summary["model"].tolist() == ["b", "a"]
    assert summary["top3_pct"].tolist() == [100.0, 50.0]
    assert summary.loc[summary["model"].eq("a"), "is_winner_1_pct"].iloc[0] == 50.0


def test_feature_wars_aggregates_models_using_each_feature():
    leaderboard = pd.DataFrame({
        "model": ["a", "b"],
        "trained_on_race_id": [1, 2],
        "races_tested": [10, 20],
        "is_winner_1_count": [2, 6],
        "winner_top3_count": [5, 14],
        "mean_winner_rank": [3.0, 2.0],
        "mrr": [0.4, 0.5],
        "is_winner_1_pct": [20.0, 30.0],
        "top3_pct": [50.0, 70.0],
    })
    entries = [
        {"name": "a", "features": ["speed", "form"]},
        {"name": "b", "features": ["speed"]},
    ]

    summary = feature_wars_summary(leaderboard, entries).set_index("feature")

    assert summary.loc["speed", "models_using_feature"] == 2
    assert summary.loc["speed", "model_race_tests"] == 30
    assert summary.loc["speed", "top3_pct"] == 100 * 19 / 30
    assert summary.loc["speed", "best_model"] == "b"
    assert summary.loc["form", "top3_pct"] == 50.0


def test_model_wars_updates_top3_feature_manifest_group(tmp_path):
    manifest = tmp_path / "winner_ranker_features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "models": {"a1": {"features": ["old"]}},
    }))
    leaderboard = tmp_path / "features.csv"
    leaderboard.write_text("feature\nspeed\nform\n")

    updated = update_feature_manifest_model(
        manifest,
        "top3",
        ["speed", "form", "speed"],
        evaluation_date="2026-08-22",
        feature_leaderboard_path=leaderboard,
    )

    payload = json.loads(updated.read_text())
    assert payload["models"]["a1"]["features"] == ["old"]
    assert payload["models"]["top3"]["features"] == ["speed", "form"]
    assert payload["models"]["top3"]["selection"]["feature_count"] == 2
    assert payload["models"]["top3"]["selection"][
        "evaluation_date_utc"
    ] == "2026-08-22"


def test_model_wars_date_parser_requires_iso_date():
    assert parse_date("2026-08-22") == "2026-08-22"
