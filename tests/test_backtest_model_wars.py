import json

import pandas as pd

from backtest_model_wars import load_model_entries, model_wars_summary, parse_date


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


def test_model_wars_date_parser_requires_iso_date():
    assert parse_date("2026-08-22") == "2026-08-22"
