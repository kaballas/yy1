import numpy as np
import pandas as pd
import pytest

from backtest_all_finished_winner_blends import (
    artifact_strategies,
    backtest_summary,
    filter_complete_races,
)


def test_artifact_strategies_keep_different_config_and_bundle_blends():
    bundle = {
        "models": {"form": [], "aware": []},
        "selected_blend_weights": {"form": 0.8, "aware": 0.2},
        "deployment_blend_weights": {"form": 1.0},
    }
    blend = {
        "model_labels": ["form", "aware"],
        "selected_weights": {"form": 0.1, "aware": 0.1, "market": 0.0},
    }

    labels, strategies = artifact_strategies(bundle, blend)

    assert labels == ["form", "aware"]
    assert sum(strategies["config_selected"].values()) == pytest.approx(0.2)
    assert strategies["bundle_selected"]["form"] == pytest.approx(0.8)
    assert strategies["equal_model_blend"] == {
        "form": 0.5,
        "aware": 0.5,
        "market": 0.0,
    }


def test_backtest_reports_ranking_and_flat_win_profit():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "runner_number": [1, 2, 1, 2],
        "is_winner": [1, 0, 0, 1],
        "fluc2": [3.0, 2.0, 4.0, 5.0],
        "form_score": [0.9, 0.1, 0.8, 0.2],
        "market_score": [0.9, 0.1, 0.8, 0.2],
    })
    strategies = {
        "form_only": {"form": 1.0, "market": 0.0},
    }

    summary, selections = backtest_summary(frame, ["form"], strategies)

    row = summary.iloc[0]
    assert row["top1_hit_rate"] == pytest.approx(0.5)
    assert row["flat_win_profit"] == pytest.approx(1.0)
    assert row["flat_win_roi"] == pytest.approx(0.5)
    assert selections["runner_number"].tolist() == [1, 1]


def test_filtering_keeps_whole_races():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "competition_id": [10, 10, 20, 20],
        "start_time_iso": [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-02T00:00:00Z",
        ],
    })

    filtered = filter_complete_races(frame, 20, None, None)

    assert filtered["race_id"].tolist() == [2, 2]
    assert np.all(filtered["competition_id"] == 20)
