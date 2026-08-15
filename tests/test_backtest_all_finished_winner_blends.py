import numpy as np
import pandas as pd
import pytest

from backtest_all_finished_winner_blends import (
    artifact_strategies,
    backtest_summary,
    best_backtest_strategy,
    blend_weights_table,
    filter_complete_races,
    parse_competition_ids,
    parse_race_numbers,
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


def test_best_backtest_strategy_matches_first_overall_strategy():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "runner_number": [1, 2, 1, 2],
        "is_winner": [1, 0, 0, 1],
        "fluc2": [2.0, 3.0, 4.0, 5.0],
        "good_score": [0.9, 0.1, 0.1, 0.9],
        "bad_score": [0.1, 0.9, 0.9, 0.1],
        "market_score": [0.5, 0.4, 0.5, 0.4],
    })
    strategies = {
        "good_only": {"good": 1.0, "bad": 0.0, "market": 0.0},
        "bad_only": {"good": 0.0, "bad": 1.0, "market": 0.0},
    }

    name, weights, metrics = best_backtest_strategy(
        frame, ["good", "bad"], strategies
    )

    assert name == "good_only"
    assert weights == strategies["good_only"]
    assert metrics["top1_hit_rate"] == pytest.approx(1.0)


def test_blend_weights_table_reports_normalized_values_actually_used():
    table = blend_weights_table(
        ["form", "aware"],
        {"selected": {"form": 0.1, "aware": 0.3, "market": 0.0}},
    )

    row = table.iloc[0]
    assert row["form"] == pytest.approx(0.25)
    assert row["aware"] == pytest.approx(0.75)
    assert row["market"] == pytest.approx(0.0)
    assert row["configured_sum"] == pytest.approx(0.4)


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


def test_competition_ids_accept_comma_separated_values_without_duplicates():
    assert parse_competition_ids("580, 570,580") == [580, 570]


def test_filtering_accepts_multiple_competition_ids():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3],
        "competition_id": [10, 10, 20, 20, 30, 30],
    })

    filtered = filter_complete_races(frame, [10, 30], None, None)

    assert filtered["race_id"].tolist() == [1, 1, 3, 3]
    assert filtered["competition_id"].tolist() == [10, 10, 30, 30]


def test_race_numbers_accept_comma_separated_values_without_duplicates():
    assert parse_race_numbers("7, 8,7") == [7, 8]


def test_filtering_accepts_race_number_with_competition_ids():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3],
        "competition_id": [10, 10, 10, 10, 20, 20],
        "race_number": [7, 7, 8, 8, 7, 7],
    })

    filtered = filter_complete_races(frame, [10, 20], None, None, [7])

    assert filtered["race_id"].tolist() == [1, 1, 3, 3]
    assert filtered["race_number"].tolist() == [7, 7, 7, 7]


def test_filtering_rejects_missing_race_number_column():
    frame = pd.DataFrame({
        "race_id": [1, 1],
        "competition_id": [10, 10],
    })

    with pytest.raises(ValueError, match="no race_number column"):
        filter_complete_races(frame, None, None, None, [7])
