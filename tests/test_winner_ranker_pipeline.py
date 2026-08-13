import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from rank_winner_models import load_active_race, ranked_output
from src.winner_ranker import (
    blend_scores,
    chronological_race_split,
    current_market_features,
    eligible_races,
    is_current_market_feature,
    market_deviation_metrics,
    rank_percentiles,
    select_blend_weights,
    select_form_features,
    winner_metrics,
)


def test_form_selection_excludes_results_identifiers_and_current_market():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "competition_id": [999] * 4,
        "runner_number": [1, 2, 1, 2],
        "finish_place": [1, 2, 2, 1],
        "is_winner": [1, 0, 0, 1],
        "top3_mask": [1, 1, 1, 1],
        "fluc2": [2.0, 5.0, 4.0, 3.0],
        "market_open_to_fluc2_move": [0.1, 0.2, 0.3, 0.4],
        "historical_market_overperformance_weighted_3": [0.1, 0.4, 0.2, 0.5],
        "form": [0.2, 0.3, 0.4, 0.5],
    })
    features, _ = select_form_features(frame, frame.columns, 0.5)

    assert features == [
        "historical_market_overperformance_weighted_3", "form"
    ]
    assert is_current_market_feature("fluc2")
    assert not is_current_market_feature(
        "historical_market_overperformance_weighted_3"
    )


def test_feature_selection_removes_exact_duplicates_deterministically():
    frame = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [1.0, 2.0, 3.0],
        "c": [3.0, 2.0, 1.0],
    })
    features, duplicates = select_form_features(frame, ["a", "b", "c"], 1.0)

    assert features == ["a", "c"]
    assert duplicates == {"b": "a"}


def test_chronological_split_keeps_whole_ordered_races():
    races = pd.DataFrame({"race_id": list(range(1, 11))})

    train, validation, test = chronological_race_split(races, 2, 3)

    assert train == [1, 2, 3, 4, 5]
    assert validation == [6, 7]
    assert test == [8, 9, 10]


def test_eligible_races_requires_one_winner_and_minimum_field():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3, 3],
        "start_time_iso": [
            "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z",
            "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z",
            "2026-01-03T00:00:00Z", "2026-01-03T00:00:00Z",
            "2026-01-03T00:00:00Z",
        ],
        "is_winner": [1, 0, 0, 0, 1, 0, 0],
    })

    races = eligible_races(frame, minimum_runners=2)

    assert races["race_id"].tolist() == [1, 3]


def test_current_market_features_use_lower_price_as_better():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1],
        "fluc2": [2.0, 4.0, 0.0],
    })

    result = current_market_features(frame)

    assert result.loc[0, "current_market_rank_pct"] == pytest.approx(1.0)
    assert result.loc[1, "current_market_rank_pct"] == pytest.approx(0.0)
    assert np.isnan(result.loc[2, "current_market_rank_pct"])
    assert result.loc[0, "current_market_log_price"] < result.loc[1, "current_market_log_price"]


def test_rank_percentile_and_winner_metrics_have_intuitive_direction():
    scores = np.asarray([0.8, 0.4, 0.1, 0.1, 0.9])
    ids = np.asarray([1, 1, 1, 2, 2])
    target = np.asarray([1, 0, 0, 0, 1])

    percentile = rank_percentiles(scores, ids)
    metrics = winner_metrics(target, percentile, ids)

    assert percentile.tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0, 1.0])
    assert metrics["top1_hit_rate"] == 1.0
    assert metrics["mrr"] == 1.0


def test_validation_blend_selection_can_choose_non_market_model():
    ids = np.asarray([1, 1, 2, 2])
    target = np.asarray([1, 0, 0, 1])
    form = np.asarray([1.0, 0.0, 0.0, 1.0])
    aware = np.asarray([0.7, 0.3, 0.3, 0.7])
    market = np.asarray([0.0, 1.0, 1.0, 0.0])

    weights, metrics = select_blend_weights(
        target, ids, form, aware, market, step=0.5
    )

    assert weights["market"] == 0.0
    assert metrics["top1_hit_rate"] == 1.0
    selected = blend_scores(form, aware, market, weights)
    assert winner_metrics(target, selected, ids)["top1_hit_rate"] == 1.0


def test_ranked_output_exposes_form_market_disagreement():
    frame = pd.DataFrame({
        "runner_number": [1, 2, 3],
        "runner_name": ["A", "B", "C"],
        "fluc2": [2.0, 5.0, 10.0],
    })
    form = np.asarray([0.0, 1.0, 0.5])
    market = np.asarray([1.0, 0.5, 0.0])

    output = ranked_output(
        frame,
        {"form": form, "deployment": form, "market": market},
        "deployment",
    )

    assert output.iloc[0]["runner_number"] == 2
    assert output.iloc[0]["market_to_form_upgrade"] == 1
    assert output.iloc[0]["contrarian_top3"] == 0


def test_deployment_ranking_is_unchanged_when_current_market_changes():
    frame = pd.DataFrame({
        "runner_number": [1, 2, 3],
        "runner_name": ["A", "B", "C"],
        "fluc2": [2.0, 5.0, 10.0],
    })
    form = np.asarray([0.1, 0.9, 0.5])
    first_market = np.asarray([1.0, 0.5, 0.0])
    reversed_market = np.asarray([0.0, 0.5, 1.0])

    first = ranked_output(
        frame,
        {"form": form, "deployment": form, "market": first_market},
        "deployment",
    )
    second = ranked_output(
        frame,
        {"form": form, "deployment": form, "market": reversed_market},
        "deployment",
    )

    assert first["runner_number"].tolist() == [2, 3, 1]
    assert second["runner_number"].tolist() == [2, 3, 1]


def test_market_deviation_reports_corrected_and_damaged_picks():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3],
        "runner_number": [1, 2, 1, 2, 1, 2],
        "is_winner": [0, 1, 1, 0, 0, 1],
        "market_rank": [1, 2, 1, 2, 1, 2],
        # Correct race 1, damage race 2, and keep the market choice in race 3.
        "selected_rank": [2, 1, 2, 1, 1, 2],
    })

    metrics = market_deviation_metrics(frame, "selected")

    assert metrics["top_pick_changes"] == 2
    assert metrics["market_losses_corrected"] == 1
    assert metrics["market_wins_damaged"] == 1
    assert metrics["net_winners_gained"] == 0


def test_live_loader_excludes_scratched_and_never_requests_outcome(tmp_path):
    database = tmp_path / "races.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE race_runners (race_id INTEGER, start_time_iso TEXT, "
            "competition_id INTEGER, competition_name TEXT, race_number INTEGER, "
            "race_name TEXT, runner_number INTEGER, runner_name TEXT, "
            "runner_mask INTEGER, status TEXT, source_betting_status TEXT, "
            "active_field_size INTEGER, fluc2 REAL, "
            "derived_racing_features_version TEXT, "
            "form REAL, is_winner INTEGER)"
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (1, '2026-01-01Z', 12, 'Track', 1, "
            "'Race', ?, ?, ?, 'finished', 'RESULTED', 2, ?, 'v3', ?, ?)",
            [
                (1, "Active", 1, 5.0, 0.8, 1),
                (2, "Scratched", 0, 2.0, 0.9, 0),
            ],
        )

    frame = load_active_race(database, 1, ["form"])

    assert frame["runner_name"].tolist() == ["Active"]
    assert "is_winner" not in frame.columns


def test_live_loader_accepts_only_complete_priced_field(tmp_path):
    database = tmp_path / "races.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE race_runners (race_id INTEGER, start_time_iso TEXT, "
            "competition_id INTEGER, competition_name TEXT, race_number INTEGER, "
            "race_name TEXT, runner_number INTEGER, runner_name TEXT, "
            "runner_mask INTEGER, status TEXT, source_betting_status TEXT, "
            "active_field_size INTEGER, fluc2 REAL, "
            "derived_racing_features_version TEXT, form REAL)"
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (1, '2026-01-01Z', 12, 'Track', 1, "
            "'Race', ?, ?, 0, 'no_result', 'PRICED', 2, ?, 'v3', ?)",
            [(1, "A", 3.0, 0.8), (3, "C", 8.0, 0.4)],
        )

    frame = load_active_race(database, 1, ["form"])

    assert frame["runner_name"].tolist() == ["A", "C"]
