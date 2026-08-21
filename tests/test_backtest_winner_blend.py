import numpy as np
import pandas as pd
import pytest

from backtest_winner_blend import (
    candidate_form_weights,
    cohort_metrics,
    filter_competitions,
    load_prediction_cohort,
    parse_competition_ids,
    select_form_weight,
    validate_holdout_order,
    winner_ranks_for_weights,
)


def test_competition_ids_accept_one_or_multiple_values_without_duplicates():
    assert parse_competition_ids("6") == [6]
    assert parse_competition_ids("6, 10,6") == [6, 10]


def test_filter_competitions_keeps_requested_complete_races():
    frame = prediction_frame()
    frame["competition_id"] = [6, 6, 10, 10, 6, 6]

    filtered = filter_competitions(frame, [6], "Validation")

    assert filtered["race_id"].tolist() == [1, 1, 3, 3]
    assert filtered["competition_id"].tolist() == [6, 6, 6, 6]


def test_filter_competitions_requires_column_and_matching_rows():
    with pytest.raises(ValueError, match="missing competition_id"):
        filter_competitions(prediction_frame(), [6], "Validation")
    frame = prediction_frame().assign(competition_id=10)
    with pytest.raises(ValueError, match="no races"):
        filter_competitions(frame, [6], "Test")


def prediction_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "race_id": [1, 1, 2, 2, 3, 3],
        "runner_number": [1, 2, 1, 2, 1, 2],
        "is_winner": [1, 0, 0, 1, 1, 0],
        # Form gets races 1 and 3; market-aware gets race 2.
        "form_score": [1.0, 0.0, 0.0, 1.0, 0.8, 0.2],
        "market_aware_score": [0.0, 1.0, 0.9, 0.1, 0.0, 1.0],
        "market_score": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        "market_rank": [1, 2, 1, 2, 2, 1],
    })


def test_candidate_weight_grid_has_exact_endpoints():
    weights = candidate_form_weights(0.25, minimum_form_weight=0.25)

    assert weights.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_weight_search_uses_only_form_and_market_aware_scores():
    frame = prediction_frame()
    weights = candidate_form_weights(0.1, 0.0)

    selected, sweep = select_form_weight(
        frame, "form_score", "market_aware_score", weights, "top1"
    )

    assert selected > 0.5
    assert sweep.loc[sweep["form_weight"] == selected, "top1_hit_rate"].iloc[0] == 1.0
    assert "market_score" not in sweep.columns


def test_winner_rank_ties_follow_stable_runner_order():
    frame = pd.DataFrame({
        "race_id": [1, 1],
        "runner_number": [1, 2],
        "is_winner": [0, 1],
        "form_score": [0.5, 0.5],
        "market_aware_score": [0.5, 0.5],
    })

    ranks = winner_ranks_for_weights(
        frame, "form_score", "market_aware_score", np.asarray([0.0, 0.5, 1.0])
    )

    assert ranks[:, 0].tolist() == [2, 2, 2]


def test_cohort_metrics_reports_raw_market_only_as_benchmark():
    metrics, deviation = cohort_metrics(
        prediction_frame(), "form_score", "market_aware_score", 0.75
    )

    assert set(metrics) == {
        "form_only", "market_aware_only", "equal_blend", "tuned_blend",
        "raw_market_benchmark",
    }
    assert deviation is not None


def test_prediction_loader_rejects_races_without_one_winner(tmp_path):
    frame = prediction_frame()
    frame["is_winner"] = 0
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exactly one winner"):
        load_prediction_cohort(path)


def test_holdout_must_be_disjoint_and_strictly_later():
    validation = pd.DataFrame({
        "race_id": [1], "start_time_iso": ["2026-01-02T00:00:00Z"],
    })
    earlier_test = pd.DataFrame({
        "race_id": [2], "start_time_iso": ["2026-01-01T00:00:00Z"],
    })
    overlapping_test = pd.DataFrame({
        "race_id": [1], "start_time_iso": ["2026-01-03T00:00:00Z"],
    })

    with pytest.raises(ValueError, match="strictly after"):
        validate_holdout_order(validation, earlier_test)
    with pytest.raises(ValueError, match="overlap"):
        validate_holdout_order(validation, overlapping_test)
