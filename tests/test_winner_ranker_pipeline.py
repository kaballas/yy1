import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from rank_winner_models import (
    completed_winner_model_results,
    consensus_representative_model_columns,
    load_active_race,
    load_original_race_models,
    model_rank_total_summary,
    model_prediction_tie_diagnostics,
    number_one_rank_summary,
    number_one_summary,
    ranked_output,
    select_historical_cohort,
    terminal_display_table,
    terminal_table_text,
)


def test_consensus_display_selects_nearest_models_and_honors_zero_limit():
    output = pd.DataFrame({
        "consensus_rank": [1, 2, 3],
        "race_1_rank": [1, 2, 3],
        "race_2_rank": [1, 3, 2],
        "race_3_rank": [3, 2, 1],
    })
    columns = ["race_3_rank", "race_2_rank", "race_1_rank"]

    assert consensus_representative_model_columns(output, columns, 2) == [
        "race_1_rank", "race_2_rank",
    ]
    assert consensus_representative_model_columns(output, columns, 0) == columns
    with pytest.raises(ValueError, match="zero or greater"):
        consensus_representative_model_columns(output, columns, -1)
from src.winner_ranker import (
    blend_scores,
    blend_named_scores,
    chronological_race_split,
    current_market_features,
    eligible_races,
    is_current_market_feature,
    market_deviation_metrics,
    model_feature_matrix,
    finishing_relevance,
    rank_percentiles,
    ranking_targets,
    select_blend_weights,
    select_form_features,
    validate_ranker_groups,
    winner_field_size_slices,
    winner_metrics,
    winner_race_report,
    xgb_ensemble_feature_importance,
)


def test_original_race_model_manifest_loads_saved_models_and_features(tmp_path):
    model = tmp_path / "race_10.json"
    second_model = tmp_path / "race_10_member_2.json"
    model.write_text("{}")
    second_model.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "models": [{
            "name": "race_10",
            "model": str(model),
            "models": [str(model), str(second_model)],
            "details": {"input_features": ["speed", "weight"]},
        }],
    }))

    features, paths = load_original_race_models(manifest)

    assert features == {"race_10": ["speed", "weight"]}
    assert paths == {"race_10": [
        str(model.resolve()), str(second_model.resolve()),
    ]}


def test_model_feature_matrix_uses_manifest_order_and_engineered_values():
    frame = pd.DataFrame({
        "race_id": [1, 1],
        "fluc2": [2.0, 4.0],
        "speed": [7.0, 8.0],
    })

    matrix = model_feature_matrix(
        frame, ["current_market_rank_pct", "speed", "fluc2"]
    )

    assert matrix.columns.tolist() == [
        "current_market_rank_pct", "speed", "fluc2",
    ]
    assert matrix["current_market_rank_pct"].tolist() == [1.0, 0.0]


def test_named_blend_includes_dynamic_model_groups():
    result = blend_named_scores(
        {
            "form": np.asarray([1.0, 0.0]),
            "fun": np.asarray([0.0, 1.0]),
            "market": np.asarray([0.5, 0.5]),
        },
        {"form": 0.25, "fun": 0.75, "market": 0.0},
    )

    np.testing.assert_allclose(result, [0.25, 0.75])


def test_finish_order_relevance_uses_rank_label_for_nonfinishers():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1, 1],
        "finish_place": [1, 2, 3, np.nan],
        "rank_label": ["1", "2", "3", "5"],
        "is_winner": [1, 0, 0, 0],
    })

    relevance = finishing_relevance(frame)

    np.testing.assert_allclose(relevance, [1.0, 2 / 3, 1 / 3, 0.0])


def test_margin_aware_target_matches_frozen_formula():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1],
        "finish_place": [1, 2, 3],
        "rank_label": ["1", "2", "3"],
        "is_winner": [1, 0, 0],
        "beaten_margin": [2.5, 5.0, 10.0],
    })

    target = ranking_targets(frame, "margin_aware_finish_order")
    expected_finish = np.asarray([1.0, 0.5, 0.0])
    expected_margin = np.exp(-np.asarray([0.0, 5.0, 10.0]) / 5.0)

    np.testing.assert_allclose(
        target, 0.75 * expected_finish + 0.25 * expected_margin
    )


def test_margin_aware_target_rejects_missing_current_race_margin():
    frame = pd.DataFrame({
        "race_id": [1, 1],
        "finish_place": [1, 2],
        "rank_label": ["1", "2"],
        "is_winner": [1, 0],
    })

    with pytest.raises(ValueError, match="current-race beaten-margin"):
        ranking_targets(frame, "margin_aware_finish_order")


def test_historical_cohort_broadens_when_exact_race_number_has_under_minimum():
    predictions = pd.DataFrame({
        "race_id": list(range(1, 32)),
        "competition_id": [256] * 31,
        "race_number": [7] * 3 + [
            number for number in range(1, 30) if number != 7
        ][:28],
    })

    cohort, scope, exact_races = select_historical_cohort(
        predictions, competition_id=256, race_number=7
    )

    assert scope == "competition_id"
    assert exact_races == 3
    assert cohort["race_id"].nunique() == 31


def test_historical_cohort_keeps_exact_race_number_at_minimum():
    predictions = pd.DataFrame({
        "race_id": list(range(1, 32)),
        "competition_id": [279] * 31,
        "race_number": [9] * 30 + [1],
    })

    cohort, scope, exact_races = select_historical_cohort(
        predictions, competition_id=279, race_number=9
    )

    assert scope == "competition_id+race_number"
    assert exact_races == 30
    assert cohort["race_id"].nunique() == 30


def test_historical_cohort_uses_competition_when_exact_is_absent():
    predictions = pd.DataFrame({
        "race_id": list(range(1, 33)),
        "competition_id": [4] * 30 + [9, 10],
        "race_number": [
            number for number in range(1, 32) if number != 4
        ][:30] + [4, 4],
    })

    cohort, scope, exact_races = select_historical_cohort(
        predictions, competition_id=4, race_number=4
    )

    assert scope == "competition_id"
    assert exact_races == 0
    assert cohort["race_id"].tolist() == list(range(1, 31))


def test_historical_cohort_rejects_too_small_competition_fallback():
    predictions = pd.DataFrame({
        "race_id": [1, 2, 3],
        "competition_id": [4, 9, 10],
        "race_number": [1, 8, 8],
    })

    with pytest.raises(ValueError, match="found 1 races; minimum=30"):
        select_historical_cohort(
            predictions, competition_id=4, race_number=8
        )


def test_terminal_display_removes_rank_suffix_without_mutating_output():
    output = pd.DataFrame({
        "display_rank": [1],
        "runner_number": [7],
        "tuned_rank": [1],
        "market_rank": [2],
    })

    table = terminal_display_table(output, list(output.columns))

    assert table.columns.tolist() == ["display", "runner_number", "tuned", "market"]
    assert output.columns.tolist() == [
        "display_rank", "runner_number", "tuned_rank", "market_rank",
    ]


def test_terminal_table_colors_only_rank_ones():
    output = pd.DataFrame({
        "display_rank": [1, 2],
        "runner_number": [1, 7],
        "tuned_rank": [2, 1],
        "market_rank": [1, 2],
        "contrarian_top3": [1, 0],
    })

    rendered = terminal_table_text(output, list(output.columns), color=True)

    assert rendered.count("\033[31m1\033[0m") == 3
    assert "runner_number" in rendered
    assert "contrarian_top3" in rendered


def test_number_one_summary_counts_visible_rankings_but_not_display_order():
    output = pd.DataFrame({
        "display_rank": [1, 2, 3],
        "runner_number": [4, 8, 1],
        "runner_name": ["A", "B", "C"],
        "fluc2": [3.0, 5.0, 7.0],
        "tuned_rank": [1, 2, 3],
        "form_rank": [2, 1, 3],
        "market_rank": [1, 3, 2],
    })

    summary = number_one_summary(output, list(output.columns))

    assert summary["runner_number"].tolist() == [4, 8]
    assert summary["number_ones"].tolist() == [2, 1]
    assert summary["picked_first_by"].tolist() == ["tuned,market", "form"]


def test_ranked_output_does_not_award_number_one_for_constant_model_scores():
    frame = pd.DataFrame({
        "runner_number": [1, 2, 3],
        "runner_name": ["A", "B", "C"],
        "fluc2": [3.0, 4.0, 5.0],
    })
    output = ranked_output(
        frame,
        {
            "consensus": np.array([0.5, 0.5, 0.5]),
            "deployment": np.array([0.5, 0.5, 0.5]),
            "market": np.array([3, 2, 1]),
        },
        "consensus",
    )

    assert output["consensus_rank"].tolist() == [2.0, 2.0, 2.0]
    summary = number_one_summary(
        output, ["consensus_rank", "market_rank"]
    )
    assert summary["picked_first_by"].tolist() == ["market"]


def test_model_rank_totals_sum_only_dynamic_model_columns():
    output = pd.DataFrame({
        "runner_number": [1, 2, 3],
        "runner_name": ["A", "B", "C"],
        "fluc2": [3.0, 5.0, 7.0],
        "tuned_rank": [1, 2, 3],
        "x1_rank": [3, 1, 2],
        "x2_rank": [2, 1, 3],
        "market_rank": [1, 2, 3],
    })

    summary = model_rank_total_summary(output, ["x1_rank", "x2_rank"])

    assert summary["runner_number"].tolist() == [2, 1, 3]
    assert summary["model_rank_total"].tolist() == [2, 5, 5]
    assert summary["models_counted"].tolist() == [2, 2, 2]
    assert summary["number_ones"].tolist() == [2, 0, 0]


def test_number_one_ranking_sorts_by_votes_then_average_rank():
    rank_totals = pd.DataFrame({
        "consensus_rank": [1, 2, 3],
        "runner_number": [1, 2, 3],
        "runner_name": ["A", "B", "C"],
        "fluc2": [3.0, 5.0, 7.0],
        "models_counted": [10, 10, 10],
        "average_model_rank": [1.8, 2.1, 2.0],
        "number_ones": [3, 4, 4],
    })

    summary = number_one_rank_summary(rank_totals)

    assert summary["runner_number"].tolist() == [3, 2, 1]
    assert summary["number_one_rank"].tolist() == [1, 2, 3]
    assert summary["number_one_pct"].tolist() == [40.0, 40.0, 30.0]
    assert summary["models_with_unique_top"].tolist() == [11, 11, 11]
    assert summary["number_one_vote_share_pct"].tolist() == pytest.approx([
        100 * 4 / 11, 100 * 4 / 11, 100 * 3 / 11,
    ])


def test_model_prediction_tie_diagnostics_counts_abstaining_models():
    diagnostics = model_prediction_tie_diagnostics(
        {
            "constant": np.array([0.5, 0.5, 0.5]),
            "tied": np.array([0.8, 0.8, 0.2]),
            "unique": np.array([0.9, 0.5, 0.1]),
        },
        ["constant", "tied", "unique"],
    )
    assert diagnostics == {
        "models_constant": 1,
        "models_unique_top": 1,
        "models_tied_top": 2,
    }


def test_completed_race_lists_models_that_ranked_actual_winner_first():
    frame = pd.DataFrame({
        "runner_number": [4, 8], "runner_name": ["Winner", "Other"],
        "status": ["finished", "finished"], "is_winner": [1, 0],
    })
    output = pd.DataFrame({
        "runner_number": [4, 8], "x1_rank": [1, 2], "x2_rank": [2, 1],
    })

    winner, results = completed_winner_model_results(
        frame.loc[frame["is_winner"] == 1].iloc[0],
        output,
        {"x1": ["speed", "form"], "x2": ["market"]},
    )

    assert winner["runner_name"] == "Winner"
    assert results.set_index("model").loc["x1", "winner_correct"]
    assert not results.set_index("model").loc["x2", "winner_correct"]
    assert results.set_index("model").loc["x1", "features"] == '["speed", "form"]'


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
    assert is_current_market_feature("finish_rank_minus_market_rank")
    assert is_current_market_feature("finish_market_rank_abs_gap")
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


def test_ranker_group_validation_reports_learnable_pairs():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1, 2, 2],
        "is_winner": [1, 0, 0, 0, 1],
    })

    audit = validate_ranker_groups(
        frame, frame["is_winner"].to_numpy(), np.asarray([3, 2])
    )

    assert audit == {
        "rows": 5,
        "races": 2,
        "minimum_runners": 2,
        "median_runners": 2.5,
        "maximum_runners": 3,
        "singleton_races": 0,
        "winner_loser_pairs": 3,
    }


def test_ranker_group_validation_rejects_noncontiguous_races():
    frame = pd.DataFrame({
        "race_id": [1, 2, 1, 2],
        "is_winner": [1, 1, 0, 0],
    })

    with pytest.raises(ValueError, match="contiguous"):
        validate_ranker_groups(frame)


def test_ranker_group_validation_rejects_wrong_group_sizes():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2],
        "is_winner": [1, 0, 0, 1],
    })

    with pytest.raises(ValueError, match="do not match"):
        validate_ranker_groups(frame, groups=np.asarray([1, 3]))


def test_winner_race_report_and_field_slices_include_random_baseline():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, 2],
        "is_winner": [1, 0, 0, 1, 0],
    })
    scores = np.asarray([0.9, 0.1, 0.8, 0.7, 0.1])

    report = winner_race_report(frame, frame["is_winner"].to_numpy(), scores)
    slices = winner_field_size_slices(report)

    assert report["winner_rank"].tolist() == [1, 2]
    assert report["random_top1_expected"].tolist() == pytest.approx([0.5, 1 / 3])
    assert slices.loc[0, "top1_hit_rate"] == pytest.approx(0.5)
    assert slices.loc[0, "random_top1_expected"] == pytest.approx(5 / 12)


def test_xgb_feature_importance_exposes_multiple_importance_types():
    class Booster:
        def get_score(self, importance_type):
            return {"speed": 2.0} if importance_type != "cover" else {"weight": 3.0}

    class Model:
        def get_booster(self):
            return Booster()

    result = xgb_ensemble_feature_importance([Model()], "form")

    assert set(result["importance_type"]) == {"gain", "cover", "weight", "total_gain"}
    assert set(result["model"]) == {"form"}


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


def test_rank_percentiles_and_metrics_do_not_favor_first_row_on_ties():
    scores = np.asarray([0.5, 0.5, 0.5])
    ids = np.asarray([1, 1, 1])
    target = np.asarray([1, 0, 0])

    percentile = rank_percentiles(scores, ids)
    metrics = winner_metrics(target, percentile, ids)

    assert percentile.tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert metrics["top1_hit_rate"] == 0.0
    assert metrics["mean_winner_rank"] == pytest.approx(2.0)


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


def test_ranked_output_keeps_dynamic_model_scores_visible():
    frame = pd.DataFrame({
        "runner_number": [1, 2],
        "runner_name": ["A", "B"],
        "fluc2": [2.0, 4.0],
    })
    form = np.asarray([0.8, 0.2])
    fun = np.asarray([0.1, 0.9])

    output = ranked_output(
        frame,
        {
            "form": form,
            "fun": fun,
            "deployment": form,
            "market": np.asarray([1.0, 0.0]),
        },
        "fun",
    )

    assert {"fun_score", "fun_rank", "form_score", "form_rank"} <= set(output)
    assert output.iloc[0]["runner_number"] == 2


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
