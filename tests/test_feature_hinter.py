import numpy as np
import pandas as pd
import pytest

from feature_hinter import (
    candidate_features,
    validate_competition_scope,
    direction_metrics,
    evaluate_features,
    exclusion_reason,
    filter_results_by_minimum_races,
    parse_competition_ids,
    ranked_indices,
    usable_races,
    winner_rank_one_payload,
)


def sample_races():
    return pd.DataFrame({
        "race_id": [1] * 5 + [2] * 5,
        "runner_number": [1, 2, 3, 4, 5] * 2,
        "runner_name": list("ABCDE") * 2,
        "top3_mask": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        "is_winner": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "higher": [5, 4, 3, 2, 1, 1, 5, 4, 3, 2],
        "lower": [1, 2, 3, 4, 5, 5, 1, 2, 3, 4],
        "constant": [1] * 10,
    })


def test_feature_direction_uses_top3_capture_and_aggregates_by_race():
    result = evaluate_features(sample_races(), ["higher", "lower", "constant"])
    indexed = result.set_index("feature")

    assert indexed.loc["higher", "direction"] == "DESC"
    assert indexed.loc["lower", "direction"] == "ASC"
    assert indexed.loc["higher", "total_top3_hits"] == 6
    assert indexed.loc["higher", "possible_top3_hits"] == 6
    assert indexed.loc["higher", "races_with_3_of_3"] == 2
    assert indexed.loc["higher", "winner_hit_rate"] == 1.0
    assert indexed.loc["higher", "mean_winner_rank"] == 1.0
    assert indexed.loc["higher", "winner_top3_rate"] == 1.0
    assert "constant" not in indexed.index


def test_nulls_are_always_ranked_last_in_both_directions():
    values = np.asarray([np.nan, 2.0, 1.0, np.inf])

    assert ranked_indices(values, "ASC").tolist() == [2, 1, 0, 3]
    assert ranked_indices(values, "DESC").tolist() == [1, 2, 0, 3]


def test_top3_direction_tie_defaults_to_ascending_not_winner_performance():
    frame = pd.DataFrame({
        "race_id": [1, 1, 1, 1], "runner_number": [1, 2, 3, 4],
        "runner_name": list("ABCD"), "top3_mask": [1, 1, 0, 1],
        "is_winner": [0, 0, 0, 1], "feature": [1, 2, 3, 4],
    })

    result = evaluate_features(frame, ["feature"]).iloc[0]

    assert result["direction"] == "ASC"
    assert result["winner_hit_rate"] == 0.0
    assert result["mean_winner_rank"] == 4.0


def test_schema_filter_excludes_leakage_ids_text_but_keeps_history():
    schema = [
        ("race_id", "INTEGER"), ("finish_place", "INTEGER"),
        ("selection_id", "INTEGER"), ("runner_name", "TEXT"),
        ("recent_1_place", "INTEGER"), ("recent_finish_percentile_avg_3", "REAL"),
        ("speed_rating", "REAL"),
    ]

    assert candidate_features(schema) == [
        "recent_1_place", "recent_finish_percentile_avg_3", "speed_rating"
    ]
    assert exclusion_reason("horse_id", "INTEGER") == "identifier column"


def test_invalid_target_race_is_rejected_without_affecting_valid_race():
    frame = sample_races()
    frame.loc[frame["race_id"] == 2, "top3_mask"] = 0

    usable, invalid = usable_races(frame)

    assert usable["race_id"].unique().tolist() == [1]
    assert invalid == [2]


def test_competition_ids_accept_comma_separated_values_without_duplicates():
    assert parse_competition_ids("580, 570,580") == [580, 570]


def test_competition_999_requires_and_accepts_explicit_entity_mode():
    with pytest.raises(ValueError, match="--allow-competition-999"):
        validate_competition_scope([999], False)
    assert validate_competition_scope([999], True) == (
        "competition_999_mode=derived_market_miss_entity"
    )
    assert validate_competition_scope([330], False) is None


def test_single_race_leaderboard_warns_that_direction_is_hindsight(capsys):
    results = evaluate_features(sample_races().loc[lambda x: x["race_id"] == 1], [
        "higher"
    ])

    from feature_hinter import print_leaderboard
    print_leaderboard(results, 5)

    output = capsys.readouterr().out
    assert "WARNING SINGLE-RACE HINDSIGHT" in output
    assert "Dir ASC: lowest feature values" in output
    assert "Top3 Hits" in output
    assert "Winner Rank" in output


def test_minimum_races_filter_can_remove_low_coverage_features():
    results = pd.DataFrame({
        "feature": ["sparse", "complete"],
        "races_tested": [1, 10],
    })

    filtered = filter_results_by_minimum_races(results, 10)

    assert filtered["feature"].tolist() == ["complete"]


def test_winner_rank_one_json_payload_contains_only_first_rank_features():
    frame = sample_races().loc[lambda x: x["race_id"] == 1].copy()
    results = evaluate_features(frame, ["higher", "lower"])
    # Force one feature to represent a non-winning first selection.
    results.loc[results["feature"] == "lower", "winner_rank_total"] = 2

    payload = winner_rank_one_payload(results, frame)["winner_rank_1_features"]

    assert payload["winner"] == {"runner_number": 1, "runner_name": "A"}
    assert payload["feature_names"] == ["higher"]
    assert payload["features"][0]["direction"] == "DESC"
