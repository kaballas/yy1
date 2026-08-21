import sqlite3

import numpy as np
import pandas as pd

from update_derived_racing_features import (
    CALCULATION_VERSION,
    add_race_aggregate_features,
    add_preparation_features,
    add_market_disagreement_features,
    target_selection,
)


def selected_rowids(connection: sqlite3.Connection, force: bool) -> tuple[list[int], str]:
    target_where, selection_mode = target_selection(force)
    rows = connection.execute(
        f'SELECT rowid FROM "race_runners" WHERE {target_where} ORDER BY rowid'
    )
    return [int(row[0]) for row in rows], selection_mode


def test_default_selects_whole_pending_races_and_force_selects_every_row():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        'CREATE TABLE "race_runners" ("race_id" INTEGER, "status" TEXT, '
        '"derived_racing_features_version" TEXT)'
    )
    connection.executemany(
        'INSERT INTO "race_runners" VALUES (?, ?, ?)',
        [
            (1, "finished", CALCULATION_VERSION),
            (1, "finished", CALCULATION_VERSION),
            (2, "no_result", CALCULATION_VERSION),
            (2, "no_result", CALCULATION_VERSION),
            (3, "finished", CALCULATION_VERSION),
            (3, "no_result", CALCULATION_VERSION),
            (None, "no_result", CALCULATION_VERSION),
            (None, "finished", CALCULATION_VERSION),
            # One missing marker selects the entire finished race.
            (4, "finished", CALCULATION_VERSION),
            (4, "finished", None),
            # An old marker also selects the entire race.
            (5, "finished", "old"),
            (5, "finished", "old"),
        ],
    )

    default_rows, default_mode = selected_rowids(connection, force=False)
    forced_rows, forced_mode = selected_rowids(connection, force=True)

    assert default_mode == "pending"
    assert default_rows == [3, 4, 5, 6, 7, 9, 10, 11, 12]
    assert forced_mode == "force"
    assert forced_rows == list(range(1, 13))


def test_market_disagreement_features_are_signed_and_absolute_rank_gaps():
    frame = pd.DataFrame({
        "fluc2_price_rank": [5, 1, np.nan],
        "recent_similar_distance_speed_rank": [1, 3, 2],
        "recent_weighted_avg_margin_rank": [7, 1, 4],
    })

    result = add_market_disagreement_features(frame)

    assert result["distance_speed_rank_minus_market_rank"].tolist()[:2] == [-4, 2]
    assert result["distance_speed_market_rank_abs_gap"].tolist()[:2] == [4, 2]
    assert result["margin_rank_minus_market_rank"].tolist()[:2] == [2, 0]
    assert np.isnan(result.loc[2, "margin_market_rank_abs_gap"])


def test_total_prize_money_is_summed_per_race_without_combining_unknown_races():
    frame = pd.DataFrame({
        "race_id": [1, 1, 2, 2, None, None],
        "prize_money": [100, "250", None, None, 40, 60],
        "speed_rating": [70, 80, None, None, 60, 65],
        "weight_kg": [55, 57, 54, 56, 58, 59],
        "field_size": [3, 3, 2, 2, 1, 1],
        "active_field_size": [2, 2, 2, 2, 1, 1],
    })

    result = add_race_aggregate_features(frame)

    assert result["total_prize_money"].iloc[:2].tolist() == [350, 350]
    assert result["total_prize_money"].iloc[2:4].isna().all()
    assert result["total_prize_money"].iloc[4:].tolist() == [40, 60]
    assert result["race_field_avg_speed_rating"].iloc[:2].tolist() == [75, 75]
    assert result["weight_vs_field_mean"].iloc[:2].tolist() == [-1, 1]
    assert result["num_scratchings"].iloc[:2].tolist() == [1, 1]


def test_preparation_features_use_only_dates_before_the_current_race():
    frame = pd.DataFrame({
        "start_time_iso": ["2026-08-20"] * 4,
        "recent_1_date": ["2026-03-01", "2026-08-01", "2026-08-01", None],
        "recent_2_date": ["2026-02-15", "2026-04-01", "2026-07-15", None],
        "recent_3_date": [None, None, "2026-07-01", None],
        "recent_4_date": [None] * 4,
        "recent_5_date": [None] * 4,
        "recent_6_date": [None] * 4,
    })

    result = add_preparation_features(frame)

    assert result["preparation_run_number"].iloc[:3].tolist() == [1, 2, 4]
    assert result["runs_this_preparation_before_race"].iloc[:3].tolist() == [0, 1, 3]
    assert result["first_up_flag"].iloc[:3].tolist() == [1, 0, 0]
    assert result["second_up_flag"].iloc[:3].tolist() == [0, 1, 0]
    assert result["third_up_flag"].iloc[:3].tolist() == [0, 0, 0]
    assert result.loc[3, "preparation_run_number"] != result.loc[3, "preparation_run_number"]


def test_missing_version_column_makes_every_race_pending_for_dry_run_migration():
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE "race_runners" ("race_id" INTEGER, "status" TEXT)')
    connection.executemany(
        'INSERT INTO "race_runners" VALUES (?, ?)',
        [(1, "finished"), (1, "finished"), (2, "no_result")],
    )
    where, mode = target_selection(False, version_column_available=False)
    rows = connection.execute(
        f'SELECT rowid FROM "race_runners" WHERE {where} ORDER BY rowid'
    ).fetchall()

    assert mode == "pending"
    assert [row[0] for row in rows] == [1, 2, 3]
