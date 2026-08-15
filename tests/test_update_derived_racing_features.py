import sqlite3

import numpy as np
import pandas as pd

from update_derived_racing_features import (
    CALCULATION_VERSION,
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
