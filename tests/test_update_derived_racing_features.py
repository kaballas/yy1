import sqlite3

from update_derived_racing_features import target_selection


def selected_rowids(connection: sqlite3.Connection, force: bool) -> tuple[list[int], str]:
    target_where, selection_mode = target_selection(force)
    rows = connection.execute(
        f'SELECT rowid FROM "race_runners" WHERE {target_where} ORDER BY rowid'
    )
    return [int(row[0]) for row in rows], selection_mode


def test_default_selects_whole_unfinished_races_and_force_selects_every_row():
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE "race_runners" ("race_id" INTEGER, "status" TEXT)')
    connection.executemany(
        'INSERT INTO "race_runners" VALUES (?, ?)',
        [
            (1, "finished"),
            (1, "finished"),
            (2, "no_result"),
            (2, "no_result"),
            (3, "finished"),
            (3, "no_result"),
            (None, "no_result"),
            (None, "finished"),
        ],
    )

    default_rows, default_mode = selected_rowids(connection, force=False)
    forced_rows, forced_mode = selected_rowids(connection, force=True)

    assert default_mode == "unfinished"
    assert default_rows == [3, 4, 5, 6, 7]
    assert forced_mode == "force"
    assert forced_rows == list(range(1, 9))
