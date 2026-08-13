import sqlite3

from update_derived_racing_features import target_selection


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
            (1, "finished", "2026-08-13-v3"),
            (1, "finished", "2026-08-13-v3"),
            (2, "no_result", "2026-08-13-v3"),
            (2, "no_result", "2026-08-13-v3"),
            (3, "finished", "2026-08-13-v3"),
            (3, "no_result", "2026-08-13-v3"),
            (None, "no_result", "2026-08-13-v3"),
            (None, "finished", "2026-08-13-v3"),
            # One missing marker selects the entire finished race.
            (4, "finished", "2026-08-13-v3"),
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
