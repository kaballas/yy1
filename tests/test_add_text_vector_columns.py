import sqlite3

from add_text_vector_columns import add_vector_columns, main, planned_vector_columns


def make_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE race_runners ("
            "race_id INTEGER, runner_name TEXT, notes VARCHAR(20), speed REAL, "
            "result_code TEXT, notes_vec BLOB)"
        )


def test_plan_includes_only_text_features_without_existing_vectors(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)

    additions, existing = planned_vector_columns(database)

    assert additions == [("runner_name", "runner_name_vec")]
    assert existing == ["notes_vec"]


def test_apply_adds_nullable_blob_columns_and_is_idempotent(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)

    additions, _ = add_vector_columns(database)
    second_additions, existing = add_vector_columns(database)

    assert additions == [("runner_name", "runner_name_vec")]
    assert second_additions == []
    assert set(existing) == {"runner_name_vec", "notes_vec"}
    with sqlite3.connect(database) as connection:
        schema = {
            row[1]: (row[2], row[3])
            for row in connection.execute("PRAGMA table_info(race_runners)")
        }
    assert schema["runner_name_vec"] == ("BLOB", 0)


def test_cli_is_dry_run_by_default(tmp_path, capsys):
    database = tmp_path / "races.sqlite"
    make_database(database)

    assert main(["--db", str(database)]) == 0

    assert "would_add=1" in capsys.readouterr().out
    additions, _ = planned_vector_columns(database)
    assert additions == [("runner_name", "runner_name_vec")]
