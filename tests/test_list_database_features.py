import json
import sqlite3

import pytest

from list_database_features import inspect_columns, main, sqlite_affinity


def make_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE race_runners ("
            "race_id INTEGER, speed REAL, rating NUMERIC, runner_name TEXT, "
            "notes VARCHAR(20), raw BLOB, mystery)"
        )


def test_affinity_uses_sqlite_rules():
    assert sqlite_affinity("BIGINT") == "INTEGER"
    assert sqlite_affinity("VARCHAR(20)") == "TEXT"
    assert sqlite_affinity("DOUBLE") == "REAL"
    assert sqlite_affinity("BOOLEAN") == "NUMERIC"
    assert sqlite_affinity("") == "BLOB"


def test_inspect_columns_classifies_schema_and_roles(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)

    columns = {column.name: column for column in inspect_columns(database)}

    assert columns["speed"].category == "numeric"
    assert columns["runner_name"].category == "text"
    assert columns["notes"].category == "text"
    assert columns["raw"].category == "other"
    assert not columns["race_id"].is_feature
    assert columns["speed"].is_feature


def test_json_features_only_omits_known_non_features(tmp_path, capsys):
    database = tmp_path / "races.sqlite"
    make_database(database)

    assert main(["--db", str(database), "--features-only", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    names = [column["name"] for column in payload["columns"]]
    assert "race_id" not in names
    assert "speed" in names


def test_missing_table_is_clear(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)

    with pytest.raises(ValueError, match="does not exist"):
        inspect_columns(database, "missing")
