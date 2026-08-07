"""Focused tests for the SQLite-to-CSV training handoff."""

import csv
import sqlite3

import numpy as np

from src.database import export_rows_to_csv, load_rows_from_csv


def test_exported_training_records_are_reloaded_from_csv(tmp_path):
    database_path = tmp_path / "races.sqlite"
    csv_path = tmp_path / "training.csv"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE rows ("
            "race_id INTEGER, start_time_iso TEXT, is_validation INTEGER, "
            "runner_number INTEGER, speed REAL, top3_mask INTEGER, fluc2 REAL)"
        )
        connection.executemany(
            "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (20, "2026-01-02T01:00:00Z", 0, 2, None, 0, None),
                (10, "2026-01-01T01:00:00Z", 0, 2, 1.5, 0, 4.2),
                (10, "2026-01-01T01:00:00Z", 0, 1, 2.5, 1, 3.1),
            ],
        )
        connection.execute("CREATE VIEW training_rows AS SELECT * FROM rows")
        connection.commit()
    finally:
        connection.close()

    assert export_rows_to_csv(
        database_path, ["speed"], "training_rows", csv_path
    ) == 3
    x, y, race_ids, times, validation_flags, market = load_rows_from_csv(
        csv_path, ["speed"]
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == [
            "race_id",
            "start_time_iso",
            "is_validation",
            "runner_number",
            "speed",
            "top3_mask",
            "market_fluc2_baseline",
        ]
    np.testing.assert_array_equal(race_ids, [10, 10, 20])
    np.testing.assert_array_equal(y, [1, 0, 0])
    np.testing.assert_array_equal(validation_flags, [0, 0, 0])
    np.testing.assert_allclose(x[:2, 0], [2.5, 1.5])
    assert np.isnan(x[2, 0])
    np.testing.assert_allclose(market[:2], [3.1, 4.2])
    assert np.isnan(market[2])
    assert times[0] < times[2]


def test_csv_loader_rejects_a_different_feature_schema(tmp_path):
    csv_path = tmp_path / "training.csv"
    csv_path.write_text(
        "race_id,start_time_iso,is_validation,runner_number,wrong,top3_mask,"
        "market_fluc2_baseline\n"
        "1,2026-01-01T00:00:00Z,0,1,2.0,1,3.0\n",
        encoding="utf-8",
    )

    try:
        load_rows_from_csv(csv_path, ["speed"])
    except ValueError as error:
        assert "CSV schema mismatch" in str(error)
    else:
        raise AssertionError("Expected the CSV schema mismatch to fail closed")


def test_market_baseline_header_is_unique_when_fluc2_is_a_feature(tmp_path):
    database_path = tmp_path / "races.sqlite"
    csv_path = tmp_path / "training.csv"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE rows ("
            "race_id INTEGER, start_time_iso TEXT, is_validation INTEGER, "
            "runner_number INTEGER, fluc2 REAL, top3_mask INTEGER)"
        )
        connection.execute(
            "INSERT INTO rows VALUES (1, '2026-01-01T00:00:00Z', 0, 1, 2.5, 1)"
        )
        connection.execute("CREATE VIEW training_rows AS SELECT * FROM rows")
        connection.commit()
    finally:
        connection.close()

    export_rows_to_csv(database_path, ["fluc2"], "training_rows", csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert len(header) == len(set(header))
    assert header[-2:] == ["top3_mask", "market_fluc2_baseline"]

    x, _, _, _, _, market = load_rows_from_csv(csv_path, ["fluc2"])
    np.testing.assert_allclose(x[:, 0], [2.5])
    np.testing.assert_allclose(market, [2.5])
