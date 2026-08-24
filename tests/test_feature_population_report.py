import csv
import json
import math
import sqlite3

from feature_population_report import analyze_population, write_csv


def _database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                race_id INTEGER,
                competition_id INTEGER,
                status TEXT,
                runner_mask INTEGER,
                full_feature REAL,
                partial_feature REAL,
                text_feature TEXT,
                top3_mask INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO race_runners VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (1, 6, "finished", 1, 1.0, 2.0, "yes", 1),
                (1, 6, "finished", 1, 0.0, None, "", 0),
                (2, 7, "no_result", 0, 3.0, float("inf"), None, None),
            ],
        )


def test_default_report_ranks_numeric_features_by_usable_population(tmp_path):
    database = tmp_path / "races.sqlite"
    _database(database)

    rows, unavailable = analyze_population(database)

    assert unavailable == []
    assert [row.feature for row in rows] == ["full_feature", "partial_feature"]
    assert rows[0].populated_rows == 3
    assert rows[0].population_pct == 100.0
    assert rows[1].populated_rows == 1
    assert math.isclose(rows[1].population_pct, 100 / 3)


def test_manifest_union_filters_rows_and_reports_unknown_features(tmp_path):
    database = tmp_path / "races.sqlite"
    manifest = tmp_path / "features.json"
    _database(database)
    manifest.write_text(
        json.dumps(
            {
                "base_features": ["full_feature"],
                "models": {
                    "a1": {
                        "features": [
                            "full_feature",
                            "partial_feature",
                            "missing_feature",
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rows, unavailable = analyze_population(
        database,
        manifest=manifest,
        competition_ids=[6],
        statuses=["finished"],
        active_only=True,
    )

    assert unavailable == ["missing_feature"]
    assert rows[0].feature == "full_feature"
    assert rows[0].total_rows == 2
    assert rows[0].population_pct == 100.0
    assert rows[1].feature == "partial_feature"
    assert rows[1].population_pct == 50.0


def test_all_columns_treats_blank_text_as_missing_and_writes_csv(tmp_path):
    database = tmp_path / "races.sqlite"
    output = tmp_path / "report.csv"
    _database(database)

    rows, _ = analyze_population(database, all_columns=True)
    text = next(row for row in rows if row.feature == "text_feature")
    assert text.populated_rows == 1

    write_csv(output, rows)
    with output.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["feature"] == rows[0].feature
    assert set(saved[0]) == {
        "feature",
        "population_pct",
        "populated_rows",
        "missing_rows",
        "total_rows",
        "declared_type",
    }
