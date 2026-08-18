import sqlite3

from build_market_mover_manifest import (
    BASE_FEATURES,
    build_manifest,
    parse_feature_list,
)


def test_builds_market_base_plus_one_numeric_feature(tmp_path):
    database = tmp_path / "races.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                race_id INTEGER,
                runner_name TEXT,
                open_price REAL,
                fluc1 REAL,
                fluc2 REAL,
                speed_rating REAL,
                weight_kg REAL,
                top3_mask INTEGER,
                is_winner INTEGER
            )
            """
        )

    manifest = build_manifest(database, BASE_FEATURES[:3])

    assert list(manifest["models"]) == ["t1", "t2"]
    assert manifest["models"]["t1"]["features"] == [
        *BASE_FEATURES[:3],
        "speed_rating",
    ]
    assert manifest["models"]["t2"]["features"] == [
        *BASE_FEATURES[:3],
        "weight_kg",
    ]
    tested_features = {
        config["features"][-1] for config in manifest["models"].values()
    }
    assert tested_features.isdisjoint(BASE_FEATURES[:3])


def test_excludes_requested_test_features(tmp_path):
    database = tmp_path / "races.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                open_price REAL, fluc1 REAL, fluc2 REAL,
                speed_rating REAL, weight_kg REAL
            )
            """
        )

    manifest = build_manifest(
        database,
        ["open_price", "fluc1", "fluc2"],
        ["speed_rating"],
    )

    assert manifest["excluded_features"] == ["speed_rating"]
    assert list(manifest["models"]) == ["t1"]
    assert manifest["models"]["t1"]["features"][-1] == "weight_kg"


def test_parses_excluded_feature_list():
    assert parse_feature_list("speed_rating, weight_kg,speed_rating") == [
        "speed_rating",
        "weight_kg",
    ]
