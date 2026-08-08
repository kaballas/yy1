import json
import sqlite3

from audit_production_features import audit_features, write_completed_manifest


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_all_other_numeric_features_are_in_zero_bucket(tmp_path):
    database = tmp_path / "races.sqlite"
    manifest = tmp_path / "features.json"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE race_runners (
                race_id INTEGER,
                active_a REAL,
                explicitly_zeroed REAL,
                newly_discovered INTEGER,
                top3_mask INTEGER,
                runner_name TEXT
            )
            """
        )
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "features": ["active_a", "explicitly_zeroed"],
            "zeroed_features": ["explicitly_zeroed"],
        },
    )

    result = audit_features(manifest, database)

    assert result.active_features == ("active_a",)
    assert result.completed_features == (
        "active_a",
        "explicitly_zeroed",
        "newly_discovered",
    )
    assert result.features_to_add == ("newly_discovered",)
    assert result.zeroed_features == ("explicitly_zeroed", "newly_discovered")
    assert not result.has_configuration_errors


def test_write_completed_manifest_adds_zero_bucket_to_model_inputs(tmp_path):
    database = tmp_path / "races.sqlite"
    manifest = tmp_path / "features.json"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE race_runners (active_a REAL, other_a REAL, other_b INTEGER)"
        )
    _write_json(manifest, {"features": ["active_a"], "zeroed_features": []})

    result = audit_features(manifest, database)
    write_completed_manifest(manifest, result)
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["features"] == ["active_a", "other_a", "other_b"]
    assert payload["zeroed_features"] == ["other_a", "other_b"]
