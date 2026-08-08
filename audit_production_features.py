#!/usr/bin/env python3
"""Audit and optionally complete the TabFM feature/zero-feature manifest."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURES = ROOT / "tabfm_features.json"
DEFAULT_DB = "/home/theo/yy1/db/race_runners.sqlite"


# These columns are deliberately not feature candidates. Keep the reason visible
# in output so a newly ignored column cannot disappear silently.
IGNORED_COLUMNS = {
    "feature_schema_version": "schema metadata",
    "race_id": "race identifier",
    "race_number": "eligibility/meeting metadata",
    "race_name": "race metadata text",
    "competition_id": "meeting identifier",
    "competition_name": "meeting metadata text",
    "country": "race metadata text",
    "class_name": "race metadata text",
    "grade": "race metadata text",
    "tempo": "race metadata text",
    "track_status": "race metadata text",
    "start_time_iso": "race timestamp",
    "winner_index": "outcome-derived target metadata",
    "is_trainable": "dataset control flag",
    "source_betting_status": "dataset status metadata",
    "selection_id": "runner identifier",
    "runner_number": "runner identifier",
    "runner_name": "runner identity text",
    "runner_country": "runner metadata text",
    "jockey": "identity text; categorical preprocessing not implemented",
    "trainer": "identity text; categorical preprocessing not implemented",
    "trainer_location": "identity text; categorical preprocessing not implemented",
    "sex": "categorical text",
    "colour": "categorical text",
    "sire": "identity text; categorical preprocessing not implemented",
    "dam": "identity text; categorical preprocessing not implemented",
    "blinkers": "categorical text",
    "last_six": "raw form text",
    "form_fig": "raw form text",
    "expected_settling_position": "raw categorical text",
    "finish_place": "post-race outcome",
    "result_code": "post-race outcome/status",
    "status": "post-race status",
    "sp_starting_price": "post-race starting-price field",
    "runner_mask": "training mask/target metadata",
    "rank_label": "target label text",
    "top3_mask": "training target",
    "is_winner": "training target",
    "awards": "raw JSON/text",
    "is_validation": "legacy partition flag",
}


@dataclass(frozen=True)
class ColumnInfo:
    position: int
    name: str
    declared_type: str

    @property
    def is_numeric(self) -> bool:
        declared = self.declared_type.upper()
        return any(
            token in declared
            for token in ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")
        )


@dataclass(frozen=True)
class AuditResult:
    database_columns: tuple[str, ...]
    numeric_candidate_columns: tuple[str, ...]
    master_features: tuple[str, ...]
    active_features: tuple[str, ...]
    completed_features: tuple[str, ...]
    zeroed_features: tuple[str, ...]
    features_to_add: tuple[str, ...]
    master_features_missing_database: tuple[str, ...]

    @property
    def has_configuration_errors(self) -> bool:
        return bool(self.master_features_missing_database)


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return payload


def load_master_features(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = _load_json_object(path, "Feature manifest")
    features = payload.get("features")
    if (
        not isinstance(features, list)
        or any(not isinstance(feature, str) or not feature for feature in features)
        or len(features) != len(set(features))
    ):
        raise ValueError("Feature manifest must have unique string features")
    zeroed = payload.get("zeroed_features", [])
    if (
        not isinstance(zeroed, list)
        or any(not isinstance(feature, str) or not feature for feature in zeroed)
        or len(zeroed) != len(set(zeroed))
    ):
        raise ValueError("Feature manifest must have unique string zeroed_features")
    missing = sorted(set(zeroed) - set(features))
    if missing:
        raise ValueError(
            "zeroed_features must also be present in features: " + ", ".join(missing)
        )
    return tuple(features), tuple(zeroed)


def load_database_columns(db_path: Path, table: str) -> tuple[ColumnInfo, ...]:
    if not table or not table.replace("_", "").isalnum():
        raise ValueError(f"Unsafe table name: {table!r}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    if not rows:
        raise ValueError(f"Table {table!r} does not exist or has no columns")
    return tuple(
        ColumnInfo(position=int(row[0]), name=str(row[1]), declared_type=str(row[2]))
        for row in rows
    )


def audit_features(
    features_path: Path,
    db_path: Path,
    table: str = "race_runners",
) -> AuditResult:
    master_features, configured_zeroed = load_master_features(features_path)
    columns = load_database_columns(db_path, table)

    database_names = {column.name for column in columns}
    master = set(master_features)
    configured_zeroed_set = set(configured_zeroed)
    active_features = tuple(
        feature for feature in master_features if feature not in configured_zeroed_set
    )
    active = set(active_features)
    numeric_candidates = tuple(
        column.name
        for column in columns
        if column.is_numeric and column.name not in IGNORED_COLUMNS
    )
    features_to_add = tuple(
        feature for feature in numeric_candidates if feature not in master
    )
    completed_features = master_features + features_to_add
    # Anything not explicitly active is retained as a model input but neutralized.
    # Preserve manifest/database order so regenerated manifests are deterministic.
    zeroed_features = tuple(
        feature for feature in completed_features if feature not in active
    )

    return AuditResult(
        database_columns=tuple(column.name for column in columns),
        numeric_candidate_columns=numeric_candidates,
        master_features=master_features,
        active_features=active_features,
        completed_features=completed_features,
        zeroed_features=zeroed_features,
        features_to_add=features_to_add,
        master_features_missing_database=tuple(sorted(master - database_names)),
    )


def write_completed_manifest(path: Path, result: AuditResult) -> None:
    """Write the complete input list and computed zero bucket atomically."""
    payload = _load_json_object(path, "Feature manifest")
    payload["features"] = list(result.completed_features)
    payload["zeroed_features"] = list(result.zeroed_features)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _print_section(title: str, values: tuple[str, ...]) -> None:
    print(f"\n{title} ({len(values)})")
    if not values:
        print("  none")
        return
    for value in values:
        print(f"  {value}")


def print_report(result: AuditResult, show_ignored: bool = False) -> None:
    print(
        f"active_features={len(result.active_features)} "
        f"zeroed_features={len(result.zeroed_features)} "
        f"master_features={len(result.master_features)} "
        f"database_columns={len(result.database_columns)} "
        f"numeric_candidates={len(result.numeric_candidate_columns)}"
    )
    _print_section(
        "ACTIVE (NON-ZEROED) FEATURES",
        result.active_features,
    )
    _print_section(
        "ZERO BUCKET (ALL OTHER NUMERIC FEATURES)",
        result.zeroed_features,
    )
    _print_section(
        "DATABASE FEATURES TO ADD TO MANIFEST",
        result.features_to_add,
    )
    _print_section(
        "MASTER FEATURES MISSING FROM DATABASE",
        result.master_features_missing_database,
    )
    if show_ignored:
        print(f"\nINTENTIONALLY IGNORED COLUMNS ({len(IGNORED_COLUMNS)})")
        for name, reason in sorted(IGNORED_COLUMNS.items()):
            print(f"  {name}: {reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep configured active TabFM features and place every other eligible "
            "numeric race_runners column in the zeroed_features bucket."
        )
    )
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--table", default="race_runners")
    parser.add_argument("--show-ignored", action="store_true")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the completed features and zeroed_features lists to the manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_features(
        args.features_json, args.db, args.table
    )
    print_report(result, args.show_ignored)
    if result.has_configuration_errors:
        return 1
    if args.write:
        write_completed_manifest(args.features_json, result)
        print(f"\nwrote_manifest={args.features_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
