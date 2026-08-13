#!/usr/bin/env python3
"""Add/update leakage-safe derived racing features in race_runners."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from src.advanced_racing_features import (
    ADVANCED_FEATURE_NAMES,
    derive_entity_history_features,
    derive_sectional_class_features,
)
from src.derived_racing_features import derive_racing_features


FEATURES_TO_STORE = (
    "form_barrier_percentile_weighted_6",
    *ADVANCED_FEATURE_NAMES,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transactionally calculate derived pre-race feature columns."
    )
    parser.add_argument("--db", type=Path, default=Path("db/race_runners.sqlite"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = args.db.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    input_columns = list(dict.fromkeys([
        "rowid", "distance_m", "race_name", "grade", "start_time_iso",
        "status", "runner_mask", "top3_mask", "active_field_size", "field_size",
        "jockey", "trainer",
        *(f"recent_{run}_{stem}" for run in range(1, 7) for stem in (
            "place", "margin", "total_runners", "barrier", "starting_price",
            "distance_m", "last600", "time", "class",
        )),
    ]))
    with sqlite3.connect(database) as connection:
        existing = {
            str(row[1]) for row in connection.execute(
                'PRAGMA table_info("race_runners")'
            )
        }
        missing_inputs = sorted(set(input_columns[1:]) - existing)
        if missing_inputs:
            raise ValueError("Database is missing inputs: " + ", ".join(missing_inputs))
        quoted = ", ".join(f'"{name}"' for name in input_columns)
        frame = pd.read_sql_query(
            f'SELECT {quoted} FROM "race_runners" ORDER BY rowid', connection
        )
        derived = pd.concat([
            derive_racing_features(frame),
            derive_sectional_class_features(frame),
            derive_entity_history_features(frame),
        ], axis=1)
        print(
            f"database={database} rows={len(frame):,} "
            + " ".join(
                f"{name}_coverage={derived[name].notna().mean():.4f}"
                for name in FEATURES_TO_STORE
            )
        )
        if args.dry_run:
            print("dry_run=yes database_modified=no")
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            changes_before = connection.total_changes
            for name in FEATURES_TO_STORE:
                if name not in existing:
                    connection.execute(
                        f'ALTER TABLE "race_runners" ADD COLUMN "{name}" REAL'
                    )
                values = [
                    (
                        None if pd.isna(value) else float(value),
                        int(rowid),
                    )
                    for value, rowid in zip(derived[name], frame["rowid"])
                ]
                connection.executemany(
                    f'UPDATE "race_runners" SET "{name}" = ? '
                    f'WHERE rowid = ? AND "{name}" IS NOT ?',
                    [(value, rowid, value) for value, rowid in values],
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        checks = []
        for name in FEATURES_TO_STORE:
            count, minimum, maximum = connection.execute(
                f'SELECT COUNT("{name}"), MIN("{name}"), MAX("{name}") '
                'FROM "race_runners"'
            ).fetchone()
            checks.append(
                f"{name}:non_null={int(count):,},min={minimum},max={maximum}"
            )
        changed_rows = connection.total_changes - changes_before
        print(
            f"database_modified={'yes' if changed_rows else 'no'} "
            f"changed_rows={changed_rows:,} "
            + " ".join(checks)
        )


if __name__ == "__main__":
    main()
