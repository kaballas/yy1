#!/usr/bin/env python3
"""Add/update leakage-safe derived racing features in race_runners."""

from __future__ import annotations

import argparse
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

from src.advanced_racing_features import (
    ADVANCED_FEATURE_NAMES,
    derive_context_features,
    derive_entity_history_features,
    derive_sectional_class_features,
)
from src.derived_racing_features import DERIVED_FEATURE_NAMES, derive_racing_features


FEATURES_TO_STORE = (
    *DERIVED_FEATURE_NAMES,
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
        "rowid", "race_id", "competition_id", "competition_name", "distance_m",
        "race_name", "grade", "track_status", "start_time_iso", "draw_number", "weight_kg",
        "status", "runner_mask", "top3_mask", "active_field_size", "field_size",
        "jockey", "trainer", "finish_place", "career_starts", "career_wins",
        "career_seconds", "career_thirds", "place_percentage",
        *(f"recent_{run}_{stem}" for run in range(1, 7) for stem in (
            "place", "margin", "total_runners", "barrier", "starting_price",
            "distance_m", "last600", "time", "class", "weight_kg",
            "track_name", "track_status",
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
        # Missing-history rows legitimately produce all-null reductions. Those
        # are represented as NULL and are not actionable runtime warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            base = pd.concat([derive_racing_features(frame),
                              derive_sectional_class_features(frame),
                              derive_entity_history_features(frame)], axis=1)
            derived = pd.concat([base, derive_context_features(frame, base)], axis=1)
        missing_outputs = [name for name in FEATURES_TO_STORE if name not in derived.columns]
        if missing_outputs:
            raise ValueError("Registered features were not generated: " + ", ".join(missing_outputs))
        print(f"database={database} rows={len(frame):,}")
        low10, low25, constant = [], [], []
        for name in FEATURES_TO_STORE:
            values = derived[name]
            coverage = values.notna().mean()
            if coverage < .10: low10.append(name)
            if coverage < .25: low25.append(name)
            if values.nunique(dropna=True) <= 1: constant.append(name)
            print(f"{name}: non_null={values.notna().sum():,} coverage={coverage:.4f} "
                  f"min={values.min()} max={values.max()} mean={values.mean()}")
        if args.dry_run:
            print("dry_run=yes database_modified=no")
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            changes_before = connection.total_changes
            # Another updater may have committed while this process waited for
            # BEGIN IMMEDIATE. Refresh under the acquired write lock so repeated
            # and concurrent executions cannot attempt duplicate ALTERs.
            locked_existing = {
                str(row[1]) for row in connection.execute(
                    'PRAGMA table_info("race_runners")'
                )
            }
            for name in FEATURES_TO_STORE:
                if name not in locked_existing:
                    connection.execute(
                        f'ALTER TABLE "race_runners" ADD COLUMN "{name}" REAL'
                    )
            # Stage one row per runner, then update all features in one table scan.
            # This retains atomicity while avoiding features*rows UPDATE statements.
            temp_columns = ", ".join(f'"{name}" REAL' for name in FEATURES_TO_STORE)
            connection.execute(
                f'CREATE TEMP TABLE "derived_feature_updates" '
                f'("source_rowid" INTEGER PRIMARY KEY, {temp_columns})'
            )
            insert_columns = ("source_rowid", *FEATURES_TO_STORE)
            placeholders = ",".join("?" for _ in insert_columns)
            quoted_insert_columns = ",".join(f'"{name}"' for name in insert_columns)
            insert_sql = (f'INSERT INTO "derived_feature_updates" '
                          f'({quoted_insert_columns}) VALUES ({placeholders})')
            for start in range(0, len(frame), 1000):
                stop = min(start + 1000, len(frame))
                batch = []
                for position in range(start, stop):
                    batch.append((int(frame["rowid"].iloc[position]), *(
                        None if pd.isna(derived[name].iloc[position])
                        else float(derived[name].iloc[position])
                        for name in FEATURES_TO_STORE
                    )))
                connection.executemany(insert_sql, batch)
            assignments = ", ".join(
                f'"{name}" = updates."{name}"' for name in FEATURES_TO_STORE
            )
            connection.execute(
                f'UPDATE "race_runners" AS runners SET {assignments} '
                f'FROM "derived_feature_updates" AS updates '
                f'WHERE runners.rowid = updates."source_rowid"'
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        changed_rows = connection.total_changes - changes_before
        print(
            f"database_modified={'yes' if changed_rows else 'no'} "
            f"changed_rows={changed_rows:,} total_features_added={sum(name not in locked_existing for name in FEATURES_TO_STORE)} "
            f"total_features_updated={len(FEATURES_TO_STORE)} "
            f"features_below_10pct_coverage={','.join(low10) or 'none'} "
            f"features_below_25pct_coverage={','.join(low25) or 'none'} "
            f"effectively_constant={','.join(constant) or 'none'}"
        )


if __name__ == "__main__":
    main()
