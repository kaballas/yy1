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
    race_relative_runner_mask,
)
from src.derived_racing_features import DERIVED_FEATURE_NAMES, derive_racing_features


FEATURES_TO_STORE = (
    *DERIVED_FEATURE_NAMES,
    *ADVANCED_FEATURE_NAMES,
)
CALCULATION_VERSION_COLUMN = "derived_racing_features_version"
# Increment this whenever a formula or registry change requires existing rows to
# be rebuilt. A version marker is reliable where feature NULLs are not: many
# leakage-safe features are legitimately NULL because a horse has no history.
CALCULATION_VERSION = "2026-08-13-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transactionally calculate derived pre-race feature columns."
    )
    parser.add_argument("--db", type=Path, default=Path("db/race_runners.sqlite"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Recalculate every race instead of only unfinished/missing-version races.",
    )
    return parser.parse_args()


def target_selection(
    force: bool,
    version_column_available: bool = True,
) -> tuple[str, str]:
    if force:
        return "1 = 1", "force"

    # Recalculate every unfinished race on every run because its source data can
    # continue changing. Also calculate finished races that arrived after the
    # last update or carry an older formula version. Select the whole race so
    # within-race ranks always see the complete field. Rows without a race ID
    # can only select themselves.
    unfinished = 'COALESCE("status", \'\') <> \'finished\''
    version_pending = (
        f'("{CALCULATION_VERSION_COLUMN}" IS NULL OR '
        f'"{CALCULATION_VERSION_COLUMN}" <> \'{CALCULATION_VERSION}\')'
        if version_column_available else "1 = 1"
    )
    pending = f"({unfinished} OR {version_pending})"
    return (
        f'"race_id" IN (SELECT "race_id" FROM "race_runners" WHERE {pending}) '
        f'OR ("race_id" IS NULL AND {pending})',
        "pending",
    )


def main() -> None:
    args = parse_args()
    database = args.db.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    input_columns = list(dict.fromkeys([
        "rowid", "race_id", "competition_id", "competition_name", "distance_m",
        "race_name", "grade", "track_status", "start_time_iso", "draw_number", "weight_kg",
        "status", "source_betting_status", "runner_mask", "top3_mask",
        "active_field_size", "field_size",
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
        # One-time migration: a database containing the complete feature registry
        # was produced by the old all-row updater. Mark those rows complete rather
        # than pointlessly rebuilding them merely because the marker is new.
        if CALCULATION_VERSION_COLUMN not in existing and not args.dry_run:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    f'ALTER TABLE "race_runners" ADD COLUMN '
                    f'"{CALCULATION_VERSION_COLUMN}" TEXT'
                )
                if set(FEATURES_TO_STORE) <= existing:
                    connection.execute(
                        f'UPDATE "race_runners" SET "{CALCULATION_VERSION_COLUMN}" = ?',
                        (CALCULATION_VERSION,),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            existing.add(CALCULATION_VERSION_COLUMN)

        target_where, selection_mode = target_selection(
            args.force,
            version_column_available=CALCULATION_VERSION_COLUMN in existing,
        )

        pending_rows, pending_races = connection.execute(
            f'SELECT COUNT(*), COUNT(DISTINCT "race_id") FROM "race_runners" '
            f'WHERE {target_where}'
        ).fetchone()
        print(
            f"database={database} selection_mode={selection_mode} "
            f"calculation_version={CALCULATION_VERSION} "
            f"races_to_process={int(pending_races):,} rows_to_process={int(pending_rows):,}"
        )
        if not pending_rows:
            print("nothing_to_update=yes database_modified=no")
            return
        quoted = ", ".join(f'"{name}"' for name in input_columns)
        frame = pd.read_sql_query(
            f'SELECT {quoted} FROM "race_runners" WHERE {target_where} ORDER BY rowid',
            connection,
        )
        context_eligible = race_relative_runner_mask(frame)
        stored_mask = pd.to_numeric(
            frame["runner_mask"], errors="coerce"
        ).eq(1)
        live_fallback = context_eligible & ~stored_mask
        print(
            f"race_relative_stored_active_rows={int(stored_mask.sum()):,} "
            f"race_relative_live_fallback_rows={int(live_fallback.sum()):,} "
            f"race_relative_live_fallback_races="
            f"{frame.loc[live_fallback, 'race_id'].nunique(dropna=True):,}"
        )
        # Entity aggregates require earlier results, but only their lightweight
        # columns are loaded/calculated across history. Expensive six-start and
        # sectional calculations remain restricted to target races.
        entity_columns = (
            "rowid", "start_time_iso", "status", "runner_mask", "top3_mask",
            "finish_place", "active_field_size", "field_size", "jockey", "trainer",
        )
        entity_quoted = ", ".join(f'"{name}"' for name in entity_columns)
        entity_frame = pd.read_sql_query(
            f'SELECT {entity_quoted} FROM "race_runners" ORDER BY rowid', connection
        )
        entity_frame.index = entity_frame["rowid"].astype(int)
        # Missing-history rows legitimately produce all-null reductions. Those
        # are represented as NULL and are not actionable runtime warnings.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            entity_all = derive_entity_history_features(entity_frame)
            entity_target = entity_all.loc[frame["rowid"].astype(int)].copy()
            entity_target.index = frame.index
            base = pd.concat([derive_racing_features(frame),
                              derive_sectional_class_features(frame),
                              entity_target], axis=1)
            derived = pd.concat([base, derive_context_features(frame, base)], axis=1)
        missing_outputs = [name for name in FEATURES_TO_STORE if name not in derived.columns]
        if missing_outputs:
            raise ValueError("Registered features were not generated: " + ", ".join(missing_outputs))
        print(f"processed_rows={len(frame):,}")
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
            if CALCULATION_VERSION_COLUMN not in locked_existing:
                connection.execute(
                    f'ALTER TABLE "race_runners" ADD COLUMN '
                    f'"{CALCULATION_VERSION_COLUMN}" TEXT'
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
            staged = derived.loc[:, FEATURES_TO_STORE].astype(object)
            staged = staged.where(pd.notna(staged), None)
            staged.insert(0, "source_rowid", frame["rowid"].astype(int).to_numpy())
            connection.executemany(
                insert_sql, staged.itertuples(index=False, name=None)
            )
            assignments = ", ".join(
                f'"{name}" = updates."{name}"' for name in FEATURES_TO_STORE
            )
            assignments += f', "{CALCULATION_VERSION_COLUMN}" = ?'
            connection.execute(
                f'UPDATE "race_runners" AS runners SET {assignments} '
                f'FROM "derived_feature_updates" AS updates '
                f'WHERE runners.rowid = updates."source_rowid"',
                (CALCULATION_VERSION,),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        changed_rows = connection.total_changes - changes_before
        print(
            f"database_modified={'yes' if changed_rows else 'no'} "
            f"selection_mode={selection_mode} races_processed={int(pending_races):,} "
            f"rows_processed={len(frame):,} "
            f"changed_rows={changed_rows:,} total_features_added={sum(name not in locked_existing for name in FEATURES_TO_STORE)} "
            f"total_features_updated={len(FEATURES_TO_STORE)} "
            f"features_below_10pct_coverage={','.join(low10) or 'none'} "
            f"features_below_25pct_coverage={','.join(low25) or 'none'} "
            f"effectively_constant={','.join(constant) or 'none'}"
        )


if __name__ == "__main__":
    main()
