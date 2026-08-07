"""TabFM training database helpers."""

from __future__ import annotations

import csv
import os
import sqlite3
from pathlib import Path
import numpy as np
from src.constants import (
    TRAINING_ROWS_VIEW,
    VALIDATION_COHORTS,
    VALIDATION_ROWS_VIEW,
)
from src.utilities import parse_iso_timestamp


def validate_feature_columns(db_path: Path, features: list[str]) -> None:
    """Validate required and numeric feature columns in the source database."""
    print(db_path)
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        schema = {
            row[1]: row[2].upper()
            for row in connection.execute("PRAGMA table_info(race_runners)")
        }
    finally:
        connection.close()
    required = {
        "race_id", "start_time_iso", "runner_number", "top3_mask",
        "is_validation", "fluc2",
    }
    missing_required = sorted(required - set(schema))
    missing = sorted(set(features) - set(schema))
    non_numeric = sorted(
        column for column in features
        if column in schema and schema[column] not in {"INTEGER", "REAL", "NUMERIC"}
    )
    if missing_required:
        raise ValueError(
            "race_runners is missing required training columns: "
            + ", ".join(missing_required)
        )
    if missing:
        raise ValueError(f"Features missing from race_runners: {', '.join(missing)}")
    if non_numeric:
        raise ValueError(f"Features are not numeric: {', '.join(non_numeric)}")


def load_race_number_eligible_ids(
    db_path: Path, minimum: int | None
) -> set[int]:
    """Return whole races whose race_number is at least ``minimum``."""
    if minimum is None:
        return set()
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        columns = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("race_runners")')
        }
        if "race_number" not in columns:
            raise ValueError("race_runners is missing race_number")
        inconsistent = connection.execute(
            'SELECT race_id FROM "race_runners" GROUP BY race_id '
            'HAVING MIN(race_number) <> MAX(race_number) LIMIT 1'
        ).fetchone()
        if inconsistent is not None:
            raise ValueError(
                f"race_id {int(inconsistent[0])} has inconsistent race_number values"
            )
        return {
            int(row[0])
            for row in connection.execute(
                'SELECT race_id FROM "race_runners" GROUP BY race_id '
                'HAVING MIN(race_number) >= ?',
                (minimum,),
            )
        }
    finally:
        connection.close()


def load_race_numbers(db_path: Path, race_ids: list[int] | set[int]) -> dict[int, int | None]:
    """Load one race_number per race for schedule diagnostics."""
    requested = sorted({int(race_id) for race_id in race_ids})
    if not requested:
        return {}
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        result: dict[int, int | None] = {}
        # Keep well below SQLite's default parameter limit.
        for start in range(0, len(requested), 500):
            chunk = requested[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT race_id, MIN(race_number) "
                f"FROM race_runners WHERE race_id IN ({placeholders}) "
                "GROUP BY race_id",
                chunk,
            )
            result.update(
                {int(race_id): (None if race_number is None else int(race_number))
                 for race_id, race_number in rows}
            )
        return result
    finally:
        connection.close()


def quote_identifier(value: str) -> str:
    """Quote a SQLite identifier without changing its value."""
    return '"' + value.replace('"', '""') + '"'


def require_rows_view(connection: sqlite3.Connection, view_name: str) -> None:
    """Fail clearly when a required database row view is not installed."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'view' AND name = ?",
        (view_name,),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Database is missing required view {view_name!r}"
        )


def require_training_rows_view(connection: sqlite3.Connection) -> None:
    """Backward-compatible check for callers that use the training view."""
    require_rows_view(connection, TRAINING_ROWS_VIEW)


def print_race_selection_logic(db_path: Path, minimum_race_number: int | None) -> None:
    """Print the database and in-memory rules that produce each race partition."""
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        require_training_rows_view(connection)
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = ?",
            (TRAINING_ROWS_VIEW,),
        ).fetchone()
    finally:
        connection.close()

    view_sql = str(row[0]).strip() if row and row[0] else "<definition unavailable>"
    view_name = quote_identifier(TRAINING_ROWS_VIEW)
    print(f"training_rows_view_sql:\n{view_sql}", flush=True)
    print(
        "training_race_sql:\n"
        f"SELECT DISTINCT race_id FROM {view_name} "
        "ORDER BY race_id",
        flush=True,
    )
    print(
        "training_race_post_sql_logic: exclude races with fewer than 3 runners "
        "or other than exactly 3 "
        "top3_mask=1 rows; "
        + (
            f"require whole-race race_number >= {minimum_race_number}"
            if minimum_race_number is not None
            else "no --min-race-number filter"
        ),
        flush=True,
    )
    print("\n")
    print(
        "validation_race_sql:\n"
        f"SELECT DISTINCT race_id FROM {quote_identifier(VALIDATION_ROWS_VIEW)} "
        " ORDER BY start_time_iso, race_id",
        flush=True,
    )
    print("\n")
    print(
        "validation_race_post_sql_logic: exclude races with fewer than 3 runners "
        "or other than exactly 3 top3_mask=1 rows; preserve all "
        "remaining races (no max-valid truncation). Cohorts come from "
        "model_validation_races.validation_cohort when present; an uncovered "
        "race is labelled legacy_combined. Checkpoint selection requires the "
        "chronological_representative cohort and never falls back to combined "
        "legacy validation metrics.",
        flush=True,
    )
    print("\n")


def load_rows(
    db_path: Path,
    feature_columns: list[str],
    view_name: str = TRAINING_ROWS_VIEW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ordered feature and target rows from a required database view."""
    columns = [
        "race_id", "start_time_iso", "is_validation", *feature_columns, "top3_mask"
    ]
    selected_columns = ", ".join(quote_identifier(column) for column in columns)
    sql = (
        f"SELECT {selected_columns} FROM {quote_identifier(view_name)} "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    loader_name = (
        "training_rows_loader_sql"
        if view_name == TRAINING_ROWS_VIEW
        else "validation_rows_loader_sql"
    )
    print(f"{loader_name}:\n{sql}", flush=True)
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        require_rows_view(connection, view_name)
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError(f"No completed rows found in {view_name}")
    print("\n")
    race_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    times = np.asarray([parse_iso_timestamp(row[1]) for row in rows], dtype=object)
    validation_flags = np.asarray([row[2] for row in rows], dtype=np.int8)
    x = np.asarray(
        [[np.nan if value is None else float(value) for value in row[3:-1]] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([row[-1] for row in rows], dtype=np.int64)
    return x, y, race_ids, times, validation_flags


def training_csv_columns(feature_columns: list[str]) -> list[str]:
    """Return the stable, inspectable schema used by training CSV snapshots."""
    return [
        "race_id",
        "start_time_iso",
        "is_validation",
        "runner_number",
        *feature_columns,
        "top3_mask",
        "fluc2",
    ]


def export_rows_to_csv(
    db_path: Path,
    feature_columns: list[str],
    view_name: str,
    csv_path: Path,
) -> int:
    """Atomically export an ordered database view snapshot for model loading."""
    columns = training_csv_columns(feature_columns)
    selected_columns = ", ".join(quote_identifier(column) for column in columns)
    sql = (
        f"SELECT {selected_columns} FROM {quote_identifier(view_name)} "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    print(f"{view_name}_csv_export_sql:\n{sql}", flush=True)
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        require_rows_view(connection, view_name)
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError(f"No completed rows found in {view_name}")

    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, csv_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(
        f"csv_export view={view_name} path={csv_path} rows={len(rows):,}",
        flush=True,
    )
    return len(rows)


def load_rows_from_csv(
    csv_path: Path,
    feature_columns: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load all training arrays, including market prices, from a CSV snapshot."""
    expected_columns = training_csv_columns(feature_columns)
    with csv_path.resolve().open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != expected_columns:
            raise ValueError(
                f"CSV schema mismatch for {csv_path}: expected {expected_columns}, "
                f"found {header}"
            )
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No records found in training CSV {csv_path}")
    expected_width = len(expected_columns)
    invalid_width = next(
        (row_number for row_number, row in enumerate(rows, start=2)
         if len(row) != expected_width),
        None,
    )
    if invalid_width is not None:
        raise ValueError(
            f"CSV row {invalid_width} in {csv_path} does not have "
            f"{expected_width} columns"
        )

    feature_start = 4
    feature_end = feature_start + len(feature_columns)

    def optional_float(value: str) -> float:
        return np.nan if value == "" else float(value)

    try:
        race_ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        times = np.asarray(
            [parse_iso_timestamp(row[1]) for row in rows], dtype=object
        )
        validation_flags = np.asarray([int(row[2]) for row in rows], dtype=np.int8)
        x = np.asarray(
            [
                [optional_float(value) for value in row[feature_start:feature_end]]
                for row in rows
            ],
            dtype=np.float32,
        )
        y = np.asarray([int(row[feature_end]) for row in rows], dtype=np.int64)
        market_fluc2 = np.asarray(
            [optional_float(row[feature_end + 1]) for row in rows],
            dtype=np.float32,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid typed value in training CSV {csv_path}: {error}") from error
    print(f"csv_load path={csv_path.resolve()} rows={len(rows):,}", flush=True)
    return x, y, race_ids, times, validation_flags, market_fluc2


def load_market_fluc2(
    db_path: Path,
    expected_race_ids: np.ndarray,
    view_name: str = TRAINING_ROWS_VIEW,
) -> np.ndarray:
    """Load the market-baseline price without making it a model feature."""
    sql = (
        f"SELECT race_id, fluc2 FROM {quote_identifier(view_name)} "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        require_rows_view(connection, view_name)
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    market_race_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    if not np.array_equal(market_race_ids, expected_race_ids):
        raise RuntimeError(f"fluc2 baseline rows do not align with {view_name} rows")
    return np.asarray(
        [np.nan if row[1] is None else float(row[1]) for row in rows],
        dtype=np.float32,
    )


def load_context_rows(
    db_path: Path,
    feature_columns: list[str],
    context_race_ids: list[int],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Load the explicitly selected fixed context independently of partition views."""
    if not context_race_ids:
        raise ValueError("Validation context JSON contains no race IDs")
    columns = [
        "race_id", "start_time_iso", "is_validation", *feature_columns,
        "top3_mask", "fluc2",
    ]
    selected_columns = ", ".join(quote_identifier(column) for column in columns)
    placeholders = ", ".join("?" for _ in context_race_ids)
    sql = (
        f"SELECT {selected_columns} FROM race_runners "
        f"WHERE race_id IN ({placeholders}) AND status = 'finished' "
        "AND top3_mask IN (0, 1) "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    print(
        "validation_context_rows_loader_sql:\n"
        f"SELECT {selected_columns} FROM race_runners "
        "WHERE race_id IN (<context-json race IDs>) AND status = 'finished' "
        "AND top3_mask IN (0, 1) "
        "ORDER BY start_time_iso, race_id, runner_number",
        flush=True,
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(sql, list(map(int, context_race_ids))).fetchall()
    finally:
        connection.close()
    loaded_race_ids = {int(row[0]) for row in rows}
    missing = sorted(set(map(int, context_race_ids)) - loaded_race_ids)
    if missing:
        raise ValueError(
            "Validation context race IDs missing completed rows in race_runners: "
            + ", ".join(map(str, missing))
        )
    race_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    times = np.asarray([parse_iso_timestamp(row[1]) for row in rows], dtype=object)
    validation_flags = np.asarray([row[2] for row in rows], dtype=np.int8)
    x = np.asarray(
        [
            [np.nan if value is None else float(value) for value in row[3:-2]]
            for row in rows
        ],
        dtype=np.float32,
    )
    y = np.asarray([row[-2] for row in rows], dtype=np.int64)
    fluc2 = np.asarray(
        [np.nan if row[-1] is None else float(row[-1]) for row in rows],
        dtype=np.float32,
    )
    return x, y, race_ids, times, validation_flags, fluc2


def load_validation_cohorts(
    db_path: Path, validation_race_ids: np.ndarray
) -> tuple[np.ndarray, str]:
    """Return one cohort label per validation row, falling back when unassigned."""
    validation_race_ids = np.asarray(validation_race_ids, dtype=np.int64)
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'model_validation_races'"
        ).fetchone()
        if exists is None:
            return (
                np.full(len(validation_race_ids), "legacy_combined", dtype=object),
                "legacy_is_validation_flags",
            )
        rows = connection.execute(
            "SELECT race_id, validation_cohort FROM model_validation_races"
        ).fetchall()
    finally:
        connection.close()

    cohort_by_race: dict[int, str] = {}
    for race_id_value, cohort_value in rows:
        race_id = int(race_id_value)
        cohort = str(cohort_value)
        if cohort not in VALIDATION_COHORTS:
            raise ValueError(
                f"model_validation_races has invalid cohort {cohort!r} for race {race_id}"
            )
        cohort_by_race[race_id] = cohort
    selected_races = set(map(int, validation_race_ids))
    missing = sorted(selected_races - set(cohort_by_race))
    source = (
        "model_validation_races.validation_cohort_with_legacy_fallback"
        if missing
        else "model_validation_races.validation_cohort"
    )
    return (
        np.asarray(
            [
                cohort_by_race.get(int(race_id), "legacy_combined")
                for race_id in validation_race_ids
            ]
        ),
        source,
    )
