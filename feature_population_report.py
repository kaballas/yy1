#!/usr/bin/env python3
"""Report how completely each feature is populated in race_runners."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "db" / "race_runners.sqlite"
DEFAULT_OUTPUT = ROOT / "outputs" / "feature_population_report.csv"

# Columns used to identify rows, control training, or store outcomes are not
# model inputs. They can still be requested explicitly with --all-columns.
NON_FEATURE_COLUMNS = {
    "feature_schema_version",
    "race_id",
    "competition_id",
    "selection_id",
    "runner_number",
    "winner_index",
    "is_trainable",
    "finish_place",
    "result_code",
    "status",
    "sp_starting_price",
    "runner_mask",
    "rank_label",
    "top3_mask",
    "is_winner",
    "is_validation",
}

NUMERIC_TYPE_TOKENS = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")


@dataclass(frozen=True)
class Column:
    position: int
    name: str
    declared_type: str

    @property
    def numeric(self) -> bool:
        declared = self.declared_type.upper()
        return any(token in declared for token in NUMERIC_TYPE_TOKENS)


@dataclass(frozen=True)
class FeaturePopulation:
    feature: str
    declared_type: str
    populated_rows: int
    missing_rows: int
    total_rows: int
    population_pct: float


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def validate_identifier(value: str, description: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe {description}: {value!r}")
    return value


def load_columns(connection: sqlite3.Connection, table: str) -> list[Column]:
    validate_identifier(table, "table name")
    rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    if not rows:
        raise ValueError(f"Table {table!r} does not exist or has no columns")
    return [
        Column(int(row[0]), str(row[1]), str(row[2]) or "UNTYPED")
        for row in rows
    ]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def load_manifest_features(path: Path) -> list[str]:
    """Read feature names from the manifest formats used in this repository."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Feature manifest must contain a JSON object: {path}")

    features: list[str] = []
    features.extend(_string_list(payload.get("features")))
    features.extend(_string_list(payload.get("base_features")))
    models = payload.get("models")
    if isinstance(models, dict):
        for model in models.values():
            if isinstance(model, dict):
                features.extend(_string_list(model.get("features")))
    elif isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            features.extend(_string_list(model.get("features")))
            details = model.get("details")
            if isinstance(details, dict):
                features.extend(_string_list(details.get("input_features")))

    unique = list(dict.fromkeys(features))
    if not unique:
        raise ValueError(f"No feature lists were found in manifest: {path}")
    return unique


def resolve_feature_names(
    columns: Iterable[Column],
    manifest: Path | None,
    all_columns: bool,
) -> tuple[list[str], list[str]]:
    columns = list(columns)
    schema_names = {column.name for column in columns}
    if manifest is not None:
        requested = load_manifest_features(manifest)
        return (
            [name for name in requested if name in schema_names],
            [name for name in requested if name not in schema_names],
        )
    if all_columns:
        return [column.name for column in columns], []
    return [
        column.name
        for column in columns
        if column.numeric and column.name not in NON_FEATURE_COLUMNS
    ], []


def build_filter(
    competition_ids: list[int], statuses: list[str], active_only: bool
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if competition_ids:
        placeholders = ", ".join("?" for _ in competition_ids)
        clauses.append(f"competition_id IN ({placeholders})")
        parameters.extend(competition_ids)
    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        clauses.append(f"status IN ({placeholders})")
        parameters.extend(statuses)
    if active_only:
        clauses.append("runner_mask = 1")
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters


def populated_expression(column: Column) -> str:
    quoted = quote_identifier(column.name)
    if column.numeric:
        # SQLite normally converts NaN to NULL. The finite bound also rejects
        # positive/negative infinity, which pandas/XGBoost cannot safely consume.
        return (
            f"CASE WHEN {quoted} IS NOT NULL "
            f"AND typeof({quoted}) IN ('integer', 'real') "
            f"AND abs(CAST({quoted} AS REAL)) <= 1.7976931348623157e308 "
            "THEN 1 ELSE 0 END"
        )
    return (
        f"CASE WHEN {quoted} IS NOT NULL "
        f"AND trim(CAST({quoted} AS TEXT)) <> '' THEN 1 ELSE 0 END"
    )


def analyze_population(
    database: Path,
    table: str = "race_runners",
    manifest: Path | None = None,
    all_columns: bool = False,
    competition_ids: list[int] | None = None,
    statuses: list[str] | None = None,
    active_only: bool = False,
    chunk_size: int = 150,
) -> tuple[list[FeaturePopulation], list[str]]:
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    table = validate_identifier(table, "table name")
    where_sql, parameters = build_filter(
        competition_ids or [], statuses or [], active_only
    )
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        columns = load_columns(connection, table)
        selected, unavailable = resolve_feature_names(columns, manifest, all_columns)
        by_name = {column.name: column for column in columns}
        total_rows = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier(table)}{where_sql}",
                parameters,
            ).fetchone()[0]
        )
        populated_by_name: dict[str, int] = {}
        for start in range(0, len(selected), chunk_size):
            names = selected[start : start + chunk_size]
            expressions = ", ".join(
                f"SUM({populated_expression(by_name[name])})" for name in names
            )
            values = connection.execute(
                f"SELECT {expressions} FROM {quote_identifier(table)}{where_sql}",
                parameters,
            ).fetchone()
            populated_by_name.update(
                (name, int(value or 0)) for name, value in zip(names, values)
            )

    rows = [
        FeaturePopulation(
            feature=name,
            declared_type=by_name[name].declared_type,
            populated_rows=populated_by_name[name],
            missing_rows=total_rows - populated_by_name[name],
            total_rows=total_rows,
            population_pct=(
                100.0 * populated_by_name[name] / total_rows
                if total_rows
                else math.nan
            ),
        )
        for name in selected
    ]
    rows.sort(key=lambda row: (-row.population_pct, row.feature))
    return rows, unavailable


def write_csv(path: Path, rows: Iterable[FeaturePopulation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "feature",
                "population_pct",
                "populated_rows",
                "missing_rows",
                "total_rows",
                "declared_type",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.feature,
                    f"{row.population_pct:.6f}",
                    row.populated_rows,
                    row.missing_rows,
                    row.total_rows,
                    row.declared_type,
                ]
            )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank database features by their percentage of populated, usable values."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--table", default="race_runners")
    parser.add_argument(
        "--features-json",
        type=Path,
        help="Limit the report to the union of features declared in a JSON manifest.",
    )
    parser.add_argument(
        "--all-columns",
        action="store_true",
        help="Include metadata, text, identifiers, controls, and outcomes.",
    )
    parser.add_argument(
        "--competition-id",
        type=int,
        action="append",
        default=[],
        help="Restrict rows to this competition; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Restrict rows to this status; repeat for multiple statuses.",
    )
    parser.add_argument(
        "--active-only", action="store_true", help="Use only runner_mask = 1 rows."
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Rows to print to the terminal; 0 prints every feature (default: 50).",
    )
    args = parser.parse_args()
    if args.features_json is not None and args.all_columns:
        parser.error("--features-json and --all-columns are mutually exclusive")
    if args.top < 0:
        parser.error("--top cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    rows, unavailable = analyze_population(
        database=args.db,
        table=args.table,
        manifest=args.features_json,
        all_columns=args.all_columns,
        competition_ids=args.competition_id,
        statuses=args.status,
        active_only=args.active_only,
    )
    write_csv(args.output_csv, rows)
    total_rows = rows[0].total_rows if rows else 0
    complete = sum(row.population_pct == 100.0 for row in rows)
    print("FEATURE POPULATION REPORT")
    print(f"database={args.db.resolve()}")
    print(f"table={args.table} rows={total_rows:,} features={len(rows):,}")
    print(f"fully_populated_features={complete:,}")
    if unavailable:
        print(
            f"WARNING manifest_features_missing_from_database={len(unavailable)} "
            + ",".join(unavailable)
        )
    print("\n  population%  populated     missing  feature")
    shown = rows if args.top == 0 else rows[: args.top]
    for row in shown:
        print(
            f"  {row.population_pct:10.2f}  {row.populated_rows:9,}  "
            f"{row.missing_rows:10,}  {row.feature}"
        )
    if len(shown) < len(rows):
        print(f"\nShowing {len(shown):,} of {len(rows):,}; CSV contains every feature.")
    print(f"report={args.output_csv.resolve()}")
    return 1 if unavailable else 0


if __name__ == "__main__":
    raise SystemExit(main())
