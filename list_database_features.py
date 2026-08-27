#!/usr/bin/env python3
"""List numeric and text columns available in a SQLite table."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from feature_population_report import NON_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "db" / "race_runners.sqlite"
NUMERIC_AFFINITIES = {"INTEGER", "REAL", "NUMERIC"}


@dataclass(frozen=True)
class DatabaseColumn:
    name: str
    declared_type: str
    affinity: str
    category: str
    is_feature: bool


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sqlite_affinity(declared_type: str) -> str:
    """Return SQLite's documented type affinity for a declared column type."""
    value = declared_type.upper().strip()
    if "INT" in value:
        return "INTEGER"
    if any(token in value for token in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if "BLOB" in value or not value:
        return "BLOB"
    if any(token in value for token in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def column_category(affinity: str) -> str:
    if affinity in NUMERIC_AFFINITIES:
        return "numeric"
    if affinity == "TEXT":
        return "text"
    return "other"


def inspect_columns(database: Path, table: str = "race_runners") -> list[DatabaseColumn]:
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        rows = connection.execute(
            f"PRAGMA table_info({quote_identifier(table)})"
        ).fetchall()
    if not rows:
        raise ValueError(f"Table or view {table!r} does not exist or has no columns")
    result: list[DatabaseColumn] = []
    for row in rows:
        name = str(row[1])
        declared_type = str(row[2] or "UNTYPED")
        affinity = sqlite_affinity("" if declared_type == "UNTYPED" else declared_type)
        result.append(
            DatabaseColumn(
                name=name,
                declared_type=declared_type,
                affinity=affinity,
                category=column_category(affinity),
                is_feature=name not in NON_FEATURE_COLUMNS,
            )
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default="race_runners")
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Exclude known identifiers, control fields, and result/target columns.",
    )
    parser.add_argument(
        "--format", choices=("text", "json"), default="text"
    )
    return parser.parse_args(argv)


def print_text(database: Path, table: str, columns: list[DatabaseColumn]) -> None:
    print(f"database={database.resolve()} table={table}")
    for category, heading in (("numeric", "NUMERICAL FEATURES"), ("text", "TEXT FEATURES"), ("other", "OTHER COLUMNS")):
        selected = [column for column in columns if column.category == category]
        if not selected:
            continue
        print(f"\n{heading} ({len(selected)})")
        width = max(len(column.name) for column in selected)
        for column in selected:
            role = "feature" if column.is_feature else "non-feature"
            print(
                f"{column.name:<{width}}  declared_type={column.declared_type:<12} "
                f"affinity={column.affinity:<7} role={role}"
            )
    counts = {
        category: sum(column.category == category for column in columns)
        for category in ("numeric", "text", "other")
    }
    print(
        f"\nTOTAL columns={len(columns)} numeric={counts['numeric']} "
        f"text={counts['text']} other={counts['other']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        columns = inspect_columns(args.db, args.table)
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.features_only:
        columns = [column for column in columns if column.is_feature]
    if args.format == "json":
        print(
            json.dumps(
                {
                    "database": str(args.db.resolve()),
                    "table": args.table,
                    "columns": [asdict(column) for column in columns],
                },
                indent=2,
            )
        )
    else:
        print_text(args.db, args.table, columns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
