#!/usr/bin/env python3
"""Add nullable BLOB columns for Word2Vec encodings of text features."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Sequence

from list_database_features import DEFAULT_DATABASE, inspect_columns, quote_identifier


def planned_vector_columns(
    database: Path, table: str = "race_runners"
) -> tuple[list[tuple[str, str]], list[str]]:
    columns = inspect_columns(database, table)
    names = {column.name for column in columns}
    additions: list[tuple[str, str]] = []
    existing: list[str] = []
    for column in columns:
        if column.category != "text" or not column.is_feature:
            continue
        vector_name = f"{column.name}_vec"
        if vector_name in names:
            existing.append(vector_name)
        else:
            additions.append((column.name, vector_name))
    return additions, existing


def add_vector_columns(
    database: Path, table: str = "race_runners"
) -> tuple[list[tuple[str, str]], list[str]]:
    additions, existing = planned_vector_columns(database, table)
    if not additions:
        return additions, existing
    with sqlite3.connect(database) as connection:
        connection.execute("BEGIN IMMEDIATE")
        for _source, vector_name in additions:
            connection.execute(
                f"ALTER TABLE {quote_identifier(table)} "
                f"ADD COLUMN {quote_identifier(vector_name)} BLOB"
            )
    return additions, existing


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--table", default="race_runners")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migration. Without this flag, only show the plan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.apply:
            additions, existing = add_vector_columns(args.db, args.table)
        else:
            additions, existing = planned_vector_columns(args.db, args.table)
    except (OSError, sqlite3.Error, ValueError) as error:
        raise SystemExit(str(error)) from error

    action = "added" if args.apply else "would_add"
    print(f"database={args.db.resolve()} table={args.table}")
    for source, vector_name in additions:
        print(f"{action} source={source} vector_column={vector_name} type=BLOB")
    print(
        f"SUMMARY {action}={len(additions)} already_existing={len(existing)} "
        f"storage=float32_blob nullable=true"
    )
    if not args.apply and additions:
        print("Dry run only; rerun with --apply to change the schema.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
