#!/usr/bin/env python3
"""Expand the chronological checkpoint-selection cohort safely."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.config import DEFAULT_DB


SELECTION_SQL = """
WITH complete AS (
    SELECT
        race_id,
        MIN(start_time_iso) AS start_time,
        MIN(competition_id) AS competition_id
    FROM tabfm_validation_runners
    GROUP BY race_id
    HAVING COUNT(*) >= 4
       AND SUM(top3_mask) = 3
),
eligible AS (
    SELECT c.*
    FROM complete AS c
    WHERE (
        SELECT COUNT(*)
        FROM complete AS earlier
        WHERE earlier.competition_id = c.competition_id
          AND earlier.start_time < c.start_time
    ) >= :context_races
)
SELECT e.race_id, e.start_time, e.competition_id
FROM eligible AS e
LEFT JOIN model_validation_races AS m ON m.race_id = e.race_id
WHERE m.race_id IS NULL
ORDER BY e.start_time DESC, e.race_id DESC
LIMIT :needed
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add eligible validation races to model_validation_races as "
            "chronological_representative races."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--target-races",
        type=int,
        default=100,
        help="Desired total chronological cohort size (default: 100).",
    )
    parser.add_argument(
        "--context-races",
        type=int,
        default=9,
        help="Required strictly earlier same-competition races (default: 9).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the races that would be added without changing the database.",
    )
    return parser.parse_args()


def validate_schema(connection: sqlite3.Connection) -> None:
    required = {"model_validation_races", "tabfm_validation_runners"}
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE name IN (?, ?)", tuple(required)
    ).fetchall()
    missing = required - {str(row[0]) for row in rows}
    if missing:
        raise RuntimeError("Database is missing: " + ", ".join(sorted(missing)))


def main() -> int:
    args = parse_args()
    if args.target_races < 1:
        raise SystemExit("--target-races must be positive")
    if args.context_races < 0:
        raise SystemExit("--context-races must be non-negative")
    if not args.db.is_file():
        raise SystemExit(f"Database does not exist: {args.db}")

    connection = sqlite3.connect(args.db)
    try:
        validate_schema(connection)
        current = int(
            connection.execute(
                """
                SELECT COUNT(DISTINCT m.race_id)
                FROM model_validation_races AS m
                JOIN tabfm_validation_runners AS v ON v.race_id = m.race_id
                WHERE m.validation_cohort = 'chronological_representative'
                """
            ).fetchone()[0]
        )
        needed = max(0, args.target_races - current)
        selected = connection.execute(
            SELECTION_SQL,
            {"context_races": args.context_races, "needed": needed},
        ).fetchall()

        print(
            f"Current chronological races: {current}\n"
            f"Target chronological races:  {args.target_races}\n"
            f"Eligible races selected:     {len(selected)}"
        )
        for race_id, start_time, competition_id in selected:
            print(
                f"  race_id={race_id} start={start_time} "
                f"competition_id={competition_id}"
            )

        if args.dry_run:
            print("DRY RUN: database unchanged")
            return 0
        if not selected:
            if current >= args.target_races:
                print("Target already satisfied; database unchanged")
                return 0
            raise RuntimeError(
                "Not enough unassigned eligible races to reach the requested target"
            )

        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            """
            INSERT INTO model_validation_races (race_id, validation_cohort)
            VALUES (?, 'chronological_representative')
            """,
            [(int(row[0]),) for row in selected],
        )
        connection.commit()
        final_count = current + len(selected)
        print(f"Updated chronological cohort: {final_count} races")
        if final_count < args.target_races:
            print(
                f"WARNING: only {final_count} eligible races are available; "
                f"requested {args.target_races}"
            )
        return 0
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
