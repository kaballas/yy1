"""TabFM training progress helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np


def load_progress_race(
    db_path: Path, race_id: int, feature_columns: list[str]
) -> tuple[np.ndarray, list[sqlite3.Row]]:
    """Load one race for unchanged per-epoch progress reporting."""
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = ", ".join(f'"{column}"' for column in feature_columns)
        rows = connection.execute(
            f"""
            SELECT runner_number, runner_name, fluc2, is_trainable, top3_mask,
                   finish_place,
                   {columns}
            FROM race_runners
            WHERE race_id = ?
            ORDER BY COALESCE(runner_number, 999999), selection_id
            """,
            (race_id,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"--progress-race-id {race_id} was not found")
    x = np.asarray(
        [
            [np.nan if row[column] is None else float(row[column]) for column in feature_columns]
            for row in rows
        ],
        dtype=np.float32,
    )
    return x, rows


def print_progress_race(
    race_id: int, rows: list[sqlite3.Row], probability: np.ndarray
) -> None:
    """Print the ranked progress-race table."""
    order = np.argsort(-probability, kind="stable")
    print(f"PROGRESS RACE {race_id}", flush=True)
    print(
        "Rank  No.  Runner                           ModelScore  Fluc2  Actual  Finish",
        flush=True,
    )
    for rank, index in enumerate(order, start=1):
        row = rows[int(index)]
        runner_number = "-" if row["runner_number"] is None else str(row["runner_number"])
        runner_name = row["runner_name"] or "-"
        fluc2 = "-" if row["fluc2"] is None else f"{row['fluc2']:.2f}"
        result = "TOP3" if row["top3_mask"] == 1 else ""
        finish = "-" if row["finish_place"] is None else str(row["finish_place"])
        print(
            f"{rank:>4}  {runner_number:>3}  {runner_name[:31]:<31} "
            f"{probability[index]:>10.6f}  {fluc2:>6}  "
            f"{result:>6}  {finish:>6}",
            flush=True,
        )
