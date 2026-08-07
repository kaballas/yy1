#!/usr/bin/env python3
"""Compare one TabFM checkpoint across causal context-window sizes.

Each validation race is scored using only complete, eligible training races
that started strictly earlier.  This makes 10/50/100-context comparisons
directly comparable and prevents future-data leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import warnings

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_DB
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)

from predict_race import (
    apply_checkpoint_preprocessing,
    load_model,
    read_feature_columns,
)
from src.constants import TRAINING_ROWS_VIEW, VALIDATION_ROWS_VIEW
from src.database import load_race_number_eligible_ids, load_rows, quote_identifier
from src.metrics import probability_metrics
from src.prediction import predict_with_chronological_context
from src.sampling import eligible_query_race_ids_from_context
from src.validation import build_race_indices, exclude_invalid_races
from src.utilities import parse_iso_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--race-id", type=int,
        help="Evaluate only this finished, labelled race ID.",
    )
    target.add_argument(
        "--competition-id", type=int,
        help="Evaluate every finished, labelled race in this competition.",
    )
    parser.add_argument(
        "--context-races", type=int, nargs="+", default=[10, 50, 100],
        help="Context-window sizes to compare (default: 10 50 100).",
    )
    parser.add_argument(
        "--context-source", choices=("training", "all_finished"), default="training",
        help=(
            "Context pool: training reproduces the training-view contract; "
            "all_finished uses every earlier completed race with a valid target."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-validation-races", type=int, default=0,
        help="Optional chronological cap for a smoke test; 0 evaluates all eligible races.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/context_window_backtest.csv"))
    return parser.parse_args()


def race_times(race_ids: np.ndarray, times: np.ndarray) -> dict[int, object]:
    result: dict[int, object] = {}
    for race_id_value, value in zip(race_ids, times):
        race_id = int(race_id_value)
        previous = result.setdefault(race_id, value)
        if previous != value:
            raise ValueError(f"Race {race_id} has inconsistent start times")
    return result


def load_all_finished_context_rows(
    db_path: Path, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the full historical completed-race pool for alternate context tests."""
    columns = ["race_id", "start_time_iso", *feature_columns, "top3_mask"]
    sql = (
        f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
        "FROM race_runners WHERE status = 'finished' AND top3_mask IN (0, 1) "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(sql).fetchall()
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("No finished labelled rows are available for all_finished context")
    race_ids = np.asarray([row[0] for row in rows], dtype=np.int64)
    times = np.asarray([parse_iso_timestamp(row[1]) for row in rows], dtype=object)
    x = np.asarray(
        [[np.nan if value is None else float(value) for value in row[2:-1]] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([row[-1] for row in rows], dtype=np.int64)
    return x, y, race_ids, times


def load_finished_target_race(
    db_path: Path, race_id: int, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one completed, labelled target without requiring validation membership."""
    columns = ["race_id", "start_time_iso", *feature_columns, "top3_mask"]
    sql = (
        f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
        "FROM race_runners WHERE race_id = ? AND status = 'finished' "
        "AND top3_mask IN (0, 1) ORDER BY start_time_iso, race_id, runner_number"
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(sql, (race_id,)).fetchall()
    finally:
        connection.close()
    if not rows:
        raise SystemExit(
            f"Race {race_id} is not a finished race with complete binary top3_mask labels."
        )
    return (
        np.asarray([[np.nan if value is None else float(value) for value in row[2:-1]] for row in rows], dtype=np.float32),
        np.asarray([row[-1] for row in rows], dtype=np.int64),
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([parse_iso_timestamp(row[1]) for row in rows], dtype=object),
    )


def load_finished_competition_targets(
    db_path: Path, competition_id: int, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all completed, labelled target rows for one competition."""
    columns = ["race_id", "start_time_iso", *feature_columns, "top3_mask"]
    sql = (
        f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
        "FROM race_runners WHERE competition_id = ? AND status = 'finished' "
        "AND top3_mask IN (0, 1) ORDER BY start_time_iso, race_id, runner_number"
    )
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(sql, (competition_id,)).fetchall()
    finally:
        connection.close()
    if not rows:
        raise SystemExit(
            f"Competition {competition_id} has no finished races with binary top3_mask labels."
        )
    return (
        np.asarray([[np.nan if value is None else float(value) for value in row[2:-1]] for row in rows], dtype=np.float32),
        np.asarray([row[-1] for row in rows], dtype=np.int64),
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([parse_iso_timestamp(row[1]) for row in rows], dtype=object),
    )


def print_target_runner_ranking(
    db_path: Path, race_id: int, probabilities: np.ndarray, labels: np.ndarray, context_races: int
) -> None:
    """Print the model's runner ranking for one focused target race."""
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT runner_number, runner_name, fluc2, finish_place "
            "FROM race_runners WHERE race_id = ? AND status = 'finished' "
            "AND top3_mask IN (0, 1) ORDER BY runner_number",
            (race_id,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != len(probabilities) or len(rows) != len(labels):
        raise RuntimeError("Target runner display rows do not align with prediction rows")
    ranking = pd.DataFrame(rows, columns=["runner_number", "runner_name", "fluc2", "finish_place"])
    ranking["actual_top3"] = labels.astype(int)
    ranking["model_score"] = probabilities
    ranking = ranking.sort_values(
        ["model_score", "runner_number"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "predicted_rank", np.arange(1, len(ranking) + 1))
    print(f"\nRUNNER PREDICTIONS race_id={race_id} context_races={context_races}")
    print(ranking.to_string(index=False))


def main() -> int:
    args = parse_args()
    windows = sorted(set(args.context_races))
    if not windows or windows[0] < 1:
        raise SystemExit("--context-races must contain positive integers")
    if args.max_validation_races < 0:
        raise SystemExit("--max-validation-races must be zero or positive")
    device = torch.device(args.device)
    model, metadata = load_model(args.model, device, strict=True)
    feature_columns = read_feature_columns(
        argparse.Namespace(feature_columns=None, feature_columns_file=None), metadata
    )

    train_x, train_y, train_ids, train_times, _ = load_rows(
        args.db, feature_columns, TRAINING_ROWS_VIEW
    )
    if args.race_id is None and args.competition_id is None:
        valid_x, valid_y, valid_ids, valid_times, _ = load_rows(
            args.db, feature_columns, VALIDATION_ROWS_VIEW
        )
    elif args.race_id is not None:
        valid_x, valid_y, valid_ids, valid_times = load_finished_target_race(
            args.db, args.race_id, feature_columns
        )
    else:
        valid_x, valid_y, valid_ids, valid_times = load_finished_competition_targets(
            args.db, args.competition_id, feature_columns
        )
    if args.context_source == "all_finished":
        train_x, train_y, train_ids, train_times = load_all_finished_context_rows(
            args.db, feature_columns
        )
    if (
        args.context_source == "training"
        and args.race_id is None
        and args.competition_id is None
        and set(map(int, train_ids)).intersection(map(int, valid_ids))
    ):
        raise SystemExit("Training and validation views overlap on race IDs")

    minimum = metadata.get("optimizer_min_race_number")
    train_mask = np.ones(len(train_ids), dtype=bool)
    if args.context_source == "training" and minimum is not None:
        eligible = load_race_number_eligible_ids(args.db, int(minimum))
        train_mask &= np.isin(train_ids, list(eligible))
    train_mask, _ = exclude_invalid_races(train_y, train_ids, train_mask, "Training pool")
    valid_mask = np.ones(len(valid_ids), dtype=bool)
    valid_mask, _ = exclude_invalid_races(valid_y, valid_ids, valid_mask, "Validation set")

    train_indices = build_race_indices(train_ids, train_mask)
    valid_indices_all = build_race_indices(valid_ids, valid_mask)
    context_pool_race_count = len(train_indices)
    validation_race_count = len(valid_indices_all)
    target_scope = (
        f"race_id={args.race_id}" if args.race_id is not None
        else f"competition_id={args.competition_id}" if args.competition_id is not None
        else "validation_view"
    )
    print(
        "RACE AVAILABILITY "
        f"context_source={args.context_source} "
        f"target_scope={target_scope} "
        f"context_pool_races={context_pool_race_count} "
        f"validation_races={validation_race_count}",
        flush=True,
    )
    times = race_times(
        np.concatenate((train_ids, valid_ids)), np.concatenate((train_times, valid_times))
    )
    train_x = apply_checkpoint_preprocessing(train_x, feature_columns, metadata)
    valid_x = apply_checkpoint_preprocessing(valid_x, feature_columns, metadata)

    rows: list[dict[str, float | int | str]] = []
    all_validation_races = list(valid_indices_all)
    earlier_context_counts = {
        race_id: sum(times[context_race_id] < times[race_id] for context_race_id in train_indices)
        for race_id in all_validation_races
    }
    if args.race_id is not None:
        print(
            f"TARGET CONTEXT AVAILABILITY race_id={args.race_id} "
            f"earlier_context_races={earlier_context_counts[args.race_id]} "
            f"total_context_pool_races={context_pool_race_count}",
            flush=True,
        )
    for window in windows:
        context_eligible_validation = eligible_query_race_ids_from_context(
            all_validation_races, list(train_indices), times, window
        )
        eligible_validation = sorted(
            context_eligible_validation, key=lambda race_id: (times[race_id], race_id)
        )
        if args.max_validation_races:
            eligible_validation = eligible_validation[:args.max_validation_races]
        if not eligible_validation:
            print(
                f"context_races={window} skipped=no eligible validation races "
                f"available_earlier_context_races={earlier_context_counts.get(args.race_id, 'varies')}",
                flush=True,
            )
            continue
        query_indices = {race_id: valid_indices_all[race_id] for race_id in eligible_validation}
        print(
            f"BACKTEST context_races={window} validation_races={len(query_indices)} "
            f"skipped_early={len(all_validation_races) - len(context_eligible_validation)}",
            flush=True,
        )
        probabilities = predict_with_chronological_context(
            model, train_x, train_y, train_indices, times,
            valid_x, query_indices, window, device,
        )
        query_mask = np.isin(valid_ids, eligible_validation)
        if args.race_id is not None:
            print_target_runner_ranking(
                args.db, args.race_id, probabilities[query_mask], valid_y[query_mask], window
            )
        metrics = probability_metrics(valid_y[query_mask], probabilities[query_mask], valid_ids[query_mask])
        row = {
            "model": str(args.model.resolve()),
            "context_source": args.context_source,
            "target_scope": target_scope,
            "context_pool_races": context_pool_race_count,
            "validation_races_available": validation_race_count,
            "context_races": window,
            "validation_races": int(metrics["complete_races"]),
            "skipped_early_validation_races": len(all_validation_races) - len(context_eligible_validation),
            **metrics,
        }
        rows.append(row)
        print(
            f"RESULT context_races={window} top3_recall={metrics['top3_recall']:.4f} "
            f"contained_top5={metrics['contained_top5_rate']:.4f} "
            f"auc={metrics['roc_auc']:.4f} logloss={metrics['logloss']:.5f}",
            flush=True,
        )
    if not rows:
        raise SystemExit("No requested context window had scoreable validation races")
    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(f"Saved {len(results)} context-window results to {args.output}")
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
