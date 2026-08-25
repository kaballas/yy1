#!/usr/bin/env python3
"""Tune graph winner models on one date window, refit, and rank one race."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from evaluate_graph_winner_features import (
    DEFAULT_MANIFEST,
    graph_experiment_feature_sets,
    load_baseline_features,
    load_joined_rows,
    quote_identifier,
    selection_eval_metrics,
    train_experiment_ensemble,
    validate_identifier,
)
from src.advanced_racing_features import race_relative_runner_mask
from src.config import DEFAULT_DB
from src.winner_ranker import (
    eligible_races,
    ensemble_rank_scores,
    group_sizes,
    model_feature_matrix,
    rows_for_races,
    validate_ranker_groups,
)
from train_winner_ranker_pipeline import model_parameters


ROOT = Path(__file__).resolve().parent


def utc_timestamp(value: str, option: str, timezone: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {option}: {value!r}") from exc
    if parsed.tzinfo is None:
        try:
            parsed = parsed.tz_localize(timezone)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"Invalid --timezone: {timezone!r}") from exc
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def load_target_race(
    database: Path,
    graph_table: str,
    race_id: int,
    required_features: Sequence[str],
) -> pd.DataFrame:
    """Load one finished or complete live race through the source-row graph join."""
    graph_table = validate_identifier(graph_table, "graph table")
    metadata = (
        "race_id", "start_time_iso", "competition_id", "competition_name",
        "race_number", "race_name", "runner_number", "runner_name",
        "career_starts", "status", "runner_mask", "is_winner",
        "source_betting_status", "active_field_size",
    )
    graph_features = [name for name in required_features if name.startswith("graph_")]
    race_features = [
        name
        for name in required_features
        if not name.startswith("graph_") and name not in metadata
    ]
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        race_schema = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("race_runners")')
        }
        graph_schema = {
            str(row[1])
            for row in connection.execute(
                f"PRAGMA table_info({quote_identifier(graph_table)})"
            )
        }
        missing_race = sorted(set([*metadata, *race_features]) - race_schema)
        missing_graph = sorted(set(graph_features) - graph_schema)
        if missing_race:
            raise ValueError("race_runners is missing: " + ", ".join(missing_race))
        if missing_graph:
            raise ValueError(f"{graph_table} is missing: " + ", ".join(missing_graph))
        selected = [
            "r.rowid AS source_rowid",
            *(f"r.{quote_identifier(name)}" for name in metadata),
            *(f"r.{quote_identifier(name)}" for name in race_features),
            "g.snapshot_date",
            *(f"g.{quote_identifier(name)}" for name in graph_features),
        ]
        frame = pd.read_sql_query(
            f"SELECT {', '.join(selected)} FROM race_runners AS r "
            f"JOIN {quote_identifier(graph_table)} AS g "
            "ON g.source_rowid = r.rowid WHERE r.race_id = ? "
            "ORDER BY r.runner_number",
            connection,
            params=(race_id,),
        )
    if frame.empty:
        raise ValueError(
            f"Race {race_id} has no rows in {graph_table}; rebuild graph features "
            "after importing this race"
        )
    if frame["source_rowid"].duplicated().any():
        raise ValueError("Graph join produced duplicate target rows")
    start = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    snapshot = pd.to_datetime(frame["snapshot_date"], utc=True, errors="coerce")
    if start.isna().any() or snapshot.isna().any() or snapshot.ge(start).any():
        raise ValueError("Target race has an invalid or non-causal graph snapshot")
    active = race_relative_runner_mask(frame)
    target = frame.loc[active].reset_index(drop=True)
    if target.empty:
        raise ValueError(
            f"Race {race_id} is not finished and does not yet have a verified "
            "complete PRICED/OFF field"
        )
    return target


def refit_ensembles(
    args: argparse.Namespace,
    feature_sets: dict[str, list[str]],
    training: pd.DataFrame,
    selected_trees: dict[str, list[int]],
    output_dir: Path,
) -> dict[str, list[XGBRanker]]:
    targets = training["is_winner"].to_numpy(dtype=np.int64)
    groups = group_sizes(training)
    validate_ranker_groups(training, targets, groups)
    result: dict[str, list[XGBRanker]] = {}
    for label, features in feature_sets.items():
        matrix = model_feature_matrix(training, features)
        members: list[XGBRanker] = []
        for member, trees in enumerate(selected_trees[label]):
            seed = args.seed + member * 1009
            model = XGBRanker(**model_parameters(args, seed, trees))
            model.fit(matrix, targets, group=groups, verbose=False)
            path = output_dir / f"{label}_one_month_seed_{seed}.json"
            model.save_model(path)
            members.append(model)
            print(f"refit={label} member={member + 1} trees={trees} path={path}")
        result[label] = members
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--graph-table", default="graph_features")
    parser.add_argument("--features-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-model", default="c2")
    parser.add_argument(
        "--models", nargs="+",
        choices=("graph_a", "graph_b", "graph_c", "graph_d"),
        default=["graph_a", "graph_b", "graph_c", "graph_d"],
    )
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument(
        "--timezone", default="Australia/Sydney",
        help="Timezone for date boundaries without an explicit offset.",
    )
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument("--validation-races", type=int, default=100)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument(
        "--selection-objective", choices=("top1", "top3", "map"), default="top1"
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "graph_one_month_prediction",
    )
    args = parser.parse_args()
    for name in (
        "race_id", "validation_races", "minimum_runners", "ensemble_size",
        "max_estimators", "early_stopping_rounds", "jobs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.models = list(dict.fromkeys(args.models))
    return args


def main() -> int:
    args = parse_args()
    train_start = utc_timestamp(args.train_start, "train-start", args.timezone)
    train_end = utc_timestamp(args.train_end, "train-end", args.timezone)
    if train_end <= train_start:
        raise ValueError("--train-end must be later than --train-start")

    baseline = load_baseline_features(args.features_json.resolve(), args.baseline_model)
    all_feature_sets = graph_experiment_feature_sets(baseline)
    feature_sets = {label: all_feature_sets[label] for label in args.models}
    required = list(dict.fromkeys(
        feature for features in feature_sets.values() for feature in features
    ))
    finished = load_joined_rows(args.db.resolve(), args.graph_table, required)
    starts = pd.to_datetime(finished["start_time_iso"], utc=True, errors="raise")
    window = finished.loc[starts.ge(train_start) & starts.lt(train_end)].copy()
    races = eligible_races(window, args.minimum_runners)
    if len(races) <= args.validation_races:
        raise ValueError(
            f"Training window has {len(races):,} eligible races, which is not more "
            f"than --validation-races={args.validation_races:,}"
        )
    race_ids = races["race_id"].astype(int).tolist()
    tuning_ids = race_ids[:-args.validation_races]
    validation_ids = race_ids[-args.validation_races:]
    tuning = rows_for_races(window, tuning_ids)
    validation = rows_for_races(window, validation_ids)
    full_training = rows_for_races(window, race_ids)
    target = load_target_race(
        args.db.resolve(), args.graph_table, args.race_id, required
    )
    target_start = pd.to_datetime(target["start_time_iso"].iloc[0], utc=True)
    if target_start < train_end:
        raise ValueError(
            f"Target starts at {target_start}; it must be at or after train-end "
            f"{train_end}"
        )
    for cohort in (tuning, validation, full_training):
        validate_ranker_groups(
            cohort,
            cohort["is_winner"].to_numpy(dtype=np.int64),
            group_sizes(cohort),
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"training_window=[{train_start}, {train_end}) "
        f"eligible_races={len(race_ids):,} tuning_races={len(tuning_ids):,} "
        f"validation_races={len(validation_ids):,} target_race={args.race_id} "
        f"target_start={target_start} active_runners={len(target):,}"
    )

    selected_trees: dict[str, list[int]] = {}
    for label, features in feature_sets.items():
        _, trees = train_experiment_ensemble(
            args, label, features, tuning, validation, output_dir
        )
        selected_trees[label] = trees
    models = refit_ensembles(
        args, feature_sets, full_training, selected_trees, output_dir
    )

    target_race_ids = target["race_id"].to_numpy(dtype=np.int64)
    output = target.loc[:, ["runner_number", "runner_name"]].copy()
    finished_target = (
        target["status"].astype("string").str.casefold().eq("finished").all()
        and pd.to_numeric(target["is_winner"], errors="coerce").eq(1).sum() == 1
    )
    if finished_target:
        output["actual_winner"] = pd.to_numeric(
            target["is_winner"], errors="coerce"
        ).eq(1)
    rank_columns: list[str] = []
    for label, members in models.items():
        scores = ensemble_rank_scores(
            members, model_feature_matrix(target, feature_sets[label]), target_race_ids
        )
        output[f"{label}_score"] = scores
        rank_column = f"{label}_rank"
        output[rank_column] = pd.Series(scores).rank(
            method="average", ascending=False
        ).to_numpy()
        rank_columns.append(rank_column)
    output["average_graph_rank"] = output[rank_columns].mean(axis=1)
    output = output.sort_values(
        ["average_graph_rank", rank_columns[0], "runner_number"], kind="stable"
    ).reset_index(drop=True)
    output.insert(0, "display", np.arange(1, len(output) + 1))
    prediction_path = output_dir / f"race_{args.race_id}_prediction.csv"
    output.to_csv(prediction_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "database": str(args.db.resolve()),
        "graph_table": args.graph_table,
        "training_window": {"start": str(train_start), "end": str(train_end)},
        "training_races": len(race_ids),
        "validation_races_for_tree_selection": len(validation_ids),
        "target_race_id": args.race_id,
        "target_start": str(target_start),
        "feature_sets": feature_sets,
        "selected_trees": selected_trees,
        "selection_metric": selection_eval_metrics(args.selection_objective)[-1],
        "prediction_csv": str(prediction_path),
    }
    (output_dir / "one_month_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nONE-MONTH GRAPH MODEL RANKING")
    print(
        output.drop(columns=[column for column in output if column.endswith("_score")])
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    if finished_target:
        winner = output.loc[output["actual_winner"]].iloc[0]
        winner_ranks = {
            label: float(winner[f"{label}_rank"]) for label in feature_sets
        }
        print(
            f"actual_winner={int(winner['runner_number'])} {winner['runner_name']} "
            f"model_ranks={json.dumps(winner_ranks, sort_keys=True)}"
        )
    print(f"prediction={prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
