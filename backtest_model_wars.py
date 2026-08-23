#!/usr/bin/env python3
"""Pit every saved single-race analysis model against one finished race date."""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_DB


def parse_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=parse_date)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--models-dir", type=Path, default=Path("models_test"))
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Leaderboard CSV; defaults inside --models-dir.",
    )
    parser.add_argument(
        "--race-results-csv",
        type=Path,
        help="Model-by-race results CSV; defaults inside --models-dir.",
    )
    return parser.parse_args()


def load_model_entries(models_dir: Path) -> list[dict[str, Any]]:
    """Load model paths and exact input features from manifest or sidecars."""
    resolved = models_dir.resolve()
    manifest_path = resolved / "per_race_models_manifest.json"
    entries: list[dict[str, Any]] = []
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("models", []):
            details = item.get("details", {})
            features = details.get("input_features")
            features_file = Path(str(item.get("features_file", "")))
            if not features and features_file.is_file():
                sidecar = json.loads(features_file.read_text(encoding="utf-8"))
                features = sidecar.get("input_features", [])
            configured_paths = item.get("models")
            if not isinstance(configured_paths, list) or not configured_paths:
                configured_paths = [item.get("model", "")]
            model_paths = [Path(str(path)) for path in configured_paths]
            if model_paths and all(path.is_file() for path in model_paths) and features:
                entries.append({
                    "name": str(item.get("name", model_paths[0].stem)),
                    "model": model_paths[0],
                    "models": model_paths,
                    "features": list(features),
                    "trained_on_race_id": int(item.get("trained_on_race_id", 0)),
                })
    if entries:
        return entries
    for sidecar_path in sorted(resolved.glob("race_*_features.json")):
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        model_path = Path(str(sidecar.get("model", "")))
        features = list(sidecar.get("input_features", []))
        if model_path.is_file() and features:
            entries.append({
                "name": model_path.stem,
                "model": model_path,
                "models": [model_path],
                "features": features,
                "trained_on_race_id": int(sidecar.get("race_id", 0)),
            })
    if not entries:
        raise ValueError(f"No usable race models found in {resolved}")
    return entries


def model_wars_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual-winner ranks for every competing model."""
    summary = results.groupby(
        ["model", "trained_on_race_id"], as_index=False, sort=False
    ).agg(
        races_tested=("race_id", "nunique"),
        is_winner_1_count=("is_winner_1", "sum"),
        winner_top3_count=("winner_top3", "sum"),
        mean_winner_rank=("winner_rank", "mean"),
        mrr=("winner_rank", lambda ranks: float(np.mean(1.0 / ranks))),
    )
    summary["is_winner_1_pct"] = (
        100.0 * summary["is_winner_1_count"] / summary["races_tested"]
    )
    summary["top3_pct"] = (
        100.0 * summary["winner_top3_count"] / summary["races_tested"]
    )
    return summary.sort_values(
        ["top3_pct", "is_winner_1_pct", "mrr", "model"],
        ascending=[False, False, False, True],
        kind="stable",
        ignore_index=True,
    )


def run_model_wars(
    database: Path,
    exact_date: str,
    entries: list[dict[str, Any]],
    minimum_runners: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from xgboost import XGBRanker
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("xgboost is required: pip install xgboost") from exc
    from src.winner_ranker import (
        database_numeric_columns,
        eligible_races,
        load_training_rows,
        model_feature_matrix,
        rank_percentiles,
        rows_for_races,
    )

    print(
        f"loading_finished_races date_utc={exact_date} database={database.resolve()}",
        flush=True,
    )
    numeric_columns = database_numeric_columns(database)
    finished = load_training_rows(database, numeric_columns)
    times = pd.to_datetime(finished["start_time_iso"], errors="coerce", utc=True)
    if times.isna().any():
        raise ValueError("Database contains invalid start_time_iso values")
    requested = pd.Timestamp(exact_date, tz="UTC")
    dated = finished.loc[times.dt.normalize().eq(requested)].copy()
    races = eligible_races(dated, minimum_runners)
    if races.empty:
        raise ValueError(f"No eligible status=finished races exist on {exact_date}")
    frame = rows_for_races(dated, races["race_id"].astype(int).tolist())
    print(
        f"race_cohort_ready races={len(races):,} runners={len(frame):,} "
        f"models_to_test={len(entries):,}",
        flush=True,
    )
    result_rows: list[dict[str, Any]] = []
    wars_started = time.perf_counter()
    for model_number, entry in enumerate(entries, start=1):
        model_started = time.perf_counter()
        models = []
        for model_path in entry.get("models", [entry["model"]]):
            model = XGBRanker()
            model.load_model(model_path)
            models.append(model)
        model_rows: list[dict[str, Any]] = []
        for race_id, race in frame.groupby("race_id", sort=False):
            race_ids = np.full(len(race), int(race_id), dtype=np.int64)
            matrix = model_feature_matrix(race, entry["features"])
            member_scores = [
                rank_percentiles(
                    np.asarray(model.predict(matrix), dtype=np.float64), race_ids
                )
                for model in models
            ]
            scores = np.mean(np.stack(member_scores), axis=0)
            targets = race["is_winner"].to_numpy(dtype=np.int64)
            winner_index = int(np.flatnonzero(targets == 1)[0])
            order = np.argsort(-scores, kind="stable")
            winner_rank = int(np.flatnonzero(order == winner_index)[0]) + 1
            selected_index = int(order[0])
            model_rows.append({
                "model": entry["name"],
                "trained_on_race_id": entry["trained_on_race_id"],
                "race_id": int(race_id),
                "competition_id": int(race.iloc[0]["competition_id"]),
                "race_number": int(race.iloc[0]["race_number"]),
                "winner_runner_number": int(race.iloc[winner_index]["runner_number"]),
                "selected_runner_number": int(race.iloc[selected_index]["runner_number"]),
                "winner_rank": winner_rank,
                "is_winner_1": int(winner_rank == 1),
                "winner_top3": int(winner_rank <= 3),
            })
        result_rows.extend(model_rows)
        winner_count = sum(row["is_winner_1"] for row in model_rows)
        top3_count = sum(row["winner_top3"] for row in model_rows)
        race_count = len(model_rows)
        print(
            f"model_result={model_number:,}/{len(entries):,} "
            f"model={entry['name']} "
            f"trained_on_race_id={entry['trained_on_race_id']} "
            f"races={race_count:,} "
            f"is_winner_1={winner_count:,} "
            f"is_winner_1_pct={100.0 * winner_count / race_count:.2f}% "
            f"winner_top3={top3_count:,} "
            f"top3_pct={100.0 * top3_count / race_count:.2f}% "
            f"model_seconds={time.perf_counter() - model_started:.2f} "
            f"total_seconds={time.perf_counter() - wars_started:.2f}",
            flush=True,
        )
    results = pd.DataFrame(result_rows)
    return model_wars_summary(results), results


def main() -> None:
    args = parse_args()
    if args.minimum_runners < 2:
        raise ValueError("--minimum-runners must be at least 2")
    models_dir = args.models_dir.resolve()
    entries = load_model_entries(models_dir)
    print(
        "MODEL WARS START\n"
        f"date_utc={args.date} models_to_test={len(entries):,} "
        f"models_dir={models_dir}",
        flush=True,
    )
    leaderboard, results = run_model_wars(
        args.db.resolve(), args.date, entries, args.minimum_runners
    )
    leaderboard_path = (
        args.output_csv or models_dir / f"model_wars_{args.date}.csv"
    ).resolve()
    results_path = (
        args.race_results_csv
        or models_dir / f"model_wars_{args.date}_race_results.csv"
    ).resolve()
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(leaderboard_path, index=False)
    results.to_csv(results_path, index=False)
    print("MODEL WARS")
    print(
        f"date_utc={args.date} models={len(entries):,} "
        f"races={results['race_id'].nunique():,}\n"
        f"leaderboard={leaderboard_path}\n"
        f"race_results={results_path}"
    )
    print(leaderboard.to_string(
        index=False, float_format=lambda value: f"{value:.2f}"
    ))


if __name__ == "__main__":
    main()
