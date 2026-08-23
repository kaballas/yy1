#!/usr/bin/env python3
"""Pit every saved single-race analysis model against one finished race date."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
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
    parser.add_argument(
        "--feature-output-csv",
        type=Path,
        help="Feature leaderboard CSV; defaults inside --models-dir.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Number of leading feature results printed (default: 20).",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path("winner_ranker_features.json"),
        help="Feature manifest updated with the selected top-feature model.",
    )
    parser.add_argument(
        "--top-feature-model-name",
        default="top3",
        help="Model group created/replaced in --feature-manifest (default: top3).",
    )
    parser.add_argument(
        "--no-update-feature-manifest",
        action="store_true",
        help="Calculate feature results without updating winner_ranker_features.json.",
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


def feature_wars_summary(
    leaderboard: pd.DataFrame,
    entries: list[dict[str, Any]],
) -> pd.DataFrame:
    """Attribute model-wars performance to each configured input feature."""
    features_by_model = {
        str(entry["name"]): list(dict.fromkeys(map(str, entry["features"])))
        for entry in entries
    }
    rows: list[dict[str, Any]] = []
    for model_row in leaderboard.itertuples(index=False):
        for feature in features_by_model.get(str(model_row.model), []):
            rows.append({
                "feature": feature,
                "model": str(model_row.model),
                "races_tested": int(model_row.races_tested),
                "is_winner_1_count": int(model_row.is_winner_1_count),
                "winner_top3_count": int(model_row.winner_top3_count),
                "is_winner_1_pct": float(model_row.is_winner_1_pct),
                "top3_pct": float(model_row.top3_pct),
                "mrr": float(model_row.mrr),
            })
    columns = [
        "feature_rank", "feature", "models_using_feature", "model_race_tests",
        "is_winner_1_count", "winner_top3_count", "is_winner_1_pct",
        "top3_pct", "mean_model_is_winner_1_pct", "mean_model_top3_pct",
        "mean_model_mrr", "best_model", "best_model_top3_pct",
        "best_model_is_winner_1_pct",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    expanded = pd.DataFrame(rows)
    summary = expanded.groupby("feature", as_index=False, sort=False).agg(
        models_using_feature=("model", "nunique"),
        model_race_tests=("races_tested", "sum"),
        is_winner_1_count=("is_winner_1_count", "sum"),
        winner_top3_count=("winner_top3_count", "sum"),
        mean_model_is_winner_1_pct=("is_winner_1_pct", "mean"),
        mean_model_top3_pct=("top3_pct", "mean"),
        mean_model_mrr=("mrr", "mean"),
    )
    summary["is_winner_1_pct"] = (
        100.0 * summary["is_winner_1_count"] / summary["model_race_tests"]
    )
    summary["top3_pct"] = (
        100.0 * summary["winner_top3_count"] / summary["model_race_tests"]
    )
    best = expanded.sort_values(
        ["feature", "top3_pct", "is_winner_1_pct", "mrr", "model"],
        ascending=[True, False, False, False, True],
        kind="stable",
    ).groupby("feature", as_index=False, sort=False).head(1)
    best = best.loc[:, [
        "feature", "model", "top3_pct", "is_winner_1_pct",
    ]].rename(columns={
        "model": "best_model",
        "top3_pct": "best_model_top3_pct",
        "is_winner_1_pct": "best_model_is_winner_1_pct",
    })
    summary = summary.merge(best, on="feature", how="left", validate="one_to_one")
    summary = summary.sort_values(
        ["top3_pct", "is_winner_1_pct", "mean_model_mrr", "models_using_feature", "feature"],
        ascending=[False, False, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    summary.insert(0, "feature_rank", np.arange(1, len(summary) + 1))
    return summary.loc[:, columns]


def update_feature_manifest_model(
    manifest_path: Path,
    model_name: str,
    features: list[str],
    *,
    evaluation_date: str,
    feature_leaderboard_path: Path,
) -> Path:
    """Atomically create or replace one model group in the feature manifest."""
    resolved = manifest_path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Feature manifest does not exist: {resolved}")
    if not model_name.strip():
        raise ValueError("--top-feature-model-name must not be empty")
    selected = list(dict.fromkeys(map(str, features)))
    if not selected:
        raise ValueError("Cannot update feature manifest with no selected features")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Feature manifest must contain a JSON object: {resolved}")
    models = payload.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError("Feature manifest models must be a JSON object")
    models[model_name] = {
        "features": selected,
        "selection": {
            "method": "model_wars_weighted_top3_pct",
            "evaluation_date_utc": evaluation_date,
            "feature_count": len(selected),
            "feature_leaderboard": str(feature_leaderboard_path.resolve()),
        },
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".tmp", dir=resolved.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, resolved)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return resolved


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
            winner_rank = float(pd.Series(scores).rank(
                method="average", ascending=False
            ).iloc[winner_index])
            top_positions = np.flatnonzero(np.isclose(
                scores, scores.max(), rtol=0.0, atol=1e-12
            ))
            selected_runner_number = (
                int(race.iloc[int(top_positions[0])]["runner_number"])
                if len(top_positions) == 1 else pd.NA
            )
            model_rows.append({
                "model": entry["name"],
                "trained_on_race_id": entry["trained_on_race_id"],
                "race_id": int(race_id),
                "competition_id": int(race.iloc[0]["competition_id"]),
                "race_number": int(race.iloc[0]["race_number"]),
                "winner_runner_number": int(race.iloc[winner_index]["runner_number"]),
                "selected_runner_number": selected_runner_number,
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
    if args.top_features < 1:
        raise ValueError("--top-features must be positive")
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
    feature_leaderboard = feature_wars_summary(leaderboard, entries)
    leaderboard_path = (
        args.output_csv or models_dir / f"model_wars_{args.date}.csv"
    ).resolve()
    results_path = (
        args.race_results_csv
        or models_dir / f"model_wars_{args.date}_race_results.csv"
    ).resolve()
    feature_path = (
        args.feature_output_csv
        or models_dir / f"model_wars_{args.date}_features.csv"
    ).resolve()
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(leaderboard_path, index=False)
    results.to_csv(results_path, index=False)
    feature_leaderboard.to_csv(feature_path, index=False)
    updated_manifest: Path | None = None
    selected_feature_count = min(args.top_features, len(feature_leaderboard))
    if not args.no_update_feature_manifest:
        updated_manifest = update_feature_manifest_model(
            args.feature_manifest,
            args.top_feature_model_name,
            feature_leaderboard.head(args.top_features)["feature"].tolist(),
            evaluation_date=args.date,
            feature_leaderboard_path=feature_path,
        )
    print("MODEL WARS")
    print(
        f"date_utc={args.date} models={len(entries):,} "
        f"races={results['race_id'].nunique():,}\n"
        f"leaderboard={leaderboard_path}\n"
        f"race_results={results_path}\n"
        f"feature_leaderboard={feature_path}"
        + (
            f"\nfeature_manifest_updated={updated_manifest} "
            f"model_group={args.top_feature_model_name} "
            f"features={selected_feature_count}"
            if updated_manifest is not None
            else "\nfeature_manifest_updated=no"
        )
    )
    print(leaderboard.to_string(
        index=False, float_format=lambda value: f"{value:.2f}"
    ))
    print(f"\nTOP {selected_feature_count} FEATURES")
    print(feature_leaderboard.head(args.top_features).to_string(
        index=False, float_format=lambda value: f"{value:.2f}"
    ))


if __name__ == "__main__":
    main()
