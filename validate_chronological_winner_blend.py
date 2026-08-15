#!/usr/bin/env python3
"""Validate the frozen winner-ranker manifest on a sealed chronological test."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_DB
from src.winner_ranker import (
    chronological_race_split,
    database_numeric_columns,
    eligible_races,
    ensemble_rank_scores,
    group_sizes,
    load_training_rows,
    market_deviation_metrics,
    market_scores,
    model_feature_matrix,
    rank_percentiles,
    rows_for_races,
    validate_ranker_groups,
    winner_metrics,
)
from train_tune_all_finished_winner_ranker import (
    fit_ensemble,
    load_model_feature_sets,
    tune_dynamic_model_blend,
    tune_tree_counts,
)
from train_winner_ranker_pipeline import score_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--feature-manifest", type=Path, default=Path("winner_ranker_features.json")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/winner_ranker_chronological"),
    )
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--tree-count-validation-races", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective", choices=("top1", "mrr", "top3", "composite"),
        default="top1",
    )
    parser.add_argument("--weight-step", type=float, default=0.001)
    parser.add_argument("--minimum-form-weight", type=float, default=0.0)
    parser.add_argument("--skip-feature-update", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def cohort_time_bounds(frame: pd.DataFrame) -> dict[str, str]:
    times = pd.to_datetime(frame["start_time_iso"], errors="raise", utc=True)
    return {"first": times.min().isoformat(), "last": times.max().isoformat()}


def validate_chronology(
    train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame
) -> None:
    cohorts = [train, validation, test]
    race_sets = [set(frame["race_id"]) for frame in cohorts]
    if any(race_sets[left] & race_sets[right]
           for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Chronological train/validation/test race sets overlap")
    bounds = [cohort_time_bounds(frame) for frame in cohorts]
    if not (bounds[0]["last"] < bounds[1]["first"]
            and bounds[1]["last"] < bounds[2]["first"]):
        raise ValueError("Chronological cohorts are not strictly ordered")


def score_models(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    models: dict[str, list[Any]],
) -> dict[str, np.ndarray]:
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    return {
        label: ensemble_rank_scores(
            models[label], model_feature_matrix(frame, features), race_ids
        )
        for label, features in feature_sets.items()
    }


def evaluate_fixed_blend(
    predictions: pd.DataFrame,
    model_labels: list[str],
    selected_weights: dict[str, float],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    weights = np.asarray([selected_weights[label] for label in model_labels])
    matrix = predictions.loc[:, [
        f"{label}_score" for label in model_labels
    ]].to_numpy(dtype=np.float64)
    blend = matrix @ weights
    targets = predictions["is_winner"].to_numpy(dtype=np.int64)
    race_ids = predictions["race_id"].to_numpy(dtype=np.int64)
    metrics = {
        "raw_market_benchmark": winner_metrics(
            targets, predictions["market_score"], race_ids
        ),
        "frozen_validation_blend": winner_metrics(targets, blend, race_ids),
    }
    audit = predictions[[
        "race_id", "runner_number", "is_winner", "market_rank",
    ]].copy()
    audit["frozen_validation_blend_rank"] = (
        pd.Series(blend).groupby(audit["race_id"], sort=False).rank(
            method="first", ascending=False
        ).astype(int)
    )
    return metrics, market_deviation_metrics(audit, "frozen_validation_blend")


def main() -> None:
    args = parse_args()
    if args.validation_races < 1 or args.test_races < 1:
        raise ValueError("validation-races and test-races must be positive")
    database = args.db.resolve()
    manifest = args.feature_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_feature_update:
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("update_derived_racing_features.py")),
            "--db", str(database),
        ], check=True)

    numeric_columns = database_numeric_columns(database)
    frame = load_training_rows(database, numeric_columns)
    race_ids = eligible_races(frame, args.minimum_runners)
    train_ids, validation_ids, test_ids = chronological_race_split(
        race_ids, args.validation_races, args.test_races
    )
    train = rows_for_races(frame, train_ids)
    validation = rows_for_races(frame, validation_ids)
    test = rows_for_races(frame, test_ids)
    validate_chronology(train, validation, test)
    feature_sets = load_model_feature_sets(manifest, numeric_columns)
    model_labels = list(feature_sets)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    print(
        f"frozen_manifest={manifest} sha256={manifest_hash}\n"
        f"eligible_races={len(race_ids):,} train_races={len(train_ids):,} "
        f"validation_races={len(validation_ids):,} sealed_test_races={len(test_ids):,}\n"
        f"train_bounds={json.dumps(cohort_time_bounds(train))}\n"
        f"validation_bounds={json.dumps(cohort_time_bounds(validation))}\n"
        f"sealed_test_bounds={json.dumps(cohort_time_bounds(test))}",
        flush=True,
    )

    train_y = train["is_winner"].to_numpy(dtype=np.int64)
    train_groups = group_sizes(train)
    validate_ranker_groups(train, train_y, train_groups)
    models: dict[str, list[Any]] = {}
    tree_counts: dict[str, list[int]] = {}
    model_paths: dict[str, list[str]] = {}
    for model_index, (label, features) in enumerate(feature_sets.items()):
        seed_offset = model_index * 50_000
        counts = tune_tree_counts(args, label, train, features, seed_offset)
        fitted = fit_ensemble(
            args, label, model_feature_matrix(train, features), train_y,
            train_groups, counts, seed_offset,
        )
        paths: list[str] = []
        for member, model in enumerate(fitted):
            seed = args.seed + seed_offset + member * 1009
            path = output_dir / f"{label}_chronological_train_seed_{seed}.json"
            model.save_model(path)
            paths.append(str(path))
        models[label] = fitted
        tree_counts[label] = counts
        model_paths[label] = paths
        print(f"trained={label} trees={json.dumps(counts)}", flush=True)

    validation_scores = score_models(validation, feature_sets, models)
    validation_race_ids = validation["race_id"].to_numpy(dtype=np.int64)
    validation_scores["market"] = rank_percentiles(
        market_scores(validation), validation_race_ids
    )
    validation_predictions = score_table(
        validation, validation["is_winner"].to_numpy(dtype=np.int64),
        validation_scores,
    )
    selected_weights, sweep = tune_dynamic_model_blend(
        validation_predictions, model_labels, args.weight_step,
        args.objective, args.minimum_form_weight,
    )
    # Weight selection is now frozen. Only after this point is the sealed test
    # scored and its labels passed to metric functions.
    test_scores = score_models(test, feature_sets, models)
    test_race_ids = test["race_id"].to_numpy(dtype=np.int64)
    test_scores["market"] = rank_percentiles(market_scores(test), test_race_ids)
    test_predictions = score_table(
        test, test["is_winner"].to_numpy(dtype=np.int64), test_scores
    )
    validation_metrics, validation_deviation = evaluate_fixed_blend(
        validation_predictions, model_labels, selected_weights
    )
    test_metrics, test_deviation = evaluate_fixed_blend(
        test_predictions, model_labels, selected_weights
    )

    print("selected_validation_weights=" + json.dumps(selected_weights, sort_keys=True))
    for name, metrics in (
        ("VALIDATION (WEIGHT SELECTION)", validation_metrics),
        ("SEALED TEST (NO RETUNING)", test_metrics),
    ):
        print(name)
        print(pd.DataFrame(metrics).T.to_string(float_format=lambda value: f"{value:.5f}"))
    print("SEALED TEST MARKET DEVIATION")
    print(pd.Series(test_deviation).to_string(float_format=lambda value: f"{value:.5f}"))

    validation_path = output_dir / "validation_predictions.csv"
    test_path = output_dir / "sealed_test_predictions.csv"
    sweep_path = output_dir / "validation_weight_sweep.csv"
    result_path = output_dir / "chronological_validation.json"
    validation_predictions.to_csv(validation_path, index=False)
    test_predictions.to_csv(test_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    result = {
        "schema_version": 1,
        "selection_cohort": "chronological_validation",
        "sealed_test_retuned": False,
        "feature_manifest": str(manifest),
        "feature_manifest_sha256": manifest_hash,
        "model_labels": model_labels,
        "selected_weights": selected_weights,
        "tree_counts_selected_inside_training": tree_counts,
        "model_paths": model_paths,
        "race_counts": {
            "train": len(train_ids), "validation": len(validation_ids),
            "sealed_test": len(test_ids),
        },
        "time_bounds": {
            "train": cohort_time_bounds(train),
            "validation": cohort_time_bounds(validation),
            "sealed_test": cohort_time_bounds(test),
        },
        "validation_metrics": validation_metrics,
        "validation_market_deviation": validation_deviation,
        "sealed_test_metrics": test_metrics,
        "sealed_test_market_deviation": test_deviation,
    }
    result_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_result={result_path}", flush=True)


if __name__ == "__main__":
    main()
