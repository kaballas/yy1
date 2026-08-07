#!/usr/bin/env python3
"""Evaluate trained TabFM top-3 classification checkpoints.

This is the race-model equivalent of the generic TabFM OpenML classification
example. It evaluates each checkpoint using its saved feature schema,
preprocessing statistics and causal context contract, then compares its race
ranking performance with the fluc2 market baseline.

Only load checkpoints you trust. PyTorch checkpoints may contain pickled data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.constants import TRAINING_ROWS_VIEW, VALIDATION_ROWS_VIEW
from src.database import (
    load_market_fluc2,
    load_rows,
    load_validation_cohorts,
    validate_feature_columns,
)
from src.metrics import probability_metrics, race_top3_metrics
from src.model import TabFM
from src.prediction import market_rank_scores, predict_with_chronological_context
from src.preprocessing import transform, zero_feature_columns
from src.sampling import eligible_query_race_ids_from_context
from src.validation import build_race_indices


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "db/race_runners.sqlite"
DEFAULT_MODELS_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        help=(
            "Checkpoint to test. Repeat for multiple models. When omitted, all "
            "*.pt files in --models-dir are tested."
        ),
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--cohort",
        choices=(
            "all",
            "chronological_representative",
            "market_miss_stress",
            "legacy_combined",
        ),
        default="chronological_representative",
        help="Validation cohort to test. Default: chronological_representative.",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=0,
        help="Test only the most recent N eligible races. Use 0 for all races.",
    )
    parser.add_argument(
        "--context-races",
        type=int,
        default=None,
        help="Override the checkpoint's saved number of context races.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for row-level classification metrics.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable results.",
    )
    args = parser.parse_args()

    if args.max_races < 0:
        parser.error("--max-races cannot be negative")
    if args.context_races is not None and args.context_races < 1:
        parser.error("--context-races must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")
    return args


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = args.checkpoint or sorted(args.models_dir.glob("*.pt"))
    paths = [path.resolve() for path in paths]
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found. Supply --checkpoint or add *.pt files to "
            f"{args.models_dir.resolve()}."
        )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint files not found: {missing}")
    return paths


def complete_race_mask(y: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Keep complete top-3 races with at least one negative runner."""
    keep = np.zeros(len(y), dtype=bool)
    for race_id in np.unique(race_ids):
        indices = np.flatnonzero(race_ids == race_id)
        targets = y[indices]
        if (
            len(indices) >= 4
            and np.isin(targets, (0, 1)).all()
            and int(targets.sum()) == 3
        ):
            keep[indices] = True
    return keep


def context_window(bundle: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    value = bundle.get(
        "validation_context_races_per_prediction",
        bundle.get("context_races_per_step"),
    )
    if value is None:
        manifest = bundle.get("context_manifest_race_ids", [])
        value = len(manifest)
    value = int(value)
    if value < 1:
        raise ValueError("Checkpoint does not contain a valid context-race window")
    return value


def row_classification_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    prediction = probability >= threshold
    positive = target == 1
    negative = ~positive

    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & negative))
    true_negative = int(np.sum(~prediction & negative))
    false_negative = int(np.sum(~prediction & positive))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "threshold": threshold,
        "accuracy": float(np.mean(prediction == positive)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": float(np.mean((probability - target) ** 2)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TabFM, dict[str, Any]]:
    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("Checkpoint must be a TabFM bundle dictionary")
    required = {
        "model_state_dict",
        "model_kwargs",
        "feature_columns",
        "median",
        "scale",
    }
    missing = sorted(required - set(bundle))
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {missing}")
    if bundle.get("label") != "top3_mask":
        raise ValueError(
            f"Checkpoint label must be top3_mask, found {bundle.get('label')!r}"
        )

    model = TabFM(**dict(bundle["model_kwargs"])).to(device)
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    model.eval()
    return model, bundle


def evaluate_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device)
    model, bundle = load_model(checkpoint_path, device)
    features = list(bundle["feature_columns"])
    validate_feature_columns(args.db, features)

    train_x, train_y, train_race_ids, train_times, _ = load_rows(
        args.db, features, TRAINING_ROWS_VIEW
    )
    valid_x, valid_y, valid_race_ids, valid_times, _ = load_rows(
        args.db, features, VALIDATION_ROWS_VIEW
    )

    overlap = sorted(set(map(int, train_race_ids)) & set(map(int, valid_race_ids)))
    if overlap:
        raise ValueError(
            f"Training and validation views overlap on {len(overlap)} races: "
            f"{overlap[:10]}"
        )

    train_mask = complete_race_mask(train_y, train_race_ids)
    valid_mask = complete_race_mask(valid_y, valid_race_ids)
    train_x = train_x[train_mask]
    train_y = train_y[train_mask]
    train_race_ids = train_race_ids[train_mask]
    train_times = train_times[train_mask]

    validation_cohorts, cohort_source = load_validation_cohorts(
        args.db, valid_race_ids
    )
    valid_fluc2 = load_market_fluc2(
        args.db, valid_race_ids, VALIDATION_ROWS_VIEW
    )
    valid_x = valid_x[valid_mask]
    valid_y = valid_y[valid_mask]
    valid_race_ids = valid_race_ids[valid_mask]
    valid_times = valid_times[valid_mask]
    valid_fluc2 = valid_fluc2[valid_mask]
    validation_cohorts = validation_cohorts[valid_mask]

    if args.cohort != "all":
        cohort_mask = validation_cohorts == args.cohort
        if not np.any(cohort_mask):
            available = sorted(set(map(str, validation_cohorts)))
            raise ValueError(
                f"Cohort {args.cohort!r} is empty. Available cohorts: {available}"
            )
        valid_x = valid_x[cohort_mask]
        valid_y = valid_y[cohort_mask]
        valid_race_ids = valid_race_ids[cohort_mask]
        valid_times = valid_times[cohort_mask]
        valid_fluc2 = valid_fluc2[cohort_mask]
        validation_cohorts = validation_cohorts[cohort_mask]

    median = np.asarray(bundle["median"], dtype=np.float32)
    scale = np.asarray(bundle["scale"], dtype=np.float32)
    if median.shape != (len(features),) or scale.shape != (len(features),):
        raise ValueError("Checkpoint preprocessing dimensions do not match its features")
    train_x = transform(train_x, median, scale)
    valid_x = transform(valid_x, median, scale)
    zeroed_features = list(bundle.get("zeroed_features", []))
    train_x = zero_feature_columns(train_x, features, zeroed_features)
    valid_x = zero_feature_columns(valid_x, features, zeroed_features)

    race_time_by_id: dict[int, object] = {}
    for race_id_value, race_time in zip(
        np.concatenate((train_race_ids, valid_race_ids)),
        np.concatenate((train_times, valid_times)),
    ):
        race_id = int(race_id_value)
        previous = race_time_by_id.setdefault(race_id, race_time)
        if previous != race_time:
            raise ValueError(f"Race {race_id} has inconsistent start times")

    training_race_indices = build_race_indices(
        train_race_ids, np.ones(len(train_race_ids), dtype=bool)
    )
    context_races = context_window(bundle, args.context_races)
    candidate_races = list(dict.fromkeys(map(int, valid_race_ids)))
    eligible_races = eligible_query_race_ids_from_context(
        candidate_races,
        list(training_race_indices),
        race_time_by_id,
        context_races,
    )
    if args.max_races:
        eligible_races = sorted(
            eligible_races,
            key=lambda race_id: (race_time_by_id[race_id], race_id),
        )[-args.max_races :]
    if not eligible_races:
        raise ValueError("No validation races have enough earlier training context")

    selected = np.isin(valid_race_ids, eligible_races)
    test_x = valid_x[selected]
    test_y = valid_y[selected]
    test_race_ids = valid_race_ids[selected]
    test_fluc2 = valid_fluc2[selected]
    test_cohorts = validation_cohorts[selected]
    query_race_indices = build_race_indices(
        test_race_ids, np.ones(len(test_race_ids), dtype=bool)
    )

    probability = predict_with_chronological_context(
        model,
        train_x,
        train_y,
        training_race_indices,
        race_time_by_id,
        test_x,
        query_race_indices,
        context_races,
        device,
    )
    model_metrics = probability_metrics(test_y, probability, test_race_ids)
    row_metrics = row_classification_metrics(test_y, probability, args.threshold)

    market_scores = market_rank_scores(test_fluc2)
    market_metrics = race_top3_metrics(test_y, market_scores, test_race_ids)
    market_coverage = float(np.isfinite(market_scores).mean())

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "device": str(device),
        "cohort": args.cohort,
        "cohort_source": cohort_source,
        "cohort_labels_present": sorted(set(map(str, test_cohorts))),
        "features": len(features),
        "context_races_per_prediction": context_races,
        "test_races": int(len(set(map(int, test_race_ids)))),
        "test_rows": int(len(test_y)),
        "model": model_metrics,
        "row_classification": row_metrics,
        "market_fluc2": market_metrics,
        "market_price_coverage": market_coverage,
        "lift_vs_market": {
            "top3_recall": float(
                model_metrics["top3_recall"] - market_metrics["top3_recall"]
            ),
            "exact_top3_set_rate": float(
                model_metrics["exact_top3_set_rate"]
                - market_metrics["exact_top3_set_rate"]
            ),
            "contained_top5_rate": float(
                model_metrics["contained_top5_rate"]
                - market_metrics["contained_top5_rate"]
            ),
        },
    }
    return result


def print_result(result: dict[str, Any]) -> None:
    model = result["model"]
    row = result["row_classification"]
    market = result["market_fluc2"]
    lift = result["lift_vs_market"]

    print("\n" + "=" * 88)
    print(f"CHECKPOINT: {result['checkpoint_name']}")
    print(
        f"cohort={result['cohort']} races={result['test_races']} "
        f"rows={result['test_rows']} features={result['features']} "
        f"context_races={result['context_races_per_prediction']} "
        f"device={result['device']}"
    )
    print("-" * 88)
    print(
        f"{'source':<14} {'top3_recall':>12} {'exact_top3':>12} "
        f"{'inside_top4':>12} {'inside_top5':>12} {'inside_top6':>12}"
    )
    print(
        f"{'model':<14} {model['top3_recall']:>12.5f} "
        f"{model['exact_top3_set_rate']:>12.5f} "
        f"{model['contained_top4_rate']:>12.5f} "
        f"{model['contained_top5_rate']:>12.5f} "
        f"{model['contained_top6_rate']:>12.5f}"
    )
    print(
        f"{'fluc2 market':<14} {market['top3_recall']:>12.5f} "
        f"{market['exact_top3_set_rate']:>12.5f} "
        f"{market['contained_top4_rate']:>12.5f} "
        f"{market['contained_top5_rate']:>12.5f} "
        f"{market['contained_top6_rate']:>12.5f}"
    )
    print("-" * 88)
    print(
        f"ROC_AUC={model['roc_auc']:.5f} log_loss={model['logloss']:.5f} "
        f"Brier={row['brier_score']:.5f} threshold={row['threshold']:.3f}"
    )
    print(
        f"accuracy={row['accuracy']:.5f} precision={row['precision']:.5f} "
        f"recall={row['recall']:.5f} F1={row['f1']:.5f}"
    )
    print(
        f"confusion_matrix TN={row['true_negative']} FP={row['false_positive']} "
        f"FN={row['false_negative']} TP={row['true_positive']}"
    )
    print(
        f"lift_vs_market top3_recall={lift['top3_recall']:+.5f} "
        f"exact_top3={lift['exact_top3_set_rate']:+.5f} "
        f"inside_top5={lift['contained_top5_rate']:+.5f} "
        f"market_coverage={result['market_price_coverage']:.5f}"
    )


def main() -> int:
    args = parse_args()
    results = []
    failures = []

    for checkpoint_path in checkpoint_paths(args):
        try:
            result = evaluate_checkpoint(checkpoint_path, args)
        except Exception as error:  # Continue so one bad model does not hide others.
            failures.append(
                {"checkpoint": str(checkpoint_path), "error": str(error)}
            )
            print(f"\nFAILED {checkpoint_path.name}: {error}")
            continue
        results.append(result)
        print_result(result)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"results": results, "failures": failures}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nsaved_results={args.output_json.resolve()}")

    if not results:
        raise SystemExit("No checkpoint completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
