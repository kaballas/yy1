#!/usr/bin/env python3
"""Tune a current-market-free winner blend from saved all-finished OOF scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from train_tune_all_finished_winner_ranker import dynamic_blend_analysis
from src.winner_ranker import current_market_free_model_labels


DEFAULT_OUTPUT_DIR = Path("outputs/winner_ranker_all_finished")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "winner_ranker_bundle.json",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "all_finished_oof_predictions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "market_free_blend.json",
    )
    parser.add_argument(
        "--objective", choices=("top1", "mrr", "top3", "composite"), default="top1"
    )
    parser.add_argument("--weight-step", type=float, default=0.001)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Required artifact does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact must contain a JSON object: {resolved}")
    return payload


def main() -> None:
    args = parse_args()
    if args.weight_step <= 0 or args.weight_step > 1:
        raise ValueError("weight-step must be in the interval (0, 1]")
    bundle = load_json(args.bundle)
    model_features = bundle.get("model_features")
    if not isinstance(model_features, dict):
        raise ValueError("Bundle does not contain model_features")
    labels = current_market_free_model_labels({
        str(label): list(features)
        for label, features in model_features.items()
        if isinstance(label, str) and isinstance(features, list)
    })
    if not labels:
        raise ValueError(
            "Bundle has no models without target-race market inputs; add a "
            "current-market-free feature group and retrain it first"
        )

    predictions_path = args.predictions.resolve()
    if not predictions_path.is_file():
        raise ValueError(f"Required artifact does not exist: {predictions_path}")
    oof = pd.read_csv(predictions_path)
    required = {
        "race_id", "runner_number", "is_winner", "market_score", "market_rank",
        *(f"{label}_score" for label in labels),
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError("OOF predictions are missing: " + ", ".join(missing))

    selected, _, model_metrics, metrics, deviation = dynamic_blend_analysis(
        oof, labels, args.weight_step, args.objective, minimum_form_weight=0.0
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "blend": "all_finished_current_market_free_model_groups",
        "objective": args.objective,
        "target_race_market_inputs": False,
        "selected_weights": selected,
        "model_labels": labels,
        "model_oof_metrics": model_metrics,
        "oof_metrics": {
            "current_market_free_blend": metrics["tuned_dynamic_blend"],
            "raw_market_benchmark": metrics["raw_market_benchmark"],
        },
        "oof_market_deviation": deviation,
        "selection_cohort": "all_eligible_finished_races_grouped_oof",
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("CURRENT-MARKET-FREE WINNER BLEND")
    print("model_labels=" + ",".join(labels))
    print("selected_weights=" + json.dumps(selected, sort_keys=True))
    print(
        pd.DataFrame(payload["oof_metrics"]).T[
            ["top1_hit_rate", "top3_hit_rate", "mrr", "mean_winner_rank"]
        ].to_string(float_format=lambda value: f"{value:.5f}")
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
