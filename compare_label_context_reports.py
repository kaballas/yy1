#!/usr/bin/env python3
"""Compare provenance-clean A/B/C summaries produced by evaluate_model_stages.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final label-context A/B/C comparison table."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=SUMMARY.json",
        help="Repeat for A, B, and C (or any named model).",
    )
    parser.add_argument(
        "--allow-unsafe-reports",
        action="store_true",
        help="Allow classroom or train-overlapping summaries with a visible warning.",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--ranking-tolerance", type=float, default=1e-4,
        help="Treat ranking metrics within this absolute tolerance as effectively equal.",
    )
    return parser.parse_args()


def load_spec(spec: str) -> tuple[str, dict[str, Any]]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=SUMMARY.json, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise ValueError(f"Expected NAME=SUMMARY.json, got {spec!r}")
    path = Path(raw_path).resolve()
    return name.strip(), json.loads(path.read_text())


def find(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}={value!r} row, found {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    if args.ranking_tolerance < 0:
        raise ValueError("--ranking-tolerance must be non-negative")
    loaded = [load_spec(spec) for spec in args.model]
    names = [name for name, _ in loaded]
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique")

    unsafe = [
        name for name, summary in loaded
        if summary.get("experiment_only") or (summary.get("overlap_count") or 0) > 0
    ]
    if unsafe and not args.allow_unsafe_reports:
        raise ValueError(
            "Refusing non-generalisation comparison for unsafe reports "
            f"{unsafe}; use clean checkpoint_validation summaries"
        )
    if unsafe:
        print("WARNING UNSAFE COMPARISON: " + ", ".join(unsafe))

    rows = []
    transition_rows = []
    for name, summary in loaded:
        full = find(summary["stage_metrics"], "configuration", "full_model")
        attention_matches = [
            row for row in summary.get("attention_metrics", [])
            if row.get("view") == "head_average"
        ]
        attention = attention_matches[0] if attention_matches else {}
        rows.append({
            "Model": name,
            "Top3 recall": full["top3_recall"],
            "Exact top3": full["exact_top3_set"],
            "NDCG@3": full["ndcg3"],
            "Pairwise ranking accuracy": full["pairwise_ranking_accuracy"],
            "Log loss": full["race_logloss"],
            "Mean sum(p)": full["mean_probability_sum"],
            "Mean |sum(p)-3|": full["mean_abs_sum_error"],
            "Effective runners": attention.get("mean_effective_runners", np.nan),
            "Effective races": attention.get("mean_effective_races", np.nan),
            "Query-attention cosine": attention.get("mean_pairwise_query_cosine", np.nan),
            "Top-10 attention overlap": attention.get("mean_top10_jaccard", np.nan),
            "Races": full["races"],
        })
        for transition, counts in summary["stage_change_counts"].items():
            transition_rows.append({"Model": name, "transition": transition, **counts})

    comparison = pd.DataFrame(rows)
    transitions = pd.DataFrame(transition_rows)
    print("LABEL-CONTEXT A/B/C COMPARISON")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nSTAGE-LEVEL TOP-3 SET EFFECT")
    print(transitions.to_string(index=False))
    by_name = {row["Model"].upper(): row for row in rows}
    if {"A", "B", "C"}.issubset(by_name):
        ranking_columns = (
            "Top3 recall", "NDCG@3", "Exact top3", "Pairwise ranking accuracy"
        )
        values = np.asarray([
            [by_name[name][column] for column in ranking_columns]
            for name in ("A", "B", "C")
        ])
        effectively_equal = bool(
            np.all(np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
                   <= args.ranking_tolerance)
        )
        if effectively_equal:
            decision = "A/B/C ranking is effectively equal; prefer simpler model B."
        else:
            winner = max(
                ("A", "B", "C"),
                key=lambda name: tuple(by_name[name][column] for column in ranking_columns),
            )
            if winner == "B":
                decision = (
                    "Model B wins clean-validation ranking; remove the explicit "
                    "label-context branch for a ranking-first production objective."
                )
                best_context_logloss = min(
                    by_name["A"]["Log loss"], by_name["C"]["Log loss"]
                )
                if best_context_logloss < by_name["B"]["Log loss"]:
                    decision += (
                        " A/C has better log loss, so revisit the choice only if "
                        "calibrated probabilities outrank race ranking as the objective."
                    )
            elif winner == "C":
                decision = (
                    "Model C wins clean-validation ranking; retain the explicit "
                    "label-context branch with its sharper retrieval configuration."
                )
            else:
                decision = (
                    "Model A wins clean-validation ranking; retain the branch but "
                    "do not adopt C's sharper retrieval configuration."
                )
        print("\nDECISION RULE")
        print(decision)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output_csv, index=False)
        transitions.to_csv(
            args.output_csv.with_name(args.output_csv.stem + ".stages.csv"),
            index=False,
        )
        print(f"saved_comparison={args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
