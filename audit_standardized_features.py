#!/usr/bin/env python3
"""Audit fitted feature scaling without changing the preprocessing contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.raceformer_preprocessing import raceformer_base_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report standardized feature tails and evidence-based preprocessing "
            "recommendations from a trusted TabFM checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--training-csv", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--show-all", action="store_true",
        help="Print all feature rows instead of only features with |z| > 3.",
    )
    return parser.parse_args()


def trusted_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError("Scaling audit requires a dictionary checkpoint bundle")
    return bundle


def standardize(
    frame: pd.DataFrame,
    features: list[str],
    median: np.ndarray,
    scale: np.ndarray,
    zeroed: set[str],
    preprocessing: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    raw = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if preprocessing and int(preprocessing.get("version", 1)) >= 2:
        diagnostics = raceformer_base_diagnostics(
            raw,
            features,
            preprocessing,
            legacy_median=median,
            legacy_scale=scale,
        )
        # Audit pre-clipping tails while honoring every saved feature transform.
        standardized = diagnostics["unclipped_standardized"].astype(np.float64)
    else:
        filled = np.where(np.isnan(raw), median, raw)
        standardized = (filled - median) / scale
    for index, feature in enumerate(features):
        if feature in zeroed:
            standardized[:, index] = 0.0
    return raw, standardized


def recommendation(
    feature: str, raw: np.ndarray, z: np.ndarray, *, already_log_transformed: bool
) -> str:
    finite_raw = raw[np.isfinite(raw)]
    if not len(finite_raw):
        return "remove or zero: no observed finite values"
    unique = np.unique(finite_raw)
    if len(unique) <= 2 and set(unique.tolist()) <= {0.0, 1.0}:
        return "binary treatment (retain 0/1; do not z-score)"
    extreme10 = float(np.mean(np.abs(z) > 10))
    extreme5 = float(np.mean(np.abs(z) > 5))
    nonnegative = float(np.min(finite_raw)) >= 0
    median_raw = float(np.median(finite_raw))
    p99_raw = float(np.quantile(finite_raw, 0.99))
    count_like = any(
        token in feature
        for token in ("starts", "wins", "seconds", "thirds", "count", "prize_money")
    )
    bounded_rate = any(
        token in feature for token in ("percentage", "_rate", "percentile", "_pct")
    )
    if already_log_transformed:
        if extreme5 > 0:
            return "retain saved log1p; held-out evidence may justify tighter clipping"
        return "retain saved log1p plus robust scaling"
    if nonnegative and (count_like or p99_raw > 10 * max(abs(median_raw), 1e-6)):
        if extreme5 > 0:
            return "log1p before fitted scaling; compare robust scaling on held-out races"
    if bounded_rate and extreme5 > 0:
        return "robust scaling; verify denominator/sparse-zero semantics"
    if extreme10 > 0:
        return "robust scaling plus train-fitted winsorisation candidate; inspect raw semantics"
    if extreme5 > 0:
        return "robust scaling candidate; retain current scaling until held-out comparison"
    return "current median-imputation/std scaling"


def dataset_rows(
    dataset: str,
    features: list[str],
    raw: np.ndarray,
    z: np.ndarray,
    log1p_features: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for index, feature in enumerate(features):
        values = z[:, index]
        raw_values = raw[:, index]
        rows.append({
            "dataset": dataset,
            "feature": feature,
            "min": float(np.min(values)),
            "p1": float(np.quantile(values, 0.01)),
            "p5": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
            "pct_abs_z_gt_3": 100 * float(np.mean(np.abs(values) > 3)),
            "pct_abs_z_gt_5": 100 * float(np.mean(np.abs(values) > 5)),
            "pct_abs_z_gt_10": 100 * float(np.mean(np.abs(values) > 10)),
            "raw_missing_pct": 100 * float(np.mean(~np.isfinite(raw_values))),
            "raw_unique": int(len(np.unique(raw_values[np.isfinite(raw_values)]))),
            "recommendation": recommendation(
                feature,
                raw_values,
                values,
                already_log_transformed=feature in log1p_features,
            ),
        })
    return rows


def main() -> None:
    args = parse_args()
    bundle = trusted_bundle(args.checkpoint)
    features = list(bundle["feature_columns"])
    median = np.asarray(bundle["median"], dtype=np.float64)
    scale = np.asarray(bundle["scale"], dtype=np.float64)
    zeroed = set(bundle.get("zeroed_features", []))
    preprocessing = bundle.get("preprocessing")
    log1p_features = set(
        preprocessing.get("log1p_features", []) if preprocessing else []
    )
    if median.shape != (len(features),) or scale.shape != (len(features),):
        raise ValueError("Checkpoint preprocessing vectors do not match feature count")
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Checkpoint scale contains invalid values")

    rows: list[dict[str, Any]] = []
    for dataset, path in (
        ("training", args.training_csv),
        ("validation", args.validation_csv),
    ):
        frame = pd.read_csv(path)
        missing = sorted(set(features) - set(frame.columns))
        if missing:
            raise ValueError(f"{dataset} CSV is missing features: {missing[:10]}")
        raw, z = standardize(
            frame, features, median, scale, zeroed, preprocessing
        )
        rows.extend(dataset_rows(dataset, features, raw, z, log1p_features))

    report = pd.DataFrame(rows)
    print("STANDARDIZED FEATURE DISTRIBUTION AUDIT")
    print(f"checkpoint={args.checkpoint.resolve()} features={len(features)}")
    version = int(preprocessing.get("version", 1)) if preprocessing else 1
    clip = preprocessing.get("clip") if preprocessing else None
    print(
        f"preprocessing_contract=checkpoint_version_{version} clip={clip}; "
        "tail statistics are measured before clipping after all saved transforms; "
        "this audit makes no preprocessing changes."
    )
    shown = report if args.show_all else report.loc[
        (report["pct_abs_z_gt_3"] > 0)
        | (report["min"] < -5)
        | (report["max"] > 5)
    ]
    shown = shown.sort_values(
        ["dataset", "pct_abs_z_gt_10", "pct_abs_z_gt_5", "pct_abs_z_gt_3"],
        ascending=[True, False, False, False],
    )
    print(shown.to_string(index=False))
    print(
        "Recommendations are candidates for a new preprocessing version only. "
        "Any accepted transform must be fitted on training data, embedded in the "
        "checkpoint, and applied identically to validation and live inference."
    )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output_csv, index=False)
        print(f"saved_report={args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
