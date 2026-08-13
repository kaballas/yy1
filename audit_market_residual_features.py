#!/usr/bin/env python3
"""Audit repeatable non-market ranking adjustments to the fluc2 baseline.

Each feature receives a one-dimensional coefficient fitted only on races before
the fold. The coefficient minimizes a regularized pairwise logistic loss on
top-three-versus-other runner pairs. It is then evaluated on the next unseen
chronological block. The newest races remain sealed and are never loaded into a
fit or metric cohort. Candidate features derived from either the current market
or historical prices are excluded.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import probability_metrics
from src.derived_racing_features import derive_racing_features


OUTCOME_OR_CONTROL = {
    "winner_index", "is_trainable", "selection_id", "runner_number",
    "finish_place", "runner_mask", "top3_mask", "is_winner", "is_validation",
}
IDENTIFIERS = {
    "race_id", "race_number", "competition_id", "feature_schema_version",
}
CURRENT_MARKET = {
    "open_price", "fluc1", "fluc2", "sp_starting_price",
    "open_price_rank", "fluc1_price_rank", "fluc2_price_rank",
    "race_consensus_score", "race_consensus_rank", "race_overlay_score",
    "race_overlay_rank", "race_signal_agreement_score",
    "race_signal_agreement_rank",
}
DERIVATION_ONLY_MARKET_INPUTS = tuple(
    f"recent_{run}_starting_price" for run in range(1, 7)
)


def is_market_derived(name: str) -> bool:
    """Return whether a candidate contains current or historical market data."""
    lowered = name.lower()
    return (
        name in CURRENT_MARKET
        or "starting_price" in lowered
        or "market" in lowered
        or lowered.startswith(("open_price", "fluc1", "fluc2"))
        or lowered.startswith(("race_overlay", "race_consensus", "race_signal"))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find non-market features whose fitted adjustments repeatedly beat "
            "the fluc2 ranking on later temporal folds."
        )
    )
    parser.add_argument("--db", type=Path, default=Path("db/race_runners.sqlite"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-races", type=int, default=500)
    parser.add_argument("--sealed-test-races", type=int, default=1000)
    parser.add_argument("--minimum-training-races", type=int, default=1000)
    parser.add_argument("--minimum-coverage", type=float, default=0.20)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--maximum-absolute-alpha", type=float, default=1.0)
    parser.add_argument("--top-features", type=int, default=40)
    parser.add_argument("--output-csv", type=Path, default=Path(
        "outputs/market_residual_feature_audit.csv"
    ))
    parser.add_argument("--fold-output-csv", type=Path, default=Path(
        "outputs/market_residual_feature_audit_folds.csv"
    ))
    parser.add_argument(
        "--bundle", action="append", default=[], metavar="NAME=FEATURE,FEATURE",
        help="Also audit a jointly fitted feature bundle; may be repeated.",
    )
    parser.add_argument("--bundle-output-csv", type=Path, default=Path(
        "outputs/market_residual_feature_audit_bundles.csv"
    ))
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "folds": args.folds,
        "fold-races": args.fold_races,
        "sealed-test-races": args.sealed_test_races,
        "minimum-training-races": args.minimum_training_races,
        "top-features": args.top_features,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))
    if not 0 < args.minimum_coverage <= 1:
        raise ValueError("--minimum-coverage must be in (0, 1]")
    if args.ridge < 0 or args.maximum_absolute_alpha <= 0:
        raise ValueError("--ridge must be non-negative and alpha limit positive")


def _numeric_candidates(connection: sqlite3.Connection) -> list[str]:
    columns = connection.execute('PRAGMA table_info("race_runners")').fetchall()
    result = []
    for row in columns:
        name = str(row[1])
        declared = str(row[2]).upper()
        numeric = any(
            token in declared for token in ("INT", "REAL", "FLOA", "DOUB", "NUM")
        )
        if not numeric:
            continue
        if name in OUTCOME_OR_CONTROL | IDENTIFIERS or is_market_derived(name):
            continue
        result.append(name)
    return result


def load_frame(database: Path) -> tuple[pd.DataFrame, list[str]]:
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        candidates = _numeric_candidates(connection)
        selected = list(dict.fromkeys([
            "race_id", "start_time_iso", "runner_number", "top3_mask", "fluc2",
            *candidates, *DERIVATION_ONLY_MARKET_INPUTS,
        ]))
        quoted = ", ".join(f'"{name}"' for name in selected)
        frame = pd.read_sql_query(
            f"""
            SELECT {quoted}
            FROM race_runners
            WHERE status = 'finished' AND runner_mask = 1
            ORDER BY start_time_iso, race_id, runner_number
            """,
            connection,
        )
    for name in [
        "top3_mask", "fluc2", *candidates, *DERIVATION_ONLY_MARKET_INPUTS,
    ]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame, candidates


def eligible_frame(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("race_id", sort=False)
    size = grouped["race_id"].transform("size")
    labelled = grouped["top3_mask"].transform("count")
    positives = grouped["top3_mask"].transform("sum")
    priced = grouped["fluc2"].transform(
        lambda values: bool(np.isfinite(values).all() and (values > 0).all())
    )
    mask = (size >= 4) & (labelled == size) & (positives == 3) & priced.astype(bool)
    return frame.loc[mask].reset_index(drop=True)


def race_percentile(values: pd.Series, race_ids: pd.Series) -> np.ndarray:
    valid = values.notna() & np.isfinite(values)
    safe = values.where(valid)
    ranks = safe.groupby(race_ids, sort=False).rank(method="average")
    counts = valid.groupby(race_ids, sort=False).transform("sum")
    result = np.zeros(len(values), dtype=np.float32)
    usable = valid & (counts > 1)
    result[usable.to_numpy()] = (
        2.0 * (ranks[usable].to_numpy(dtype=np.float64) - 1.0)
        / (counts[usable].to_numpy(dtype=np.float64) - 1.0)
        - 1.0
    ).astype(np.float32)
    return result


def pair_indices(
    targets: np.ndarray, race_ids: np.ndarray, selected_races: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for race_id in dict.fromkeys(map(int, race_ids)):
        if race_id not in selected_races:
            continue
        rows = np.flatnonzero(race_ids == race_id)
        pos = rows[targets[rows] == 1]
        neg = rows[targets[rows] == 0]
        positives.append(np.repeat(pos, len(neg)))
        negatives.append(np.tile(neg, len(pos)))
    return np.concatenate(positives), np.concatenate(negatives)


def fit_pairwise_alpha(
    market_score: np.ndarray,
    feature_score: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    ridge: float,
    limit: float,
) -> float:
    market_difference = (
        market_score[positive_rows] - market_score[negative_rows]
    ).astype(np.float64)
    feature_difference = (
        feature_score[positive_rows] - feature_score[negative_rows]
    ).astype(np.float64)
    if np.max(np.abs(feature_difference), initial=0.0) < 1e-12:
        return 0.0
    alpha = 0.0
    for _ in range(12):
        margin = np.clip(market_difference + alpha * feature_difference, -30, 30)
        error_probability = 1.0 / (1.0 + np.exp(margin))
        gradient = float(np.mean(-feature_difference * error_probability) + ridge * alpha)
        hessian = float(
            np.mean(
                np.square(feature_difference)
                * error_probability * (1.0 - error_probability)
            )
            + ridge
        )
        if hessian < 1e-12:
            break
        updated = float(np.clip(alpha - gradient / hessian, -limit, limit))
        if abs(updated - alpha) < 1e-7:
            alpha = updated
            break
        alpha = updated
    return alpha


def fit_pairwise_weights(
    market_score: np.ndarray,
    feature_scores: np.ndarray,
    positive_rows: np.ndarray,
    negative_rows: np.ndarray,
    ridge: float,
    limit: float,
) -> np.ndarray:
    market_difference = (
        market_score[positive_rows] - market_score[negative_rows]
    ).astype(np.float64)
    differences = (
        feature_scores[positive_rows] - feature_scores[negative_rows]
    ).astype(np.float64)
    weights = np.zeros(feature_scores.shape[1], dtype=np.float64)
    identity = np.eye(len(weights), dtype=np.float64)
    for _ in range(15):
        margin = np.clip(market_difference + differences @ weights, -30, 30)
        error_probability = 1.0 / (1.0 + np.exp(margin))
        gradient = (
            -(differences.T @ error_probability) / len(differences)
            + ridge * weights
        )
        curvature = error_probability * (1.0 - error_probability)
        hessian = (
            (differences.T * curvature) @ differences / len(differences)
            + ridge * identity
        )
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        updated = np.clip(weights - step, -limit, limit)
        if np.max(np.abs(updated - weights), initial=0.0) < 1e-7:
            weights = updated
            break
        weights = updated
    return weights


def parse_bundles(values: list[str]) -> list[tuple[str, list[str]]]:
    bundles = []
    for value in values:
        if "=" not in value:
            raise ValueError("--bundle must use NAME=FEATURE,FEATURE")
        name, feature_text = value.split("=", 1)
        features = [item.strip() for item in feature_text.split(",") if item.strip()]
        if not name.strip() or not features or len(features) != len(set(features)):
            raise ValueError("--bundle needs a name and unique feature names")
        bundles.append((name.strip(), features))
    return bundles


def composite(metrics: dict[str, float | int]) -> float:
    return (
        0.50 * float(metrics["top3_recall"])
        + 0.25 * float(metrics["ndcg3"])
        + 0.25 * float(metrics["pairwise_ranking_accuracy"])
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    frame, candidates = load_frame(args.db)
    frame = eligible_frame(frame)
    derived = derive_racing_features(frame)
    for name in derived.columns:
        frame[name] = derived[name]
    candidates = [
        name for name in dict.fromkeys([*candidates, *derived.columns.tolist()])
        if not is_market_derived(name)
    ]
    ordered_races = list(dict.fromkeys(map(int, frame["race_id"])))
    required = (
        args.minimum_training_races + args.folds * args.fold_races
        + args.sealed_test_races
    )
    if len(ordered_races) < required:
        raise ValueError(
            f"Need at least {required} eligible races for this audit; "
            f"found {len(ordered_races)}"
        )
    sealed_ids = ordered_races[-args.sealed_test_races:]
    sealed_start = frame.loc[
        frame["race_id"] == sealed_ids[0], "start_time_iso"
    ].iloc[0]
    audit_races = ordered_races[:-args.sealed_test_races]
    validation_total = args.folds * args.fold_races
    first_validation = len(audit_races) - validation_total
    if first_validation < args.minimum_training_races:
        raise ValueError("Rolling folds leave too few initial training races")

    audit_mask = frame["race_id"].isin(audit_races)
    coverage = frame.loc[audit_mask, candidates].notna().mean()
    candidates = [
        name for name in candidates
        if float(coverage[name]) >= args.minimum_coverage
        and frame.loc[audit_mask, name].nunique(dropna=True) > 1
    ]
    frame = frame.loc[frame["race_id"].isin(audit_races)].reset_index(drop=True)
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    targets = frame["top3_mask"].to_numpy(dtype=np.int64)
    market_score = -race_percentile(frame["fluc2"], frame["race_id"])

    folds = []
    for fold in range(args.folds):
        start = first_validation + fold * args.fold_races
        stop = start + args.fold_races
        training_ids = set(audit_races[:start])
        validation_ids = set(audit_races[start:stop])
        positive_rows, negative_rows = pair_indices(
            targets, race_ids, training_ids
        )
        validation_mask = np.isin(race_ids, list(validation_ids))
        market_metrics = probability_metrics(
            targets[validation_mask], market_score[validation_mask],
            race_ids[validation_mask],
        )
        folds.append({
            "fold": fold + 1,
            "training_ids": training_ids,
            "validation_ids": validation_ids,
            "positive_rows": positive_rows,
            "negative_rows": negative_rows,
            "validation_mask": validation_mask,
            "market_composite": composite(market_metrics),
            "validation_start": frame.loc[
                frame["race_id"] == audit_races[start], "start_time_iso"
            ].iloc[0],
            "validation_end": frame.loc[
                frame["race_id"] == audit_races[stop - 1], "start_time_iso"
            ].iloc[0],
        })

    records: list[dict[str, float | int | str]] = []
    print(
        "MARKET RESIDUAL FEATURE AUDIT\n"
        f"eligible_races={len(ordered_races):,} audit_races={len(audit_races):,} "
        f"sealed_test_races={len(sealed_ids):,} candidates={len(candidates)} "
        f"folds={args.folds} fold_races={args.fold_races}",
        flush=True,
    )
    for index, feature in enumerate(candidates, start=1):
        feature_score = race_percentile(frame[feature], frame["race_id"])
        for fold in folds:
            alpha = fit_pairwise_alpha(
                market_score, feature_score,
                fold["positive_rows"], fold["negative_rows"],
                args.ridge, args.maximum_absolute_alpha,
            )
            mask = fold["validation_mask"]
            adjusted = market_score[mask] + alpha * feature_score[mask]
            metrics = probability_metrics(
                targets[mask], adjusted, race_ids[mask]
            )
            score = composite(metrics)
            records.append({
                "feature": feature,
                "fold": int(fold["fold"]),
                "validation_start": str(fold["validation_start"]),
                "validation_end": str(fold["validation_end"]),
                "training_races": len(fold["training_ids"]),
                "validation_races": len(fold["validation_ids"]),
                "coverage": float(coverage[feature]),
                "alpha": alpha,
                "market_composite": float(fold["market_composite"]),
                "adjusted_composite": score,
                "composite_delta": score - float(fold["market_composite"]),
                "top3_recall": float(metrics["top3_recall"]),
                "ndcg3": float(metrics["ndcg3"]),
                "pairwise_ranking_accuracy": float(
                    metrics["pairwise_ranking_accuracy"]
                ),
            })
        if index % 20 == 0 or index == len(candidates):
            print(f"audited_features={index}/{len(candidates)}", flush=True)

    details = pd.DataFrame(records)
    summaries = []
    for feature, group in details.groupby("feature", sort=False):
        deltas = group["composite_delta"].to_numpy(dtype=np.float64)
        alphas = group["alpha"].to_numpy(dtype=np.float64)
        nonzero = alphas[np.abs(alphas) > 1e-8]
        dominant_sign_count = (
            max(int((nonzero > 0).sum()), int((nonzero < 0).sum()))
            if len(nonzero) else 0
        )
        summaries.append({
            "feature": feature,
            "coverage": float(group["coverage"].iloc[0]),
            "mean_composite_delta": float(deltas.mean()),
            "median_composite_delta": float(np.median(deltas)),
            "worst_composite_delta": float(deltas.min()),
            "best_composite_delta": float(deltas.max()),
            "positive_folds": int((deltas > 0).sum()),
            "nonnegative_folds": int((deltas >= 0).sum()),
            "alpha_sign_consistency": dominant_sign_count,
            "mean_alpha": float(alphas.mean()),
            "mean_absolute_alpha": float(np.abs(alphas).mean()),
            "stable_candidate": bool(
                (deltas > 0).sum() >= max(1, args.folds - 1)
                and deltas.mean() > 0
                and deltas.min() > -0.002
                and dominant_sign_count >= max(1, args.folds - 1)
            ),
        })
    summary = pd.DataFrame(summaries).sort_values(
        ["stable_candidate", "mean_composite_delta", "worst_composite_delta"],
        ascending=[False, False, False],
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.fold_output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    details.to_csv(args.fold_output_csv, index=False)

    bundle_records = []
    for bundle_name, bundle_features in parse_bundles(args.bundle):
        missing = sorted(set(bundle_features) - set(candidates))
        if missing:
            raise ValueError(
                f"Bundle {bundle_name!r} has unavailable features: {missing}"
            )
        scores = np.column_stack([
            race_percentile(frame[name], frame["race_id"])
            for name in bundle_features
        ])
        for fold in folds:
            weights = fit_pairwise_weights(
                market_score, scores, fold["positive_rows"], fold["negative_rows"],
                args.ridge, args.maximum_absolute_alpha,
            )
            mask = fold["validation_mask"]
            adjusted = market_score[mask] + scores[mask] @ weights
            metrics = probability_metrics(
                targets[mask], adjusted, race_ids[mask]
            )
            score = composite(metrics)
            bundle_records.append({
                "bundle": bundle_name,
                "features": ",".join(bundle_features),
                "fold": int(fold["fold"]),
                "validation_start": str(fold["validation_start"]),
                "validation_end": str(fold["validation_end"]),
                "weights": ",".join(f"{weight:.8f}" for weight in weights),
                "market_composite": float(fold["market_composite"]),
                "adjusted_composite": score,
                "composite_delta": score - float(fold["market_composite"]),
                "top3_recall": float(metrics["top3_recall"]),
                "ndcg3": float(metrics["ndcg3"]),
                "pairwise_ranking_accuracy": float(
                    metrics["pairwise_ranking_accuracy"]
                ),
            })
    bundle_details = pd.DataFrame(bundle_records)
    if not bundle_details.empty:
        bundle_summary = bundle_details.groupby(
            ["bundle", "features"], as_index=False
        ).agg(
            mean_composite_delta=("composite_delta", "mean"),
            worst_composite_delta=("composite_delta", "min"),
            best_composite_delta=("composite_delta", "max"),
            positive_folds=("composite_delta", lambda values: int((values > 0).sum())),
        ).sort_values(
            ["mean_composite_delta", "worst_composite_delta"], ascending=False
        )
        args.bundle_output_csv.parent.mkdir(parents=True, exist_ok=True)
        bundle_details.to_csv(args.bundle_output_csv, index=False)
        print("\nJOINT FEATURE BUNDLES")
        print(bundle_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nTOP REPEATABLE NON-MARKET SIGNALS")
    print(summary.head(args.top_features).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(
        f"\nsealed_test_start_race={sealed_ids[0]} "
        f"sealed_test_start={sealed_start} "
        "sealed_test_evaluated=no\n"
        f"saved_summary={args.output_csv.resolve()}\n"
        f"saved_folds={args.fold_output_csv.resolve()}"
        + (
            f"\nsaved_bundles={args.bundle_output_csv.resolve()}"
            if not bundle_details.empty else ""
        )
    )


if __name__ == "__main__":
    main()
