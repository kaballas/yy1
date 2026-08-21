#!/usr/bin/env python3
"""Cross-fit, tune, and refit winner rankers on all eligible finished races.

Every eligible finished race receives out-of-fold form and market-aware scores
from models that did not train on that race. Those OOF scores tune a two-model
blend with raw market weight fixed at zero. Finally, both ensembles are refit
on every eligible finished race for live prediction.

Because all races participate in tuning, this mode intentionally has no sealed
test cohort. Its OOF metrics are model-selection diagnostics, not an untouched
future-performance estimate. Grouped folds cover all races but are not a
chronological backtest; use train_winner_ranker_pipeline.py for that audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required: pip install xgboost") from exc

from backtest_winner_blend import (
    candidate_form_weights,
)
from src.config import DEFAULT_DB
from src.winner_ranker import (
    MARKET_ENGINEERED_FEATURES,
    database_numeric_columns,
    eligible_races,
    ensemble_rank_scores,
    group_sizes,
    is_current_market_feature,
    load_training_rows,
    market_scores,
    market_deviation_metrics,
    model_feature_matrix,
    rank_percentiles,
    rows_for_races,
    select_form_features,
    validate_ranker_groups,
    winner_field_size_slices,
    winner_metrics,
    winner_race_report,
    xgb_ensemble_feature_importance,
)
from train_winner_ranker_pipeline import model_parameters, score_table


CPU_THREADS = os.cpu_count() or 1
DEFAULT_JOBS = max(1, int(CPU_THREADS * 0.80))


class OOFCohortMismatchError(ValueError):
    """Raised when saved OOF rows cannot be reused for the current cohort."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/winner_ranker_all_finished"),
    )
    parser.add_argument(
        "--source-bundle",
        type=Path,
        default=Path("outputs/winner_ranker/winner_ranker_bundle.json"),
        help=(
            "Provides validated tree counts when --no-tune-tree-counts is used."
        ),
    )
    parser.add_argument(
        "--features-json",
        type=Path,
        default=Path(__file__).resolve().with_name("winner_ranker_features.json"),
        help="JSON manifest defining the exact ordered features for each model.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--minimum-feature-coverage", type=float, default=0.20)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument(
        "--default-form-estimators",
        type=int,
        default=120,
        help="Non-market fallback used only with --no-tune-tree-counts.",
    )
    parser.add_argument(
        "--default-market-aware-estimators",
        type=int,
        default=50,
        help="Market-aware fallback used only with --no-tune-tree-counts.",
    )
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument(
        "--tree-count-validation-races",
        type=int,
        default=1000,
        help=(
            "Maximum chronological inner-validation races used to select tree "
            "counts inside each outer OOF fold."
        ),
    )
    parser.add_argument(
        "--no-tune-tree-counts",
        action="store_true",
        help=(
            "Disable nested early stopping and use counts from --source-bundle "
            "or the estimator fallbacks."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=(
            "XGBoost CPU threads. Defaults to 80%% of available logical CPU "
            "threads."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--objective", choices=("top1", "mrr", "top3", "composite"),
        default="top1",
    )
    parser.add_argument("--weight-step", type=float, default=0.001)
    parser.add_argument("--minimum-form-weight", type=float, default=0.0)
    parser.add_argument("--skip-feature-update", action="store_true")
    parser.add_argument(
        "--ranker-diagnostics", action="store_true",
        help=(
            "Save grouped OOF race reports, field-size failure slices, and "
            "gain/cover/split feature importance for final XGBoost models."
        ),
    )
    parser.add_argument(
        "--retune-only",
        action="store_true",
        help="Retune all JSON model groups from saved OOF scores without refitting.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help=(
            "Train only these JSON model groups (for example --models f). "
            "Unlisted groups are excluded from training, blending, and output."
        ),
    )
    parser.add_argument(
        "--reuse-unselected-models",
        action="store_true",
        help=(
            "With --models, preserve unlisted groups from an existing output "
            "bundle and matching OOF file instead of excluding them."
        ),
    )
    return parser.parse_args()


def crossfit_fold_ids(race_ids: list[int], folds: int) -> list[list[int]]:
    """Assign whole races deterministically and evenly across grouped folds.

    Chronological round-robin allocation prevents one fold from being only old
    or only new races while guaranteeing that all runners from a race stay in
    one holdout fold.
    """
    if folds < 2:
        raise ValueError("folds must be at least two")
    if len(race_ids) < folds:
        raise ValueError(f"Need at least {folds} races; found {len(race_ids)}")
    result = [[] for _ in range(folds)]
    for index, race_id in enumerate(race_ids):
        result[index % folds].append(int(race_id))
    return result


def tree_counts(
    source_bundle: Path,
    label: str,
    ensemble_size: int,
    fallback: int,
) -> list[int]:
    """Load previously validated tree counts, with a deterministic fallback."""
    counts: list[int] = []
    if source_bundle.resolve().is_file():
        payload = json.loads(source_bundle.resolve().read_text(encoding="utf-8"))
        counts = [
            int(value) for value in payload.get("best_tree_counts", {}).get(label, [])
            if int(value) > 0
        ]
    if not counts:
        counts = [int(fallback)]
    return [counts[index % len(counts)] for index in range(ensemble_size)]


def inner_tree_count_split(
    race_ids: list[int], maximum_validation_races: int
) -> tuple[list[int], list[int]]:
    """Create a chronological inner split without consuming the outer holdout."""
    if maximum_validation_races < 1:
        raise ValueError("tree-count-validation-races must be positive")
    if len(race_ids) < 2:
        raise ValueError("Nested tree-count tuning requires at least two races")
    validation_count = min(
        maximum_validation_races,
        max(1, len(race_ids) // 5),
    )
    return race_ids[:-validation_count], race_ids[-validation_count:]


def tree_count_eval_metrics(objective: str) -> list[str]:
    """Put the tree-count selection metric last, as required by XGBoost."""
    if objective == "top1":
        return ["map", "ndcg@3", "ndcg@1"]
    if objective == "top3":
        return ["map", "ndcg@1", "ndcg@3"]
    if objective in {"mrr", "composite"}:
        # With exactly one relevant runner per race, MAP is reciprocal rank.
        return ["ndcg@1", "ndcg@3", "map"]
    raise ValueError(f"Unknown tree-count objective: {objective}")


def tune_tree_counts(
    args: argparse.Namespace,
    label: str,
    training: pd.DataFrame,
    configured_features: list[str],
    seed_offset: int,
) -> list[int]:
    """Early-stop ensemble members on an inner chronological validation slice."""
    ordered_race_ids = training.groupby("race_id", sort=False).head(1)[
        "race_id"
    ].astype(int).tolist()
    inner_train_ids, inner_validation_ids = inner_tree_count_split(
        ordered_race_ids, args.tree_count_validation_races
    )
    inner_train = rows_for_races(training, inner_train_ids)
    inner_validation = rows_for_races(training, inner_validation_ids)
    inner_train_y = inner_train["is_winner"].to_numpy(dtype=np.int64)
    inner_validation_y = inner_validation["is_winner"].to_numpy(dtype=np.int64)
    inner_train_groups = group_sizes(inner_train)
    inner_validation_groups = group_sizes(inner_validation)
    validate_ranker_groups(inner_train, inner_train_y, inner_train_groups)
    validate_ranker_groups(
        inner_validation, inner_validation_y, inner_validation_groups
    )
    train_matrix = model_feature_matrix(inner_train, configured_features)
    validation_matrix = model_feature_matrix(
        inner_validation, configured_features
    )
    counts: list[int] = []
    for member in range(args.ensemble_size):
        seed = args.seed + seed_offset + member * 1009
        parameters = model_parameters(args, seed, args.max_estimators)
        parameters["eval_metric"] = tree_count_eval_metrics(args.objective)
        model = XGBRanker(
            **parameters,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model.fit(
            train_matrix,
            inner_train_y,
            group=inner_train_groups,
            eval_set=[
                (train_matrix, inner_train_y),
                (validation_matrix, inner_validation_y),
            ],
            eval_group=[inner_train_groups, inner_validation_groups],
            verbose=False,
        )
        counts.append(int(model.best_iteration) + 1)
    print(
        f"tree_count_tuning={label} "
        f"inner_train_races={len(inner_train_ids):,} "
        f"inner_validation_races={len(inner_validation_ids):,} "
        f"selection_metric={tree_count_eval_metrics(args.objective)[-1]} "
        f"selected_trees={json.dumps(counts)}",
        flush=True,
    )
    return counts


def aggregate_tree_counts(fold_counts: list[list[int]]) -> list[int]:
    """Take a deterministic per-member median of nested-fold tree counts."""
    if not fold_counts or not fold_counts[0]:
        raise ValueError("No nested-fold tree counts to aggregate")
    width = len(fold_counts[0])
    if any(len(counts) != width for counts in fold_counts):
        raise ValueError("Nested-fold tree-count ensembles have inconsistent sizes")
    values = np.asarray(fold_counts, dtype=np.int64)
    return [int(math.floor(value + 0.5)) for value in np.median(values, axis=0)]


def fit_ensemble(
    args: argparse.Namespace,
    label: str,
    matrix: pd.DataFrame,
    targets: np.ndarray,
    groups: np.ndarray,
    estimators: list[int],
    seed_offset: int,
) -> list[XGBRanker]:
    models: list[XGBRanker] = []
    for member, trees in enumerate(estimators):
        seed = args.seed + seed_offset + member * 1009
        model = XGBRanker(**model_parameters(args, seed, trees))
        model.fit(matrix, targets, group=groups, verbose=False)
        models.append(model)
    return models


def save_ensemble(
    models: list[XGBRanker], label: str, output_dir: Path, seed: int
) -> list[str]:
    paths: list[str] = []
    for member, model in enumerate(models):
        member_seed = seed + member * 1009
        path = output_dir / f"{label}_all_finished_seed_{member_seed}.json"
        model.save_model(path)
        paths.append(str(path.resolve()))
        print(f"saved_model={path}", flush=True)
    return paths


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def load_model_feature_sets(
    manifest_path: Path, eligible_features: list[str]
) -> dict[str, list[str]]:
    """Load and validate the exact ordered input columns for each model."""
    path = manifest_path.resolve()
    if not path.is_file():
        raise ValueError(f"Feature manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} must have schema_version 1")
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError(f"{path} must contain a models object")

    if not models:
        raise ValueError(f"{path} models object must not be empty")
    feature_sets: dict[str, list[str]] = {}
    for label, model in models.items():
        if not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", label):
            raise ValueError(f"{path} has invalid model name: {label!r}")
        if label == "market" or label.endswith("_evaluation"):
            raise ValueError(f"{path} model name is reserved: {label}")
        features = model.get("features") if isinstance(model, dict) else None
        if not isinstance(features, list) or not features:
            raise ValueError(f"{path} models.{label}.features must be a non-empty list")
        if not all(isinstance(feature, str) and feature for feature in features):
            raise ValueError(f"{path} models.{label}.features contains an invalid name")
        if len(features) != len(set(features)):
            raise ValueError(f"{path} models.{label}.features contains duplicates")
        feature_sets[label] = list(features)

    eligible = set(eligible_features) | set(MARKET_ENGINEERED_FEATURES)
    for label, features in feature_sets.items():
        unavailable = [feature for feature in features if feature not in eligible]
        if unavailable:
            raise ValueError(
                f"Feature manifest has unavailable {label} features: "
                + ", ".join(unavailable)
            )
    return feature_sets


def select_requested_model_groups(
    feature_sets: dict[str, list[str]],
    requested: list[str] | None,
    reuse_unselected: bool,
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """Resolve an exclusive run or explicit selective-retraining run."""
    if reuse_unselected and not requested:
        raise ValueError("--reuse-unselected-models requires --models")
    if requested and len(requested) != len(set(requested)):
        raise ValueError("--models contains duplicate model names")
    requested_labels = set(requested or feature_sets)
    unknown = sorted(requested_labels - set(feature_sets))
    if unknown:
        raise ValueError(
            "Requested models are absent from the feature manifest: "
            + ", ".join(unknown)
        )
    training = [label for label in feature_sets if label in requested_labels]
    if reuse_unselected:
        reused = [label for label in feature_sets if label not in requested_labels]
        return feature_sets, training, reused
    selected = {label: feature_sets[label] for label in training}
    return selected, training, []


def normalize_requested_models(requested: list[str] | None) -> list[str] | None:
    """Accept both ``--models f x1`` and ``--models f,x1`` forms."""
    if requested is None:
        return None
    labels: list[str] = []
    for value in requested:
        parts = [part.strip() for part in value.split(",")]
        if any(not part for part in parts):
            raise ValueError("--models contains an empty model name")
        labels.extend(parts)
    return labels


def print_model_feature_report(
    feature_file: Path, feature_sets: dict[str, list[str]]
) -> None:
    """Print complete, machine-readable feature lists and their saved location."""
    print("MODEL FEATURES", flush=True)
    print(f"feature_file={feature_file.resolve()}", flush=True)
    for label, features in feature_sets.items():
        print(
            f"model={label} feature_count={len(features)} "
            f"features={json.dumps(features)}",
            flush=True,
        )


def merge_reused_oof_scores(
    fresh: pd.DataFrame,
    existing: pd.DataFrame,
    reused_labels: list[str],
) -> pd.DataFrame:
    """Attach validated OOF scores for model groups not being retrained."""
    if not reused_labels:
        return fresh
    keys = ["race_id", "runner_number"]
    reused_columns = [
        column
        for label in reused_labels
        for column in (f"{label}_score", f"{label}_rank")
    ]
    missing = sorted(set([*keys, *reused_columns]) - set(existing.columns))
    if missing:
        raise ValueError(
            "Existing OOF predictions cannot preserve requested models; missing: "
            + ", ".join(missing)
        )
    if fresh.duplicated(keys).any() or existing.duplicated(keys).any():
        raise ValueError("OOF predictions contain duplicate race/runner keys")
    existing_subset = existing.loc[:, [*keys, *reused_columns]]
    merged = fresh.merge(
        existing_subset, on=keys, how="left", validate="one_to_one", indicator=True
    )
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched or len(merged) != len(existing_subset):
        raise OOFCohortMismatchError(
            "Existing OOF cohort does not match the current eligible cohort: "
            f"fresh_rows={len(fresh):,} existing_rows={len(existing_subset):,} "
            f"unmatched_fresh_rows={unmatched:,}"
        )
    merged = merged.drop(columns="_merge")
    if merged[reused_columns].isna().any().any():
        raise ValueError("Existing OOF predictions contain missing reused scores")
    return merged


def _blend_objective_values(metrics: pd.DataFrame, objective: str) -> np.ndarray:
    if objective == "top1":
        return metrics["top1_hit_rate"].to_numpy()
    if objective == "top3":
        return metrics["top3_hit_rate"].to_numpy()
    if objective == "mrr":
        return metrics["mrr"].to_numpy()
    return (
        0.50 * metrics["top1_hit_rate"].to_numpy()
        + 0.30 * metrics["mrr"].to_numpy()
        + 0.20 * metrics["top3_hit_rate"].to_numpy()
    )


def blend_metrics_for_weights(
    frame: pd.DataFrame,
    score_columns: list[str],
    weights: np.ndarray,
) -> pd.DataFrame:
    """Evaluate many dynamic model blends with equal weight per race."""
    candidate_weights = np.asarray(weights, dtype=np.float64)
    if candidate_weights.ndim != 2 or candidate_weights.shape[1] != len(score_columns):
        raise ValueError("Blend weight matrix does not match model score columns")
    candidate_count = len(candidate_weights)
    top1 = np.zeros(candidate_count, dtype=np.float64)
    top3 = np.zeros(candidate_count, dtype=np.float64)
    reciprocal = np.zeros(candidate_count, dtype=np.float64)
    rank_total = np.zeros(candidate_count, dtype=np.float64)
    race_count = 0
    for _, race in frame.groupby("race_id", sort=False):
        model_scores = race.loc[:, score_columns].to_numpy(dtype=np.float64)
        targets = race["is_winner"].to_numpy(dtype=np.int64)
        winner = int(np.flatnonzero(targets == 1)[0])
        blended = model_scores @ candidate_weights.T
        winner_scores = blended[winner]
        positions = np.arange(len(race))[:, None]
        ranks = 1 + np.sum(
            (blended > winner_scores)
            | ((blended == winner_scores) & (positions < winner)),
            axis=0,
        )
        top1 += ranks == 1
        top3 += ranks <= 3
        reciprocal += 1.0 / ranks
        rank_total += ranks
        race_count += 1
    return pd.DataFrame({
        "top1_hit_rate": top1 / race_count,
        "top3_hit_rate": top3 / race_count,
        "mrr": reciprocal / race_count,
        "mean_winner_rank": rank_total / race_count,
    })


def _best_blend_index(metrics: pd.DataFrame, objective: str) -> int:
    candidates = metrics.copy()
    candidates["objective_value"] = _blend_objective_values(candidates, objective)
    for column, ascending in (
        ("objective_value", False),
        ("mrr", False),
        ("top1_hit_rate", False),
        ("top3_hit_rate", False),
        ("mean_winner_rank", True),
    ):
        best = candidates[column].min() if ascending else candidates[column].max()
        candidates = candidates.loc[np.isclose(
            candidates[column], best, rtol=0.0, atol=1e-15
        )]
    return int(candidates.index[0])


def _simplex_lattice(model_count: int, maximum_candidates: int = 25_000) -> np.ndarray:
    """Create a deterministic global simplex grid with a bounded row count."""
    units = 100
    while math.comb(units + model_count - 1, model_count - 1) > maximum_candidates:
        units -= 1
    allocations: list[list[int]] = []

    def extend(prefix: list[int], remaining: int, dimensions: int) -> None:
        if dimensions == 1:
            allocations.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            extend([*prefix, value], remaining - value, dimensions - 1)

    extend([], units, model_count)
    return np.asarray(allocations, dtype=np.float64) / units


def _quantize_blend_weights(weights: np.ndarray, step: float) -> np.ndarray:
    """Snap simplex rows to exact requested-grid units while preserving sum one."""
    units = int(round(1.0 / step))
    scaled = np.asarray(weights, dtype=np.float64) * units
    allocations = np.floor(scaled).astype(np.int64)
    missing = units - allocations.sum(axis=1)
    fractions = scaled - allocations
    for row_index, count in enumerate(missing):
        if count > 0:
            order = np.argsort(-fractions[row_index], kind="stable")
            allocations[row_index, order[:count]] += 1
        elif count < 0:
            order = np.argsort(fractions[row_index], kind="stable")
            for column in order:
                removable = min(allocations[row_index, column], -count)
                allocations[row_index, column] -= removable
                count += removable
                if count == 0:
                    break
    return allocations.astype(np.float64) / units


def tune_dynamic_model_blend(
    frame: pd.DataFrame,
    model_labels: list[str],
    step: float,
    objective: str,
    minimum_form_weight: float = 0.0,
    refinement_passes: int = 5,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune a convex blend over every configured model group."""
    if not model_labels:
        raise ValueError("At least one model is required for blend tuning")
    if not 0.0 <= minimum_form_weight <= 1.0:
        raise ValueError("minimum-form-weight must be in [0, 1]")
    if "form" not in model_labels and minimum_form_weight > 0:
        raise ValueError("minimum-form-weight requires a configured form model")
    grid = candidate_form_weights(step, 0.0)
    model_count = len(model_labels)
    initial = np.unique(
        _quantize_blend_weights(_simplex_lattice(model_count), step), axis=0
    )
    if "form" in model_labels and minimum_form_weight > 0:
        initial = initial[
            initial[:, model_labels.index("form")] >= minimum_form_weight - 1e-12
        ]

    score_columns = [f"{label}_score" for label in model_labels]
    sweep_parts: list[pd.DataFrame] = []

    def evaluate(candidate_weights: np.ndarray, phase: str) -> tuple[np.ndarray, int]:
        metrics = blend_metrics_for_weights(frame, score_columns, candidate_weights)
        metrics["objective_value"] = _blend_objective_values(metrics, objective)
        for index, label in enumerate(model_labels):
            metrics[f"{label}_weight"] = candidate_weights[:, index]
        metrics.insert(0, "phase", phase)
        sweep_parts.append(metrics)
        return candidate_weights, _best_blend_index(metrics, objective)

    initial, best_index = evaluate(initial, "global_simplex")
    current = initial[best_index]
    for pass_number in range(1, refinement_passes + 1):
        refinements: list[np.ndarray] = []
        for model_index in range(model_count):
            moves = grid[:, None] * current[None, :]
            moves[:, model_index] += 1.0 - grid
            refinements.append(moves)
        candidates_for_pass = np.unique(
            _quantize_blend_weights(np.vstack(refinements), step), axis=0
        )
        if "form" in model_labels and minimum_form_weight > 0:
            candidates_for_pass = candidates_for_pass[
                candidates_for_pass[:, model_labels.index("form")]
                >= minimum_form_weight - 1e-12
            ]
        candidates_for_pass, best_index = evaluate(
            candidates_for_pass, f"refinement_{pass_number}"
        )
        updated = candidates_for_pass[best_index]
        if np.allclose(updated, current, rtol=0.0, atol=1e-15):
            break
        current = updated

    selected = {
        label: float(current[index]) for index, label in enumerate(model_labels)
    }
    selected["market"] = 0.0
    return selected, pd.concat(sweep_parts, ignore_index=True)


def dynamic_blend_analysis(
    oof: pd.DataFrame,
    model_labels: list[str],
    step: float,
    objective: str,
    minimum_form_weight: float,
) -> tuple[
    dict[str, float],
    pd.DataFrame,
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    """Tune and evaluate every configured OOF model score."""
    targets = oof["is_winner"].to_numpy(dtype=np.int64)
    race_ids = oof["race_id"].to_numpy(dtype=np.int64)
    model_metrics = {
        label: winner_metrics(targets, oof[f"{label}_score"], race_ids)
        for label in model_labels
    }
    selected_weights, sweep = tune_dynamic_model_blend(
        oof, model_labels, step, objective, minimum_form_weight
    )
    selected = np.asarray([selected_weights[label] for label in model_labels])
    score_matrix = oof.loc[:, [
        f"{label}_score" for label in model_labels
    ]].to_numpy(dtype=np.float64)
    tuned_score = score_matrix @ selected
    metrics = {
        **{f"{label}_only": values for label, values in model_metrics.items()},
        "equal_dynamic_blend": winner_metrics(
            targets, score_matrix.mean(axis=1), race_ids
        ),
        "tuned_dynamic_blend": winner_metrics(targets, tuned_score, race_ids),
        "raw_market_benchmark": winner_metrics(
            targets, oof["market_score"], race_ids
        ),
    }
    audit = oof[["race_id", "runner_number", "is_winner", "market_rank"]].copy()
    audit["tuned_dynamic_blend_rank"] = (
        pd.Series(tuned_score).groupby(audit["race_id"], sort=False).rank(
            method="first", ascending=False
        ).astype(int)
    )
    deviation = market_deviation_metrics(audit, "tuned_dynamic_blend")
    return selected_weights, sweep, model_metrics, metrics, deviation


def save_oof_ranker_diagnostics(
    oof: pd.DataFrame,
    model_labels: list[str],
    selected_weights: dict[str, float],
    output_dir: Path,
) -> None:
    """Persist model/blend race failures and equal-race field-size slices."""
    targets = oof["is_winner"].to_numpy(dtype=np.int64)
    model_matrix = oof.loc[:, [
        f"{name}_score" for name in model_labels
    ]].to_numpy(dtype=np.float64)
    diagnostic_scores = {
        **{
            label: oof[f"{label}_score"].to_numpy(dtype=np.float64)
            for label in model_labels
        },
        "tuned_dynamic_blend": model_matrix @ np.asarray([
            selected_weights[name] for name in model_labels
        ]),
        "raw_market_benchmark": oof["market_score"].to_numpy(dtype=np.float64),
    }
    for label, score in diagnostic_scores.items():
        report = winner_race_report(oof, targets, score)
        report.to_csv(output_dir / f"oof_{label}_race_report.csv", index=False)
        slices = winner_field_size_slices(report)
        slices.to_csv(
            output_dir / f"oof_{label}_field_size_slices.csv", index=False
        )
        print(f"RANKER DIAGNOSTICS OOF {label.upper()}")
        print(slices.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


def retune_saved_predictions(
    args: argparse.Namespace,
    output_dir: Path,
    feature_manifest: Path,
    bundle_path: Path,
) -> None:
    """Update dynamic blend artifacts from existing OOF predictions."""
    manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
    model_labels = list(manifest.get("models", {}))
    if not model_labels:
        raise ValueError(f"No model groups in feature manifest: {feature_manifest}")
    oof_path = output_dir / "all_finished_oof_predictions.csv"
    recommendation_path = output_dir / "all_finished_blend.json"
    sweep_path = output_dir / "all_finished_weight_sweep.csv"
    for path in (oof_path, bundle_path):
        if not path.is_file():
            raise ValueError(f"Retune input does not exist: {path}")
    oof = pd.read_csv(oof_path)
    required = {
        "race_id", "runner_number", "is_winner", "market_score", "market_rank",
        *(f"{label}_score" for label in model_labels),
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError("Saved OOF predictions are missing: " + ", ".join(missing))
    selected, sweep, model_metrics, metrics, deviation = dynamic_blend_analysis(
        oof, model_labels, args.weight_step, args.objective,
        args.minimum_form_weight,
    )
    if args.ranker_diagnostics:
        save_oof_ranker_diagnostics(oof, model_labels, selected, output_dir)
    recommendation = (
        json.loads(recommendation_path.read_text(encoding="utf-8"))
        if recommendation_path.is_file() else {}
    )
    recommendation.update({
        "schema_version": 2,
        "blend": "all_finished_crossfit_dynamic_model_groups",
        "objective": args.objective,
        "selected_weights": selected,
        "model_labels": model_labels,
        "model_oof_metrics": model_metrics,
        "oof_metrics": metrics,
        "oof_market_deviation": deviation,
    })
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["selected_blend_weights"] = selected
    bundle["all_finished_tuned_blend_weights"] = selected
    bundle.setdefault("all_finished_crossfit", {}).update(recommendation)
    sweep.to_csv(sweep_path, index=False)
    recommendation_path.write_text(
        json.dumps(_jsonable(recommendation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle_path.write_text(
        json.dumps(_jsonable(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("RETUNED DYNAMIC MODEL BLEND")
    print("selected_weights=" + json.dumps(selected, sort_keys=True))
    print(pd.DataFrame(metrics).T[[
        "top1_hit_rate", "top3_hit_rate", "mrr",
        "mean_winner_rank", "race_logloss",
    ]].to_string(float_format=lambda value: f"{value:.5f}"))
    print(
        f"saved_blend={recommendation_path}\n"
        f"saved_weight_sweep={sweep_path}\n"
        f"updated_bundle={bundle_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    print(
        f"cpu_threads={CPU_THREADS}\n"
        f"xgboost_jobs={args.jobs}\n"
        f"cpu_target={'80%' if args.jobs == DEFAULT_JOBS else 'manual'}",
        flush=True,
    )
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1")
    args.models = normalize_requested_models(args.models)
    if args.ensemble_size < 1:
        raise ValueError("ensemble-size must be positive")
    if args.max_estimators < 1:
        raise ValueError("max-estimators must be positive")
    if args.early_stopping_rounds < 1:
        raise ValueError("early-stopping-rounds must be positive")
    if args.tree_count_validation_races < 1:
        raise ValueError("tree-count-validation-races must be positive")
    database = args.db.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "winner_ranker_bundle.json"
    oof_path = output_dir / "all_finished_oof_predictions.csv"
    sweep_path = output_dir / "all_finished_weight_sweep.csv"
    recommendation_path = output_dir / "all_finished_blend.json"
    feature_manifest = args.features_json.resolve()
    if args.reuse_unselected_models and not args.models:
        raise ValueError("--reuse-unselected-models requires --models")
    if args.retune_only and args.models:
        raise ValueError("--retune-only and --models cannot be used together")
    if args.retune_only:
        retune_saved_predictions(args, output_dir, feature_manifest, bundle_path)
        return
    if not args.skip_feature_update:
        subprocess.run([
            sys.executable,
            str(Path(__file__).with_name("update_derived_racing_features.py")),
            "--db", str(database),
        ], check=True)

    numeric_columns = database_numeric_columns(database)
    frame = load_training_rows(database, numeric_columns)
    races = eligible_races(frame, args.minimum_runners)
    eligible_ids = races["race_id"].astype(int).tolist()
    all_finished = rows_for_races(frame, eligible_ids)
    all_finished_audit = validate_ranker_groups(
        all_finished,
        all_finished["is_winner"].to_numpy(dtype=np.int64),
        group_sizes(all_finished),
    )
    eligible_features, duplicates = select_form_features(
        all_finished, numeric_columns, args.minimum_feature_coverage
    )
    # Duplicate detection is useful diagnostics, but a JSON group may
    # intentionally select either copy without also selecting its twin.
    eligible_features.extend(
        feature for feature in duplicates if feature not in eligible_features
    )
    for feature in numeric_columns:
        if not is_current_market_feature(feature) or feature == "sp_starting_price":
            continue
        values = pd.to_numeric(all_finished[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if (
            float(values.notna().mean()) >= args.minimum_feature_coverage
            and int(values.nunique(dropna=True)) > 1
            and feature not in eligible_features
        ):
            eligible_features.append(feature)
    if not eligible_features:
        raise ValueError("No eligible model features")
    feature_sets = load_model_feature_sets(feature_manifest, eligible_features)
    feature_sets, training_labels, reused_labels = select_requested_model_groups(
        feature_sets, args.models, args.reuse_unselected_models
    )
    requested_labels = set(training_labels)
    existing_bundle: dict[str, Any] = {}
    existing_oof: pd.DataFrame | None = None
    if reused_labels:
        for path in (bundle_path, oof_path):
            if not path.is_file():
                raise ValueError(
                    f"Selective retraining requires existing artifact: {path}"
                )
        existing_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        existing_features = existing_bundle.get("model_features", {})
        existing_models = existing_bundle.get("models", {})
        for label in reused_labels:
            if list(existing_features.get(label, [])) != feature_sets[label]:
                raise ValueError(
                    f"Cannot reuse {label}: its JSON features differ from the "
                    "existing bundle; include it in --models"
                )
            paths = list(existing_models.get(label, []))
            if not paths or any(not Path(path).is_file() for path in paths):
                raise ValueError(
                    f"Cannot reuse {label}: existing saved model files are missing"
                )
        existing_oof = pd.read_csv(oof_path)
        # Fail before expensive cross-fitting if saved scores cannot be safely
        # paired with the current cohort.
        try:
            merge_reused_oof_scores(
                all_finished.loc[:, ["race_id", "runner_number"]].copy(),
                existing_oof,
                reused_labels,
            )
        except OOFCohortMismatchError as exc:
            raise ValueError(
                "Cannot reuse unselected models because the existing OOF cohort "
                "does not match the current eligible cohort. Run without "
                "--reuse-unselected-models for an exclusive model run, or rebuild "
                "the reused model scores."
            ) from exc
    print_model_feature_report(feature_manifest, feature_sets)
    fold_ids = crossfit_fold_ids(eligible_ids, args.folds)
    fixed_tree_counts_by_model = (
        {
            label: tree_counts(
                args.source_bundle,
                label,
                args.ensemble_size,
                (
                    args.default_market_aware_estimators
                    if label == "market_aware"
                    else args.default_form_estimators
                ),
            )
            for label in feature_sets
        }
        if args.no_tune_tree_counts else {}
    )
    tree_count_mode = (
        "source_bundle_or_fallback"
        if args.no_tune_tree_counts else "nested_early_stopping"
    )
    tree_count_selection_metric = (
        "not_applicable"
        if args.no_tune_tree_counts
        else tree_count_eval_metrics(args.objective)[-1]
    )
    fixed_counts_line = (
        "fixed_tree_counts_by_model="
        f"{json.dumps(fixed_tree_counts_by_model)}\n"
        if args.no_tune_tree_counts else ""
    )
    print(
        f"source=status_finished active_runner_only=yes "
        f"eligible_races={len(eligible_ids):,} rows={len(all_finished):,} "
        f"folds={args.folds} models={len(feature_sets)} "
        f"duplicates_removed={len(duplicates)}\n"
        f"models_retrained={json.dumps(training_labels)} "
        f"models_reused={json.dumps(reused_labels)}\n"
        f"tree_count_mode={tree_count_mode} "
        f"selection_metric={tree_count_selection_metric} "
        f"max_estimators={args.max_estimators} "
        f"early_stopping_rounds={args.early_stopping_rounds} "
        f"maximum_inner_validation_races="
        f"{args.tree_count_validation_races}\n"
        f"{fixed_counts_line}"
        f"ranker_group_audit={json.dumps(all_finished_audit)}\n"
        "crossfit_guarantee=each_race_scored_by_models_not_trained_on_that_race "
        "sealed_test=no",
        flush=True,
    )

    oof_parts: list[pd.DataFrame] = []
    nested_tree_counts: dict[str, list[list[int]]] = {
        label: [] for label in training_labels
    }
    all_id_set = set(eligible_ids)
    for fold_number, holdout_ids in enumerate(fold_ids, start=1):
        holdout_set = set(holdout_ids)
        training_ids = [race_id for race_id in eligible_ids if race_id not in holdout_set]
        if set(training_ids) & holdout_set or set(training_ids) | holdout_set != all_id_set:
            raise AssertionError("Cross-fit race partition is invalid")
        training = rows_for_races(all_finished, training_ids)
        holdout = rows_for_races(all_finished, holdout_ids)
        train_y = training["is_winner"].to_numpy(dtype=np.int64)
        train_groups = group_sizes(training)
        validate_ranker_groups(training, train_y, train_groups)
        validate_ranker_groups(
            holdout,
            holdout["is_winner"].to_numpy(dtype=np.int64),
            group_sizes(holdout),
        )
        holdout_ids_array = holdout["race_id"].to_numpy(dtype=np.int64)
        model_scores: dict[str, np.ndarray] = {}
        for model_index, (label, configured_features) in enumerate(
            feature_sets.items()
        ):
            if label not in requested_labels:
                continue
            seed_offset = fold_number * 100_000 + model_index * 10_000
            if args.no_tune_tree_counts:
                fold_tree_counts = fixed_tree_counts_by_model[label]
            else:
                fold_tree_counts = tune_tree_counts(
                    args,
                    label,
                    training,
                    configured_features,
                    seed_offset,
                )
                nested_tree_counts[label].append(fold_tree_counts)
            models = fit_ensemble(
                args,
                label,
                model_feature_matrix(training, configured_features),
                train_y,
                train_groups,
                fold_tree_counts,
                seed_offset,
            )
            model_scores[label] = ensemble_rank_scores(
                models,
                model_feature_matrix(holdout, configured_features),
                holdout_ids_array,
            )
        market_score = rank_percentiles(
            market_scores(holdout), holdout_ids_array
        )
        part = score_table(
            holdout,
            holdout["is_winner"].to_numpy(dtype=np.int64),
            {**model_scores, "market": market_score},
        )
        part["crossfit_fold"] = fold_number
        oof_parts.append(part)
        print(
            f"crossfit_fold={fold_number}/{args.folds} "
            f"train_races={len(training_ids):,} holdout_races={len(holdout_ids):,}",
            flush=True,
        )

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["start_time_iso", "race_id", "runner_number"], kind="stable",
        ignore_index=True,
    )
    if oof["race_id"].nunique() != len(eligible_ids):
        raise AssertionError("Not every eligible finished race received OOF scores")
    oof_tree_counts_by_model = {
        label: (
            fixed_tree_counts_by_model[label]
            if args.no_tune_tree_counts
            else aggregate_tree_counts(nested_tree_counts[label])
        )
        for label in training_labels
    }
    print(
        "oof_tree_counts_by_model="
        + json.dumps(oof_tree_counts_by_model, sort_keys=True),
        flush=True,
    )
    if existing_oof is not None:
        oof = merge_reused_oof_scores(oof, existing_oof, reused_labels)
    model_labels = list(feature_sets)
    (
        selected_weights,
        sweep,
        model_oof_metrics,
        metrics,
        deviation,
    ) = dynamic_blend_analysis(
        oof,
        model_labels,
        args.weight_step,
        args.objective,
        args.minimum_form_weight,
    )
    print("ALL-FINISHED MODEL OOF METRICS")
    print(pd.DataFrame(model_oof_metrics).T[[
        "top1_hit_rate", "top3_hit_rate", "mrr",
        "mean_winner_rank", "race_logloss",
    ]].to_string(float_format=lambda value: f"{value:.5f}"), flush=True)

    print(
        "all_finished_selected_weights="
        + json.dumps(selected_weights, sort_keys=True)
    )
    print("ALL-FINISHED GROUPED OOF METRICS")
    print(pd.DataFrame(metrics).T[[
        "top1_hit_rate", "top3_hit_rate", "mrr",
        "mean_winner_rank", "race_logloss",
    ]].to_string(float_format=lambda value: f"{value:.5f}"), flush=True)
    if args.ranker_diagnostics:
        save_oof_ranker_diagnostics(
            oof, model_labels, selected_weights, output_dir
        )

    # OOF fold medians are appropriate for unbiased cross-fit diagnostics. The
    # deployable models have a different job: predict races after the complete
    # historical cohort. Tune their capacity on the latest chronological slice
    # of all available history, then refit at that fixed capacity on every race.
    deployment_tree_counts_by_model = (
        dict(oof_tree_counts_by_model)
        if args.no_tune_tree_counts
        else {
            label: tune_tree_counts(
                args,
                label,
                all_finished,
                feature_sets[label],
                1_000_000 + model_index * 10_000,
            )
            for model_index, label in enumerate(training_labels)
        }
    )
    print(
        "deployment_tree_counts_by_model="
        + json.dumps(deployment_tree_counts_by_model, sort_keys=True),
        flush=True,
    )

    all_y = all_finished["is_winner"].to_numpy(dtype=np.int64)
    all_groups = group_sizes(all_finished)
    model_paths: dict[str, list[str]] = {
        label: list(existing_bundle.get("models", {}).get(label, []))
        for label in reused_labels
    }
    importance_parts: list[pd.DataFrame] = []
    for model_index, (label, configured_features) in enumerate(feature_sets.items()):
        if label not in requested_labels:
            continue
        seed_offset = model_index * 50_000
        final_models = fit_ensemble(
            args,
            label,
            model_feature_matrix(all_finished, configured_features),
            all_y,
            all_groups,
            deployment_tree_counts_by_model[label],
            seed_offset,
        )
        model_paths[label] = save_ensemble(
            final_models, label, output_dir, args.seed + seed_offset
        )
        if args.ranker_diagnostics:
            importance_parts.append(
                xgb_ensemble_feature_importance(final_models, label)
            )

    if importance_parts:
        pd.concat(importance_parts, ignore_index=True).to_csv(
            output_dir / "all_finished_feature_importance.csv", index=False
        )

    versions = sorted(
        str(value) for value in all_finished[
            "derived_racing_features_version"
        ].dropna().unique()
    )
    recommendation = {
        "schema_version": 2,
        "blend": "all_finished_crossfit_dynamic_model_groups",
        "selection_cohort": "all_eligible_finished_races_grouped_oof",
        "raw_market_weight_fixed": 0.0,
        "objective": args.objective,
        "selected_weights": selected_weights,
        "model_labels": model_labels,
        "model_oof_metrics": model_oof_metrics,
        "oof_metrics": metrics,
        "oof_market_deviation": deviation,
        "eligible_finished_races": len(eligible_ids),
        "eligible_finished_rows": len(all_finished),
        "crossfit_folds": args.folds,
        "tree_count_selection": tree_count_mode,
        "tree_count_selection_metric": tree_count_selection_metric,
        "oof_median_tree_counts": oof_tree_counts_by_model,
        "tuned_model_tree_counts": deployment_tree_counts_by_model,
        "deployment_tree_count_selection": (
            "source_bundle_or_fallback"
            if args.no_tune_tree_counts
            else "full_history_chronological_tail_early_stopping"
        ),
        "tree_count_max_estimators": args.max_estimators,
        "tree_count_early_stopping_rounds": args.early_stopping_rounds,
        "tree_count_maximum_inner_validation_races": (
            args.tree_count_validation_races
        ),
        "sealed_test_available": False,
    }
    deployment_model = "form" if "form" in feature_sets else next(iter(feature_sets))
    deployment_uses_market = any(
        feature in MARKET_ENGINEERED_FEATURES or is_current_market_feature(feature)
        for feature in feature_sets[deployment_model]
    )
    best_tree_counts = {
        label: list(existing_bundle.get("best_tree_counts", {}).get(label, []))
        for label in reused_labels
    }
    best_tree_counts.update({
        label: deployment_tree_counts_by_model[label] for label in training_labels
    })
    bundle = {
        "schema_version": 3,
        "objective": "single_winner_ranking",
        "training_scope": "all_eligible_finished_races",
        "competition_scope": "all_eligible_races",
        "competition_id_feature_used": False,
        "form_features": feature_sets.get("form", []),
        "model_features": feature_sets,
        "feature_manifest": str(feature_manifest),
        "deployment_default": deployment_model,
        "deployment_uses_current_market": deployment_uses_market,
        "deployment_blend_weights": {
            label: 1.0 if label == deployment_model else 0.0
            for label in [*feature_sets, "market"]
        },
        "selected_blend_weights": selected_weights,
        "all_finished_tuned_blend_weights": selected_weights,
        "market_engineered_features": list(MARKET_ENGINEERED_FEATURES),
        "feature_duplicates_removed": duplicates,
        "derived_feature_versions": versions,
        "models": model_paths,
        "best_tree_counts": best_tree_counts,
        "tree_count_selection": tree_count_mode,
        "all_finished_crossfit": recommendation,
        "database": str(database),
        "seed": args.seed,
    }
    oof.to_csv(oof_path, index=False)
    sweep.to_csv(sweep_path, index=False)
    recommendation_path.write_text(
        json.dumps(_jsonable(recommendation), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle_path.write_text(
        json.dumps(_jsonable(bundle), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"saved_bundle={bundle_path}\n"
        f"feature_file={feature_manifest}\n"
        f"saved_blend={recommendation_path}\n"
        f"saved_oof_predictions={oof_path}\n"
        f"saved_weight_sweep={sweep_path}\n"
        "rank_command=python rank_winner_models.py --race-id RACE_ID "
        f"--ranking tuned --bundle {bundle_path} "
        f"--blend-config {recommendation_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
