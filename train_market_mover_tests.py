#!/usr/bin/env python3
"""Run leakage-aware chronological winner or Top-3 feature selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required: pip install xgboost") from exc

from src.config import DEFAULT_DB
from src.winner_ranker import (
    categorical_levels,
    database_numeric_columns,
    database_text_columns,
    eligible_races,
    group_sizes,
    is_current_market_feature,
    load_training_rows,
    model_feature_matrix,
    prepare_categorical_features,
    rows_for_races,
)
from train_winner_ranker_pipeline import model_parameters


CPU_THREADS = os.cpu_count() or 1
DEFAULT_JOBS = max(1, int(CPU_THREADS * 0.80))


def parse_competition_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "competition IDs must be comma-separated positive integers"
        )
    try:
        competition_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "competition IDs must be comma-separated positive integers"
        ) from exc
    if any(competition_id < 1 for competition_id in competition_ids):
        raise argparse.ArgumentTypeError(
            "competition IDs must be comma-separated positive integers"
        )
    return list(dict.fromkeys(competition_ids))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features-json", type=Path, default=Path("test.json"))
    parser.add_argument(
        "--results-json", type=Path,
        default=Path("outputs/market_mover_test_results.json"),
    )
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument(
        "--test-races",
        type=int,
        default=200,
        help=(
            "Latest chronological races reserved for one sealed evaluation "
            "after forward selection (default: 200)."
        ),
    )
    parser.add_argument(
        "--competition-id", "--competition-ids",
        dest="competition_ids",
        type=parse_competition_ids,
        metavar="ID[,ID...]",
        help=(
            "Train and validate only on races with these competition IDs. "
            "Accepts one ID or a comma-separated list."
        ),
    )
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--max-estimators", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
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
        "--selection-objective",
        choices=("winner", "top3"),
        default="winner",
        help=(
            "Primary training/selection target: winner uses is_winner and "
            "winner top-1; top3 uses top3_mask and top-3 capture."
        ),
    )
    parser.add_argument(
        "--minimum-uplift",
        type=float,
        default=0.01,
        help=(
            "Minimum absolute primary-metric change in the selected direction "
            "required to add a feature (default: 0.01, or one percentage point)."
        ),
    )
    parser.add_argument(
        "--include-current-market",
        action="store_true",
        help=(
            "Allow current-race prices and market-derived features. By default "
            "they are excluded so the selected model remains market-free."
        ),
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--forward-select",
        action="store_true",
        help=(
            "Greedily add the best improving feature, then retest all remaining "
            "features against the expanded baseline until none improves it."
        ),
    )
    parser.add_argument(
        "--reverse-select",
        action="store_true",
        help=(
            "With --forward-select, greedily add the candidate with the lowest "
            "primary score each round instead of the highest. This deliberately "
            "constructs a degrading/adversarial feature set."
        ),
    )
    parser.add_argument(
        "--add-all-improving-after-round-one",
        action="store_true",
        help=(
            "After testing round one, add every candidate meeting --minimum-uplift "
            "instead of only the best. The combined feature set is refitted, then "
            "later rounds return to normal best-one greedy selection."
        ),
    )
    parser.add_argument(
        "--max-forward-rounds",
        type=int,
        help="Optional safety limit on successful --forward-select rounds.",
    )
    parser.add_argument(
        "--limit-models", type=int,
        help="Run only the first N manifest models (useful for a quick check).",
    )
    return parser.parse_args()


def load_feature_sets(
    path: Path, allow_forward_pool: bool = False
) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("models"), dict):
        raise ValueError(f"Invalid model feature manifest: {path}")
    result: dict[str, list[str]] = {}
    for label, config in payload["models"].items():
        features = config.get("features") if isinstance(config, dict) else None
        if not isinstance(features, list) or not features:
            raise ValueError(f"Invalid features for model {label}")
        if not all(isinstance(feature, str) and feature for feature in features):
            raise ValueError(f"Invalid feature name for model {label}")
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicate features for model {label}")
        result[str(label)] = features
    if not result:
        raise ValueError("Feature manifest contains no models")
    declared_base = payload.get("base_features")
    if declared_base is not None:
        if (
            not isinstance(declared_base, list)
            or not declared_base
            or len(declared_base) != len(set(declared_base))
        ):
            raise ValueError("Manifest base_features must be a non-empty unique list")
        if not allow_forward_pool:
            for label, features in result.items():
                tested = [feature for feature in features if feature not in declared_base]
                if features[:len(declared_base)] != declared_base or len(tested) != 1:
                    raise ValueError(
                        f"{label} must contain the declared base_features followed "
                        "by exactly one non-base tested feature"
                    )
    declared_excluded = payload.get("excluded_features", [])
    if not isinstance(declared_excluded, list) or len(declared_excluded) != len(
        set(declared_excluded)
    ):
        raise ValueError("Manifest excluded_features must be a unique list")
    if not allow_forward_pool:
        tested_features = {features[-1] for features in result.values()}
        overlap = sorted(tested_features & set(declared_excluded))
        if overlap:
            raise ValueError(
                "Excluded features cannot be tested: " + ", ".join(overlap)
            )
    return result


def forward_feature_pool(
    path: Path,
    feature_sets: dict[str, list[str]],
    include_current_market: bool = False,
) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    """Resolve a declared base and unique candidate pool from any model shapes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared_base = payload.get("base_features")
    if not isinstance(declared_base, list) or not declared_base:
        inferred_base, _ = infer_ablation_features(feature_sets)
        declared_base = inferred_base
    if (
        not all(isinstance(feature, str) and feature for feature in declared_base)
        or len(declared_base) != len(set(declared_base))
    ):
        raise ValueError("Manifest base_features must be a non-empty unique list")
    market_base = [
        feature for feature in declared_base if is_current_market_feature(feature)
    ]
    if market_base and not include_current_market:
        raise ValueError(
            "Manifest base_features contain current-market inputs; either remove "
            "them or pass --include-current-market: " + ", ".join(market_base)
        )
    excluded = payload.get("excluded_features", [])
    if not isinstance(excluded, list):
        raise ValueError("Manifest excluded_features must be a list")
    forbidden = set(declared_base) | set(excluded)

    candidates: list[str] = []
    for features in feature_sets.values():
        for feature in features:
            if (
                feature not in forbidden
                and (include_current_market or not is_current_market_feature(feature))
                and feature not in candidates
            ):
                candidates.append(feature)
    if not candidates:
        raise ValueError("No non-base, non-excluded forward-selection candidates found")
    candidate_sets = {
        f"candidate_{index}": [*declared_base, feature]
        for index, feature in enumerate(candidates, start=1)
    }
    additions = {
        label: features[-1] for label, features in candidate_sets.items()
    }
    return list(declared_base), candidate_sets, additions


def infer_ablation_features(
    feature_sets: dict[str, list[str]],
) -> tuple[list[str], dict[str, str]]:
    """Infer shared bases and require exactly one tested feature per model."""
    first_features = next(iter(feature_sets.values()))
    common = set(first_features)
    for features in feature_sets.values():
        common.intersection_update(features)
    base_features = [feature for feature in first_features if feature in common]
    if not base_features:
        raise ValueError("Models do not share any base features")

    additions: dict[str, str] = {}
    for label, features in feature_sets.items():
        extra = [feature for feature in features if feature not in common]
        missing_base = [feature for feature in base_features if feature not in features]
        if missing_base or len(extra) != 1:
            raise ValueError(
                f"{label} must contain all shared base features plus exactly one "
                f"tested feature; found tested features={extra}"
            )
        additions[label] = extra[0]
    return base_features, additions


def recommended_validation_races(total_races: int) -> int:
    """Recommend a useful holdout while retaining most races for training."""
    if total_races < 2:
        raise ValueError("At least two eligible races are required")
    return min(1000, max(1, int(round(total_races * 0.20))))


def top3_capture(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, Any]:
    scored = frame.loc[:, ["race_id", "top3_mask", "is_winner"]].copy()
    scored["score"] = np.asarray(scores, dtype=np.float64)
    hits = 0.0
    races = 0
    races_3of3 = 0
    races_2plus = 0
    winner_hits = 0.0
    races_with_score_ties = 0
    for _, race in scored.groupby("race_id", sort=False):
        ordered = race.sort_values("score", ascending=False, kind="stable")
        score_counts = ordered.groupby("score", sort=False, dropna=False).size()
        races_with_score_ties += int((score_counts > 1).any())

        # Give expected credit at tied score boundaries. This prevents arbitrary
        # database/runner order from making a race-constant feature look useful.
        slots = 3
        race_hits = 0.0
        for _, tier in ordered.groupby("score", sort=False, dropna=False):
            if slots <= 0:
                break
            take_fraction = min(slots, len(tier)) / len(tier)
            race_hits += float(pd.to_numeric(tier["top3_mask"]).sum()) * take_fraction
            slots -= min(slots, len(tier))
        top_tier = ordered.loc[ordered["score"] == ordered.iloc[0]["score"]]
        winner_hits += (
            float(pd.to_numeric(top_tier["is_winner"]).sum()) / len(top_tier)
        )
        hits += race_hits
        races += 1
        races_3of3 += int(np.isclose(race_hits, 3.0))
        races_2plus += int(race_hits >= 2.0)
    possible = races * 3
    return {
        "validation_races": races,
        "top3_hits": hits,
        "possible_top3_hits": possible,
        "top3_capture_rate": hits / possible if possible else 0.0,
        "races_with_3_of_3": races_3of3,
        "races_with_2plus_of_3": races_2plus,
        "winner_hits": winner_hits,
        "winner_hit_rate": winner_hits / races if races else 0.0,
        "races_with_score_ties": races_with_score_ties,
        "tie_handling": "expected_credit",
    }


def best_improving_result(
    results: list[dict[str, Any]],
    baseline_rate: float,
    metric: str = "top3_capture_rate",
    minimum_uplift: float = 0.0,
    reverse: bool = False,
) -> dict[str, Any] | None:
    """Return the best material directional change, or no selection."""
    directional = [
        result for result in results
        if (
            float(baseline_rate) - float(result[metric])
            if reverse
            else float(result[metric]) - float(baseline_rate)
        ) + 1e-12 >= minimum_uplift
        and (
            float(result[metric]) < float(baseline_rate)
            if reverse
            else float(result[metric]) > float(baseline_rate)
        )
    ]
    if not directional:
        return None
    secondary = (
        "top3_capture_rate" if metric == "winner_hit_rate" else "winner_hit_rate"
    )
    if reverse:
        return min(
            directional,
            key=lambda result: (
                float(result[metric]),
                float(result[secondary]),
                int(result["candidate_order"]),
            ),
        )
    return max(
        directional,
        key=lambda result: (
            float(result[metric]),
            float(result[secondary]),
            -int(result["candidate_order"]),
        ),
    )


def all_qualifying_results(
    results: list[dict[str, Any]], reverse: bool = False
) -> list[dict[str, Any]]:
    """Return candidates already marked as meeting the round threshold."""
    qualifying_status = "lowers" if reverse else "improves"
    return [result for result in results if result.get("status") == qualifying_status]


def forward_selection_model_parameters(
    parameter_args: SimpleNamespace,
    seed: int,
    max_estimators: int,
    selection_objective: str = "top3",
) -> dict[str, Any]:
    """Return stable parameters for nested forward-feature comparisons."""
    parameters = model_parameters(parameter_args, seed, max_estimators)
    # Sampling makes a candidate fit non-nested: merely adding a column changes
    # which baseline columns and rows each tree sees. Full sampling lets XGBoost
    # ignore an unhelpful candidate and makes forward comparisons interpretable.
    parameters["colsample_bytree"] = 1.0
    parameters["subsample"] = 1.0
    parameters["eval_metric"] = (
        ["ndcg@1", "ndcg@3", "map"]
        if selection_objective == "winner"
        else ["ndcg@1", "map", "ndcg@3"]
    )
    return parameters


def enable_native_categorical(
    parameters: dict[str, Any], matrix: pd.DataFrame
) -> dict[str, Any]:
    """Enable XGBoost categorical splits when the manifest matrix needs them."""
    result = dict(parameters)
    if any(isinstance(dtype, pd.CategoricalDtype) for dtype in matrix.dtypes):
        result["enable_categorical"] = True
    return result


def fit_feature_set(
    args: argparse.Namespace,
    parameter_args: SimpleNamespace,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    features: list[str],
    selection_objective: str = "top3",
) -> dict[str, Any]:
    parameters = forward_selection_model_parameters(
        parameter_args, args.seed, args.max_estimators, selection_objective
    )
    train_matrix = model_feature_matrix(training, features)
    validation_matrix = model_feature_matrix(validation, features)
    model = XGBRanker(
        **enable_native_categorical(parameters, train_matrix),
        early_stopping_rounds=args.early_stopping_rounds,
    )
    model.fit(
        train_matrix,
        train_y,
        group=train_groups,
        eval_set=[(validation_matrix, validation_y)],
        eval_group=[validation_groups],
        verbose=False,
    )
    metrics = top3_capture(validation, model.predict(validation_matrix))
    result = {
        "features": list(features),
        "best_iteration": int(model.best_iteration) + 1,
        **metrics,
    }
    del model
    return result


def refit_and_evaluate_feature_set(
    args: argparse.Namespace,
    parameter_args: SimpleNamespace,
    training: pd.DataFrame,
    test: pd.DataFrame,
    train_y: np.ndarray,
    train_groups: np.ndarray,
    features: list[str],
    estimators: int,
    selection_objective: str,
) -> dict[str, Any]:
    """Refit a selected design with fixed trees and score the sealed test once."""
    parameters = forward_selection_model_parameters(
        parameter_args, args.seed, estimators, selection_objective
    )
    train_matrix = model_feature_matrix(training, features)
    test_matrix = model_feature_matrix(test, features)
    model = XGBRanker(**enable_native_categorical(parameters, train_matrix))
    model.fit(train_matrix, train_y, group=train_groups, verbose=False)
    metrics = top3_capture(test, model.predict(test_matrix))
    metrics["test_races"] = metrics.pop("validation_races")
    del model
    return {
        "features": list(features),
        "estimators": int(estimators),
        **metrics,
    }


def validate_production_selection_scope(
    competition_ids: list[int] | None,
) -> None:
    """Reject the known post-result market-miss label as a selection cohort."""
    if competition_ids and 999 in competition_ids:
        raise ValueError(
            "competition_id=999 is assigned after results to races where the "
            "market top three completely missed the actual top three. It may be "
            "used for diagnostics, but not production feature selection. Select "
            "a genuine competition (for example --competition-id 6) or omit the "
            "competition filter."
        )


def save_forward_results(
    path: Path,
    database: Path,
    manifest: Path,
    train_races: int,
    validation_races: int,
    competition_ids: list[int] | None,
    initial_base_features: list[str],
    selected_features: list[str],
    rounds: list[dict[str, Any]],
    completed: bool,
    *,
    selection_objective: str,
    selection_direction: str,
    minimum_uplift: float,
    test_races: int,
    sealed_test: dict[str, Any] | None = None,
) -> None:
    target = "is_winner" if selection_objective == "winner" else "top3_mask"
    primary_metric = (
        "winner_hit_rate"
        if selection_objective == "winner"
        else "top3_capture_rate"
    )
    payload = {
        "schema_version": 2,
        "mode": "greedy_forward_selection",
        "target": target,
        "primary_metric": primary_metric,
        "selection_direction": selection_direction,
        "early_stopping_metric": (
            "map" if selection_objective == "winner" else "ndcg@3"
        ),
        "minimum_uplift": minimum_uplift,
        "database": str(database),
        "feature_manifest": str(manifest),
        "train_races": train_races,
        "validation_races": validation_races,
        "test_races": test_races,
        "competition_ids": competition_ids,
        "initial_base_features": initial_base_features,
        "selected_features": selected_features,
        "final_features": [*initial_base_features, *selected_features],
        "rounds": rounds,
        "sealed_test": sealed_test,
        "completed": completed,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_forward_selection(
    args: argparse.Namespace,
    parameter_args: SimpleNamespace,
    database: Path,
    manifest_path: Path,
    results_path: Path,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    train_race_count: int,
    validation_race_count: int,
    base_features: list[str],
    feature_sets: dict[str, list[str]],
    added_features: dict[str, str],
) -> None:
    primary_metric = (
        "winner_hit_rate"
        if args.selection_objective == "winner"
        else "top3_capture_rate"
    )
    primary_label = "winner#1" if args.selection_objective == "winner" else "top3"
    selection_direction = "minimize" if args.reverse_select else "maximize"
    current_features = list(base_features)
    remaining = [
        (label, added_features[label]) for label in feature_sets
    ]
    selected_features: list[str] = []
    rounds: list[dict[str, Any]] = []
    baseline = fit_feature_set(
        args, parameter_args, training, validation, train_y, validation_y,
        train_groups, validation_groups, current_features,
        args.selection_objective,
    )
    initial_baseline = dict(baseline)
    print(
        "\nFORWARD BASELINE "
        f"top3={baseline['top3_capture_rate']:.2%} "
        f"winner#1={baseline['winner_hit_rate']:.2%} "
        f"tied_races={baseline['races_with_score_ties']}/{baseline['validation_races']} "
        f"features={len(current_features)}",
        flush=True,
    )
    print("forward_sampling=full colsample_bytree=1.0 subsample=1.0", flush=True)
    print(
        f"selection_objective={args.selection_objective} "
        f"primary_metric={primary_metric} "
        f"selection_direction={selection_direction} "
        f"minimum_uplift={args.minimum_uplift:.2%} "
        f"sealed_test_races={test['race_id'].nunique():,}",
        flush=True,
    )

    def save_progress(
        completed: bool, sealed_test: dict[str, Any] | None = None
    ) -> None:
        save_forward_results(
            results_path, database, manifest_path, train_race_count,
            validation_race_count, args.competition_ids, base_features,
            selected_features, rounds, completed=completed,
            selection_objective=args.selection_objective,
            selection_direction=selection_direction,
            minimum_uplift=args.minimum_uplift,
            test_races=int(test["race_id"].nunique()),
            sealed_test=sealed_test,
        )

    round_number = 1
    while remaining:
        if (
            args.max_forward_rounds is not None
            and round_number > args.max_forward_rounds
        ):
            break
        baseline_rate = float(baseline[primary_metric])
        round_record: dict[str, Any] = {
            "round": round_number,
            "baseline": baseline,
            "candidate_results": [],
            "selected_feature": None,
            "selection_direction": selection_direction,
        }
        rounds.append(round_record)
        print(
            f"\nFORWARD ROUND {round_number} "
            f"baseline_{primary_label}={baseline_rate:.2%} "
            f"remaining={len(remaining):,}",
            flush=True,
        )
        candidate_results: list[dict[str, Any]] = round_record["candidate_results"]
        for candidate_order, (label, feature) in enumerate(remaining):
            tested_features = [*current_features, feature]
            result = fit_feature_set(
                args, parameter_args, training, validation, train_y,
                validation_y, train_groups, validation_groups, tested_features,
                args.selection_objective,
            )
            result.update({
                "model": label,
                "added_feature": feature,
                "candidate_order": candidate_order,
                "top3_uplift_vs_round_baseline": (
                    float(result["top3_capture_rate"])
                    - float(baseline["top3_capture_rate"])
                ),
                "winner_uplift_vs_round_baseline": (
                    float(result["winner_hit_rate"])
                    - float(baseline["winner_hit_rate"])
                ),
            })
            primary_delta = float(result[primary_metric]) - baseline_rate
            directional_change = (
                -primary_delta if args.reverse_select else primary_delta
            )
            result["status"] = (
                "lowers" if args.reverse_select else "improves"
            ) if directional_change + 1e-12 >= args.minimum_uplift else "skipped"
            candidate_results.append(result)
            print(
                f"  [{candidate_order + 1:>3}/{len(remaining)}] "
                f"feature={feature:<50} "
                f"top3={result['top3_capture_rate']:.2%} "
                f"winner#1={result['winner_hit_rate']:.2%} "
                f"primary_delta={primary_delta:+.2%} "
                f"status={result['status'].upper()}",
                flush=True,
            )
            save_progress(completed=False)

        best = best_improving_result(
            candidate_results,
            baseline_rate,
            metric=primary_metric,
            minimum_uplift=args.minimum_uplift,
            reverse=args.reverse_select,
        )
        if best is None:
            round_record["stop_reason"] = (
                "no_remaining_feature_met_minimum_primary_decrease"
                if args.reverse_select
                else "no_remaining_feature_met_minimum_primary_uplift"
            )
            if args.reverse_select:
                print(
                    "No remaining feature met the minimum primary-metric "
                    "decrease; stopping."
                )
            else:
                print(
                    "No remaining feature met the minimum primary-metric "
                    "uplift; stopping."
                )
            break

        if round_number == 1 and args.add_all_improving_after_round_one:
            qualifying = all_qualifying_results(candidate_results)
            chosen_features = [str(result["added_feature"]) for result in qualifying]
            for result in qualifying:
                result["status"] = "bulk_selected"
            round_record["selection_mode"] = "all_qualifying_candidates"
            round_record["selected_features"] = chosen_features
            round_record["selected_primary_metric"] = primary_metric
            selected_features.extend(chosen_features)
            current_features.extend(chosen_features)
            chosen_set = set(chosen_features)
            remaining = [
                (label, feature) for label, feature in remaining
                if feature not in chosen_set
            ]
            # Individual improvements are not additive. Refit the combined set
            # so the next round compares candidates with the true joint baseline.
            baseline = fit_feature_set(
                args, parameter_args, training, validation, train_y,
                validation_y, train_groups, validation_groups,
                current_features, args.selection_objective,
            )
            round_record["combined_baseline"] = baseline
            round_record["selected_primary_rate"] = baseline[primary_metric]
            print(
                f"BULK SELECTED round=1 features={len(chosen_features)} "
                f"joint_{primary_label}={baseline[primary_metric]:.2%} "
                f"total_features={len(current_features)}\n"
                f"bulk_selected_features={json.dumps(chosen_features)}",
                flush=True,
            )
            save_progress(completed=False)
            round_number += 1
            continue

        chosen = str(best["added_feature"])
        best["status"] = "selected"
        round_record["selected_feature"] = chosen
        round_record["selected_primary_metric"] = primary_metric
        round_record["selected_primary_rate"] = best[primary_metric]
        selected_features.append(chosen)
        current_features.append(chosen)
        remaining = [(label, feature) for label, feature in remaining if feature != chosen]
        baseline = {
            key: value for key, value in best.items()
            if key not in {
                "model", "added_feature", "candidate_order",
                "top3_uplift_vs_round_baseline",
                "winner_uplift_vs_round_baseline", "status",
            }
        }
        print(
            f"SELECTED round={round_number} feature={chosen} "
            f"new_{primary_label}={baseline[primary_metric]:.2%} "
            f"total_features={len(current_features)}",
            flush=True,
        )
        round_number += 1
    if (
        args.max_forward_rounds is not None
        and round_number > args.max_forward_rounds
        and remaining
    ):
        print(f"Stopped at --max-forward-rounds={args.max_forward_rounds}.")

    refit_training = pd.concat([training, validation], ignore_index=True)
    refit_y = refit_training[
        "is_winner" if args.selection_objective == "winner" else "top3_mask"
    ].to_numpy(dtype=np.int64)
    sealed_selected = refit_and_evaluate_feature_set(
        args,
        parameter_args,
        refit_training,
        test,
        refit_y,
        group_sizes(refit_training),
        current_features,
        int(baseline["best_iteration"]),
        args.selection_objective,
    )
    sealed_baseline = refit_and_evaluate_feature_set(
        args,
        parameter_args,
        refit_training,
        test,
        refit_y,
        group_sizes(refit_training),
        base_features,
        int(initial_baseline["best_iteration"]),
        args.selection_objective,
    )
    sealed_test = {
        "baseline": sealed_baseline,
        "selected": sealed_selected,
        "winner_uplift": (
            float(sealed_selected["winner_hit_rate"])
            - float(sealed_baseline["winner_hit_rate"])
        ),
        "top3_uplift": (
            float(sealed_selected["top3_capture_rate"])
            - float(sealed_baseline["top3_capture_rate"])
        ),
    }
    save_progress(completed=True, sealed_test=sealed_test)

    print("\n" + "=" * 88)
    print("FINAL FORWARD-SELECTED FEATURE LIST")
    print("=" * 88)
    print(json.dumps(current_features, indent=2))
    print(f"\nadded_features={json.dumps(selected_features)}")
    print(
        f"selection_{primary_label}={baseline[primary_metric]:.2%}\n"
        f"sealed_baseline_top3={sealed_baseline['top3_capture_rate']:.2%}\n"
        f"sealed_selected_top3={sealed_selected['top3_capture_rate']:.2%} "
        f"uplift={sealed_test['top3_uplift']:+.2%}\n"
        f"sealed_baseline_winner#1={sealed_baseline['winner_hit_rate']:.2%}\n"
        f"sealed_selected_winner#1={sealed_selected['winner_hit_rate']:.2%} "
        f"uplift={sealed_test['winner_uplift']:+.2%}\n"
        "NOTE sealed-test metrics were evaluated once after feature selection."
    )
    print(f"results={results_path}")


def save_results(
    path: Path,
    database: Path,
    manifest: Path,
    train_races: int,
    validation_races: int,
    base_features: list[str],
    competition_ids: list[int] | None,
    base_result: dict[str, Any],
    results: list[dict[str, Any]],
    completed: bool,
    *,
    selection_objective: str,
    minimum_uplift: float,
    test_races: int,
    sealed_test: dict[str, Any] | None = None,
) -> None:
    primary_metric = (
        "winner_hit_rate"
        if selection_objective == "winner"
        else "top3_capture_rate"
    )
    ranked = sorted(
        results,
        key=lambda row: (-float(row[primary_metric]), str(row["model"])),
    )
    payload = {
        "schema_version": 2,
        "target": "is_winner" if selection_objective == "winner" else "top3_mask",
        "primary_metric": primary_metric,
        "early_stopping_metric": (
            "map" if selection_objective == "winner" else "ndcg@3"
        ),
        "minimum_uplift": minimum_uplift,
        "database": str(database),
        "feature_manifest": str(manifest),
        "train_races": train_races,
        "validation_races": validation_races,
        "test_races": test_races,
        "base_features": base_features,
        "competition_ids": competition_ids,
        "base_result": base_result,
        "completed": completed,
        "models_completed": len(results),
        "selected_models": sum(row.get("status") == "selected" for row in results),
        "skipped_models": sum(row.get("status") == "skipped" for row in results),
        "results": ranked,
        "sealed_test": sealed_test,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    print(
        f"cpu_threads={CPU_THREADS}\n"
        f"xgboost_jobs={args.jobs}\n"
        f"cpu_target={'80%' if args.jobs == DEFAULT_JOBS else 'manual'}",
        flush=True,
    )
    if args.validation_races < 1 or args.test_races < 1 or args.max_estimators < 1:
        raise ValueError(
            "validation-races, test-races and max-estimators must be positive"
        )
    if args.early_stopping_rounds < 1 or args.jobs < 1 or args.top < 1:
        raise ValueError("early-stopping-rounds, jobs and top must be positive")
    if args.max_forward_rounds is not None and args.max_forward_rounds < 1:
        raise ValueError("max-forward-rounds must be positive")
    if args.max_forward_rounds is not None and not args.forward_select:
        raise ValueError("--max-forward-rounds requires --forward-select")
    if args.reverse_select and not args.forward_select:
        raise ValueError("--reverse-select requires --forward-select")
    if args.add_all_improving_after_round_one and not args.forward_select:
        raise ValueError(
            "--add-all-improving-after-round-one requires --forward-select"
        )
    if args.add_all_improving_after_round_one and args.reverse_select:
        raise ValueError(
            "--add-all-improving-after-round-one cannot be combined with "
            "--reverse-select"
        )
    if not 0.0 < args.minimum_uplift < 1.0:
        raise ValueError("--minimum-uplift must be between zero and one")
    validate_production_selection_scope(args.competition_ids)

    database = args.db.resolve()
    manifest_path = args.features_json.resolve()
    results_path = args.results_json.resolve()
    feature_sets = load_feature_sets(
        manifest_path, allow_forward_pool=args.forward_select
    )
    if args.forward_select:
        base_features, feature_sets, added_features = forward_feature_pool(
            manifest_path,
            feature_sets,
            include_current_market=args.include_current_market,
        )
    else:
        base_features, added_features = infer_ablation_features(feature_sets)
    if args.limit_models is not None:
        if args.limit_models < 1:
            raise ValueError("limit-models must be positive")
        feature_sets = dict(list(feature_sets.items())[:args.limit_models])

    numeric_columns = database_numeric_columns(database)
    text_columns = database_text_columns(database)
    manifest_features = list(dict.fromkeys(
        feature for features in feature_sets.values() for feature in features
    ))
    manifest_categorical = [
        feature for feature in manifest_features if feature in set(text_columns)
    ]
    frame = load_training_rows(
        database,
        numeric_columns,
        categorical_columns=manifest_categorical,
    )
    if args.competition_ids:
        frame = frame.loc[
            pd.to_numeric(frame["competition_id"], errors="coerce").isin(
                args.competition_ids
            )
        ].copy()
        if frame.empty:
            raise ValueError(
                "No finished active runners found for competition IDs: "
                + ", ".join(map(str, args.competition_ids))
            )
    if "top3_mask" not in frame:
        raise ValueError("race_runners does not contain top3_mask")
    valid_top3 = pd.to_numeric(frame["top3_mask"], errors="coerce")
    frame = frame.loc[valid_top3.isin([0, 1])].copy()
    frame["top3_mask"] = valid_top3.loc[frame.index].astype(np.int8)
    races = eligible_races(frame, args.minimum_runners)
    held_out_races = args.validation_races + args.test_races
    if len(races) <= held_out_races:
        suggested = recommended_validation_races(len(races))
        raise ValueError(
            f"--validation-races={args.validation_races:,} plus "
            f"--test-races={args.test_races:,} leaves no training cohort; "
            f"found {len(races):,} eligible races. Reduce the two holdouts; a "
            f"rough validation suggestion is {suggested:,} races."
        )
    ordered_ids = races["race_id"].astype(int).tolist()
    train_ids = ordered_ids[:-held_out_races]
    validation_ids = ordered_ids[-held_out_races:-args.test_races]
    test_ids = ordered_ids[-args.test_races:]
    training = rows_for_races(frame, train_ids)
    validation = rows_for_races(frame, validation_ids)
    test = rows_for_races(frame, test_ids)
    saved_categorical_levels = categorical_levels(
        training, manifest_categorical
    )
    training = prepare_categorical_features(training, saved_categorical_levels)
    validation = prepare_categorical_features(validation, saved_categorical_levels)
    test = prepare_categorical_features(test, saved_categorical_levels)
    target_column = (
        "is_winner" if args.selection_objective == "winner" else "top3_mask"
    )
    train_y = training[target_column].to_numpy(dtype=np.int64)
    validation_y = validation[target_column].to_numpy(dtype=np.int64)
    train_groups = group_sizes(training)
    validation_groups = group_sizes(validation)

    print(
        f"database={database}\nmanifest={manifest_path}\n"
        f"models={len(feature_sets):,} train_races={len(train_ids):,} "
        f"validation_races={len(validation_ids):,} "
        f"sealed_test_races={len(test_ids):,}\n"
        f"competition_ids={args.competition_ids or 'all'}\n"
        f"target={target_column} selection_objective={args.selection_objective} "
        f"minimum_uplift={args.minimum_uplift:.2%}\n"
        f"current_market_features={'included' if args.include_current_market else 'excluded'}\n"
        f"base_features={json.dumps(base_features)}\n"
        f"native_categorical_features={json.dumps(manifest_categorical)}",
        flush=True,
    )

    parameter_args = SimpleNamespace(jobs=args.jobs)
    if args.forward_select:
        run_forward_selection(
            args, parameter_args, database, manifest_path, results_path,
            training, validation, test, train_y, validation_y, train_groups,
            validation_groups, len(train_ids), len(validation_ids),
            base_features, feature_sets, added_features,
        )
        return

    results: list[dict[str, Any]] = []
    total = len(feature_sets)
    base_parameters = forward_selection_model_parameters(
        parameter_args, args.seed, args.max_estimators, args.selection_objective
    )
    base_train_matrix = model_feature_matrix(training, base_features)
    base_validation_matrix = model_feature_matrix(validation, base_features)
    base_model = XGBRanker(
        **enable_native_categorical(base_parameters, base_train_matrix),
        early_stopping_rounds=args.early_stopping_rounds,
    )
    base_model.fit(
        base_train_matrix,
        train_y,
        group=train_groups,
        eval_set=[(base_validation_matrix, validation_y)],
        eval_group=[validation_groups],
        verbose=False,
    )
    base_metrics = top3_capture(
        validation, base_model.predict(base_validation_matrix)
    )
    base_result = {
        "model": "base",
        "status": "baseline",
        "features": base_features,
        "best_iteration": int(base_model.best_iteration) + 1,
        **base_metrics,
    }
    print(
        "\nBASE-ONLY RESULT "
        f"top3={base_metrics['top3_capture_rate']:.2%} "
        f"hits={base_metrics['top3_hits']:g}/{base_metrics['possible_top3_hits']} "
        f"winner#1={base_metrics['winner_hit_rate']:.2%} "
        f"winner_hits={base_metrics['winner_hits']:g}/{base_metrics['validation_races']} "
        f"tied_races={base_metrics['races_with_score_ties']}/{base_metrics['validation_races']} "
        f"trees={base_result['best_iteration']}\n",
        flush=True,
    )
    del base_model
    save_results(
        results_path, database, manifest_path, len(train_ids),
        len(validation_ids), base_features, args.competition_ids,
        base_result, results, completed=False,
        selection_objective=args.selection_objective,
        minimum_uplift=args.minimum_uplift,
        test_races=len(test_ids),
    )

    for index, (label, features) in enumerate(feature_sets.items(), start=1):
        unavailable = [feature for feature in features if feature not in frame.columns]
        if unavailable:
            raise ValueError(f"{label} has unavailable features: {', '.join(unavailable)}")
        parameters = forward_selection_model_parameters(
            parameter_args, args.seed, args.max_estimators, args.selection_objective
        )
        train_matrix = model_feature_matrix(training, features)
        validation_matrix = model_feature_matrix(validation, features)
        model = XGBRanker(
            **enable_native_categorical(parameters, train_matrix),
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model.fit(
            train_matrix,
            train_y,
            group=train_groups,
            eval_set=[(validation_matrix, validation_y)],
            eval_group=[validation_groups],
            verbose=False,
        )
        scores = model.predict(validation_matrix)
        metrics = top3_capture(validation, scores)
        extra_feature = added_features[label]
        result = {
            "model": label,
            "added_feature": extra_feature,
            "features": features,
            "best_iteration": int(model.best_iteration) + 1,
            **metrics,
            "top3_uplift_vs_base": (
                float(metrics["top3_capture_rate"])
                - float(base_metrics["top3_capture_rate"])
            ),
            "winner_uplift_vs_base": (
                float(metrics["winner_hit_rate"])
                - float(base_metrics["winner_hit_rate"])
            ),
        }
        result["status"] = (
            "selected"
            if float(result[
                "winner_uplift_vs_base"
                if args.selection_objective == "winner"
                else "top3_uplift_vs_base"
            ]) + 1e-12 >= args.minimum_uplift
            else "skipped"
        )
        results.append(result)
        save_results(
            results_path, database, manifest_path, len(train_ids),
            len(validation_ids), base_features, args.competition_ids,
            base_result, results, completed=index == total,
            selection_objective=args.selection_objective,
            minimum_uplift=args.minimum_uplift,
            test_races=len(test_ids),
        )
        primary_uplift = float(result[
            "winner_uplift_vs_base"
            if args.selection_objective == "winner"
            else "top3_uplift_vs_base"
        ])
        print(
            f"[{index:>3}/{total}] {label:<6} "
            f"feature={str(extra_feature):<50} "
            f"top3={metrics['top3_capture_rate']:.2%} "
            f"winner#1={metrics['winner_hit_rate']:.2%} "
            f"primary_uplift={primary_uplift:+.2%} "
            f"hits={metrics['top3_hits']:g}/{metrics['possible_top3_hits']} "
            f"trees={result['best_iteration']} "
            f"status={result['status'].upper()}",
            flush=True,
        )
        del model

    selected_results = [row for row in results if row["status"] == "selected"]
    primary_metric = (
        "winner_hit_rate"
        if args.selection_objective == "winner"
        else "top3_capture_rate"
    )
    ranked = sorted(
        selected_results,
        key=lambda row: (-float(row[primary_metric]), str(row["model"])),
    )

    chosen_result = ranked[0] if ranked else base_result
    refit_training = pd.concat([training, validation], ignore_index=True)
    refit_y = refit_training[target_column].to_numpy(dtype=np.int64)
    sealed_test = refit_and_evaluate_feature_set(
        args,
        parameter_args,
        refit_training,
        test,
        refit_y,
        group_sizes(refit_training),
        list(chosen_result["features"]),
        int(chosen_result["best_iteration"]),
        args.selection_objective,
    )
    sealed_test["selected_model"] = str(chosen_result["model"])
    save_results(
        results_path, database, manifest_path, len(train_ids),
        len(validation_ids), base_features, args.competition_ids,
        base_result, results, completed=True,
        selection_objective=args.selection_objective,
        minimum_uplift=args.minimum_uplift,
        test_races=len(test_ids),
        sealed_test=sealed_test,
    )
    print("\n" + "=" * 88)
    print(f"TOP {min(args.top, len(ranked))} MARKET-MOVER FEATURE MODELS")
    print("=" * 88)
    print(
        f"{'Rank':<5} {'Model':<8} {'Added feature':<42} "
        f"{'Top3':>8} {'Primary +':>10} {'Winner #1':>10}"
    )
    uplift_field = (
        "winner_uplift_vs_base"
        if args.selection_objective == "winner"
        else "top3_uplift_vs_base"
    )
    for rank, row in enumerate(ranked[:args.top], start=1):
        print(
            f"{rank:<5} {row['model']:<8} {str(row['added_feature']):<42} "
            f"{float(row['top3_capture_rate']):>7.2%} "
            f"{float(row[uplift_field]):>+9.2%} "
            f"{float(row['winner_hit_rate']):>9.2%}"
        )
    if not ranked:
        print("No added feature met the minimum primary-metric uplift.")
    print(
        f"\nselected={len(selected_results):,} "
        f"skipped={len(results) - len(selected_results):,}"
    )
    print(
        f"sealed_test_model={sealed_test['selected_model']} "
        f"top3={sealed_test['top3_capture_rate']:.2%} "
        f"winner#1={sealed_test['winner_hit_rate']:.2%}"
    )
    print(f"\nresults={results_path}")


if __name__ == "__main__":
    main()
