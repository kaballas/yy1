#!/usr/bin/env python3
"""Train an honest chronological, current-market-free winner ranker.

Unlike the older RaceFormer experiment, this pipeline trains on the complete
race population, targets the single winner, and does not anchor predictions to
the market. The deployment ensemble never receives current-race prices. An
unanchored market-aware model and blended benchmark are available only through
an explicit diagnostic flag and can never become the deployment default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:  # pragma: no cover - CLI environment failure
    raise SystemExit("xgboost is required: pip install xgboost") from exc

from src.config import DEFAULT_DB
from src.winner_ranker import (
    MARKET_ENGINEERED_FEATURES,
    blend_scores,
    chronological_race_split,
    database_numeric_columns,
    eligible_races,
    ensemble_rank_scores,
    form_matrix,
    group_sizes,
    load_training_rows,
    market_aware_matrix,
    market_deviation_metrics,
    market_scores,
    rank_percentiles,
    rows_for_races,
    select_blend_weights,
    select_form_features,
    validate_ranker_groups,
    winner_field_size_slices,
    winner_metrics,
    winner_race_report,
    xgb_ensemble_feature_importance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/winner_ranker"))
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--minimum-feature-coverage", type=float, default=0.20)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-step", type=float, default=0.05)
    parser.add_argument(
        "--include-market-aware-benchmark", action="store_true",
        help=(
            "Also train a current-price-aware diagnostic model and select a "
            "benchmark blend. It is never used as the deployment default."
        ),
    )
    parser.add_argument("--skip-feature-update", action="store_true")
    parser.add_argument(
        "--ranker-diagnostics", action="store_true",
        help=(
            "Save per-race results, field-size failure slices, XGBoost training "
            "history, and gain/cover/split feature importance."
        ),
    )
    parser.add_argument(
        "--skip-deployment-refit", action="store_true",
        help="Keep evaluation models instead of refitting deployment models on all history.",
    )
    return parser.parse_args()


def model_parameters(args: argparse.Namespace, seed: int, estimators: int) -> dict[str, Any]:
    return {
        "objective": "rank:ndcg",
        # The final metric remains MAP so early-stopping semantics stay
        # backward compatible; NDCG curves make top-of-list learning visible.
        "eval_metric": ["ndcg@1", "ndcg@3", "map"],
        "n_estimators": estimators,
        "max_depth": 4,
        "learning_rate": 0.025,
        "subsample": 0.80,
        "colsample_bytree": 0.65,
        "min_child_weight": 12,
        "reg_lambda": 10.0,
        "reg_alpha": 0.25,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 8,
        "tree_method": "hist",
        "n_jobs": args.jobs,
        "random_state": seed,
    }


def train_evaluation_ensemble(
    args: argparse.Namespace,
    label: str,
    train_matrix: pd.DataFrame,
    train_targets: np.ndarray,
    train_groups: np.ndarray,
    validation_matrix: pd.DataFrame,
    validation_targets: np.ndarray,
    validation_groups: np.ndarray,
    output_dir: Path,
) -> tuple[list[XGBRanker], list[int], list[str]]:
    train_audit = validate_ranker_groups(train_matrix.assign(
        race_id=np.repeat(np.arange(len(train_groups)), train_groups),
        is_winner=train_targets,
    ), train_targets, train_groups)
    validation_audit = validate_ranker_groups(validation_matrix.assign(
        race_id=np.repeat(np.arange(len(validation_groups)), validation_groups),
        is_winner=validation_targets,
    ), validation_targets, validation_groups)
    print(
        f"ranker_group_audit={label} train={json.dumps(train_audit)} "
        f"validation={json.dumps(validation_audit)}",
        flush=True,
    )
    models: list[XGBRanker] = []
    iterations: list[int] = []
    paths: list[str] = []
    for member in range(args.ensemble_size):
        seed = args.seed + member * 1009
        params = model_parameters(args, seed, args.max_estimators)
        model = XGBRanker(**params, early_stopping_rounds=args.early_stopping_rounds)
        model.fit(
            train_matrix,
            train_targets,
            group=train_groups,
            eval_set=[
                (train_matrix, train_targets),
                (validation_matrix, validation_targets),
            ],
            eval_group=[train_groups, validation_groups],
            verbose=False,
        )
        iteration = int(model.best_iteration) + 1
        path = output_dir / f"{label}_evaluation_seed_{seed}.json"
        model.save_model(path)
        if args.ranker_diagnostics:
            history_path = output_dir / f"{label}_evaluation_seed_{seed}_history.json"
            history_path.write_text(
                json.dumps(model.evals_result(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(
            f"trained={label} member={member + 1}/{args.ensemble_size} "
            f"seed={seed} best_trees={iteration} path={path}",
            flush=True,
        )
        models.append(model)
        iterations.append(iteration)
        paths.append(str(path.resolve()))
    return models, iterations, paths


def refit_deployment_ensemble(
    args: argparse.Namespace,
    label: str,
    matrix: pd.DataFrame,
    targets: np.ndarray,
    groups: np.ndarray,
    iterations: list[int],
    output_dir: Path,
) -> list[str]:
    validate_ranker_groups(matrix.assign(
        race_id=np.repeat(np.arange(len(groups)), groups), is_winner=targets,
    ), targets, groups)
    paths: list[str] = []
    for member, trees in enumerate(iterations):
        seed = args.seed + member * 1009
        model = XGBRanker(**model_parameters(args, seed, trees))
        model.fit(matrix, targets, group=groups, verbose=False)
        path = output_dir / f"{label}_deployment_seed_{seed}.json"
        model.save_model(path)
        paths.append(str(path.resolve()))
        print(
            f"refit={label} member={member + 1}/{len(iterations)} "
            f"seed={seed} trees={trees} path={path}",
            flush=True,
        )
    return paths


def score_table(
    frame: pd.DataFrame,
    targets: np.ndarray,
    scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    output = frame[[
        "race_id", "start_time_iso", "competition_id", "competition_name",
        "race_number", "race_name", "runner_number", "runner_name", "fluc2",
        "status",
    ]].copy()
    output["is_winner"] = targets
    for name, score in scores.items():
        output[f"{name}_score"] = score
        output[f"{name}_rank"] = (
            pd.Series(score).groupby(output["race_id"], sort=False).rank(
                method="first", ascending=False
            ).astype(int)
        )
    return output


def metrics_for_scores(
    targets: np.ndarray,
    race_ids: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {
        name: winner_metrics(targets, score, race_ids)
        for name, score in scores.items()
    }


def print_metrics(cohort: str, metrics: dict[str, dict[str, float]]) -> None:
    print(f"WINNER RANKING {cohort.upper()}")
    table = pd.DataFrame(metrics).T[
        ["top1_hit_rate", "top3_hit_rate", "mrr", "mean_winner_rank", "race_logloss"]
    ]
    print(table.to_string(float_format=lambda value: f"{value:.5f}"), flush=True)


def save_ranker_diagnostics(
    output_dir: Path,
    cohort_name: str,
    cohort: pd.DataFrame,
    targets: np.ndarray,
    scores: dict[str, np.ndarray],
) -> None:
    """Save per-race and field-size reports for each evaluated score."""
    for label, score in scores.items():
        report = winner_race_report(cohort, targets, score)
        report.to_csv(output_dir / f"{cohort_name}_{label}_race_report.csv", index=False)
        slices = winner_field_size_slices(report)
        slices.to_csv(
            output_dir / f"{cohort_name}_{label}_field_size_slices.csv", index=False
        )
        print(f"RANKER DIAGNOSTICS {cohort_name.upper()} {label.upper()}")
        print(slices.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 1 or args.max_estimators < 1:
        raise ValueError("ensemble-size and max-estimators must be positive")
    database = args.db.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
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
    races = eligible_races(frame, args.minimum_runners)
    train_ids, validation_ids, test_ids = chronological_race_split(
        races, args.validation_races, args.test_races
    )
    train = rows_for_races(frame, train_ids)
    validation = rows_for_races(frame, validation_ids)
    test = rows_for_races(frame, test_ids)
    features, duplicates = select_form_features(
        train, numeric_columns, args.minimum_feature_coverage
    )
    if not features:
        raise ValueError("No eligible form features")
    print(
        f"eligible_races={len(races):,} train_races={len(train_ids):,} "
        f"validation_races={len(validation_ids):,} test_races={len(test_ids):,}\n"
        f"form_features={len(features)} duplicates_removed={len(duplicates)}\n"
        "competition_999_role=outcome_conditioned_diagnostic_only "
        "competition_id_feature=excluded target=is_winner",
        flush=True,
    )

    train_y = train["is_winner"].to_numpy(dtype=np.int64)
    validation_y = validation["is_winner"].to_numpy(dtype=np.int64)
    test_y = test["is_winner"].to_numpy(dtype=np.int64)
    train_groups = group_sizes(train)
    validation_groups = group_sizes(validation)
    validate_ranker_groups(train, train_y, train_groups)
    validate_ranker_groups(validation, validation_y, validation_groups)
    validate_ranker_groups(test, test_y, group_sizes(test))
    all_eval = pd.concat([train, validation, test], ignore_index=True)

    train_form = form_matrix(train, features)
    validation_form = form_matrix(validation, features)
    test_form = form_matrix(test, features)
    form_models, form_iterations, form_evaluation_paths = train_evaluation_ensemble(
        args, "form", train_form, train_y, train_groups,
        validation_form, validation_y, validation_groups, output_dir,
    )
    def form_and_market_scores(
        cohort: pd.DataFrame, form_x: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        race_ids = cohort["race_id"].to_numpy(dtype=np.int64)
        return (
            ensemble_rank_scores(form_models, form_x, race_ids),
            rank_percentiles(market_scores(cohort), race_ids),
        )

    validation_form_score, validation_market_score = form_and_market_scores(
        validation, validation_form
    )
    test_form_score, test_market_score = form_and_market_scores(
        test, test_form
    )
    # Deployment is deliberately current-market-free. This is a product
    # contract, not a blend weight selected by validation performance.
    deployment_weights = {"form": 1.0, "market_aware": 0.0, "market": 0.0}
    validation_scores = {
        "form_deployment": validation_form_score,
        "market": validation_market_score,
    }
    test_scores = {
        "form_deployment": test_form_score,
        "market": test_market_score,
    }
    aware_models: list[XGBRanker] = []
    aware_iterations: list[int] = []
    aware_evaluation_paths: list[str] = []
    benchmark_weights: dict[str, float] | None = None
    if args.include_market_aware_benchmark:
        train_aware = market_aware_matrix(train, features)
        validation_aware = market_aware_matrix(validation, features)
        test_aware = market_aware_matrix(test, features)
        aware_models, aware_iterations, aware_evaluation_paths = train_evaluation_ensemble(
            args, "market_aware", train_aware, train_y, train_groups,
            validation_aware, validation_y, validation_groups, output_dir,
        )
        validation_ids_array = validation["race_id"].to_numpy(dtype=np.int64)
        test_ids_array = test["race_id"].to_numpy(dtype=np.int64)
        validation_aware_score = ensemble_rank_scores(
            aware_models, validation_aware, validation_ids_array
        )
        test_aware_score = ensemble_rank_scores(
            aware_models, test_aware, test_ids_array
        )
        benchmark_weights, _ = select_blend_weights(
            validation_y, validation_ids_array, validation_form_score,
            validation_aware_score, validation_market_score, args.blend_step,
        )
        validation_scores["market_aware_benchmark"] = validation_aware_score
        validation_scores["blended_benchmark"] = blend_scores(
            validation_form_score, validation_aware_score,
            validation_market_score, benchmark_weights,
        )
        test_scores["market_aware_benchmark"] = test_aware_score
        test_scores["blended_benchmark"] = blend_scores(
            test_form_score, test_aware_score, test_market_score, benchmark_weights,
        )
    validation_metrics = metrics_for_scores(
        validation_y,
        validation["race_id"].to_numpy(dtype=np.int64),
        validation_scores,
    )
    test_metrics = metrics_for_scores(
        test_y,
        test["race_id"].to_numpy(dtype=np.int64),
        test_scores,
    )
    print(
        "deployment_default=form_deployment current_market_inputs=none "
        f"deployment_weights={json.dumps(deployment_weights, sort_keys=True)}"
    )
    if benchmark_weights is not None:
        print(
            "diagnostic_benchmark_weights="
            + json.dumps(benchmark_weights, sort_keys=True)
        )
    print_metrics("validation", validation_metrics)
    print_metrics("test", test_metrics)

    validation_predictions = score_table(validation, validation_y, validation_scores)
    test_predictions = score_table(test, test_y, test_scores)
    validation_deviations = {
        name: market_deviation_metrics(validation_predictions, name)
        for name in validation_scores if name != "market"
    }
    test_deviations = {
        name: market_deviation_metrics(test_predictions, name)
        for name in test_scores if name != "market"
    }
    print("MARKET DEVIATION TEST")
    print(pd.DataFrame(test_deviations).T.to_string(
        float_format=lambda value: f"{value:.5f}"
    ), flush=True)
    validation_path = output_dir / "validation_predictions.csv"
    test_path = output_dir / "test_predictions.csv"
    validation_predictions.to_csv(validation_path, index=False)
    test_predictions.to_csv(test_path, index=False)
    if args.ranker_diagnostics:
        save_ranker_diagnostics(
            output_dir, "validation", validation, validation_y, validation_scores
        )
        save_ranker_diagnostics(output_dir, "test", test, test_y, test_scores)
        importance_parts = [
            xgb_ensemble_feature_importance(form_models, "form")
        ]
        if aware_models:
            importance_parts.append(
                xgb_ensemble_feature_importance(aware_models, "market_aware")
            )
        importance = pd.concat(importance_parts, ignore_index=True)
        importance.to_csv(output_dir / "evaluation_feature_importance.csv", index=False)

    if args.skip_deployment_refit:
        form_deployment_paths = form_evaluation_paths
        aware_deployment_paths = aware_evaluation_paths
        refit_all_history = False
    else:
        all_y = all_eval["is_winner"].to_numpy(dtype=np.int64)
        all_groups = group_sizes(all_eval)
        form_deployment_paths = refit_deployment_ensemble(
            args, "form", form_matrix(all_eval, features), all_y, all_groups,
            form_iterations, output_dir,
        )
        aware_deployment_paths = (
            refit_deployment_ensemble(
                args, "market_aware", market_aware_matrix(all_eval, features),
                all_y, all_groups, aware_iterations, output_dir,
            )
            if aware_iterations else []
        )
        refit_all_history = True

    feature_versions = sorted(
        str(value) for value in frame["derived_racing_features_version"].dropna().unique()
    )
    bundle = {
        "schema_version": 2,
        "objective": "single_winner_ranking",
        "competition_scope": "all_eligible_races",
        "competition_id_feature_used": False,
        "competition_999_warning": (
            "competition_id=999 was assigned after results to market misses and "
            "is diagnostic only"
        ),
        "form_features": features,
        "deployment_default": "form",
        "deployment_uses_current_market": False,
        "deployment_blend_weights": deployment_weights,
        # Backward-compatible key consumed by older rankers; it is now pure form.
        "selected_blend_weights": deployment_weights,
        "market_engineered_features": (
            list(MARKET_ENGINEERED_FEATURES)
            if args.include_market_aware_benchmark else []
        ),
        "benchmark_blend_weights": benchmark_weights,
        "selected_validation_metrics": validation_metrics["form_deployment"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_market_deviation": validation_deviations,
        "test_market_deviation": test_deviations,
        "feature_duplicates_removed": duplicates,
        "derived_feature_versions": feature_versions,
        "models": {
            "form": form_deployment_paths,
            "market_aware": aware_deployment_paths,
            "form_evaluation": form_evaluation_paths,
            "market_aware_evaluation": aware_evaluation_paths,
        },
        "best_tree_counts": {
            "form": form_iterations,
            "market_aware": aware_iterations,
        },
        "partition": {
            "train_races": len(train_ids),
            "validation_races": len(validation_ids),
            "test_races": len(test_ids),
            "train_end": str(train["start_time_iso"].max()),
            "validation_start": str(validation["start_time_iso"].min()),
            "validation_end": str(validation["start_time_iso"].max()),
            "test_start": str(test["start_time_iso"].min()),
            "test_end": str(test["start_time_iso"].max()),
        },
        "deployment_refit_all_history": refit_all_history,
        "database": str(database),
        "seed": args.seed,
    }
    bundle_path = output_dir / "winner_ranker_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    print(
        f"saved_bundle={bundle_path}\nvalidation_predictions={validation_path}\n"
        f"test_predictions={test_path}\n"
        f"rank_command={sys.executable} {Path(__file__).with_name('rank_winner_models.py')} "
        f"--bundle {bundle_path} --race-id RACE_ID",
        flush=True,
    )


if __name__ == "__main__":
    main()
