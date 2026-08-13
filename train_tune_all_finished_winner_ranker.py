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
    cohort_metrics,
    select_form_weight,
)
from src.config import DEFAULT_DB
from src.winner_ranker import (
    MARKET_ENGINEERED_FEATURES,
    database_numeric_columns,
    eligible_races,
    ensemble_rank_scores,
    form_matrix,
    group_sizes,
    load_training_rows,
    market_aware_matrix,
    market_scores,
    rank_percentiles,
    rows_for_races,
    select_form_features,
)
from train_winner_ranker_pipeline import model_parameters, score_table


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
        help="Provides already validated tree counts when available.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--minimum-feature-coverage", type=float, default=0.20)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--default-form-estimators", type=int, default=120)
    parser.add_argument("--default-market-aware-estimators", type=int, default=50)
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


def main() -> None:
    args = parse_args()
    if args.ensemble_size < 1:
        raise ValueError("ensemble-size must be positive")
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
    eligible_ids = races["race_id"].astype(int).tolist()
    all_finished = rows_for_races(frame, eligible_ids)
    features, duplicates = select_form_features(
        all_finished, numeric_columns, args.minimum_feature_coverage
    )
    if not features:
        raise ValueError("No eligible form features")
    fold_ids = crossfit_fold_ids(eligible_ids, args.folds)
    form_trees = tree_counts(
        args.source_bundle, "form", args.ensemble_size,
        args.default_form_estimators,
    )
    aware_trees = tree_counts(
        args.source_bundle, "market_aware", args.ensemble_size,
        args.default_market_aware_estimators,
    )
    print(
        f"source=status_finished active_runner_only=yes "
        f"eligible_races={len(eligible_ids):,} rows={len(all_finished):,} "
        f"folds={args.folds} form_features={len(features)} "
        f"duplicates_removed={len(duplicates)}\n"
        f"form_tree_counts={form_trees} market_aware_tree_counts={aware_trees}\n"
        "crossfit_guarantee=each_race_scored_by_models_not_trained_on_that_race "
        "sealed_test=no",
        flush=True,
    )

    oof_parts: list[pd.DataFrame] = []
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
        form_models = fit_ensemble(
            args, "form", form_matrix(training, features), train_y,
            train_groups, form_trees, fold_number * 100_000,
        )
        aware_models = fit_ensemble(
            args, "market_aware", market_aware_matrix(training, features),
            train_y, train_groups, aware_trees, fold_number * 100_000 + 50_000,
        )
        holdout_ids_array = holdout["race_id"].to_numpy(dtype=np.int64)
        form_score = ensemble_rank_scores(
            form_models, form_matrix(holdout, features), holdout_ids_array
        )
        aware_score = ensemble_rank_scores(
            aware_models, market_aware_matrix(holdout, features), holdout_ids_array
        )
        market_score = rank_percentiles(
            market_scores(holdout), holdout_ids_array
        )
        part = score_table(
            holdout,
            holdout["is_winner"].to_numpy(dtype=np.int64),
            {
                "form": form_score,
                "market_aware": aware_score,
                "market": market_score,
            },
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
    weights = candidate_form_weights(args.weight_step, args.minimum_form_weight)
    selected_form_weight, sweep = select_form_weight(
        oof, "form_score", "market_aware_score", weights, args.objective
    )
    selected_weights = {
        "form": selected_form_weight,
        "market_aware": 1.0 - selected_form_weight,
        "market": 0.0,
    }
    metrics, deviation = cohort_metrics(
        oof, "form_score", "market_aware_score", selected_form_weight
    )
    print(
        "all_finished_selected_weights="
        + json.dumps(selected_weights, sort_keys=True)
    )
    print("ALL-FINISHED GROUPED OOF METRICS")
    print(pd.DataFrame(metrics).T[[
        "top1_hit_rate", "top3_hit_rate", "mrr",
        "mean_winner_rank", "race_logloss",
    ]].to_string(float_format=lambda value: f"{value:.5f}"), flush=True)

    all_y = all_finished["is_winner"].to_numpy(dtype=np.int64)
    all_groups = group_sizes(all_finished)
    final_form_models = fit_ensemble(
        args, "form", form_matrix(all_finished, features), all_y,
        all_groups, form_trees, 0,
    )
    final_aware_models = fit_ensemble(
        args, "market_aware", market_aware_matrix(all_finished, features), all_y,
        all_groups, aware_trees, 50_000,
    )
    form_paths = save_ensemble(final_form_models, "form", output_dir, args.seed)
    aware_paths = save_ensemble(
        final_aware_models, "market_aware", output_dir, args.seed + 50_000
    )

    versions = sorted(
        str(value) for value in all_finished[
            "derived_racing_features_version"
        ].dropna().unique()
    )
    recommendation = {
        "schema_version": 2,
        "blend": "all_finished_crossfit_form_plus_market_aware",
        "selection_cohort": "all_eligible_finished_races_grouped_oof",
        "raw_market_weight_fixed": 0.0,
        "objective": args.objective,
        "selected_weights": selected_weights,
        "oof_metrics": metrics,
        "oof_market_deviation": deviation,
        "eligible_finished_races": len(eligible_ids),
        "eligible_finished_rows": len(all_finished),
        "crossfit_folds": args.folds,
        "sealed_test_available": False,
    }
    bundle = {
        "schema_version": 3,
        "objective": "single_winner_ranking",
        "training_scope": "all_eligible_finished_races",
        "competition_scope": "all_eligible_races",
        "competition_id_feature_used": False,
        "form_features": features,
        "deployment_default": "form",
        "deployment_uses_current_market": False,
        "deployment_blend_weights": {
            "form": 1.0, "market_aware": 0.0, "market": 0.0,
        },
        "selected_blend_weights": {
            "form": 1.0, "market_aware": 0.0, "market": 0.0,
        },
        "all_finished_tuned_blend_weights": selected_weights,
        "market_engineered_features": list(MARKET_ENGINEERED_FEATURES),
        "feature_duplicates_removed": duplicates,
        "derived_feature_versions": versions,
        "models": {
            "form": form_paths,
            "market_aware": aware_paths,
            "form_evaluation": [],
            "market_aware_evaluation": [],
        },
        "best_tree_counts": {
            "form": form_trees,
            "market_aware": aware_trees,
        },
        "all_finished_crossfit": recommendation,
        "database": str(database),
        "seed": args.seed,
    }
    oof_path = output_dir / "all_finished_oof_predictions.csv"
    sweep_path = output_dir / "all_finished_weight_sweep.csv"
    recommendation_path = output_dir / "all_finished_blend.json"
    bundle_path = output_dir / "winner_ranker_bundle.json"
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
