#!/usr/bin/env python3
"""Compare frozen-feature, market-blind winner-ranking supervision targets.

Each invocation trains one target on the same final 2,000-race chronological
split. The model features, tree counts, hyperparameters, seeds, and ranking
objective are fixed so only the supervision target changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.config import DEFAULT_DB
from src.winner_ranker import (
    MARKET_ENGINEERED_FEATURES,
    RANKING_TARGETS,
    chronological_race_split,
    database_numeric_columns,
    eligible_races,
    ensemble_rank_scores,
    group_sizes,
    is_current_market_feature,
    load_training_rows,
    model_feature_matrix,
    ranking_targets,
    rows_for_races,
    validate_ranker_groups,
    winner_metrics,
)
from train_winner_ranker_pipeline import model_parameters, score_table
from validate_chronological_winner_blend import (
    cohort_time_bounds,
    validate_chronology,
)


FROZEN_TREE_COUNTS = (33, 55, 86)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=RANKING_TARGETS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--feature-manifest", type=Path, default=Path("winner_ranker_features.json")
    )
    parser.add_argument("--model", default="gpt_pick_v2")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/winner_target_comparison"),
    )
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tree-counts", nargs="+", type=int,
        default=list(FROZEN_TREE_COUNTS),
    )
    parser.add_argument(
        "--beaten-margin-column", default="beaten_margin",
        help="Current-race beaten-margin result column required by margin-aware target.",
    )
    return parser.parse_args()


def frozen_features(manifest: Path, model: str) -> tuple[list[str], str]:
    raw = manifest.resolve().read_bytes()
    payload = json.loads(raw)
    try:
        features = list(payload["models"][model]["features"])
    except KeyError as exc:
        raise ValueError(f"Feature manifest has no model {model!r}") from exc
    forbidden = [
        feature for feature in features
        if feature in MARKET_ENGINEERED_FEATURES or is_current_market_feature(feature)
    ]
    if forbidden:
        raise ValueError(
            "Frozen target comparison must be current-market-blind; forbidden="
            + ",".join(forbidden)
        )
    return features, hashlib.sha256(raw).hexdigest()


def validate_supervision(frame: pd.DataFrame, targets: np.ndarray) -> None:
    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError("supervision targets must match the runner rows")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("supervision targets must be finite and non-negative")
    if any(group.nunique() < 2 for _, group in pd.Series(values).groupby(
        frame["race_id"], sort=False
    )):
        raise ValueError("every race must contain at least two relevance levels")


def common_ranker_parameters(
    args: argparse.Namespace, seed: int, trees: int
) -> dict[str, Any]:
    parameters = model_parameters(args, seed, trees)
    # rank:pairwise accepts the exact continuous B/C relevance definitions.
    # It is deliberately shared by A, B, and C to isolate target supervision.
    parameters["objective"] = "rank:pairwise"
    parameters["eval_metric"] = "rmse"
    return parameters


def train_models(
    args: argparse.Namespace,
    matrix: pd.DataFrame,
    targets: np.ndarray,
    groups: np.ndarray,
    target_dir: Path,
) -> tuple[list[XGBRanker], list[str]]:
    models: list[XGBRanker] = []
    paths: list[str] = []
    for member, trees in enumerate(args.tree_counts):
        if trees < 1:
            raise ValueError("tree-counts must be positive")
        seed = args.seed + member * 1009
        model = XGBRanker(**common_ranker_parameters(args, seed, trees))
        model.fit(matrix, targets, group=groups, verbose=False)
        path = target_dir / f"{args.model}_{args.target}_seed_{seed}.json"
        model.save_model(path)
        models.append(model)
        paths.append(str(path.resolve()))
        print(
            f"trained_target={args.target} member={member + 1}/{len(args.tree_counts)} "
            f"seed={seed} trees={trees} path={path}",
            flush=True,
        )
    return models, paths


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
    database = args.db.resolve()
    manifest = args.feature_manifest.resolve()
    target_dir = (args.output_dir.resolve() / args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    features, manifest_hash = frozen_features(manifest, args.model)
    numeric_columns = database_numeric_columns(database)
    frame = load_training_rows(database, numeric_columns)
    races = eligible_races(frame, args.minimum_runners)
    train_ids, validation_ids, test_ids = chronological_race_split(
        races, args.validation_races, args.test_races
    )
    train = rows_for_races(frame, train_ids)
    validation = rows_for_races(frame, validation_ids)
    test = rows_for_races(frame, test_ids)
    validate_chronology(train, validation, test)

    winner_targets = {
        name: cohort["is_winner"].to_numpy(dtype=np.int64)
        for name, cohort in (
            ("train", train), ("validation", validation), ("sealed_test", test)
        )
    }
    for name, cohort in (
        ("train", train), ("validation", validation), ("sealed_test", test)
    ):
        validate_ranker_groups(
            cohort, winner_targets[name], group_sizes(cohort)
        )

    train_target = ranking_targets(
        train, args.target, args.beaten_margin_column
    )
    validate_supervision(train, train_target)
    train_matrix = model_feature_matrix(train, features)
    models, model_paths = train_models(
        args, train_matrix, train_target, group_sizes(train), target_dir
    )

    metrics: dict[str, dict[str, float]] = {}
    prediction_paths: dict[str, str] = {}
    for name, cohort in (("validation", validation), ("sealed_test", test)):
        race_ids = cohort["race_id"].to_numpy(dtype=np.int64)
        scores = ensemble_rank_scores(
            models, model_feature_matrix(cohort, features), race_ids
        )
        metrics[name] = winner_metrics(winner_targets[name], scores, race_ids)
        predictions = score_table(
            cohort, winner_targets[name], {args.target: scores}
        )
        path = target_dir / f"{name}_predictions.csv"
        predictions.to_csv(path, index=False)
        prediction_paths[name] = str(path.resolve())

    print(
        f"target={args.target} model={args.model} current_market_features=none\n"
        f"frozen_manifest={manifest} sha256={manifest_hash}\n"
        f"frozen_feature_count={len(features)} tree_counts={json.dumps(args.tree_counts)}\n"
        f"train_races={len(train_ids):,} validation_races={len(validation_ids):,} "
        f"sealed_test_races={len(test_ids):,}",
        flush=True,
    )
    print(pd.DataFrame(metrics).T[[
        "top1_hit_rate", "top3_hit_rate", "mrr",
        "mean_winner_rank", "race_logloss", "races",
    ]].to_string(float_format=lambda value: f"{value:.5f}"))

    result = {
        "schema_version": 1,
        "experiment": "frozen_market_blind_supervision_target",
        "target": args.target,
        "target_definition": {
            "winner": "is_winner",
            "finish_order": "1 - ((finish_place - 1) / (field_size - 1))",
            "margin_aware_finish_order": (
                "0.75 * finish_order + 0.25 * exp(-beaten_margin / 5)"
            ),
        }[args.target],
        "ranker_objective": "rank:pairwise",
        "model": args.model,
        "features": features,
        "feature_manifest": str(manifest),
        "feature_manifest_sha256": manifest_hash,
        "current_market_features": [],
        "tree_counts": list(args.tree_counts),
        "seeds": [args.seed + member * 1009 for member in range(len(args.tree_counts))],
        "model_paths": model_paths,
        "prediction_paths": prediction_paths,
        "race_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "sealed_test": len(test_ids),
        },
        "time_bounds": {
            "train": cohort_time_bounds(train),
            "validation": cohort_time_bounds(validation),
            "sealed_test": cohort_time_bounds(test),
        },
        "metrics": metrics,
    }
    result_path = target_dir / "result.json"
    result_path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved_result={result_path}", flush=True)


if __name__ == "__main__":
    main()
