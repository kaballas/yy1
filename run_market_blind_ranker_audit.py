#!/usr/bin/env python3
"""Run the validation-only market-blind winner-ranker audit experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from evaluate_moe_winner_rankers import paired_comparison
from src.config import DEFAULT_DB
from src.market_blind_ranker_audit import (
    DEFAULT_SEEDS, NeuralTrainingConfig, aggregate_multiseed_bootstrap,
    aggregate_seed_metrics, assert_market_blind_contract, audit_features,
    build_neural_model, canonical_json_hash, git_commit_sha,
    load_development_snapshot, metrics_from_scores, prepare_feature_matrices,
    redundant_feature_clusters, subset_preprocessing_contract,
    train_neural_seed, validate_final_selection, xgboost_group_contract,
)
from src.race_moe_snapshot import (
    create_split_snapshot,
)
from src.race_moe_data import load_finished_winner_rows


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = ROOT / "outputs/moe_winner_experiment_market_blind_v2/snapshot/manifest.json"
DEFAULT_REFERENCE_CHECKPOINT = ROOT / "outputs/moe_winner_experiment_market_blind_v2/baseline_mlp.pt"
DEFAULT_OUTPUT = ROOT / "outputs/market_blind_ranker_audit_v1"
PHASES = ("audit", "baseline", "ablations", "reduced", "architectures", "select", "all")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--snapshot-manifest", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--phase", choices=PHASES, default="audit")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--races-per-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--create-test3", action="store_true")
    parser.add_argument(
        "--release-test3", action="store_true",
        help="Score sealed Test-3 once; requires an existing final selection lock.",
    )
    return parser.parse_args(argv)


def _seeds(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) != 5 or len(set(result)) != 5:
        raise ValueError("Exactly five distinct predetermined seeds are required")
    return result


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any, *, refuse_overwrite: bool = False) -> None:
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _metadata(
    manifest_path: Path, manifest: Mapping[str, Any], features: Sequence[str],
    seeds: Sequence[int], training_config: NeuralTrainingConfig,
    reference_checkpoint: Path, zeroed_features: Sequence[str],
) -> dict[str, Any]:
    reference_model = build_neural_model("current_mlp", 390)
    return {
        "git_commit_sha": git_commit_sha(ROOT),
        "snapshot_manifest": str(manifest_path.resolve()),
        "snapshot_content_hashes": {
            name: data["content_sha256"] for name, data in manifest["splits"].items()
        },
        "snapshot_file_hashes": {
            name: data["file_sha256"] for name, data in manifest["splits"].items()
        },
        "snapshot_excluded_features": list(manifest.get("excluded_features", [])),
        "reference_checkpoint": str(reference_checkpoint.resolve()),
        "reference_zeroed_base_features": list(zeroed_features),
        "nonzero_raw_base_features": [
            feature for feature in features if feature not in set(zeroed_features)
        ],
        "exact_feature_list": list(features), "seeds": list(map(int, seeds)),
        "objective": "race_softmax_nll", "development_splits": ["training", "validation"],
        "prohibited_development_splits": ["test", "test2", "test3"],
        "immutable_historical_reference": {
            "validation_top1": 0.309, "inspected_test_top1": 0.269,
            "inspected_test_usage": "historical_evidence_only",
        },
        "reference_model_architecture": reference_model.config(),
        "reference_trainable_parameters": reference_model.trainable_parameter_count(),
        "training_parameters": vars(training_config),
        "preprocessing": {
            "implementation": "src.raceformer_preprocessing",
            "standardized_clip": training_config.standardized_clip,
            "layoff_bucket_mode": "none",
        },
    }


def _test3_cutoff(manifest: Mapping[str, Any]) -> str:
    return max(split["last_start_time"] for split in manifest["splits"].values())


def test3_races_available(database: Path, cutoff: str) -> int:
    sql = """
        SELECT COUNT(*) FROM (
            SELECT race_id
            FROM race_runners
            WHERE status = 'finished' AND start_time_iso > ?
            GROUP BY race_id
            HAVING COUNT(*) >= 4
               AND SUM(CASE WHEN is_winner = 1 THEN 1 ELSE 0 END) = 1
               AND SUM(CASE WHEN is_winner IS NULL THEN 1 ELSE 0 END) = 0
        )
    """
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        return int(connection.execute(sql, (cutoff,)).fetchone()[0])


def maybe_create_test3(
    args: argparse.Namespace, manifest: Mapping[str, Any], features: Sequence[str],
    output_dir: Path,
) -> int:
    cutoff = _test3_cutoff(manifest)
    available = test3_races_available(args.db, cutoff)
    print(f"test3_races_available={available}", flush=True)
    test3_manifest = output_dir / "test3_snapshot/manifest.json"
    if not args.create_test3 or not available:
        return available
    if test3_manifest.exists():
        existing = _json(test3_manifest)
        print(
            f"test3_snapshot=already_sealed races={existing['splits']['test3']['races']}",
            flush=True,
        )
        return available
    live = load_finished_winner_rows(args.db, features)
    test3 = live.loc[live["start_time_iso"] > cutoff].copy()
    create_split_snapshot(
        test3_manifest.parent, {"test3": test3}, features,
        database=args.db, excluded_features=manifest.get("excluded_features", []),
        required_splits=("test3",),
    )
    print(f"test3_snapshot_created={test3_manifest} scoring=no", flush=True)
    return available


def run_feature_audit(
    output_dir: Path, training: pd.DataFrame, validation: pd.DataFrame,
    features: Sequence[str], metadata: Mapping[str, Any],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    print("PHASE 1 feature audit started", flush=True)
    audit, provenance, groups = audit_features(training, validation, features)
    zeroed = set(metadata["reference_zeroed_base_features"])
    relative = set(metadata["reference_preprocessing_relative_features"])
    audit["zeroed_by_reference_contract"] = audit["feature"].isin(zeroed)
    audit["has_race_relative_transformed_input"] = audit["feature"].isin(relative)
    audit["effective_model_input"] = (
        ~audit["zeroed_by_reference_contract"]
        | audit["has_race_relative_transformed_input"]
    )
    for key, value in metadata.items():
        if key in {"exact_feature_list", "training_parameters"}:
            continue
        audit[f"experiment_{key}"] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
    audit.to_csv(output_dir / "feature_audit.csv", index=False)
    _write_json(output_dir / "feature_groups.json", {"metadata": metadata, "groups": groups})
    _write_json(output_dir / "feature_provenance.json", {
        "metadata": metadata,
        "status_counts": {
            status: sum(item["status"] == status for item in provenance.values())
            for status in ("VERIFIED_PRE_RACE", "SUSPECT", "UNKNOWN")
        },
        "unknown_features": [name for name, item in provenance.items() if item["status"] == "UNKNOWN"],
        "suspect_features": [name for name, item in provenance.items() if item["status"] == "SUSPECT"],
        "features": provenance,
    })
    print("feature correlations calculating", flush=True)
    clusters = redundant_feature_clusters(training, features)
    _write_json(output_dir / "feature_correlation_clusters.json", {
        "metadata": metadata, **clusters,
    })
    print(
        f"PHASE 1 complete features={len(features)} "
        f"unknown={sum(x['status']=='UNKNOWN' for x in provenance.values())} "
        f"suspect={sum(x['status']=='SUSPECT' for x in provenance.values())} "
        f"redundant_clusters={len(clusters['candidate_redundant_clusters'])}",
        flush=True,
    )
    return groups, provenance


def _prediction_path(output_dir: Path, label: str, seed: int) -> Path:
    return output_dir / "race_level_predictions" / f"{label}_seed_{seed}.validation.csv"


def _checkpoint_path(output_dir: Path, label: str, seed: int) -> Path:
    return output_dir / "checkpoints" / f"{label}_seed_{seed}.pt"


def _save_prediction(
    path: Path, prediction: pd.DataFrame, *, label: str, seed: int,
    metadata: Mapping[str, Any], feature_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = prediction.copy()
    frame["experiment_label"] = label; frame["seed"] = seed
    frame["git_commit_sha"] = metadata["git_commit_sha"]
    frame["validation_snapshot_content_sha256"] = metadata["snapshot_content_hashes"]["validation"]
    frame["feature_list_sha256"] = feature_hash
    frame.to_csv(path, index=False)


def run_neural_set(
    *, label: str, architecture: str, features: Sequence[str], seeds: Sequence[int],
    training: pd.DataFrame, validation: pd.DataFrame,
    training_config: NeuralTrainingConfig, metadata: Mapping[str, Any],
    output_dir: Path, device: torch.device,
    reference_preprocessing: Mapping[str, Any], reference_features: Sequence[str],
    reference_zeroed_features: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[int, pd.DataFrame]]:
    preprocessing_contract = subset_preprocessing_contract(
        reference_preprocessing, reference_features, features,
    )
    zeroed = [feature for feature in reference_zeroed_features if feature in features]
    values, preprocessing, transformed = prepare_feature_matrices(
        training, validation, features, training_config.standardized_clip,
        zero_features=zeroed, preprocessing=preprocessing_contract,
    )
    results, predictions = [], {}
    feature_hash = canonical_json_hash(list(features))
    for seed in seeds:
        print(
            f"training label={label} architecture={architecture} seed={seed} "
            f"raw_features={len(features)} transformed_features={len(transformed)}",
            flush=True,
        )
        result, prediction, state = train_neural_seed(
            architecture, seed, features, training, validation, values,
            preprocessing, transformed, training_config, device,
        )
        result.update({
            "experiment_label": label, "git_commit_sha": metadata["git_commit_sha"],
            "snapshot_content_hashes": metadata["snapshot_content_hashes"],
            "feature_list_sha256": feature_hash, "zeroed_features": zeroed,
        })
        results.append(result); predictions[int(seed)] = prediction
        _save_prediction(
            _prediction_path(output_dir, label, seed), prediction,
            label=label, seed=seed, metadata=metadata, feature_hash=feature_hash,
        )
        checkpoint_path = _checkpoint_path(output_dir, label, seed)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_type": "market_blind_ranker_audit",
            "architecture": architecture, "model_state_dict": state,
            "result": result, "features": list(features),
            "preprocessing": preprocessing, "metadata": metadata,
        }, checkpoint_path)
        print(
            f"completed label={label} seed={seed} best_epoch={result['best_epoch']} "
            f"validation_top1={result['metrics']['top1_hit_rate']:.2%}", flush=True,
        )
    return results, predictions


def _load_predictions(output_dir: Path, label: str, seeds: Sequence[int]) -> dict[int, pd.DataFrame]:
    result = {}
    for seed in seeds:
        path = _prediction_path(output_dir, label, seed)
        if not path.exists():
            raise FileNotFoundError(f"Required paired prediction is missing: {path}")
        frame = pd.read_csv(path)
        if set(frame["seed"].astype(int)) != {int(seed)}:
            raise ValueError(f"Prediction seed mismatch: {path}")
        result[int(seed)] = frame
    return result


def compare_seed_sets(
    baseline: Mapping[int, pd.DataFrame], challenger: Mapping[int, pd.DataFrame],
    seeds: Sequence[int], samples: int, bootstrap_seed: int,
) -> dict[str, Any]:
    per_seed, deltas = {}, []
    for seed in seeds:
        result = paired_comparison(
            baseline[int(seed)], challenger[int(seed)], samples, bootstrap_seed,
        )
        per_seed[str(seed)] = result; deltas.append(result["top1_difference"])
    aggregate = aggregate_multiseed_bootstrap(
        baseline, challenger, seeds, samples, bootstrap_seed,
    )
    return {
        "per_seed": per_seed, "aggregate": aggregate,
        "seeds_improved": int(sum(value > 0 for value in deltas)),
        "seeds_tied": int(sum(value == 0 for value in deltas)),
        "seeds_regressed": int(sum(value < 0 for value in deltas)),
        "per_seed_top1_deltas": deltas,
    }


def run_baseline(
    output_dir: Path, training: pd.DataFrame, validation: pd.DataFrame,
    features: Sequence[str], seeds: Sequence[int], config: NeuralTrainingConfig,
    metadata: Mapping[str, Any], device: torch.device,
    reference_preprocessing: Mapping[str, Any], reference_zeroed_features: Sequence[str],
) -> dict[str, Any]:
    print("PHASE 2 five-seed baseline started", flush=True)
    runs, _ = run_neural_set(
        label="full_current_mlp", architecture="current_mlp", features=features,
        seeds=seeds, training=training, validation=validation,
        training_config=config, metadata=metadata, output_dir=output_dir, device=device,
        reference_preprocessing=reference_preprocessing, reference_features=features,
        reference_zeroed_features=reference_zeroed_features,
    )
    result = {"metadata": metadata, "runs": runs, "summary": aggregate_seed_metrics(runs)}
    _write_json(output_dir / "seed_variance.json", result)
    print(
        f"PHASE 2 complete mean_top1={result['summary']['mean_top1']:.2%} "
        f"std={result['summary']['std_top1']:.2%} "
        f"range={result['summary']['range_top1']:.2%}", flush=True,
    )
    return result


def run_ablations(
    output_dir: Path, training: pd.DataFrame, validation: pd.DataFrame,
    features: Sequence[str], groups: Mapping[str, Sequence[str]], seeds: Sequence[int],
    config: NeuralTrainingConfig, metadata: Mapping[str, Any], device: torch.device,
    samples: int, bootstrap_seed: int,
    reference_preprocessing: Mapping[str, Any], reference_zeroed_features: Sequence[str],
) -> dict[str, Any]:
    print("PHASE 3 leave-one-group-out ablations started", flush=True)
    baseline_json = _json(output_dir / "seed_variance.json")
    baseline_predictions = _load_predictions(output_dir, "full_current_mlp", seeds)
    baseline_runs = {int(row["seed"]): row for row in baseline_json["runs"]}
    ablations = []
    for group, removed in groups.items():
        if not removed:
            continue
        selected = [feature for feature in features if feature not in set(removed)]
        label = f"minus_{group}"
        runs, challenger_predictions = run_neural_set(
            label=label, architecture="current_mlp", features=selected, seeds=seeds,
            training=training, validation=validation, training_config=config,
            metadata=metadata, output_dir=output_dir, device=device,
            reference_preprocessing=reference_preprocessing, reference_features=features,
            reference_zeroed_features=reference_zeroed_features,
        )
        paired = compare_seed_sets(
            baseline_predictions, challenger_predictions, seeds, samples, bootstrap_seed,
        )
        summary = aggregate_seed_metrics(runs)
        baseline_mrr = np.mean([baseline_runs[s]["metrics"]["mrr"] for s in seeds])
        baseline_loss = np.mean([baseline_runs[s]["metrics"]["race_logloss"] for s in seeds])
        transformed_removed = (
            baseline_json["runs"][0]["transformed_feature_count"]
            - runs[0]["transformed_feature_count"]
        )
        ablations.append({
            "group": group, "features_removed": list(removed),
            "raw_features_removed": len(removed),
            "transformed_features_removed": transformed_removed,
            "remaining_features": selected, "runs": runs, "summary": summary,
            "mean_top1_delta": summary["mean_top1"] - baseline_json["summary"]["mean_top1"],
            "mean_mrr_delta": summary["mean_mrr"] - baseline_mrr,
            "mean_logloss_delta": summary["mean_logloss"] - baseline_loss,
            "paired_comparison": paired,
        })
        print(
            f"ablation_complete group={group} removed={len(removed)} "
            f"mean_top1={summary['mean_top1']:.2%} "
            f"delta={ablations[-1]['mean_top1_delta']:+.2%} "
            f"bootstrap95={paired['aggregate']['paired_race_bootstrap_95_ci']}", flush=True,
        )
    result = {"metadata": metadata, "baseline_summary": baseline_json["summary"], "ablations": ablations}
    _write_json(output_dir / "group_ablation_results.json", result)
    return result


def plan_reduced_candidate(
    output_dir: Path, features: Sequence[str], ablation_result: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    harmful, redundant = [], []
    for row in ablation_result["ablations"]:
        low, high = row["paired_comparison"]["aggregate"]["paired_race_bootstrap_95_ci"]
        nonnegative = row["paired_comparison"]["seeds_improved"] + row["paired_comparison"]["seeds_tied"]
        if row["mean_top1_delta"] > 0 and low > 0 and nonnegative >= 4:
            harmful.append(row)
        elif low <= 0 <= high and row["mean_top1_delta"] >= -0.0025:
            redundant.append(row)
    removed_groups = []
    if harmful:
        removed_groups.append(max(harmful, key=lambda row: row["mean_top1_delta"])["group"])
    remaining_redundant = [row for row in redundant if row["group"] not in removed_groups]
    if remaining_redundant:
        removed_groups.append(max(remaining_redundant, key=lambda row: row["raw_features_removed"])["group"])
    removed_features = [
        feature for row in ablation_result["ablations"] if row["group"] in removed_groups
        for feature in row["features_removed"]
    ]
    selected = [feature for feature in features if feature not in set(removed_features)]
    plan = {
        "metadata": metadata, "decision_written_before_reduced_training": True,
        "rule": "at most one clearly harmful group, then at most one statistically tied largest redundant group",
        "removed_groups": removed_groups, "removed_features": removed_features,
        "selected_features": selected,
        "fallback": "full feature set" if not removed_groups else None,
    }
    _write_json(output_dir / "reduced_candidate_plan.json", plan, refuse_overwrite=True)
    return plan


def run_reduced(
    output_dir: Path, training: pd.DataFrame, validation: pd.DataFrame,
    features: Sequence[str], seeds: Sequence[int], config: NeuralTrainingConfig,
    metadata: Mapping[str, Any], device: torch.device, samples: int,
    bootstrap_seed: int,
    reference_preprocessing: Mapping[str, Any], reference_zeroed_features: Sequence[str],
) -> dict[str, Any]:
    print("PHASE 4 single reduced candidate started", flush=True)
    ablations = _json(output_dir / "group_ablation_results.json")
    plan_path = output_dir / "reduced_candidate_plan.json"
    plan = _json(plan_path) if plan_path.exists() else plan_reduced_candidate(
        output_dir, features, ablations, metadata,
    )
    selected = plan["selected_features"]
    if selected == list(features):
        runs = _json(output_dir / "seed_variance.json")["runs"]
        predictions = _load_predictions(output_dir, "full_current_mlp", seeds)
        label = "full_current_mlp"
    else:
        label = "reduced_current_mlp"
        runs, predictions = run_neural_set(
            label=label, architecture="current_mlp", features=selected, seeds=seeds,
            training=training, validation=validation, training_config=config,
            metadata=metadata, output_dir=output_dir, device=device,
            reference_preprocessing=reference_preprocessing, reference_features=features,
            reference_zeroed_features=reference_zeroed_features,
        )
    comparison = compare_seed_sets(
        _load_predictions(output_dir, "full_current_mlp", seeds), predictions,
        seeds, samples, bootstrap_seed,
    )
    result = {
        "metadata": metadata, "label": label, "plan": plan, "runs": runs,
        "summary": aggregate_seed_metrics(runs), "paired_against_full": comparison,
    }
    _write_json(output_dir / "reduced_candidate_results.json", result)
    return result


def run_xgboost_set(
    output_dir: Path, training: pd.DataFrame, validation: pd.DataFrame,
    features: Sequence[str], seeds: Sequence[int], config: NeuralTrainingConfig,
    metadata: Mapping[str, Any],
    reference_preprocessing: Mapping[str, Any], reference_features: Sequence[str],
    reference_zeroed_features: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[int, pd.DataFrame]]:
    from xgboost import XGBRanker

    preprocessing_contract = subset_preprocessing_contract(
        reference_preprocessing, reference_features, features,
    )
    zeroed = [feature for feature in reference_zeroed_features if feature in features]
    values, preprocessing, transformed = prepare_feature_matrices(
        training, validation, features, config.standardized_clip,
        zero_features=zeroed, preprocessing=preprocessing_contract,
    )
    contract = xgboost_group_contract(training, validation)
    train_groups = contract["training_group_sizes"]
    validation_groups = contract["validation_group_sizes"]
    train_y = training["is_winner"].to_numpy(np.int64)
    validation_y = validation["is_winner"].to_numpy(np.int64)
    results, predictions = [], {}
    parameters = {
        "objective": "rank:ndcg", "eval_metric": "ndcg@1",
        "n_estimators": 700, "learning_rate": 0.03, "max_depth": 5,
        "min_child_weight": 5.0, "subsample": 0.90, "colsample_bytree": 0.90,
        "reg_lambda": 5.0, "reg_alpha": 0.0, "tree_method": "hist",
        "n_jobs": 6,
    }
    feature_hash = canonical_json_hash(list(features))
    for seed in seeds:
        print(f"training label=xgboost_ranker seed={seed}", flush=True)
        model = XGBRanker(
            **parameters, random_state=seed, early_stopping_rounds=60,
        )
        model.fit(
            values["training"], train_y, group=train_groups,
            eval_set=[(values["validation"], validation_y)],
            eval_group=[validation_groups], verbose=False,
        )
        scores = model.predict(values["validation"])
        metrics, prediction = metrics_from_scores(validation, scores)
        result = {
            "experiment_label": "xgboost_ranker", "architecture": "xgboost_ranker",
            "seed": seed, "best_epoch": int(model.best_iteration) + 1,
            "metrics": metrics, "trainable_parameters": None,
            "raw_feature_count": len(features), "transformed_feature_count": len(transformed),
            "raw_features": list(features), "transformed_features": transformed,
            "model_configuration": {**parameters, "random_state": seed, "early_stopping_rounds": 60},
            "preprocessing_configuration": {
                key: value for key, value in preprocessing.items() if key not in {"median", "scale"}
            },
            "training_parameters": vars(config), "xgboost_group_contract": contract,
            "git_commit_sha": metadata["git_commit_sha"],
            "snapshot_content_hashes": metadata["snapshot_content_hashes"],
            "feature_list_sha256": feature_hash,
            "zeroed_features": zeroed,
        }
        results.append(result); predictions[int(seed)] = prediction
        _save_prediction(
            _prediction_path(output_dir, "xgboost_ranker", seed), prediction,
            label="xgboost_ranker", seed=seed, metadata=metadata, feature_hash=feature_hash,
        )
        model_path = _checkpoint_path(output_dir, "xgboost_ranker", seed).with_suffix(".json")
        model_path.parent.mkdir(parents=True, exist_ok=True); model.save_model(model_path)
        print(
            f"completed label=xgboost_ranker seed={seed} trees={result['best_epoch']} "
            f"validation_top1={metrics['top1_hit_rate']:.2%}", flush=True,
        )
    return results, predictions


def run_architectures(
    output_dir: Path, training: pd.DataFrame, validation: pd.DataFrame,
    seeds: Sequence[int], config: NeuralTrainingConfig, metadata: Mapping[str, Any],
    device: torch.device, samples: int, bootstrap_seed: int,
    reference_preprocessing: Mapping[str, Any], reference_features: Sequence[str],
    reference_zeroed_features: Sequence[str],
) -> dict[str, Any]:
    print("PHASE 5 simple architecture controls started", flush=True)
    reduced = _json(output_dir / "reduced_candidate_results.json")
    features = reduced["plan"]["selected_features"]
    baseline_predictions = _load_predictions(output_dir, "full_current_mlp", seeds)
    candidates: dict[str, Any] = {}
    reduced_label = reduced["label"]
    candidates[reduced_label] = {
        "runs": reduced["runs"], "summary": reduced["summary"],
        "paired_against_baseline": reduced["paired_against_full"],
    }
    for architecture in ("wider_mlp", "residual_mlp"):
        runs, predictions = run_neural_set(
            label=architecture, architecture=architecture, features=features,
            seeds=seeds, training=training, validation=validation,
            training_config=config, metadata=metadata, output_dir=output_dir, device=device,
            reference_preprocessing=reference_preprocessing,
            reference_features=reference_features,
            reference_zeroed_features=reference_zeroed_features,
        )
        candidates[architecture] = {
            "runs": runs, "summary": aggregate_seed_metrics(runs),
            "paired_against_baseline": compare_seed_sets(
                baseline_predictions, predictions, seeds, samples, bootstrap_seed,
            ),
        }
    xgb_runs, xgb_predictions = run_xgboost_set(
        output_dir, training, validation, features, seeds, config, metadata,
        reference_preprocessing, reference_features, reference_zeroed_features,
    )
    candidates["xgboost_ranker"] = {
        "runs": xgb_runs, "summary": aggregate_seed_metrics(xgb_runs),
        "paired_against_baseline": compare_seed_sets(
            baseline_predictions, xgb_predictions, seeds, samples, bootstrap_seed,
        ),
    }
    result = {
        "metadata": metadata, "selected_raw_features": features,
        "baseline": _json(output_dir / "seed_variance.json"), "candidates": candidates,
    }
    _write_json(output_dir / "architecture_comparison.json", result)
    return result


def select_final_model(
    output_dir: Path, metadata: Mapping[str, Any],
) -> dict[str, Any]:
    print("PHASE 6 validation-only model selection started", flush=True)
    comparison = _json(output_dir / "architecture_comparison.json")
    baseline = comparison["baseline"]["summary"]
    eligible = []
    for label, candidate in comparison["candidates"].items():
        paired = candidate["paired_against_baseline"]
        low, _ = paired["aggregate"]["paired_race_bootstrap_95_ci"]
        summary = candidate["summary"]
        mean_delta = summary["mean_top1"] - baseline["mean_top1"]
        nonnegative = paired["seeds_improved"] + paired["seeds_tied"]
        logloss_delta = summary["mean_logloss"] - baseline["mean_logloss"]
        if mean_delta > 0 and nonnegative >= 4 and low > 0 and logloss_delta <= 0.02:
            eligible.append((mean_delta, label, candidate))
    if eligible:
        _, selected_label, candidate = max(eligible, key=lambda row: row[0])
        selected_run = candidate["runs"][0]
        selected_challengers = [selected_label]
        reason = "Meets the predeclared validation-only improvement rule."
        features = selected_run["raw_features"]
        model_configuration = selected_run["model_configuration"]
    else:
        selected_label = "full_current_mlp"
        selected_challengers = []
        reason = "No challenger meets the predeclared validation-only improvement rule."
        selected_run = comparison["baseline"]["runs"][0]
        features = selected_run["raw_features"]
        model_configuration = selected_run["model_configuration"]
    lock_payload = {
        "status": "LOCKED", "selection_complete": True,
        "selected_model": selected_label,
        "selected_challengers": selected_challengers, "selection_reason": reason,
        "exact_feature_list": features, "model_configuration": model_configuration,
        "seeds": metadata["seeds"], "training_parameters": metadata["training_parameters"],
        "preprocessing_configuration": selected_run["preprocessing_configuration"],
        "objective": "race_softmax_nll" if selected_label != "xgboost_ranker" else "rank:ndcg",
        "git_commit_sha": metadata["git_commit_sha"],
        "snapshot_content_hashes": metadata["snapshot_content_hashes"],
        "test3_opened": False,
    }
    lock_payload["configuration_sha256"] = canonical_json_hash(lock_payload)
    validate_final_selection(lock_payload)
    _write_json(output_dir / "final_selection.json", lock_payload, refuse_overwrite=True)
    print(
        f"PHASE 6 complete selected_model={selected_label} "
        f"configuration_sha256={lock_payload['configuration_sha256']} test3_opened=no",
        flush=True,
    )
    return lock_payload


def write_report(output_dir: Path, test3_available: int) -> None:
    audit = pd.read_csv(output_dir / "feature_audit.csv")
    seed = _json(output_dir / "seed_variance.json")
    ablations = _json(output_dir / "group_ablation_results.json")
    architecture = _json(output_dir / "architecture_comparison.json")
    selection = _json(output_dir / "final_selection.json")
    lines = [
        "# Market-Blind Winner Ranker Audit v1", "",
        f"Git commit: `{selection['git_commit_sha']}`", "",
        "Development used training and validation only. Historical test/test2 were not loaded or scored.", "",
        f"Test-3 races available: **{test3_available}**. Test-3 opened: **no**.", "",
        "## Baseline seed variance", "",
        f"Mean Top-1: {seed['summary']['mean_top1']:.2%}; standard deviation: {seed['summary']['std_top1']:.2%}; "
        f"range: {seed['summary']['range_top1']:.2%}.", "",
        "## Feature audit", "",
        f"Features: {len(audit)}; verified: {(audit.pre_race_availability_status == 'VERIFIED_PRE_RACE').sum()}; "
        f"unknown: {(audit.pre_race_availability_status == 'UNKNOWN').sum()}; "
        f"suspect: {(audit.pre_race_availability_status == 'SUSPECT').sum()}.", "",
        "## Group ablations", "",
        "| Removed group | Features | Mean Top-1 | Delta | Bootstrap 95% CI |", "|---|---:|---:|---:|---:|",
    ]
    for row in ablations["ablations"]:
        low, high = row["paired_comparison"]["aggregate"]["paired_race_bootstrap_95_ci"]
        lines.append(
            f"| {row['group']} | {row['raw_features_removed']} | {row['summary']['mean_top1']:.2%} | "
            f"{row['mean_top1_delta']:+.2%} | [{low:+.2%}, {high:+.2%}] |"
        )
    lines.extend(["", "## Architecture comparison", "",
                  "| Model | Features | Parameters | Mean Top-1 | Std | Delta | Bootstrap 95% CI | Seeds +/=/- |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    baseline_mean = architecture["baseline"]["summary"]["mean_top1"]
    for label, candidate in architecture["candidates"].items():
        run = candidate["runs"][0]; paired = candidate["paired_against_baseline"]
        low, high = paired["aggregate"]["paired_race_bootstrap_95_ci"]
        lines.append(
            f"| {label} | {run['raw_feature_count']} | {run['trainable_parameters']} | "
            f"{candidate['summary']['mean_top1']:.2%} | {candidate['summary']['std_top1']:.2%} | "
            f"{candidate['summary']['mean_top1'] - baseline_mean:+.2%} | [{low:+.2%}, {high:+.2%}] | "
            f"{paired['seeds_improved']}/{paired['seeds_tied']}/{paired['seeds_regressed']} |"
        )
    lines.extend(["", "## Final selection", "",
                  f"Selected: **{selection['selected_model']}**.", "",
                  selection["selection_reason"], "",
                  f"Configuration SHA256: `{selection['configuration_sha256']}`", ""])
    (output_dir / "final_report.md").write_text("\n".join(lines) + "\n")


def release_test3(args: argparse.Namespace, output_dir: Path) -> None:
    lock_path = output_dir / "final_selection.json"
    if not lock_path.exists():
        raise PermissionError("TEST-3 IS SEALED: final model-selection lock does not exist")
    selection = _json(lock_path); validate_final_selection(selection)
    expected = selection["configuration_sha256"]
    copy_for_hash = copy.deepcopy(selection); copy_for_hash.pop("configuration_sha256")
    if canonical_json_hash(copy_for_hash) != expected:
        raise ValueError("Final model-selection lock hash is invalid")
    if selection.get("test3_opened"):
        raise PermissionError("TEST-3 has already been opened and cannot be rescored")
    test3_manifest = output_dir / "test3_snapshot/manifest.json"
    if not test3_manifest.exists():
        raise FileNotFoundError("No sealed Test-3 snapshot exists")
    raise NotImplementedError(
        "Test-3 release is correctly locked but scoring is intentionally unavailable "
        "until a sufficiently large cohort exists."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv); seeds = _seeds(args.seeds)
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir == (ROOT / "outputs/moe_winner_experiment_market_blind_v2").resolve():
        raise ValueError("Reference experiment is immutable and cannot be overwritten")
    if args.release_test3:
        release_test3(args, output_dir); return
    # Critical guard: model development loads only these two named files. The
    # historical test and test2 files are never read by this program.
    frames, manifest = load_development_snapshot(args.snapshot_manifest)
    training = frames["training"].reset_index(drop=True)
    validation = frames["validation"].reset_index(drop=True)
    features = list(manifest["feature_columns"]); assert_market_blind_contract(features)
    reference_checkpoint_path = args.reference_checkpoint.resolve()
    reference = torch.load(
        reference_checkpoint_path, map_location="cpu", weights_only=False,
    )
    if list(reference.get("raw_feature_columns", [])) != features:
        raise ValueError("Reference checkpoint features differ from immutable snapshot")
    for split, frame in (("training", training), ("validation", validation)):
        expected_ids = list(map(int, reference["partition"][f"{split}_race_ids"]))
        actual_ids = list(map(int, frame["race_id"].drop_duplicates()))
        if actual_ids != expected_ids:
            raise ValueError(f"Reference checkpoint {split} races differ from snapshot")
    if reference.get("training_objective") != "race_softmax_nll":
        raise ValueError("Reference checkpoint objective is not race_softmax_nll")
    if reference.get("market_features_enabled"):
        raise ValueError("Reference checkpoint is not market-blind")
    reference_preprocessing = reference["preprocessing"]
    reference_zeroed = list(reference.get("zeroed_features", []))
    config = NeuralTrainingConfig(
        epochs=args.epochs, races_per_batch=args.races_per_batch,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
    )
    metadata = _metadata(
        args.snapshot_manifest, manifest, features, seeds, config,
        reference_checkpoint_path, reference_zeroed,
    )
    metadata["reference_preprocessing_relative_features"] = list(
        reference_preprocessing.get("relative_features", [])
    )
    configuration = {"metadata": metadata}
    configuration["configuration_sha256"] = canonical_json_hash(configuration)
    config_path = output_dir / "configuration.json"
    _write_json(config_path, configuration)
    test3_available = maybe_create_test3(args, manifest, features, output_dir)
    audit_phase = args.phase in {"audit", "all"}
    if audit_phase:
        groups, _ = run_feature_audit(
            output_dir, training, validation, features, metadata,
        )
    else:
        groups = _json(output_dir / "feature_groups.json")["groups"]
    if args.phase == "audit":
        return
    device = torch.device(args.device)
    if args.phase in {"baseline", "all"}:
        run_baseline(
            output_dir, training, validation, features, seeds, config, metadata, device,
            reference_preprocessing, reference_zeroed,
        )
    if args.phase == "baseline": return
    if args.phase in {"ablations", "all"}:
        run_ablations(
            output_dir, training, validation, features, groups, seeds, config,
            metadata, device, args.bootstrap_samples, args.bootstrap_seed,
            reference_preprocessing, reference_zeroed,
        )
    if args.phase == "ablations": return
    if args.phase in {"reduced", "all"}:
        run_reduced(
            output_dir, training, validation, features, seeds, config, metadata,
            device, args.bootstrap_samples, args.bootstrap_seed,
            reference_preprocessing, reference_zeroed,
        )
    if args.phase == "reduced": return
    if args.phase in {"architectures", "all"}:
        run_architectures(
            output_dir, training, validation, seeds, config, metadata, device,
            args.bootstrap_samples, args.bootstrap_seed,
            reference_preprocessing, features, reference_zeroed,
        )
    if args.phase == "architectures": return
    if args.phase in {"select", "all"}:
        select_final_model(output_dir, metadata)
        write_report(output_dir, test3_available)


if __name__ == "__main__":
    main()
