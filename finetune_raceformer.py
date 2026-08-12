#!/usr/bin/env python3
"""Fine-tune an existing RaceFormerTop3 checkpoint conservatively."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from predict_raceformer import load_checkpoint
from src.config import DEFAULT_DB
from src.constants import TRAINING_ROWS_VIEW, VALIDATION_ROWS_VIEW
from src.database import export_rows_to_csv, load_rows_from_csv
from src.dataset import load_feature_manifest
from src.model.raceformer import RaceFormerTop3
from src.raceformer_preprocessing import model_feature_columns, transform_raceformer
from src.raceformer_partition import combine_disjoint_snapshots
from train_raceformer import (
    _batches,
    _pad_batch,
    _raceformer_objective,
    _selected_indices,
    _selection,
    evaluate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a RaceFormer checkpoint without refitting preprocessing."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--features-json", type=Path,
        help=(
            "Optional feature manifest. Its ordered features must match the "
            "checkpoint; zeroed_features may be overridden for fine-tuning."
        ),
    )
    parser.add_argument(
        "--layoff-bucket-mode", choices=("none", "cumulative", "exclusive"),
        help=(
            "Assert the source checkpoint's layoff representation. Fine-tuning "
            "cannot change this architecture contract."
        ),
    )
    parser.add_argument(
        "--training-csv", type=Path, default=Path("outputs/raceformer_training.csv")
    )
    parser.add_argument(
        "--validation-csv", type=Path, default=Path("outputs/raceformer_validation.csv")
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Reuse existing CSV snapshots instead of exporting database views.",
    )
    parser.add_argument(
        "--scope", choices=("full", "transformer_and_head", "head_only"),
        default="full",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--races-per-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-loss-weight", type=float)
    parser.add_argument("--cardinality-loss-weight", type=float)
    parser.add_argument("--listwise-loss-weight", type=float)
    parser.add_argument(
        "--market-residual-weight", type=float,
        help="Override the source checkpoint's residual-correction penalty.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--checkpoint-metric", choices=("loss", "composite"),
        help="Defaults to the source checkpoint's selection mode.",
    )
    parser.add_argument(
        "--save-strategy", choices=("best_finetuned", "source_guarded"),
        default="best_finetuned",
        help=(
            "Save the best actual fine-tune epoch (default), or retain source "
            "weights unless an epoch beats the source."
        ),
    )
    parser.add_argument("--max-training-races", type=int, default=0)
    parser.add_argument("--max-validation-races", type=int, default=0)
    parser.add_argument(
        "--training-competition-id",
        help=(
            "Comma-separated competitions for fine-tuning. If neither competition "
            "ID is supplied in an interactive terminal, the program prompts."
        ),
    )
    parser.add_argument(
        "--validation-competition-id",
        help="Comma-separated, disjoint competitions for validation.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--allow-in-place", action="store_true",
        help="Allow replacing the source checkpoint; a separate output is safer.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint.resolve() == args.output.resolve() and not args.allow_in_place:
        raise ValueError(
            "Refusing in-place fine-tuning; choose another --output or pass --allow-in-place"
        )
    positive = {
        "epochs": args.epochs,
        "races-per-batch": args.races_per_batch,
        "learning-rate": args.learning_rate,
        "max-grad-norm": args.max_grad_norm,
        "early-stopping-patience": args.early_stopping_patience,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError("These arguments must be positive: " + ", ".join(invalid))
    if args.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    for name in (
        "ranking_loss_weight", "cardinality_loss_weight", "listwise_loss_weight",
        "market_residual_weight",
    ):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise ValueError(f"{name.replace('_', '-')} must be non-negative")
    if args.max_training_races < 0 or args.max_validation_races < 0:
        raise ValueError("race limits must be zero or positive")


def _inherited_weight(
    args: argparse.Namespace, checkpoint: dict[str, Any], name: str, fallback: float
) -> float:
    requested = getattr(args, name)
    if requested is not None:
        return float(requested)
    return float(checkpoint.get("training_config", {}).get(name, fallback))


def configure_scope(model: RaceFormerTop3, scope: str) -> list[str]:
    for parameter in model.parameters():
        parameter.requires_grad = scope == "full"
    if scope in {"transformer_and_head", "head_only"}:
        for parameter in model.prediction_head.parameters():
            parameter.requires_grad = True
    if scope == "transformer_and_head":
        if model.race_transformer is None:
            raise ValueError("transformer_and_head requires a transformer variant")
        for parameter in model.race_transformer.parameters():
            parameter.requires_grad = True
        if model.race_token is not None:
            model.race_token.requires_grad = True
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError("Fine-tune scope selected no parameters")
    return names


def _competition_summary(
    race_ids: np.ndarray, competition_ids: np.ndarray
) -> list[tuple[int, int, int]]:
    """Return competition ID, race count, and runner count."""
    summary = []
    for competition_id in sorted(set(map(int, competition_ids))):
        selected = competition_ids == competition_id
        summary.append(
            (competition_id, len(np.unique(race_ids[selected])), int(selected.sum()))
        )
    return summary


def _resolve_competition_ids(
    args: argparse.Namespace, race_ids: np.ndarray, competition_ids: np.ndarray
) -> tuple[list[int], list[int]]:
    train_value = args.training_competition_id
    validation_value = args.validation_competition_id
    if train_value is None and validation_value is None:
        print("AVAILABLE COMPETITIONS", flush=True)
        for competition_id, races, runners in _competition_summary(
            race_ids, competition_ids
        ):
            print(
                f"  competition_id={competition_id} races={races:,} runners={runners:,}",
                flush=True,
            )
        try:
            train_value = input("Training competition IDs (comma-separated): ").strip()
            validation_value = input(
                "Validation competition IDs (comma-separated): "
            ).strip()
        except EOFError as error:
            raise ValueError(
                "Competition IDs are required. In a non-interactive run pass "
                "--training-competition-id and --validation-competition-id."
            ) from error
    elif train_value is None or validation_value is None:
        raise ValueError(
            "Pass both --training-competition-id and --validation-competition-id, "
            "or omit both to answer the interactive prompts"
        )
    try:
        train_ids = sorted({int(value.strip()) for value in str(train_value).split(",") if value.strip()})
        validation_ids = sorted({int(value.strip()) for value in str(validation_value).split(",") if value.strip()})
    except ValueError as error:
        raise ValueError("Competition IDs must be comma-separated integers") from error
    if not train_ids or not validation_ids:
        raise ValueError("Training and validation competition lists cannot be empty")
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise ValueError(f"Training and validation competitions overlap: {overlap}")
    available = set(map(int, competition_ids))
    missing = sorted((set(train_ids) | set(validation_ids)) - available)
    if missing:
        raise ValueError(
            f"Competition ID(s) not found in exported snapshots: {missing}"
        )
    return train_ids, validation_ids


def _json_number(value: np.floating | float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _print_race_data_example(
    label: str,
    stage: str,
    x: np.ndarray,
    y: np.ndarray,
    race_ids: np.ndarray,
    times: np.ndarray,
    race_groups: dict[int, np.ndarray],
    feature_names: list[str],
    competition_by_race: dict[int, int],
) -> None:
    """Print one complete eligible race at the requested processing stage."""
    if not race_groups:
        print(f"{label.upper()} RACE DATA\n<no eligible race>", flush=True)
        return
    race_id = int(next(iter(race_groups)))
    rows = np.asarray(race_groups[race_id], dtype=np.int64)
    payload = {
        "race_id": race_id,
        "competition_id": competition_by_race[race_id],
        "start_time": times[int(rows[0])].isoformat(),
        "runners": len(rows),
        "top3_targets": int(y[rows].sum()),
        "runner_data": [
            {
                "row_in_race": position,
                "top3_mask": int(y[row]),
                "features": {
                    name: _json_number(x[row, column])
                    for column, name in enumerate(feature_names)
                },
            }
            for position, row in enumerate(rows, start=1)
        ],
    }
    print(
        f"{label.upper()} RACE DATA ({stage})\n"
        + json.dumps(payload, indent=2),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, source = load_checkpoint(args.checkpoint, device)
    features = list(source.get("raw_feature_columns", source["feature_columns"]))
    zeroed = list(source.get("zeroed_features", []))
    if args.features_json is not None:
        manifest_features, manifest_zeroed = load_feature_manifest(args.features_json)
        if manifest_features != features:
            checkpoint_only = [name for name in features if name not in manifest_features]
            manifest_only = [name for name in manifest_features if name not in features]
            order_only = not checkpoint_only and not manifest_only
            detail = (
                "same columns but different order" if order_only else
                f"checkpoint_only={checkpoint_only} manifest_only={manifest_only}"
            )
            raise ValueError(
                "--features-json is incompatible with the checkpoint feature contract: "
                + detail
            )
        zeroed = manifest_zeroed
        print(
            f"feature_manifest={args.features_json.resolve()} "
            f"features={len(features)} zeroed={len(zeroed)}",
            flush=True,
        )

    if not args.no_export:
        export_rows_to_csv(args.db, features, TRAINING_ROWS_VIEW, args.training_csv)
        export_rows_to_csv(args.db, features, VALIDATION_ROWS_VIEW, args.validation_csv)
    (
        train_x, train_y, train_ids, train_times, train_competitions, *_
    ) = load_rows_from_csv(
        args.training_csv, features
    )
    (
        valid_x, valid_y, valid_ids, valid_times, valid_competitions, *_
    ) = load_rows_from_csv(
        args.validation_csv, features
    )
    all_x, all_y, all_ids, all_times = combine_disjoint_snapshots(
        train_x, train_y, train_ids, train_times,
        valid_x, valid_y, valid_ids, valid_times,
    )
    competition_by_race = {
        int(race_id): int(competition_id)
        for race_id, competition_id in zip(
            np.concatenate((train_ids, valid_ids)),
            np.concatenate((train_competitions, valid_competitions)),
        )
    }
    all_competitions = np.asarray(
        [competition_by_race[int(race_id)] for race_id in all_ids], dtype=np.int64
    )
    training_competition_ids, validation_competition_ids = _resolve_competition_ids(
        args, all_ids, all_competitions
    )
    if "competition_id" in features and "competition_id" not in zeroed:
        zeroed.append("competition_id")
        print(
            "auto_zeroed_feature=competition_id reason=competition_holdout_identifier",
            flush=True,
        )
    if training_competition_ids is not None:
        train_mask = np.isin(all_competitions, training_competition_ids)
        valid_mask = np.isin(all_competitions, validation_competition_ids)
        train_x, train_y = all_x[train_mask], all_y[train_mask]
        train_ids, train_times = all_ids[train_mask], all_times[train_mask]
        valid_x, valid_y = all_x[valid_mask], all_y[valid_mask]
        valid_ids, valid_times = all_ids[valid_mask], all_times[valid_mask]
        print(
            "competition_partition "
            f"training_competition_ids={training_competition_ids} "
            f"validation_competition_ids={validation_competition_ids} "
            f"training_races={len(np.unique(train_ids)):,} "
            f"validation_races={len(np.unique(valid_ids)):,}",
            flush=True,
        )
    train_races, invalid_train = _selected_indices(
        train_y, train_ids, args.max_training_races, "Training"
    )
    valid_races, invalid_valid = _selected_indices(
        valid_y, valid_ids, args.max_validation_races, "Validation"
    )
    _print_race_data_example(
        "training", "raw, before preprocessing",
        train_x, train_y, train_ids, train_times, train_races,
        features, competition_by_race,
    )
    _print_race_data_example(
        "validation", "raw, before preprocessing",
        valid_x, valid_y, valid_ids, valid_times, valid_races,
        features, competition_by_race,
    )
    preprocessing = source.get("preprocessing")
    bucket_features = list(
        preprocessing.get("layoff_bucket_features", []) if preprocessing else []
    )
    checkpoint_layoff_mode = (
        preprocessing.get("layoff_bucket_mode") if preprocessing else None
    )
    if checkpoint_layoff_mode is None:
        checkpoint_layoff_mode = (
            "cumulative" if "recent_days_30_plus" in bucket_features else
            "exclusive" if "recent_days_30_59" in bucket_features else "none"
        )
    if (
        args.layoff_bucket_mode is not None
        and args.layoff_bucket_mode != checkpoint_layoff_mode
    ):
        raise ValueError(
            f"--layoff-bucket-mode={args.layoff_bucket_mode} cannot fine-tune a "
            f"checkpoint with mode={checkpoint_layoff_mode}; train that feature "
            "contract from scratch"
        )
    expanded_features = list(
        source.get(
            "model_feature_columns",
            model_feature_columns(features, preprocessing),
        )
    )
    zeroed_set = set(zeroed)
    active_features = [
        name for name in expanded_features
        if name not in zeroed_set
        and not any(name == f"{zeroed_name}__race_percentile" for zeroed_name in zeroed_set)
    ]
    legacy_median = np.asarray(source["median"], dtype=np.float32)
    legacy_scale = np.asarray(source["scale"], dtype=np.float32)
    train_x = transform_raceformer(
        train_x, train_ids, features, zeroed, preprocessing,
        legacy_median=legacy_median, legacy_scale=legacy_scale,
    )
    valid_x = transform_raceformer(
        valid_x, valid_ids, features, zeroed, preprocessing,
        legacy_median=legacy_median, legacy_scale=legacy_scale,
    )
    _print_race_data_example(
        "training", "final model inputs, after preprocessing",
        train_x, train_y, train_ids, train_times, train_races,
        expanded_features, competition_by_race,
    )
    _print_race_data_example(
        "validation", "final model inputs, after preprocessing",
        valid_x, valid_y, valid_ids, valid_times, valid_races,
        expanded_features, competition_by_race,
    )
    if train_x.shape[1] != model.feature_count:
        raise ValueError(
            f"Checkpoint expects {model.feature_count} model features but preprocessing "
            f"produced {train_x.shape[1]}"
        )

    ranking_weight = _inherited_weight(args, source, "ranking_loss_weight", 0.5)
    cardinality_weight = _inherited_weight(
        args, source, "cardinality_loss_weight", 0.1
    )
    listwise_weight = _inherited_weight(args, source, "listwise_loss_weight", 0.0)
    market_residual_weight = _inherited_weight(
        args, source, "market_residual_weight", 0.0
    )
    metric_mode = args.checkpoint_metric or source.get("checkpoint_metric", "composite")
    trainable_names = configure_scope(model, args.scope)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)

    baseline_loss, baseline_metrics = evaluate(
        model, valid_x, valid_y, valid_ids, valid_races, args.races_per_batch,
        ranking_weight, cardinality_weight, device, listwise_weight,
        market_residual_weight,
    )
    source_selection = _selection(baseline_loss, baseline_metrics, metric_mode)
    source_state = copy.deepcopy(model.state_dict())
    best_selection: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int | None]] = [{
        "fine_tune_epoch": 0,
        "train_loss": None,
        "validation_loss": baseline_loss,
        **baseline_metrics,
        "selection_score": source_selection[0],
    }]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    effective_hyperparameters = {
        "model": model.config(),
        "fine_tuning": {
            "optimizer": "AdamW",
            "scope": args.scope,
            "epochs": args.epochs,
            "races_per_batch": args.races_per_batch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "early_stopping_patience": args.early_stopping_patience,
            "checkpoint_metric": metric_mode,
            "save_strategy": args.save_strategy,
            "ranking_loss_weight": ranking_weight,
            "cardinality_loss_weight": cardinality_weight,
            "listwise_loss_weight": listwise_weight,
            "market_residual_weight": market_residual_weight,
            "layoff_bucket_mode": checkpoint_layoff_mode,
            "max_training_races": args.max_training_races,
            "max_validation_races": args.max_validation_races,
            "training_competition_ids": training_competition_ids,
            "validation_competition_ids": validation_competition_ids,
            "seed": args.seed,
            "device": str(device),
        },
        "data": {
            "source_checkpoint": str(args.checkpoint.resolve()),
            "output": str(args.output.resolve()),
            "database": str(args.db.resolve()),
            "features_json": (
                str(args.features_json.resolve()) if args.features_json else None
            ),
            "training_csv": str(args.training_csv.resolve()),
            "validation_csv": str(args.validation_csv.resolve()),
            "export_snapshots": not args.no_export,
            "raw_feature_count": len(features),
            "model_feature_count": model.feature_count,
            "raw_feature_columns": features,
            "model_feature_columns": expanded_features,
            "active_trainable_features": active_features,
            "train_races": len(train_races),
            "validation_races": len(valid_races),
            "partition_mode": "competition_holdout",
        },
        "preprocessing": {
            "version": int(preprocessing.get("version", 1)) if preprocessing else 1,
            "scaler": "median_mad_with_std_fallback" if preprocessing else "legacy_median_std",
            "clip": preprocessing.get("clip") if preprocessing else None,
            "log1p_features": preprocessing.get("log1p_features", []) if preprocessing else [],
            "relative_features": preprocessing.get("relative_features", []) if preprocessing else [],
            "layoff_bucket_features": preprocessing.get(
                "layoff_bucket_features", []
            ) if preprocessing else [],
            "layoff_bucket_mode": checkpoint_layoff_mode,
            "zeroed_features": zeroed,
        },
        "parameters": {
            "trainable": trainable_count,
            "frozen": total_count - trainable_count,
            "total": total_count,
        },
        "composite_selection_weights": {
            "top3_recall": 0.50,
            "ndcg3": 0.25,
            "pairwise_ranking_accuracy": 0.25,
        },
    }
    print(
        "RACEFORMER FINE-TUNING\n"
        f"source={args.checkpoint.resolve()} source_best_epoch={source.get('best_epoch', '-')}\n"
        f"variant={model.variant} scope={args.scope} trainable={trainable_count:,}/{total_count:,} "
        f"train_races={len(train_races)} validation_races={len(valid_races)} device={device}\n"
        f"lr={args.learning_rate:g} weights=ranking:{ranking_weight:g},"
        f"cardinality:{cardinality_weight:g},listwise:{listwise_weight:g}\n"
        f"baseline valid={baseline_loss:.5f} top3={baseline_metrics['top3_recall']:.4f} "
        f"ndcg3={baseline_metrics['ndcg3']:.4f} "
        f"pairwise={baseline_metrics['pairwise_ranking_accuracy']:.4f} "
        f"select={source_selection[0]:.5f} save_strategy={args.save_strategy}",
        flush=True,
    )
    print(
        "HYPERPARAMETERS\n"
        + json.dumps(effective_hyperparameters, indent=2, sort_keys=True),
        flush=True,
    )
    print(
        f"TRAINABLE FEATURES ({len(active_features)})\n"
        + "\n".join(f"  {index:03d} {name}" for index, name in enumerate(active_features, 1)),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for groups in _batches(train_races, args.races_per_batch, rng=rng):
            bx, by, valid, _ = _pad_batch(train_x, train_y, train_ids, groups, device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss, _ = _raceformer_objective(
                model, bx, by, valid, ranking_weight, cardinality_weight,
                listwise_weight, market_residual_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite RaceFormer fine-tuning loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        validation_loss, metrics = evaluate(
            model, valid_x, valid_y, valid_ids, valid_races, args.races_per_batch,
            ranking_weight, cardinality_weight, device, listwise_weight,
            market_residual_weight,
        )
        selection = _selection(validation_loss, metrics, metric_mode)
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "fine_tune_epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
            **metrics,
            "selection_score": selection[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.5f} valid={validation_loss:.5f} "
            f"top3={metrics['top3_recall']:.4f} ndcg3={metrics['ndcg3']:.4f} "
            f"pairwise={metrics['pairwise_ranking_accuracy']:.4f} "
            f"select={selection[0]:.5f} best={'yes' if improved else 'no'} "
            f"stale={stale_epochs}/{args.early_stopping_patience}",
            flush=True,
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None or best_selection is None:
        raise RuntimeError("Fine-tuning produced no checkpoint state")
    improved_over_source = best_selection > source_selection
    if args.save_strategy == "source_guarded" and not improved_over_source:
        saved_state = source_state
        saved_epoch = 0
        saved_selection = source_selection
    else:
        saved_state = best_state
        saved_epoch = best_epoch
        saved_selection = best_selection

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = copy.deepcopy(source)
    result.update({
        "model_state_dict": saved_state,
        "zeroed_features": zeroed,
        "partition": {
            "mode": "competition_holdout",
            "training_competition_ids": training_competition_ids,
            "validation_competition_ids": validation_competition_ids,
            "training_race_ids": sorted(set(map(int, train_ids))),
            "validation_race_ids": sorted(set(map(int, valid_ids))),
        },
        "best_epoch": saved_epoch,
        "checkpoint_metric": metric_mode,
        "best_selection": saved_selection,
        "fine_tune_history": history,
        "fine_tune_config": vars(args),
        "fine_tune_provenance": {
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_best_epoch": source.get("best_epoch"),
            "scope": args.scope,
            "trainable_parameter_names": trainable_names,
            "baseline_selection": history[0]["selection_score"],
            "best_fine_tune_epoch": best_epoch,
            "best_fine_tune_selection": best_selection,
            "saved_epoch": saved_epoch,
            "save_strategy": args.save_strategy,
            "improved_over_source": improved_over_source,
        },
        "invalid_races": {
            "training": [
                {"race_id": race_id, "runners": runners, "top3": top3}
                for race_id, runners, top3 in invalid_train
            ],
            "validation": [
                {"race_id": race_id, "runners": runners, "top3": top3}
                for race_id, runners, top3 in invalid_valid
            ],
        },
    })
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(result, temporary)
    temporary.replace(output)
    history_path = output.with_suffix(".finetune-history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    audit_path = output.with_suffix(".invalid-races.json")
    audit_path.write_text(json.dumps(result["invalid_races"], indent=2) + "\n")
    print(
        f"saved checkpoint={output} best_fine_tune_epoch={best_epoch} "
        f"saved_epoch={saved_epoch} "
        f"improved_over_source={'yes' if improved_over_source else 'no'} "
        f"history={history_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
