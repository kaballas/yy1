#!/usr/bin/env python3
"""Train a current-race-only RaceFormerTop3 model."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.config import DEFAULT_DB, DEFAULT_FEATURES
from src.constants import TRAINING_ROWS_VIEW, VALIDATION_ROWS_VIEW
from src.database import export_rows_to_csv, load_rows_from_csv
from src.dataset import load_feature_manifest
from src.metrics import probability_metrics
from src.model.raceformer import RaceFormerTop3, raceformer_losses
from src.raceformer_preprocessing import (
    fit_raceformer_preprocessor,
    model_feature_columns,
    transform_raceformer,
)
from src.raceformer_partition import (
    chronological_validation_ids,
    combine_disjoint_snapshots,
    partition_by_validation_ids,
)
from src.validation import build_race_indices, invalid_race_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RaceFormerTop3 without historical labelled context or ICL."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=Path("outputs/raceformer_top3.pt"))
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
        "--variant", choices=sorted(RaceFormerTop3.VARIANTS), default="race_token"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--race-layers", type=int, default=1)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--races-per-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.1)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--standardized-clip", type=float, default=100.0,
        help="Clip robustly transformed base inputs to +/- this value.",
    )
    parser.add_argument(
        "--layoff-buckets", action=argparse.BooleanOptionalAction, default=None,
        help="Deprecated shortcut for --layoff-bucket-mode cumulative.",
    )
    parser.add_argument(
        "--layoff-bucket-mode", choices=("none", "cumulative", "exclusive"),
        default="none",
        help=(
            "Layoff indicators: cumulative thresholds, mutually exclusive bands "
            "with 0-29 as reference, or none (default)."
        ),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument(
        "--checkpoint-metric", choices=("loss", "composite"), default="composite"
    )
    parser.add_argument(
        "--max-training-races", type=int, default=0,
        help="Use only the most recent N eligible training races; 0 uses all.",
    )
    parser.add_argument(
        "--max-validation-races", type=int, default=0,
        help="Use only the most recent N eligible validation races; 0 uses all.",
    )
    parser.add_argument(
        "--chronological-validation-races", type=int, default=1000,
        help=(
            "Hold out the latest N eligible races across both database views; "
            "0 preserves the legacy view-defined partition (default: 1000)."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
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
    if (
        args.weight_decay < 0 or args.ranking_loss_weight < 0
        or args.cardinality_loss_weight < 0 or args.listwise_loss_weight < 0
    ):
        raise ValueError("weight decay and loss weights must be non-negative")
    if args.standardized_clip <= 0:
        raise ValueError("standardized clip must be positive")
    if args.max_training_races < 0 or args.max_validation_races < 0:
        raise ValueError("race limits must be zero or positive")
    if args.chronological_validation_races < 0:
        raise ValueError("chronological validation races must be zero or positive")


def _selected_indices(
    targets: np.ndarray, race_ids: np.ndarray, maximum: int, partition: str
) -> tuple[dict[int, np.ndarray], list[tuple[int, int, int]]]:
    mask = np.ones(len(race_ids), dtype=bool)
    invalid = invalid_race_targets(targets, race_ids, mask)
    if invalid:
        invalid_ids = [race_id for race_id, _, _ in invalid]
        preview = ", ".join(
            f"race_id={race_id} runners={runners} top3={top3}"
            for race_id, runners, top3 in invalid[:10]
        )
        print(
            f"WARNING skipped_invalid_{partition.lower()}_races={len(invalid)} "
            f"preview=[{preview}]", flush=True,
        )
        mask &= ~np.isin(race_ids, invalid_ids)
    indices = build_race_indices(race_ids, mask)
    # CSV snapshots are ordered by start_time/race_id. Preserve that order;
    # numeric race IDs are not themselves a chronological contract.
    ordered = list(dict.fromkeys(map(int, race_ids[mask])))
    if maximum:
        ordered = ordered[-maximum:]
    if not ordered:
        raise ValueError(f"No eligible {partition.lower()} races")
    return {race_id: indices[race_id] for race_id in ordered}, invalid


def _batches(
    race_indices: dict[int, np.ndarray], races_per_batch: int,
    *, rng: np.random.Generator | None,
) -> Iterator[list[np.ndarray]]:
    race_ids = np.asarray(list(race_indices), dtype=np.int64)
    if rng is not None:
        race_ids = rng.permutation(race_ids)
    for start in range(0, len(race_ids), races_per_batch):
        yield [race_indices[int(race_id)] for race_id in race_ids[start:start + races_per_batch]]


def _pad_batch(
    x: np.ndarray, y: np.ndarray, race_ids: np.ndarray, groups: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    maximum = max(map(len, groups))
    batch_x = np.zeros((len(groups), maximum, x.shape[1]), dtype=np.float32)
    batch_y = np.zeros((len(groups), maximum), dtype=np.float32)
    valid = np.zeros((len(groups), maximum), dtype=bool)
    flat_race_ids = []
    for batch_index, indices in enumerate(groups):
        count = len(indices)
        batch_x[batch_index, :count] = x[indices]
        batch_y[batch_index, :count] = y[indices]
        valid[batch_index, :count] = True
        flat_race_ids.extend(race_ids[indices].tolist())
    return (
        torch.from_numpy(batch_x).to(device),
        torch.from_numpy(batch_y).to(device),
        torch.from_numpy(valid).to(device),
        np.asarray(flat_race_ids, dtype=np.int64),
    )


def evaluate(
    model: RaceFormerTop3, x: np.ndarray, y: np.ndarray, race_ids: np.ndarray,
    race_indices: dict[int, np.ndarray], races_per_batch: int,
    ranking_weight: float, cardinality_weight: float, device: torch.device,
    listwise_weight: float,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    weighted_loss = 0.0
    evaluated_races = 0
    probabilities = []
    targets = []
    evaluated_race_ids = []
    with torch.inference_mode():
        for groups in _batches(race_indices, races_per_batch, rng=None):
            bx, by, valid, flat_ids = _pad_batch(x, y, race_ids, groups, device)
            logits = model(bx, valid)
            loss, _ = raceformer_losses(
                logits, by, valid, ranking_weight, cardinality_weight, listwise_weight
            )
            weighted_loss += float(loss) * len(groups)
            evaluated_races += len(groups)
            probabilities.append(torch.sigmoid(logits[valid]).cpu().numpy())
            targets.append(by[valid].cpu().numpy().astype(np.int64))
            evaluated_race_ids.append(flat_ids)
    metrics = probability_metrics(
        np.concatenate(targets), np.concatenate(probabilities),
        np.concatenate(evaluated_race_ids),
    )
    return weighted_loss / evaluated_races, metrics


def _selection(
    validation_loss: float, metrics: dict[str, float | int], mode: str
) -> tuple[float, ...]:
    if mode == "loss":
        return (-validation_loss,)
    score = (
        0.50 * float(metrics["top3_recall"])
        + 0.25 * float(metrics["ndcg3"])
        + 0.25 * float(metrics["pairwise_ranking_accuracy"])
    )
    return (score,)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    features, zero_features = load_feature_manifest(args.features_json)
    if not args.no_export:
        export_rows_to_csv(args.db, features, TRAINING_ROWS_VIEW, args.training_csv)
        export_rows_to_csv(args.db, features, VALIDATION_ROWS_VIEW, args.validation_csv)
    train_x, train_y, train_ids, train_times, *_ = load_rows_from_csv(
        args.training_csv, features
    )
    valid_x, valid_y, valid_ids, valid_times, *_ = load_rows_from_csv(
        args.validation_csv, features
    )
    validation_ids: np.ndarray | None = None
    if args.chronological_validation_races:
        all_x, all_y, all_ids, all_times = combine_disjoint_snapshots(
            train_x, train_y, train_ids, train_times,
            valid_x, valid_y, valid_ids, valid_times,
        )
        validation_ids = chronological_validation_ids(
            all_y, all_ids, args.chronological_validation_races
        )
        (
            train_x, train_y, train_ids, train_times,
            valid_x, valid_y, valid_ids, valid_times,
        ) = partition_by_validation_ids(
            all_x, all_y, all_ids, all_times, validation_ids
        )
        print(
            f"chronological_partition validation_races={len(validation_ids)} "
            f"cutoff={valid_times[0].isoformat()}", flush=True,
        )
    train_races, invalid_train = _selected_indices(
        train_y, train_ids, args.max_training_races, "Training"
    )
    valid_races, invalid_valid = _selected_indices(
        valid_y, valid_ids, args.max_validation_races, "Validation"
    )
    checkpoint_validation_ids = list(valid_races)
    train_rows = np.concatenate(list(train_races.values()))
    layoff_bucket_mode = args.layoff_bucket_mode
    if args.layoff_buckets is not None:
        shortcut_mode = "cumulative" if args.layoff_buckets else "none"
        if args.layoff_bucket_mode != "none" and args.layoff_bucket_mode != shortcut_mode:
            raise ValueError("--layoff-buckets conflicts with --layoff-bucket-mode")
        layoff_bucket_mode = shortcut_mode
    preprocessing = fit_raceformer_preprocessor(
        train_x[train_rows], features, clip=args.standardized_clip,
        layoff_bucket_mode=layoff_bucket_mode,
    )
    train_x = transform_raceformer(
        train_x, train_ids, features, zero_features, preprocessing
    )
    valid_x = transform_raceformer(
        valid_x, valid_ids, features, zero_features, preprocessing
    )
    expanded_features = model_feature_columns(features, preprocessing)

    model = RaceFormerTop3(
        len(expanded_features), args.variant, args.hidden_dim, args.model_dim,
        args.attention_heads, args.race_layers, args.feedforward_dim, args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    best_selection: tuple[float, ...] | None = None
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history = []
    zeroed_set = set(zero_features)
    active_features = [
        name for name in expanded_features
        if name not in zeroed_set
        and not any(name == f"{zeroed_name}__race_percentile" for zeroed_name in zeroed_set)
    ]
    effective_hyperparameters = {
        "model": model.config(),
        "training": {
            "optimizer": "AdamW",
            "epochs": args.epochs,
            "races_per_batch": args.races_per_batch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "early_stopping_patience": args.early_stopping_patience,
            "checkpoint_metric": args.checkpoint_metric,
            "ranking_loss_weight": args.ranking_loss_weight,
            "cardinality_loss_weight": args.cardinality_loss_weight,
            "listwise_loss_weight": args.listwise_loss_weight,
            "max_training_races": args.max_training_races,
            "max_validation_races": args.max_validation_races,
            "chronological_validation_races": args.chronological_validation_races,
            "layoff_bucket_mode": layoff_bucket_mode,
            "seed": args.seed,
            "device": str(device),
        },
        "data": {
            "database": str(args.db.resolve()),
            "features_json": str(args.features_json.resolve()),
            "output": str(args.output.resolve()),
            "training_csv": str(args.training_csv.resolve()),
            "validation_csv": str(args.validation_csv.resolve()),
            "export_snapshots": not args.no_export,
            "raw_feature_count": len(features),
            "model_feature_count": len(expanded_features),
            "raw_feature_columns": features,
            "model_feature_columns": expanded_features,
            "active_trainable_features": active_features,
            "train_races": len(train_races),
            "validation_races": len(valid_races),
            "partition_mode": (
                "chronological_latest_complete_races"
                if validation_ids is not None else "legacy_database_views"
            ),
        },
        "preprocessing": {
            "version": preprocessing["version"],
            "scaler": "median_mad_with_std_fallback",
            "clip": preprocessing["clip"],
            "log1p_features": preprocessing["log1p_features"],
            "relative_features": preprocessing["relative_features"],
            "layoff_bucket_features": preprocessing["layoff_bucket_features"],
            "layoff_bucket_mode": preprocessing["layoff_bucket_mode"],
            "zeroed_features": zero_features,
        },
        "parameters": {
            "trainable": sum(parameter.numel() for parameter in model.parameters()),
            "total": sum(parameter.numel() for parameter in model.parameters()),
        },
        "composite_selection_weights": {
            "top3_recall": 0.50,
            "ndcg3": 0.25,
            "pairwise_ranking_accuracy": 0.25,
        },
    }
    print(
        "RACEFORMER TRAINING\n"
        f"variant={args.variant} raw_features={len(features)} "
        f"model_features={len(expanded_features)} train_races={len(train_races)} "
        f"validation_races={len(valid_races)} device={device}\n"
        "historical_context=OFF icl=OFF context_labels=OFF",
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
        component_values = {"bce": [], "ranking": [], "cardinality": [], "listwise": []}
        for groups in _batches(train_races, args.races_per_batch, rng=rng):
            bx, by, valid, _ = _pad_batch(train_x, train_y, train_ids, groups, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, valid)
            loss, components = raceformer_losses(
                logits, by, valid, args.ranking_loss_weight,
                args.cardinality_loss_weight, args.listwise_loss_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite RaceFormer training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            for name, value in components.items():
                component_values[name].append(float(value.detach()))

        validation_loss, metrics = evaluate(
            model, valid_x, valid_y, valid_ids, valid_races,
            args.races_per_batch, args.ranking_loss_weight,
            args.cardinality_loss_weight, device, args.listwise_loss_weight,
        )
        selection = _selection(validation_loss, metrics, args.checkpoint_metric)
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
            **metrics,
            "selection_score": selection[0],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.5f} valid={validation_loss:.5f} "
            f"top3={metrics['top3_recall']:.4f} exact={metrics['exact_top3_set_rate']:.4f} "
            f"ndcg3={metrics['ndcg3']:.4f} pairwise={metrics['pairwise_ranking_accuracy']:.4f} "
            f"logloss={metrics['logloss']:.5f} select={selection[0]:.5f} "
            f"best={'yes' if improved else 'no'} "
            f"stale={stale_epochs}/{args.early_stopping_patience}",
            flush=True,
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "checkpoint_type": "raceformer_top3",
        "checkpoint_version": 2,
        "model_state_dict": best_state,
        "model_config": model.config(),
        "feature_columns": features,
        "raw_feature_columns": features,
        "model_feature_columns": expanded_features,
        "zeroed_features": zero_features,
        "preprocessing": preprocessing,
        # Retain these aliases for simple checkpoint inspection tools.
        "median": preprocessing["median"],
        "scale": preprocessing["scale"],
        "best_epoch": best_epoch,
        "checkpoint_metric": args.checkpoint_metric,
        "best_selection": best_selection,
        "history": history,
        "training_config": vars(args),
        "partition": {
            "mode": (
                "chronological_latest_complete_races"
                if validation_ids is not None else "legacy_database_views"
            ),
            "validation_race_ids": (
                checkpoint_validation_ids if validation_ids is not None else None
            ),
            "validation_race_count": (
                len(checkpoint_validation_ids)
            ),
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
    }
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    history_path = output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n")
    audit_path = output.with_suffix(".invalid-races.json")
    audit_path.write_text(json.dumps(checkpoint["invalid_races"], indent=2) + "\n")
    print(
        f"saved checkpoint={output} best_epoch={best_epoch} history={history_path} "
        f"invalid_race_audit={audit_path}"
    )


if __name__ == "__main__":
    main()
