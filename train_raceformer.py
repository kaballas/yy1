#!/usr/bin/env python3
"""Train a current-race-only RaceFormerTop3 model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
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
    parser.add_argument("--races-per-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.5)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.1)
    parser.add_argument("--listwise-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--market-residual-scale", type=float, default=0.25,
        help=(
            "Scale applied to learned corrections in the market_residual variant "
            "(default: 0.25)."
        ),
    )
    parser.add_argument(
        "--market-residual-weight", type=float, default=0.05,
        help=(
            "Mean-squared correction penalty for the market_residual variant "
            "(default: 0.05)."
        ),
    )
    parser.add_argument(
        "--standardized-clip", type=float, default=5.0,
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
    parser.add_argument(
        "--training-competition-id",
        help="Comma-separated competition IDs used for training.",
    )
    parser.add_argument(
        "--validation-competition-id",
        help="Comma-separated competition IDs used for validation.",
    )
    parser.add_argument(
        "--competition-split", action="store_true",
        help=(
            "Prompt for training and validation competition IDs. Explicit ID "
            "flags imply this option and are preferable for reproducible jobs."
        ),
    )
    parser.add_argument(
        "--train-all-races", action="store_true",
        help=(
            "Final-refit mode: train for exactly --epochs on every eligible race "
            "across both CSV snapshots. Disables validation selection, early "
            "stopping, and deployment-gate claims."
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
        or args.market_residual_weight < 0
    ):
        raise ValueError("weight decay and loss weights must be non-negative")
    if args.market_residual_scale <= 0:
        raise ValueError("market residual scale must be positive")
    if args.standardized_clip <= 0:
        raise ValueError("standardized clip must be positive")
    if args.max_training_races < 0 or args.max_validation_races < 0:
        raise ValueError("race limits must be zero or positive")
    if args.chronological_validation_races < 0:
        raise ValueError("chronological validation races must be zero or positive")
    if args.variant == "market_residual" and args.checkpoint_metric != "composite":
        raise ValueError("market_residual requires --checkpoint-metric composite")
    if args.train_all_races:
        if (
            args.competition_split or args.training_competition_id is not None
            or args.validation_competition_id is not None
        ):
            raise ValueError(
                "--train-all-races cannot be combined with competition split options"
            )
        if args.max_training_races or args.max_validation_races:
            raise ValueError(
                "--train-all-races cannot be combined with race-count limits"
            )


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


def fit_market_anchor(
    x: np.ndarray,
    y: np.ndarray,
    race_ids: np.ndarray,
    race_indices: dict[int, np.ndarray],
    market_feature_index: int,
    ranking_weight: float,
    cardinality_weight: float,
    listwise_weight: float,
) -> tuple[float, float, dict[str, float]]:
    """Fit a positive-slope market anchor using training races only."""
    minimum_scale = 0.25
    groups = list(race_indices.values())
    bx, by, valid, flat_ids = _pad_batch(
        x, y, race_ids, groups, torch.device("cpu")
    )
    market_z = bx[..., market_feature_index].double()
    targets = by.double()
    prevalence = float(targets[valid].mean())
    initial_bias = math.log(prevalence / (1.0 - prevalence))
    bias = torch.tensor(initial_bias, dtype=torch.float64, requires_grad=True)
    initial_unconstrained_scale = math.log(math.expm1(1.0 - minimum_scale))
    unconstrained_scale = torch.tensor(
        initial_unconstrained_scale, dtype=torch.float64, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [bias, unconstrained_scale], max_iter=100, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        scale = minimum_scale + torch.nn.functional.softplus(unconstrained_scale)
        logits = bias - scale * market_z
        loss, _ = raceformer_losses(
            logits, targets, valid, ranking_weight, cardinality_weight,
            listwise_weight,
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_bias = float(bias.detach())
    fitted_scale = float(
        minimum_scale + torch.nn.functional.softplus(unconstrained_scale.detach())
    )
    with torch.inference_mode():
        logits = fitted_bias - fitted_scale * market_z
        loss, components = raceformer_losses(
            logits, targets, valid, ranking_weight, cardinality_weight,
            listwise_weight,
        )
        probability = torch.sigmoid(logits[valid]).cpu().numpy()
    metrics = probability_metrics(
        targets[valid].cpu().numpy().astype(np.int64), probability, flat_ids
    )
    summary = {
        "fit_loss": float(loss),
        "minimum_scale": minimum_scale,
        "mean_race_probability_sum": float(
            np.mean([
                probability[flat_ids == race_id].sum()
                for race_id in np.unique(flat_ids)
            ])
        ),
        "top3_recall": float(metrics["top3_recall"]),
        "ndcg3": float(metrics["ndcg3"]),
        "pairwise_ranking_accuracy": float(metrics["pairwise_ranking_accuracy"]),
        **{f"loss_{name}": float(value) for name, value in components.items()},
    }
    return fitted_bias, fitted_scale, summary


def _raceformer_objective(
    model: RaceFormerTop3,
    bx: torch.Tensor,
    by: torch.Tensor,
    valid: torch.Tensor,
    ranking_weight: float,
    cardinality_weight: float,
    listwise_weight: float,
    market_residual_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    logits, _, correction = model.forward_parts(bx, valid)
    loss, components = raceformer_losses(
        logits, by, valid, ranking_weight, cardinality_weight, listwise_weight
    )
    residual_penalty = (
        correction[valid].square().mean()
        if model.variant == "market_residual" else logits.new_zeros(())
    )
    loss = loss + market_residual_weight * residual_penalty
    return logits, loss, {**components, "residual_penalty": residual_penalty}


def evaluate(
    model: RaceFormerTop3, x: np.ndarray, y: np.ndarray, race_ids: np.ndarray,
    race_indices: dict[int, np.ndarray], races_per_batch: int,
    ranking_weight: float, cardinality_weight: float, device: torch.device,
    listwise_weight: float,
    market_residual_weight: float = 0.0,
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
            logits, loss, _ = _raceformer_objective(
                model, bx, by, valid, ranking_weight, cardinality_weight,
                listwise_weight, market_residual_weight,
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
    return (_composite_score(metrics),)


def _composite_score(metrics: dict[str, float | int]) -> float:
    """Return the ranking composite used for checkpoint selection."""
    return (
        0.50 * float(metrics["top3_recall"])
        + 0.25 * float(metrics["ndcg3"])
        + 0.25 * float(metrics["pairwise_ranking_accuracy"])
    )


def _market_baseline_metrics(
    targets: np.ndarray,
    race_ids: np.ndarray,
    prices: np.ndarray,
    race_indices: dict[int, np.ndarray],
) -> dict[str, float | int]:
    """Evaluate the always-available closing-price ranking on eligible races."""
    rows = np.concatenate(list(race_indices.values()))
    selected_prices = np.asarray(prices[rows], dtype=np.float64)
    scores = np.full(selected_prices.shape, -np.inf, dtype=np.float64)
    valid = np.isfinite(selected_prices) & (selected_prices > 0)
    scores[valid] = 1.0 / selected_prices[valid]
    return probability_metrics(targets[rows], scores, race_ids[rows])


def _competition_summary(
    race_ids: np.ndarray, competition_ids: np.ndarray
) -> list[tuple[int, int, int]]:
    return [
        (
            competition_id,
            len(np.unique(race_ids[competition_ids == competition_id])),
            int((competition_ids == competition_id).sum()),
        )
        for competition_id in sorted(set(map(int, competition_ids)))
    ]


def _resolve_competition_split(
    args: argparse.Namespace, race_ids: np.ndarray, competition_ids: np.ndarray
) -> tuple[list[int] | None, list[int] | None]:
    train_value = args.training_competition_id
    validation_value = args.validation_competition_id
    requested = (
        args.competition_split or train_value is not None or validation_value is not None
    )
    if not requested:
        return None, None
    if train_value is None and validation_value is None:
        print("AVAILABLE COMPETITIONS", flush=True)
        for competition_id, races, runners in _competition_summary(
            race_ids, competition_ids
        ):
            print(
                f"  competition_id={competition_id} races={races:,} runners={runners:,}",
                flush=True,
            )
        if not sys.stdin.isatty():
            raise ValueError(
                "--competition-split needs an interactive terminal or both "
                "competition ID flags"
            )
        try:
            train_value = input("Training competition IDs (comma-separated): ").strip()
            validation_value = input(
                "Validation competition IDs (comma-separated): "
            ).strip()
        except EOFError as error:
            raise ValueError("Competition IDs are required") from error
    elif train_value is None or validation_value is None:
        raise ValueError(
            "Pass both --training-competition-id and --validation-competition-id"
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
        raise ValueError(
            f"Training and validation competitions overlap: {overlap}"
        )
    available = set(map(int, competition_ids))
    missing = sorted((set(train_ids) | set(validation_ids)) - available)
    if missing:
        raise ValueError(f"Competition ID(s) not found in snapshots: {missing}")
    return train_ids, validation_ids


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
    (
        train_x, train_y, train_ids, train_times, train_competitions, *rest_train
    ) = load_rows_from_csv(
        args.training_csv, features
    )
    (
        valid_x, valid_y, valid_ids, valid_times, valid_competitions, *rest_valid
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
    if args.train_all_races:
        training_competition_ids = sorted(set(map(int, all_competitions)))
        validation_competition_ids = None
    else:
        training_competition_ids, validation_competition_ids = (
            _resolve_competition_split(args, all_ids, all_competitions)
        )
    validation_ids: np.ndarray | None = None
    if args.train_all_races:
        train_x, train_y, train_ids, train_times = (
            all_x, all_y, all_ids, all_times
        )
        valid_x, valid_y = all_x[:0], all_y[:0]
        valid_ids, valid_times = all_ids[:0], all_times[:0]
        print(
            "full_data_refit=yes validation_selection=disabled "
            f"candidate_races={len(np.unique(train_ids)):,} "
            f"competitions={training_competition_ids}",
            flush=True,
        )
    elif training_competition_ids is not None:
        train_mask = np.isin(all_competitions, training_competition_ids)
        valid_mask = np.isin(all_competitions, validation_competition_ids)
        train_x, train_y = all_x[train_mask], all_y[train_mask]
        train_ids, train_times = all_ids[train_mask], all_times[train_mask]
        valid_x, valid_y = all_x[valid_mask], all_y[valid_mask]
        valid_ids, valid_times = all_ids[valid_mask], all_times[valid_mask]
        validation_ids = np.unique(valid_ids)
        if "competition_id" in features and "competition_id" not in zero_features:
            zero_features.append("competition_id")
            print(
                "auto_zeroed_feature=competition_id "
                "reason=competition_holdout_identifier",
                flush=True,
            )
        print(
            "competition_partition "
            f"training_competition_ids={training_competition_ids} "
            f"validation_competition_ids={validation_competition_ids} "
            f"training_races={len(np.unique(train_ids)):,} "
            f"validation_races={len(np.unique(valid_ids)):,}",
            flush=True,
        )
    elif args.chronological_validation_races:
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
    if args.train_all_races:
        valid_races: dict[int, np.ndarray] = {}
        invalid_valid: list[tuple[int, int, int]] = []
    else:
        valid_races, invalid_valid = _selected_indices(
            valid_y, valid_ids, args.max_validation_races, "Validation"
        )
    checkpoint_validation_ids = list(valid_races)
    if "fluc2" not in features:
        raise ValueError("RaceFormer training requires fluc2 for the market baseline")
    market_reference_y = train_y if args.train_all_races else valid_y
    market_reference_ids = train_ids if args.train_all_races else valid_ids
    market_reference_x = train_x if args.train_all_races else valid_x
    market_reference_races = train_races if args.train_all_races else valid_races
    market_metrics = _market_baseline_metrics(
        market_reference_y,
        market_reference_ids,
        market_reference_x[:, features.index("fluc2")],
        market_reference_races,
    )
    market_composite = _composite_score(market_metrics)
    print(
        "MARKET BASELINE ("
        + ("in-sample full-data" if args.train_all_races else "validation")
        + " fluc2 ranking)\n"
        f"top3={market_metrics['top3_recall']:.4f} "
        f"exact={market_metrics['exact_top3_set_rate']:.4f} "
        f"ndcg3={market_metrics['ndcg3']:.4f} "
        f"pairwise={market_metrics['pairwise_ranking_accuracy']:.4f} "
        f"composite={market_composite:.5f}",
        flush=True,
    )
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
    if not args.train_all_races:
        valid_x = transform_raceformer(
            valid_x, valid_ids, features, zero_features, preprocessing
        )
    expanded_features = model_feature_columns(features, preprocessing)

    market_anchor: dict[str, float | str] | None = None
    model_market_args: dict[str, int | float] = {}
    if args.variant == "market_residual":
        if "fluc2" in zero_features:
            raise ValueError("market_residual requires active fluc2")
        anchor_feature = "fluc2__race_percentile"
        if anchor_feature not in expanded_features:
            raise ValueError(
                "market_residual requires the fluc2 within-race percentile feature"
            )
        market_feature_index = expanded_features.index(anchor_feature)
        anchor_bias, anchor_scale, anchor_summary = fit_market_anchor(
            train_x, train_y, train_ids, train_races, market_feature_index,
            args.ranking_loss_weight, args.cardinality_loss_weight,
            args.listwise_loss_weight,
        )
        market_anchor = {
            "source_feature": "fluc2",
            "anchor_feature": anchor_feature,
            "feature_index": market_feature_index,
            "bias": anchor_bias,
            "scale": anchor_scale,
            "residual_scale": args.market_residual_scale,
            "residual_penalty_weight": args.market_residual_weight,
            **anchor_summary,
        }
        model_market_args = {
            "market_feature_index": market_feature_index,
            "market_anchor_bias": anchor_bias,
            "market_anchor_scale": anchor_scale,
            "market_residual_scale": args.market_residual_scale,
        }
        print(
            "MARKET ANCHOR (training races only)\n"
            f"feature={anchor_feature} bias={anchor_bias:+.6f} "
            f"scale={anchor_scale:.6f} "
            f"mean_probability_sum={anchor_summary['mean_race_probability_sum']:.4f} "
            f"residual_scale={args.market_residual_scale:g} "
            f"residual_penalty_weight={args.market_residual_weight:g}",
            flush=True,
        )

    model = RaceFormerTop3(
        len(expanded_features), args.variant, args.hidden_dim, args.model_dim,
        args.attention_heads, args.race_layers, args.feedforward_dim, args.dropout,
        **model_market_args,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    best_selection: tuple[float, ...] | None = (
        (market_composite,)
        if args.variant == "market_residual" and not args.train_all_races else None
    )
    best_state = (
        copy.deepcopy(model.state_dict())
        if args.variant == "market_residual" and not args.train_all_races else None
    )
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
            "train_all_races": args.train_all_races,
            "ranking_loss_weight": args.ranking_loss_weight,
            "cardinality_loss_weight": args.cardinality_loss_weight,
            "listwise_loss_weight": args.listwise_loss_weight,
            "market_residual_scale": args.market_residual_scale,
            "market_residual_weight": args.market_residual_weight,
            "max_training_races": args.max_training_races,
            "max_validation_races": args.max_validation_races,
            "chronological_validation_races": args.chronological_validation_races,
            "training_competition_ids": training_competition_ids,
            "validation_competition_ids": validation_competition_ids,
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
                "full_data_fit" if args.train_all_races else
                "competition_holdout" if training_competition_ids is not None else
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
        component_values = {
            "bce": [], "ranking": [], "cardinality": [], "listwise": [],
            "residual_penalty": [],
        }
        for groups in _batches(train_races, args.races_per_batch, rng=rng):
            bx, by, valid, _ = _pad_batch(train_x, train_y, train_ids, groups, device)
            optimizer.zero_grad(set_to_none=True)
            logits, loss, components = _raceformer_objective(
                model, bx, by, valid, args.ranking_loss_weight,
                args.cardinality_loss_weight, args.listwise_loss_weight,
                args.market_residual_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite RaceFormer training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            for name, value in components.items():
                component_values[name].append(float(value.detach()))

        if args.train_all_races:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "mean_residual_penalty": float(
                    np.mean(component_values["residual_penalty"])
                ),
                "selection_disabled_full_data_fit": True,
            }
            history.append(row)
            print(
                f"epoch={epoch:03d} train={row['train_loss']:.5f} "
                f"fixed_full_data_refit={epoch}/{args.epochs}",
                flush=True,
            )
            continue

        validation_loss, metrics = evaluate(
            model, valid_x, valid_y, valid_ids, valid_races,
            args.races_per_batch, args.ranking_loss_weight,
            args.cardinality_loss_weight, device, args.listwise_loss_weight,
            args.market_residual_weight,
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
            "composite_score": _composite_score(metrics),
            "market_composite_delta": _composite_score(metrics) - market_composite,
            "mean_residual_penalty": float(
                np.mean(component_values["residual_penalty"])
            ),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train={row['train_loss']:.5f} valid={validation_loss:.5f} "
            f"top3={metrics['top3_recall']:.4f} exact={metrics['exact_top3_set_rate']:.4f} "
            f"ndcg3={metrics['ndcg3']:.4f} pairwise={metrics['pairwise_ranking_accuracy']:.4f} "
            f"logloss={metrics['logloss']:.5f} select={selection[0]:.5f} "
            f"market_delta={row['market_composite_delta']:+.5f} "
            f"best={'yes' if improved else 'no'} "
            f"stale={stale_epochs}/{args.early_stopping_patience}",
            flush=True,
        )
        if stale_epochs >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint state")
    full_fit_metrics: dict[str, float | int] | None = None
    full_fit_loss: float | None = None
    if args.train_all_races:
        full_fit_loss, full_fit_metrics = evaluate(
            model, train_x, train_y, train_ids, train_races,
            args.races_per_batch, args.ranking_loss_weight,
            args.cardinality_loss_weight, device, args.listwise_loss_weight,
            args.market_residual_weight,
        )
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
        "checkpoint_metric": (
            "fixed_epochs_full_data" if args.train_all_races
            else args.checkpoint_metric
        ),
        "best_selection": best_selection,
        "history": history,
        "market_baseline": {
            **market_metrics,
            "composite": market_composite,
            "feature": "fluc2",
            "cohort": (
                "full_data_training_races_in_sample"
                if args.train_all_races else "checkpoint_validation_races"
            ),
        },
        "market_anchor": market_anchor,
        "full_data_fit_metrics": full_fit_metrics,
        "full_data_fit_loss": full_fit_loss,
        "training_config": vars(args),
        "partition": {
            "mode": (
                "full_data_fit" if args.train_all_races else
                "competition_holdout" if training_competition_ids is not None else
                "chronological_latest_complete_races"
                if validation_ids is not None else "legacy_database_views"
            ),
            "training_competition_ids": training_competition_ids,
            "validation_competition_ids": validation_competition_ids,
            "training_race_ids": (
                sorted(map(int, train_races))
                if training_competition_ids is not None or args.train_all_races else None
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
    if args.train_all_races:
        assert full_fit_metrics is not None and full_fit_loss is not None
        print(
            f"saved checkpoint={output} fixed_epoch={best_epoch} "
            f"history={history_path} invalid_race_audit={audit_path}\n"
            "FULL DATA REFIT validation_metrics=unavailable "
            "deployment_gate_model_beats_market=not_applicable "
            f"in_sample_composite={_composite_score(full_fit_metrics):.5f} "
            f"in_sample_loss={full_fit_loss:.5f}"
        )
    else:
        best_metrics = market_metrics if best_epoch == 0 else history[best_epoch - 1]
        best_composite = _composite_score(best_metrics)
        beats_market = best_composite > market_composite
        print(
            f"saved checkpoint={output} best_epoch={best_epoch} history={history_path} "
            f"invalid_race_audit={audit_path}\n"
            f"MARKET COMPARISON model_composite={best_composite:.5f} "
            f"market_composite={market_composite:.5f} "
            f"deployment_gate_model_beats_market={'yes' if beats_market else 'no'}"
        )


if __name__ == "__main__":
    main()
