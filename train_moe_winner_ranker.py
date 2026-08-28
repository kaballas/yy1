#!/usr/bin/env python3
"""Train a market-blind baseline MLP or race-level winner Mixture of Experts."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_DB, DEFAULT_FEATURES
from src.dataset import load_feature_manifest
from src.model.race_moe import (
    RaceMixtureOfExperts, RaceWinnerModelConfig, race_softmax_nll,
    router_balance_loss,
)
from src.race_moe_data import (
    batches, chronological_race_ids, load_finished_winner_rows,
    market_blind_features, numeric_matrix, pad_batch, race_indices,
)
from src.race_moe_evaluation import collapse_warnings, evaluate_model
from src.race_moe_snapshot import (
    create_split_snapshot, load_split_snapshot, snapshot_manifest_reference,
)
from src.raceformer_preprocessing import (
    fit_raceformer_preprocessor, model_feature_columns, transform_raceformer,
)


def _dims(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden dims must be comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("hidden dims must be positive")
    return result


def _top_k(value: str) -> int | None:
    if value.lower() in {"all", "dense"}:
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be an integer or all") from error
    if result < 1:
        raise argparse.ArgumentTypeError("top-k must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=Path("outputs/race_winner_moe.pt"))
    parser.add_argument("--model-type", choices=("baseline", "moe"), default="moe")
    parser.add_argument("--include-market-features", action="store_true")
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument(
        "--split-checkpoint", type=Path,
        help="Freeze train/validation/test race IDs from an existing checkpoint.",
    )
    snapshot = parser.add_mutually_exclusive_group()
    snapshot.add_argument(
        "--snapshot-dir", type=Path,
        help="Create a new immutable training/validation/test feature snapshot.",
    )
    snapshot.add_argument(
        "--snapshot-manifest", type=Path,
        help="Train only from an existing hash-verified immutable snapshot.",
    )
    parser.add_argument("--max-training-races", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--races-per-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--representation-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=_top_k, default=2)
    parser.add_argument("--moe-gate-temperature", type=float, default=1.0)
    parser.add_argument("--moe-router-balance-weight", type=float, default=0.01)
    parser.add_argument(
        "--moe-expert-hidden-dims", "--expert-hidden-dims",
        type=_dims, default=(64,),
    )
    parser.add_argument("--moe-router-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--moe-routing-mode", choices=("learned", "fixed_uniform"),
        default="learned",
    )
    parser.add_argument(
        "--moe-expert-context-conditioning",
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument("--moe-collapse-threshold", type=float, default=0.80)
    parser.add_argument("--moe-correlation-threshold", type=float, default=0.95)
    parser.add_argument("--standardized-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "validation-races": args.validation_races, "test-races": args.test_races,
        "epochs": args.epochs, "races-per-batch": args.races_per_batch,
        "learning-rate": args.learning_rate, "max-grad-norm": args.max_grad_norm,
        "early-stopping-patience": args.early_stopping_patience,
        "moe-num-experts": args.moe_num_experts,
        "moe-gate-temperature": args.moe_gate_temperature,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError("These arguments must be positive: " + ", ".join(bad))
    if args.max_training_races < 0 or args.weight_decay < 0:
        raise ValueError("race limit and weight decay must be non-negative")
    if args.moe_router_balance_weight < 0:
        raise ValueError("router balance weight must be non-negative")
    if args.moe_top_k is not None and args.moe_top_k > args.moe_num_experts:
        raise ValueError("--moe-top-k cannot exceed --moe-num-experts")
    if not 0 < args.moe_collapse_threshold <= 1:
        raise ValueError("collapse threshold must be in (0, 1]")
    if not 0 < args.moe_correlation_threshold <= 1:
        raise ValueError("correlation threshold must be in (0, 1]")
    if args.model_type == "baseline" and args.moe_routing_mode != "learned":
        raise ValueError("fixed-uniform routing requires --model-type moe")
    if args.split_checkpoint is not None and args.max_training_races:
        raise ValueError("--split-checkpoint cannot be combined with --max-training-races")
    if args.snapshot_manifest is not None and args.split_checkpoint is not None:
        raise ValueError("A snapshot manifest already defines the frozen split")
    if args.snapshot_manifest is not None and args.max_training_races:
        raise ValueError("A snapshot manifest cannot be combined with a race limit")


def _selection(metrics: dict[str, float | int]) -> tuple[float, float, float]:
    return (
        float(metrics["top1_hit_rate"]), float(metrics["mrr"]),
        -float(metrics["race_logloss"]),
    )


def _run_epoch(
    model: RaceMixtureOfExperts, optimizer: torch.optim.Optimizer,
    x: np.ndarray, y: np.ndarray, indices: dict[int, np.ndarray],
    args: argparse.Namespace, device: torch.device, rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    ranking_values, balance_values, total_values = [], [], []
    for groups in batches(indices, args.races_per_batch, rng):
        bx, by, valid = pad_batch(x, y, groups, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(bx, valid, return_diagnostics=True)
        ranking = race_softmax_nll(output["logits"], by, valid)
        balance = router_balance_loss(output["dense_router_weights"], valid)
        total = ranking + args.moe_router_balance_weight * balance
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite winner MoE loss")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        ranking_values.append(float(ranking.detach()))
        balance_values.append(float(balance.detach()))
        total_values.append(float(total.detach()))
    return {
        "main_ranking_loss": float(np.mean(ranking_values)),
        "router_balance_loss": float(np.mean(balance_values)),
        "total_loss": float(np.mean(total_values)),
    }


def _split_range(frame, race_ids_values: list[int]) -> dict[str, Any]:
    selected = frame.loc[frame["race_id"].isin(race_ids_values)]
    return {
        "races": len(race_ids_values), "first_race_id": race_ids_values[0],
        "last_race_id": race_ids_values[-1],
        "start_time": str(selected["start_time_iso"].iloc[0]),
        "end_time": str(selected["start_time_iso"].iloc[-1]),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    configured_features, configured_zeroed = load_feature_manifest(args.features_json)
    features, market_excluded = market_blind_features(
        configured_features, include_market=args.include_market_features
    )
    zeroed = [name for name in configured_zeroed if name in features]
    snapshot_manifest_path: Path | None = None
    snapshot_metadata: dict[str, Any] | None = None
    if args.snapshot_manifest is not None:
        partitions, snapshot_metadata = load_split_snapshot(args.snapshot_manifest)
        snapshot_manifest_path = args.snapshot_manifest.resolve()
        snapshot_features = list(snapshot_metadata["feature_columns"])
        forbidden = [
            name for name in snapshot_features
            if name not in set(features)
        ]
        if forbidden:
            raise ValueError(
                "Snapshot violates the current market-blind/identifier contract: "
                + ", ".join(forbidden)
            )
        features = snapshot_features
        zeroed = [name for name in zeroed if name in features]
        unavailable_features = [
            name for name in snapshot_metadata.get("excluded_features", [])
            if name not in set(market_excluded)
        ]
        train_ids = list(map(int, partitions["training"]["race_id"].drop_duplicates()))
        validation_ids = list(map(int, partitions["validation"]["race_id"].drop_duplicates()))
        test_ids = list(map(int, partitions["test"]["race_id"].drop_duplicates()))
        frame = pd.concat(partitions.values(), ignore_index=True)
        print(
            f"immutable_snapshot={snapshot_manifest_path} hash_verification=passed "
            f"training_races={len(train_ids):,} validation_races={len(validation_ids):,} "
            f"test_races={len(test_ids):,}", flush=True,
        )
    else:
        frame = load_finished_winner_rows(args.db, features)
    if args.snapshot_manifest is None and args.split_checkpoint is not None:
        split_source = torch.load(
            args.split_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        source_partition = split_source.get("partition", {})
        train_ids = list(map(int, source_partition.get("training_race_ids") or []))
        validation_ids = list(map(int, source_partition.get("validation_race_ids") or []))
        test_ids = list(map(int, source_partition.get("test_race_ids") or []))
        if not train_ids or not validation_ids or not test_ids:
            raise ValueError("--split-checkpoint has no complete three-way race split")
        available_ids = set(map(int, frame["race_id"]))
        missing_ids = sorted(
            (set(train_ids) | set(validation_ids) | set(test_ids)) - available_ids
        )
        if missing_ids:
            raise ValueError(
                f"Database is missing {len(missing_ids)} frozen split races"
            )
        print(
            f"frozen_split_checkpoint={args.split_checkpoint.resolve()} "
            f"training_races={len(train_ids):,} validation_races={len(validation_ids):,} "
            f"test_races={len(test_ids):,}", flush=True,
        )
    elif args.snapshot_manifest is None:
        train_ids, validation_ids, test_ids = chronological_race_ids(
            frame, args.validation_races, args.test_races
        )
    if args.snapshot_manifest is None and args.max_training_races:
        train_ids = train_ids[-args.max_training_races:]
    if args.snapshot_manifest is None:
        partitions = {
            "training": frame.loc[frame["race_id"].isin(train_ids)].copy(),
            "validation": frame.loc[frame["race_id"].isin(validation_ids)].copy(),
            "test": frame.loc[frame["race_id"].isin(test_ids)].copy(),
        }
        unavailable_features = [
            name for name in features
            if not pd.to_numeric(
                partitions["training"][name], errors="coerce"
            ).notna().any()
        ]
        if unavailable_features:
            unavailable_set = set(unavailable_features)
            features = [name for name in features if name not in unavailable_set]
            zeroed = [name for name in zeroed if name in features]
            print(
                f"unavailable_training_features_excluded={len(unavailable_features)} "
                + json.dumps(unavailable_features), flush=True,
            )
    if not features:
        raise ValueError("No numerical features have training coverage")
    if args.snapshot_dir is not None:
        snapshot_partitions = dict(partitions)
        used_ids = set(train_ids) | set(validation_ids) | set(test_ids)
        test_end = partitions["test"]["start_time_iso"].max()
        newer = frame.loc[
            (frame["start_time_iso"] > test_end)
            & ~frame["race_id"].isin(used_ids)
        ].copy()
        if not newer.empty:
            snapshot_partitions["test2"] = newer
            print(
                f"snapshot_additional_test2_races={newer['race_id'].nunique():,} "
                "selection_use=never", flush=True,
            )
        snapshot_manifest_path = create_split_snapshot(
            args.snapshot_dir, snapshot_partitions, features, database=args.db,
            excluded_features=[*market_excluded, *unavailable_features],
        )
        partitions, snapshot_metadata = load_split_snapshot(snapshot_manifest_path)
        frame = pd.concat(partitions.values(), ignore_index=True)
        print(
            f"immutable_snapshot_created={snapshot_manifest_path} "
            "hash_verification=passed", flush=True,
        )
    raw = {name: numeric_matrix(part, features) for name, part in partitions.items()}
    preprocessing = fit_raceformer_preprocessor(
        raw["training"], features, clip=args.standardized_clip,
        layoff_bucket_mode="none",
    )
    values = {
        name: transform_raceformer(
            matrix, part["race_id"].to_numpy(dtype=np.int64), features, zeroed,
            preprocessing,
        )
        for (name, part), matrix in zip(partitions.items(), raw.values())
    }
    expanded_features = model_feature_columns(features, preprocessing)
    arrays = {}
    for name, part in partitions.items():
        race_id_array = part["race_id"].to_numpy(dtype=np.int64)
        arrays[name] = (
            values[name], part["is_winner"].to_numpy(dtype=np.float32),
            race_id_array, race_indices(race_id_array),
        )

    num_experts = 1 if args.model_type == "baseline" else args.moe_num_experts
    top_k = (
        1 if args.model_type == "baseline" else
        None if args.moe_routing_mode == "fixed_uniform" else args.moe_top_k
    )
    config = RaceWinnerModelConfig(
        feature_count=len(expanded_features), model_type=args.model_type,
        encoder_hidden_dim=args.encoder_hidden_dim,
        representation_dim=args.representation_dim, dropout=args.dropout,
        num_experts=num_experts, top_k=top_k,
        gate_temperature=args.moe_gate_temperature,
        expert_hidden_dims=args.moe_expert_hidden_dims,
        router_hidden_dim=args.moe_router_hidden_dim,
        expert_context_conditioning=args.moe_expert_context_conditioning,
        routing_mode=args.moe_routing_mode,
    )
    model = RaceMixtureOfExperts(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print(
        "RACE WINNER MOE EXPERIMENT\n"
        f"model_type={args.model_type} objective=race_softmax_nll "
        f"market_blind={not args.include_market_features} "
        f"raw_features={len(features)} model_features={len(expanded_features)}\n"
        f"train_races={len(train_ids):,} validation_races={len(validation_ids):,} "
        f"sealed_test_races={len(test_ids):,} device={device}\n"
        f"num_experts={num_experts} top_k={top_k if top_k is not None else 'all'} "
        f"routing_mode={config.routing_mode} "
        f"temperature={args.moe_gate_temperature:g} "
        f"balance_weight={args.moe_router_balance_weight:g} "
        f"expert_hidden_dims={list(args.moe_expert_hidden_dims)}",
        flush=True,
    )
    print(
        f"trainable_parameters={model.trainable_parameter_count():,} "
        f"executed_forward_parameters={model.executed_parameter_count():,} "
        f"contributing_parameters={model.contributing_parameter_count():,}", flush=True,
    )
    print(
        f"market_features_excluded={len(market_excluded)} "
        + json.dumps(market_excluded), flush=True,
    )
    best_state = None; best_selection = None; best_epoch = 0; stale = 0; history = []
    rng = np.random.default_rng(args.seed)
    train_x, train_y, _, train_index = arrays["training"]
    for epoch in range(1, args.epochs + 1):
        losses = _run_epoch(
            model, optimizer, train_x, train_y, train_index, args, device, rng
        )
        vx, vy, vids, vindices = arrays["validation"]
        validation_metrics, validation_diagnostics, _ = evaluate_model(
            model, vx, vy, vids, vindices, partitions["validation"],
            args.races_per_batch, device,
        )
        selection = _selection(validation_metrics)
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection; best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch; stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch, **losses,
            "validation_metrics": validation_metrics,
            "validation_router": {
                key: validation_diagnostics[key] for key in (
                    "dominant_expert_rate", "gate_entropy",
                    "average_number_of_active_experts", "router_balance_loss",
                )
            }, "best": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} main_ranking_loss={losses['main_ranking_loss']:.5f} "
            f"router_balance_loss={losses['router_balance_loss']:.5f} "
            f"total_loss={losses['total_loss']:.5f} "
            f"validation_top1={validation_metrics['top1_hit_rate']:.4f} "
            f"top2={validation_metrics['top2_containment']:.4f} "
            f"top3={validation_metrics['top3_containment']:.4f} "
            f"mrr={validation_metrics['mrr']:.4f} "
            f"logloss={validation_metrics['race_logloss']:.5f} "
            f"dominant={validation_diagnostics['dominant_expert_rate']:.3f} "
            f"entropy={validation_diagnostics['gate_entropy']:.3f} "
            f"best={'yes' if improved else 'no'} stale={stale}/{args.early_stopping_patience}",
            flush=True,
        )
        if stale >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    results = {}; diagnostics = {}; predictions = {}
    for name in ("training", "validation", "test"):
        x, y, ids, indices = arrays[name]
        results[name], diagnostics[name], predictions[name] = evaluate_model(
            model, x, y, ids, indices, partitions[name], args.races_per_batch, device
        )
    split_ranges = {
        "training": _split_range(frame, train_ids),
        "validation": _split_range(frame, validation_ids),
        "test": _split_range(frame, test_ids),
    }
    output = args.output.resolve()
    checkpoint = {
        "checkpoint_type": "race_winner_moe",
        "checkpoint_version": 1,
        "model_state_dict": best_state, "model_config": model.config(),
        "training_objective": "race_softmax_nll",
        "router_balance_weight": args.moe_router_balance_weight,
        "raw_feature_columns": features,
        "model_feature_columns": expanded_features,
        "configured_feature_columns": configured_features,
        "market_features_excluded": market_excluded,
        "unavailable_training_features_excluded": unavailable_features,
        "market_features_enabled": args.include_market_features,
        "feature_snapshot": (
            {
                "manifest": snapshot_manifest_reference(
                    snapshot_manifest_path, output,
                ),
                "manifest_relative_to": "checkpoint_directory",
                "splits": snapshot_metadata["splits"],
                "identity_columns": snapshot_metadata["identity_columns"],
            }
            if snapshot_manifest_path is not None and snapshot_metadata is not None
            else None
        ),
        "zeroed_features": zeroed, "preprocessing": preprocessing,
        "branch_configuration": {"runner_encoder": "shared_mlp", "race_context": "masked_mean_plus_max"},
        "router_architecture": {
            "input": "runner_embedding_plus_mean_max_race_context",
            "hidden_dim": args.moe_router_hidden_dim,
            "top_k_gradient": "straight_through_dense_softmax",
        },
        "expert_architecture": {"hidden_dims": list(args.moe_expert_hidden_dims), "activation": "GELU", "dropout": args.dropout},
        "partition": {"method": "complete_races_chronological_three_way", "ranges": split_ranges,
                      "split_checkpoint": (
                          str(args.split_checkpoint.resolve())
                          if args.split_checkpoint is not None else None
                      ),
                      "training_race_ids": train_ids, "validation_race_ids": validation_ids,
                      "test_race_ids": test_ids, "test_used_for_checkpoint_selection": False},
        "best_epoch": best_epoch, "best_selection": best_selection,
        "parameter_count": {
            "trainable": model.trainable_parameter_count(),
            "executed_forward": model.executed_parameter_count(),
            "contributing": model.contributing_parameter_count(),
        },
        "history": history, "metrics": results, "router_diagnostics": diagnostics,
        "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(checkpoint, temporary); temporary.replace(output)
    report_path = output.with_suffix(".report.json")
    report = {key: value for key, value in checkpoint.items() if key not in {"model_state_dict", "preprocessing"}}
    report_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    for name, prediction in predictions.items():
        prediction.to_csv(output.with_suffix(f".{name}.predictions.csv"), index=False)
    print("\nMODEL RESULTS")
    print("split        top1     top2     top3      mrr  logloss  avg_winner_probability")
    for name in ("training", "validation", "test"):
        metric = results[name]
        print(
            f"{name:<10} {metric['top1_hit_rate']:>7.2%} {metric['top2_containment']:>8.2%} "
            f"{metric['top3_containment']:>8.2%} {metric['mrr']:>8.4f} "
            f"{metric['race_logloss']:>8.4f} {metric['average_winner_probability']:>23.4f}"
        )
    for name in ("training", "validation", "test"):
        diagnostic = diagnostics[name]
        print(f"\n{name.upper()} ROUTER DIAGNOSTICS")
        for expert in range(num_experts):
            print(
                f"Expert {expert}: usage={diagnostic['expert_usage_rate'][expert]:.1%} "
                f"mean_gate={diagnostic['mean_gate_weight_per_expert'][expert]:.1%} "
                f"top1_frequency={diagnostic['top1_routed_expert_frequency'][expert]:.1%}; "
                f"{diagnostic['specialisation_descriptions'][expert]}"
            )
        print(
            f"gate_entropy={diagnostic['gate_entropy']:.4f} "
            f"dominant_expert_rate={diagnostic['dominant_expert_rate']:.2%} "
            f"average_active_experts={diagnostic['average_number_of_active_experts']:.2f} "
            f"router_balance_loss={diagnostic['router_balance_loss']:.5f} "
            f"max_abs_expert_correlation={diagnostic['maximum_absolute_pairwise_expert_correlation']}"
        )
        for warning in collapse_warnings(
            diagnostic, args.moe_collapse_threshold, args.moe_correlation_threshold
        ):
            print("WARNING: " + warning)
    print(f"\nsaved_checkpoint={output}\nreport={report_path}", flush=True)


if __name__ == "__main__":
    main()
