#!/usr/bin/env python3
"""Train a race-winner MoE where each expert is restricted to a feature subset from JSON."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_DB, DEFAULT_FEATURES
from src.dataset import load_feature_manifest
from src.model.race_moe import race_softmax_nll
from src.model.race_moe_feature_map import (
    FeatureMappedRaceWinnerConfig,
    RaceMixtureOfExpertsFeatureMap,
    expand_feature_map_to_model_features,
    expand_feature_indices_to_model_features,
    load_feature_expert_map,
    load_router_feature_indices,
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-map-json", type=Path, required=True, help="JSON mapping feature names to expert IDs")
    parser.add_argument("--output", type=Path, default=Path("outputs/race_winner_moe_feature_map.pt"))
    parser.add_argument(
        "--include-market-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow market-derived price and implied-probability features.",
    )
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument(
        "--train-competition-id",
        "--competition-id",
        dest="train_competition_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Restrict the population to these competition IDs before chronological splitting.",
    )
    parser.add_argument(
        "--validation-competition-id",
        dest="validation_competition_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Must match --train-competition-id to select checkpoints in-distribution.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--races-per-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=_top_k, default=2)
    parser.add_argument("--moe-gate-temperature", type=float, default=1.0)
    parser.add_argument("--moe-router-balance-weight", type=float, default=0.01)
    parser.add_argument("--moe-expert-hidden-dims", type=_dims, default=(64,))
    parser.add_argument("--moe-router-hidden-dim", type=int, default=64)
    parser.add_argument("--moe-routing-mode", choices=("learned", "fixed_uniform"), default="learned")
    parser.add_argument("--standardized-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "validation-races": args.validation_races,
        "test-races": args.test_races,
        "epochs": args.epochs,
        "races-per-batch": args.races_per_batch,
        "learning-rate": args.learning_rate,
        "max-grad-norm": args.max_grad_norm,
        "early-stopping-patience": args.early_stopping_patience,
        "moe-num-experts": args.moe_num_experts,
        "moe-gate-temperature": args.moe_gate_temperature,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError("These arguments must be positive: " + ", ".join(bad))
    if args.moe_router_balance_weight < 0:
        raise ValueError("router balance weight must be non-negative")
    if args.moe_top_k is not None and args.moe_top_k > args.moe_num_experts:
        raise ValueError("--moe-top-k cannot exceed --moe-num-experts")


def _selection(metrics: dict[str, float | int]) -> tuple[float, float, float]:
    return (
        -float(metrics["race_logloss"]),
        float(metrics["mrr"]),
        float(metrics["top1_hit_rate"]),
    )


def _competition_population(
    frame: pd.DataFrame,
    competition_ids: list[int] | None,
) -> pd.DataFrame:
    """Restrict the population before creating chronological partitions."""
    if competition_ids is None:
        return frame
    competition = pd.to_numeric(frame["competition_id"], errors="coerce")
    return frame.loc[competition.isin(competition_ids)].copy()


def _run_epoch(
    model: RaceMixtureOfExpertsFeatureMap,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    indices: dict[int, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    ranking_values, balance_values, total_values = [], [], []
    for groups in batches(indices, args.races_per_batch, rng):
        bx, by, valid = pad_batch(x, y, groups, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(bx, valid, return_diagnostics=True)
        ranking = race_softmax_nll(output["logits"], by, valid)
        dense = output["dense_router_weights"]
        weights = dense[valid]
        if weights.numel() > 0 and weights.shape[-1] > 1:
            mean_load = weights.mean(dim=0)
            balance = weights.shape[-1] * mean_load.square().sum() - 1.0
        else:
            balance = dense.new_zeros(())
        total = ranking + args.moe_router_balance_weight * balance
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


def _split_range(frame: pd.DataFrame, race_ids_values: list[int]) -> dict[str, Any]:
    selected = frame.loc[frame["race_id"].isin(race_ids_values)]
    return {
        "races": len(race_ids_values),
        "first_race_id": race_ids_values[0],
        "last_race_id": race_ids_values[-1],
        "start_time": str(selected["start_time_iso"].iloc[0]),
        "end_time": str(selected["start_time_iso"].iloc[-1]),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    train_competitions = (
        None if args.train_competition_ids is None
        else sorted(set(args.train_competition_ids))
    )
    validation_competitions = (
        None if args.validation_competition_ids is None
        else sorted(set(args.validation_competition_ids))
    )
    if train_competitions != validation_competitions:
        raise ValueError(
            "--train-competition-id and --validation-competition-id must match; "
            "use a separate run for cross-competition evaluation"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    configured_features, configured_zeroed = load_feature_manifest(args.features_json)
    features, market_excluded = market_blind_features(configured_features, include_market=args.include_market_features)
    zeroed = [name for name in configured_zeroed if name in features]

    frame = _competition_population(
        load_finished_winner_rows(args.db, features),
        train_competitions,
    )
    train_ids, validation_ids, test_ids = chronological_race_ids(frame, args.validation_races, args.test_races)

    partitions = {
        "training": frame.loc[frame["race_id"].isin(train_ids)].copy(),
        "validation": frame.loc[frame["race_id"].isin(validation_ids)].copy(),
        "test": frame.loc[frame["race_id"].isin(test_ids)].copy(),
    }

    unavailable_features = [
        name for name in features
        if not pd.to_numeric(partitions["training"][name], errors="coerce").notna().any()
    ]
    if unavailable_features:
        features = [name for name in features if name not in unavailable_features]
        zeroed = [name for name in zeroed if name in features]
        print(f"unavailable_training_features_excluded={len(unavailable_features)} {json.dumps(unavailable_features)}", flush=True)

    if not features:
        raise ValueError("No numerical features have training coverage")

    raw = {name: numeric_matrix(part, features) for name, part in partitions.items()}
    preprocessing = fit_raceformer_preprocessor(raw["training"], features, clip=args.standardized_clip, layoff_bucket_mode="none")
    values = {
        name: transform_raceformer(matrix, part["race_id"].to_numpy(dtype=np.int64), features, zeroed, preprocessing)
        for (name, part), matrix in zip(partitions.items(), raw.values())
    }
    expanded_features = model_feature_columns(features, preprocessing)
    arrays = {}
    for name, part in partitions.items():
        race_id_array = part["race_id"].to_numpy(dtype=np.int64)
        arrays[name] = (values[name], part["is_winner"].to_numpy(dtype=np.float32), race_id_array, race_indices(race_id_array))

    feature_map = load_feature_expert_map(args.feature_map_json, features, args.moe_num_experts)
    feature_map = expand_feature_map_to_model_features(feature_map, features, expanded_features)
    router_features = load_router_feature_indices(args.feature_map_json, features)
    router_feature_indices = expand_feature_indices_to_model_features(
        router_features, features, expanded_features,
    )
    config = FeatureMappedRaceWinnerConfig(
        feature_count=len(expanded_features),
        num_experts=args.moe_num_experts,
        top_k=(None if args.moe_routing_mode == "fixed_uniform" else args.moe_top_k),
        gate_temperature=args.moe_gate_temperature,
        expert_hidden_dims=args.moe_expert_hidden_dims,
        router_hidden_dim=args.moe_router_hidden_dim,
        dropout=args.dropout,
        routing_mode=args.moe_routing_mode,
        feature_map=feature_map,
        router_feature_indices=router_feature_indices,
    )
    model = RaceMixtureOfExpertsFeatureMap(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(
        "RACE WINNER FEATURE-MAP MOE EXPERIMENT\n"
        f"model_type=moe_feature_map objective=race_softmax_nll "
        f"market_blind={not args.include_market_features} raw_features={len(features)} model_features={len(expanded_features)}\n"
        f"train_races={len(train_ids):,} validation_races={len(validation_ids):,} sealed_test_races={len(test_ids):,} device={device}\n"
        f"competition_ids={train_competitions or 'all'} "
        "split_population=shared\n"
        f"router_features={len(router_feature_indices)} "
        f"num_experts={args.moe_num_experts} top_k={args.moe_top_k if args.moe_top_k is not None else 'all'} routing_mode={config.routing_mode} temperature={args.moe_gate_temperature:g} balance_weight={args.moe_router_balance_weight:g} expert_hidden_dims={list(args.moe_expert_hidden_dims)}",
        flush=True,
    )
    print(f"feature_map_path={args.feature_map_json}", flush=True)
    #print(f"feature_expert_distribution={json.dumps({str(i): list(expanded_features[j] for j in indices) for i, indices in enumerate(feature_map)})}", flush=True)

    best_state = None
    best_selection = None
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.default_rng(args.seed)
    train_x, train_y, _, train_index = arrays["training"]

    for epoch in range(1, args.epochs + 1):
        losses = _run_epoch(model, optimizer, train_x, train_y, train_index, args, device, rng)
        vx, vy, vids, vindices = arrays["validation"]
        validation_metrics, validation_diagnostics, _ = evaluate_model(model, vx, vy, vids, vindices, partitions["validation"], args.races_per_batch, device)
        selection = _selection(validation_metrics)
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            **losses,
            "validation_metrics": validation_metrics,
            "validation_router": {
                key: validation_diagnostics[key] for key in ("dominant_expert_rate", "gate_entropy", "average_number_of_active_experts", "router_balance_loss")
            },
            "best": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} main_ranking_loss={losses['main_ranking_loss']:.5f} router_balance_loss={losses['router_balance_loss']:.5f} total_loss={losses['total_loss']:.5f} validation_top1={validation_metrics['top1_hit_rate']:.4f} top2={validation_metrics['top2_containment']:.4f} top3={validation_metrics['top3_containment']:.4f} mrr={validation_metrics['mrr']:.4f} logloss={validation_metrics['race_logloss']:.5f} dominant={validation_diagnostics['dominant_expert_rate']:.3f} entropy={validation_diagnostics['gate_entropy']:.3f} best={'yes' if improved else 'no'} stale={stale}/{args.early_stopping_patience}",
            flush=True,
        )
        if stale >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    model.load_state_dict(best_state)
    results = {}
    diagnostics = {}
    predictions = {}
    for name in ("training", "validation", "test"):
        x, y, ids, indices = arrays[name]
        results[name], diagnostics[name], predictions[name] = evaluate_model(model, x, y, ids, indices, partitions[name], args.races_per_batch, device)

    output = args.output.resolve()
    checkpoint = {
        "checkpoint_type": "race_winner_moe_feature_map",
        "checkpoint_version": 1,
        "model_state_dict": best_state,
        "model_config": {
            "feature_count": len(expanded_features),
            "num_experts": args.moe_num_experts,
            "top_k": args.moe_top_k,
            "expert_hidden_dims": list(args.moe_expert_hidden_dims),
            "router_hidden_dim": args.moe_router_hidden_dim,
            "feature_expert_map": {str(i): list(indices) for i, indices in enumerate(feature_map)},
            "router_feature_indices": list(router_feature_indices),
            "routing_mode": args.moe_routing_mode,
            "gate_temperature": args.moe_gate_temperature,
        },
        "feature_map_path": str(args.feature_map_json) if args.feature_map_json else None,
        "raw_feature_columns": features,
        "model_feature_columns": expanded_features,
        "preprocessing": preprocessing,
        "zeroed_features": zeroed,
        "market_features_excluded": market_excluded,
        "partition": {
            "training_race_ids": train_ids,
            "validation_race_ids": validation_ids,
            "test_race_ids": test_ids,
        },
        "best_epoch": best_epoch,
        "history": history,
        "metrics": results,
        "router_diagnostics": diagnostics,
        "training_config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp")
    torch.save(checkpoint, temp)
    temp.replace(output)

    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps({key: value for key, value in checkpoint.items() if key not in {"model_state_dict"}}, indent=2, default=str) + "\n")

    print("\nMODEL RESULTS")
    print("split        top1     top2     top3      mrr  logloss  avg_winner_probability")
    for name in ("training", "validation", "test"):
        metric = results[name]
        print(
            f"{name:<10} {metric['top1_hit_rate']:>7.2%} {metric['top2_containment']:>8.2%} "
            f"{metric['top3_containment']:>8.2%} {metric['mrr']:>8.4f} {metric['race_logloss']:>8.4f} {metric['average_winner_probability']:>23.4f}"
        )
    for name in ("training", "validation", "test"):
        diagnostic = diagnostics[name]
        print(f"\n{name.upper()} ROUTER DIAGNOSTICS")
        for expert in range(args.moe_num_experts):
            print(
                f"Expert {expert}: usage={diagnostic['expert_usage_rate'][expert]:.1%} mean_gate={diagnostic['mean_gate_weight_per_expert'][expert]:.1%} top1_frequency={diagnostic['top1_routed_expert_frequency'][expert]:.1%}; {diagnostic['specialisation_descriptions'][expert]}"
            )
        print(
            f"gate_entropy={diagnostic['gate_entropy']:.4f} dominant_expert_rate={diagnostic['dominant_expert_rate']:.2%} average_active_experts={diagnostic['average_number_of_active_experts']:.2f} router_balance_loss={diagnostic['router_balance_loss']:.5f} max_abs_expert_correlation={diagnostic['maximum_absolute_pairwise_expert_correlation']}"
        )
        for warning in collapse_warnings(diagnostic, 0.80, 0.95):
            print("WARNING: " + warning)
    predictor = Path(__file__).resolve().with_name("predict_moe_winner_ranker_feature_map.py")
    prediction_command = shlex.join([
        sys.executable,
        str(predictor),
        "--checkpoint", str(output),
        "--db", str(args.db.resolve()),
        "--race-id", "12345",
    ])
    print(
        f"\nsaved_checkpoint={output}\n"
        f"report={report_path}\n\n"
        "HOW TO PREDICT (replace 12345 with the race ID):\n"
        f"{prediction_command}",
        flush=True,
    )


if __name__ == "__main__":
    main()
