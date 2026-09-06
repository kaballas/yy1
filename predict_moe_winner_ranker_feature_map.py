#!/usr/bin/env python3
"""Rank one active race with a saved feature-mapped MoE winner checkpoint."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.advanced_racing_features import race_relative_runner_mask
from src.config import DEFAULT_DB
from src.database import quote_identifier
from src.model.race_moe_feature_map import FeatureMappedRaceWinnerConfig, RaceMixtureOfExpertsFeatureMap
from src.raceformer_preprocessing import transform_raceformer

DISPLAY_MARKET_FEATURES = ["open_price", "fluc1", "fluc2"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Predict with one checkpoint.")
    source.add_argument(
        "--models-dir", type=Path,
        help="Recursively predict with every compatible .pt checkpoint in this directory.",
    )
    parser.add_argument("--race-id", type=int, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--top", type=int,
        help="Number of runners shown per model (default: 4 with --models-dir; all with --checkpoint).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Show model scores and per-expert gate/selection/logit columns. "
            "The default view shows rank, runner identity, and market prices."
        ),
    )
    return parser.parse_args(argv)


def prediction_view(result: pd.DataFrame, *, diagnostics: bool) -> pd.DataFrame:
    """Select the concise default columns or the complete diagnostic table."""
    if diagnostics:
        return result
    return result.loc[:, [
        "rank", "runner_number", "runner_name", *DISPLAY_MARKET_FEATURES,
    ]]


def model_top_summary(
    predictions: list[tuple[str, list[int]]], top: int | None,
) -> pd.DataFrame:
    """Build one compact row per model with runner numbers in predicted order."""
    column = f"top{top}" if top is not None else "ranked_runners"
    return pd.DataFrame([
        {
            "model": model,
            column: ", ".join(map(
                str, runner_numbers if top is None else runner_numbers[:top],
            )),
        }
        for model, runner_numbers in predictions
    ])


def unique_top_summary(
    predictions: list[tuple[str, list[int]]], top: int | None,
) -> pd.DataFrame:
    """Group models that produced the same ordered runner ranking."""
    column = f"top{top}" if top is not None else "ranked_runners"
    grouped: dict[tuple[int, ...], list[str]] = {}
    for model, runner_numbers in predictions:
        selected = runner_numbers if top is None else runner_numbers[:top]
        grouped.setdefault(tuple(selected), []).append(model)
    return pd.DataFrame([
        {
            column: ", ".join(map(str, runner_numbers)),
            "model_count": len(models),
            "models": ", ".join(models),
        }
        for runner_numbers, models in grouped.items()
    ])


def expert_usage_line(router_weights: np.ndarray) -> str:
    """Format each expert's mean mixture weight across the active field."""
    usage = router_weights.mean(axis=0) * 100.0
    values = " ".join(
        f"expert_{expert}={percentage:.2f}%"
        for expert, percentage in enumerate(usage)
    )
    return f"expert_usage_mean_gate: {values}"


def expert_influence_line(
    router_weights: np.ndarray, expert_logits: np.ndarray,
) -> str:
    """Format each expert's share of field-relative weighted score movement."""
    weighted_logits = router_weights * expert_logits
    centered = weighted_logits - weighted_logits.mean(axis=0, keepdims=True)
    magnitudes = np.mean(np.abs(centered), axis=0)
    total = float(magnitudes.sum())
    influence = (
        magnitudes / total * 100.0
        if total > np.finfo(magnitudes.dtype).eps
        else np.zeros_like(magnitudes)
    )
    values = " ".join(
        f"expert_{expert}={percentage:.2f}%"
        for expert, percentage in enumerate(influence)
    )
    return f"expert_influence_score_variation: {values}"


def build_model_from_checkpoint_config(model_config: dict) -> RaceMixtureOfExpertsFeatureMap:
    """Recreate the exact model variant described by a saved checkpoint."""
    map_dict = model_config["feature_expert_map"]
    feature_map = tuple(
        tuple(int(idx) for idx in sorted(map_dict[str(expert_id)]))
        for expert_id in range(len(map_dict))
    )
    config = FeatureMappedRaceWinnerConfig(
        feature_count=model_config["feature_count"],
        num_experts=model_config["num_experts"],
        top_k=model_config["top_k"],
        gate_temperature=model_config["gate_temperature"],
        expert_hidden_dims=tuple(model_config["expert_hidden_dims"]),
        router_hidden_dim=model_config["router_hidden_dim"],
        routing_mode=model_config.get("routing_mode", "learned"),
        feature_map=feature_map,
        router_feature_indices=tuple(
            model_config.get(
                "router_feature_indices",
                range(model_config["feature_count"]),
            )
        ),
        judge_hidden_dims=tuple(model_config.get("judge_hidden_dims", ())),
    )
    return RaceMixtureOfExpertsFeatureMap(config)


def predict_checkpoint(
    checkpoint_path: Path,
    *,
    database: Path,
    race_id: int,
    device: torch.device,
    diagnostics: bool,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load one checkpoint and return its prediction details for one race."""
    checkpoint = torch.load(
        checkpoint_path.resolve(), map_location=device, weights_only=False,
    )
    if checkpoint.get("checkpoint_type") != "race_winner_moe_feature_map":
        raise ValueError("Checkpoint is not a race_winner_moe_feature_map bundle")

    model_config = checkpoint["model_config"]
    model = build_model_from_checkpoint_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    features = list(checkpoint["raw_feature_columns"])
    columns = list(dict.fromkeys([
        "race_id", "start_time_iso", "race_number", "race_name", "runner_number",
        "runner_name", "runner_mask", "status", "source_betting_status",
        *DISPLAY_MARKET_FEATURES, *features,
    ]))
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT " + ", ".join(map(quote_identifier, columns))
            + " FROM race_runners WHERE race_id = ? ORDER BY runner_number",
            connection,
            params=(race_id,),
        )
    frame = frame.loc[race_relative_runner_mask(frame)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Race {race_id} has no verified active field")

    raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    values = transform_raceformer(raw, race_ids, features, checkpoint["zeroed_features"], checkpoint["preprocessing"])

    x = torch.from_numpy(values).unsqueeze(0).to(device)
    valid = torch.ones((1, len(frame)), dtype=torch.bool, device=device)
    with torch.inference_mode():
        output = model(x, valid, return_diagnostics=True)
        output_logits = output["logits"]
        logits = output_logits[0].cpu().numpy()
        base_logits = output["base_logits"][0].cpu().numpy()
        judge_adjustment = output["judge_adjustment"][0].cpu().numpy()
        if checkpoint.get("objective") == "top3_mask_ranking":
            probability = torch.sigmoid(output_logits[0]).cpu().numpy()
        else:
            probability = F.softmax(output_logits[0], dim=0).cpu().numpy()
        weights = output["router_weights"][0].cpu().numpy()
        expert_logits = output["expert_logits"][0].cpu().numpy()

    result = frame[[
        "runner_number", "runner_name", *DISPLAY_MARKET_FEATURES,
    ]].copy()
    result["ranking_logit"] = logits
    probability_column = (
        "top3_probability"
        if checkpoint.get("objective") == "top3_mask_ranking"
        else "winner_probability"
    )
    result[probability_column] = probability
    result["rank"] = result[probability_column].rank(method="first", ascending=False).astype(int)
    if diagnostics:
        result["base_ranking_logit"] = base_logits
        result["judge_adjustment"] = judge_adjustment
        selected = output["selected_experts"][0].cpu().numpy()
        for expert in range(weights.shape[1]):
            result[f"expert_{expert}_gate"] = weights[:, expert]
            result[f"expert_{expert}_selected"] = selected[:, expert].astype(int)
            result[f"expert_{expert}_logit"] = expert_logits[:, expert]
    result = result.sort_values("rank", kind="stable")
    return model_config, frame, result, weights, expert_logits


def print_prediction(
    checkpoint_path: Path,
    *,
    model_label: str,
    database: Path,
    race_id: int,
    device: torch.device,
    diagnostics: bool,
    top: int | None,
) -> list[int]:
    """Predict and print one model's ranking."""
    model_config, frame, result, weights, expert_logits = predict_checkpoint(
        checkpoint_path,
        database=database,
        race_id=race_id,
        device=device,
        diagnostics=diagnostics,
    )
    displayed = result if top is None else result.head(top)
    first = frame.iloc[0]
    print(
        f"MODEL {model_label}\n"
        f"RACE WINNER FEATURE-MAP MOE\n"
        f"race={race_id} R{first['race_number']} {first['race_name']} "
        f"start={first['start_time_iso']} active_runners={len(frame)}\n"
        f"num_experts={model_config['num_experts']} top_k={model_config['top_k']}\n"
        f"{expert_usage_line(weights)}\n"
        f"{expert_influence_line(weights, expert_logits)}"
    )
    print(
        prediction_view(displayed, diagnostics=diagnostics).to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )
    return list(map(int, result["runner_number"]))


def main() -> None:
    args = parse_args()
    if args.top is not None and args.top < 1:
        raise ValueError("--top must be positive")
    device = torch.device(args.device)

    if args.checkpoint is not None:
        paths = [args.checkpoint]
        models_dir = None
        top = args.top
    else:
        models_dir = args.models_dir.resolve()
        if not models_dir.is_dir():
            raise ValueError(f"--models-dir is not a directory: {models_dir}")
        paths = sorted(models_dir.rglob("*.pt"))
        if not paths:
            raise ValueError(f"No .pt checkpoints found under {models_dir}")
        top = 4 if args.top is None else args.top

    printed = 0
    skipped: list[tuple[Path, str]] = []
    summaries: list[tuple[str, list[int]]] = []
    for path in paths:
        label = (
            str(path.resolve().relative_to(models_dir).with_suffix(""))
            if models_dir is not None
            else path.stem
        )
        try:
            if printed:
                print()
            runner_numbers = print_prediction(
                path,
                model_label=label,
                database=args.db,
                race_id=args.race_id,
                device=device,
                diagnostics=args.diagnostics,
                top=top,
            )
            summaries.append((label, runner_numbers))
            printed += 1
        except (KeyError, RuntimeError, ValueError) as error:
            if models_dir is None:
                raise
            skipped.append((path, str(error)))

    if printed == 0:
        reasons = "; ".join(f"{path.name}: {reason}" for path, reason in skipped)
        raise ValueError(f"No compatible checkpoints produced a prediction. {reasons}")
    if skipped:
        print("\nSKIPPED CHECKPOINTS")
        for path, reason in skipped:
            print(f"{path}: {reason}")
    summary = model_top_summary(summaries, top)
    heading = f"MODEL TOP {top} SUMMARY" if top is not None else "MODEL RANKING SUMMARY"
    print(f"\n{heading}")
    print(summary.to_string(index=False))
    if top != 3:
        top3_summary = model_top_summary(summaries, 3)
        print("\nMODEL TOP 3 SUMMARY")
        print(top3_summary.to_string(index=False))
    unique_summary = unique_top_summary(summaries, top)
    unique_heading = (
        f"UNIQUE TOP {top} SUMMARY" if top is not None
        else "UNIQUE RANKING SUMMARY"
    )
    print(f"\n{unique_heading}")
    print(unique_summary.to_string(index=False))
    if top != 3:
        unique_top3_summary = unique_top_summary(summaries, 3)
        print("\nUNIQUE TOP 3 SUMMARY")
        print(unique_top3_summary.to_string(index=False))
    if top != 1:
        unique_top1_summary = unique_top_summary(summaries, 1)
        print("\nUNIQUE TOP 1 SUMMARY")
        print(unique_top1_summary.to_string(index=False))


if __name__ == "__main__":
    main()
