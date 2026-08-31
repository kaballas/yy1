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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--race-id", type=int, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help=(
            "Show model scores and per-expert gate/selection/logit columns. "
            "The default view only shows rank, runner number, and runner name."
        ),
    )
    return parser.parse_args()


def prediction_view(result: pd.DataFrame, *, diagnostics: bool) -> pd.DataFrame:
    """Select the concise default columns or the complete diagnostic table."""
    if diagnostics:
        return result
    return result.loc[:, ["rank", "runner_number", "runner_name"]]


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
    )
    return RaceMixtureOfExpertsFeatureMap(config)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "race_winner_moe_feature_map":
        raise ValueError("Checkpoint is not a race_winner_moe_feature_map bundle")

    model_config = checkpoint["model_config"]
    model = build_model_from_checkpoint_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    features = list(checkpoint["raw_feature_columns"])
    columns = list(dict.fromkeys([
        "race_id", "start_time_iso", "race_number", "race_name", "runner_number",
        "runner_name", "runner_mask", "status", "source_betting_status", *features,
    ]))
    with sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT " + ", ".join(map(quote_identifier, columns))
            + " FROM race_runners WHERE race_id = ? ORDER BY runner_number",
            connection,
            params=(args.race_id,),
        )
    frame = frame.loc[race_relative_runner_mask(frame)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Race {args.race_id} has no verified active field")

    raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    values = transform_raceformer(raw, race_ids, features, checkpoint["zeroed_features"], checkpoint["preprocessing"])

    x = torch.from_numpy(values).unsqueeze(0).to(device)
    valid = torch.ones((1, len(frame)), dtype=torch.bool, device=device)
    with torch.inference_mode():
        output = model(x, valid, return_diagnostics=True)
        output_logits = output["logits"]
        logits = output_logits[0].cpu().numpy()
        probability = F.softmax(output_logits[0], dim=0).cpu().numpy()
        weights = output["router_weights"][0].cpu().numpy()
        expert_logits = output["expert_logits"][0].cpu().numpy()

    result = frame[["runner_number", "runner_name"]].copy()
    result["ranking_logit"] = logits
    result["winner_probability"] = probability
    result["rank"] = result["winner_probability"].rank(method="first", ascending=False).astype(int)
    if args.diagnostics:
        selected = output["selected_experts"][0].cpu().numpy()
        for expert in range(weights.shape[1]):
            result[f"expert_{expert}_gate"] = weights[:, expert]
            result[f"expert_{expert}_selected"] = selected[:, expert].astype(int)
            result[f"expert_{expert}_logit"] = expert_logits[:, expert]
    result = result.sort_values("rank", kind="stable")
    first = frame.iloc[0]
    print(
        f"RACE WINNER FEATURE-MAP MOE\n"
        f"race={args.race_id} R{first['race_number']} {first['race_name']} "
        f"start={first['start_time_iso']} active_runners={len(frame)}\n"
        f"num_experts={model_config['num_experts']} top_k={model_config['top_k']}\n"
        f"{expert_usage_line(weights)}\n"
        f"{expert_influence_line(weights, expert_logits)}"
    )
    print(
        prediction_view(result, diagnostics=args.diagnostics).to_string(
            index=False,
            float_format=lambda value: f"{value:.5f}",
        )
    )


if __name__ == "__main__":
    main()
