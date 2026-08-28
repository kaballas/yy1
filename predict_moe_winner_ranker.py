#!/usr/bin/env python3
"""Rank one active race with a saved baseline or MoE winner checkpoint."""

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
from src.model.race_moe import build_race_winner_model
from src.raceformer_preprocessing import transform_raceformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--race-id", type=int, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "race_winner_moe":
        raise ValueError("Checkpoint is not a race_winner_moe bundle")
    model = build_race_winner_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    features = list(checkpoint["raw_feature_columns"])
    columns = list(dict.fromkeys([
        "race_id", "start_time_iso", "race_number", "race_name", "runner_number",
        "runner_name", "runner_mask", "status", "source_betting_status", *features,
    ]))
    with sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True) as connection:
        frame = pd.read_sql_query(
            "SELECT " + ", ".join(map(quote_identifier, columns))
            + " FROM race_runners WHERE race_id = ? ORDER BY runner_number",
            connection, params=(args.race_id,),
        )
    frame = frame.loc[race_relative_runner_mask(frame)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Race {args.race_id} has no verified active field")
    raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    values = transform_raceformer(
        raw, race_ids, features, checkpoint["zeroed_features"], checkpoint["preprocessing"]
    )
    x = torch.from_numpy(values).unsqueeze(0).to(device)
    valid = torch.ones((1, len(frame)), dtype=torch.bool, device=device)
    with torch.inference_mode():
        output = model(x, valid, return_diagnostics=True)
        logits = output["logits"][0].cpu().numpy()
        probability = F.softmax(output["logits"][0], dim=0).cpu().numpy()
        weights = output["router_weights"][0].cpu().numpy()
        selected = output["selected_experts"][0].cpu().numpy()
        expert_logits = output["expert_logits"][0].cpu().numpy()
    result = frame[["runner_number", "runner_name"]].copy()
    result["ranking_logit"] = logits; result["winner_probability"] = probability
    result["rank"] = result["winner_probability"].rank(method="first", ascending=False).astype(int)
    for expert in range(weights.shape[1]):
        result[f"expert_{expert}_gate"] = weights[:, expert]
        result[f"expert_{expert}_selected"] = selected[:, expert].astype(int)
        result[f"expert_{expert}_logit"] = expert_logits[:, expert]
    result = result.sort_values("rank", kind="stable")
    first = frame.iloc[0]
    print(
        f"RACE WINNER {checkpoint['model_config']['model_type'].upper()}\n"
        f"race={args.race_id} R{first['race_number']} {first['race_name']} "
        f"start={first['start_time_iso']} active_runners={len(frame)}\n"
        f"objective={checkpoint['training_objective']} "
        f"market_blind={not checkpoint['market_features_enabled']} "
        f"num_experts={weights.shape[1]} top_k={checkpoint['model_config']['top_k']}"
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
