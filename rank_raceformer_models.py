#!/usr/bin/env python3
"""Rank one active field with corrected residual and unanchored RaceFormer models."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from predict_raceformer import load_checkpoint, score_frame
from src.config import DEFAULT_DB
from src.database import quote_identifier
from src.advanced_racing_features import race_relative_runner_mask


METADATA_COLUMNS = (
    "race_id", "start_time_iso", "competition_id", "race_number", "race_name",
    "runner_number", "runner_name", "runner_mask", "fluc2",
    "status", "source_betting_status", "active_field_size",
    "derived_racing_features_version",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--market-checkpoint", type=Path,
        default=Path("outputs/corrected_raceformer/raceformer_corrected_market_residual.pt"),
    )
    parser.add_argument(
        "--unanchored-checkpoint", type=Path,
        default=Path("outputs/corrected_raceformer/raceformer_corrected_unanchored.pt"),
    )
    parser.add_argument("--market-model-weight", type=float, default=0.5)
    parser.add_argument("--unanchored-model-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def checkpoint_features(checkpoint: dict[str, Any]) -> list[str]:
    return list(checkpoint.get("raw_feature_columns", checkpoint["feature_columns"]))


def load_active_race(
    database: Path, race_id: int, feature_sets: list[list[str]]
) -> pd.DataFrame:
    """Load only a verified complete active field for stored or live races."""
    requested = list(dict.fromkeys([
        *METADATA_COLUMNS,
        *(feature for features in feature_sets for feature in features),
    ]))
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        existing = {str(row[1]) for row in connection.execute(
            'PRAGMA table_info("race_runners")'
        )}
        missing = sorted(set(requested) - existing)
        if missing:
            raise ValueError("Database is missing checkpoint inputs: " + ", ".join(missing))
        columns = ", ".join(quote_identifier(column) for column in requested)
        frame = pd.read_sql_query(
            f"SELECT {columns} FROM race_runners "
            "WHERE race_id = ? ORDER BY runner_number",
            connection,
            params=(race_id,),
        )
    if not frame.empty:
        frame = frame.loc[race_relative_runner_mask(frame)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"Race {race_id} has no verified complete active field or does not exist"
        )
    return frame


def rank_percentile(rank: pd.Series) -> pd.Series:
    """Return one for best and zero for worst, retaining deterministic ties."""
    count = len(rank)
    if count == 1:
        return pd.Series(1.0, index=rank.index)
    return (count - pd.to_numeric(rank, errors="raise")) / (count - 1)


def combine_rankings(
    market_scored: pd.DataFrame,
    unanchored_scored: pd.DataFrame,
    market_weight: float,
    unanchored_weight: float,
) -> pd.DataFrame:
    """Combine model rank percentiles, avoiding incompatible probability scales."""
    if market_weight < 0 or unanchored_weight < 0 or market_weight + unanchored_weight <= 0:
        raise ValueError("Model weights must be non-negative with a positive sum")
    market = market_scored[[
        "runner_number", "runner_name", "fluc2", "market_rank", "probability",
        "model_rank", "anchor_logit", "residual_logit",
    ]].rename(columns={
        "probability": "market_model_probability",
        "model_rank": "market_model_rank",
    })
    unanchored = unanchored_scored[[
        "runner_number", "probability", "model_rank",
    ]].rename(columns={
        "probability": "unanchored_probability",
        "model_rank": "unanchored_rank",
    })
    result = market.merge(unanchored, on="runner_number", validate="one_to_one")
    result["market_model_rank_pct"] = rank_percentile(result["market_model_rank"])
    result["unanchored_rank_pct"] = rank_percentile(result["unanchored_rank"])
    denominator = market_weight + unanchored_weight
    result["consensus_score"] = (
        market_weight * result["market_model_rank_pct"]
        + unanchored_weight * result["unanchored_rank_pct"]
    ) / denominator
    # Stable runner-number order breaks exact consensus ties deterministically.
    result = result.sort_values("runner_number", kind="stable")
    result["consensus_rank"] = result["consensus_score"].rank(
        method="first", ascending=False
    ).astype(int)
    result["model_rank_disagreement"] = (
        result["market_model_rank"] - result["unanchored_rank"]
    ).abs()
    return result.sort_values(["consensus_rank", "runner_number"], kind="stable")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    market_model, market_checkpoint = load_checkpoint(args.market_checkpoint, device)
    unanchored_model, unanchored_checkpoint = load_checkpoint(
        args.unanchored_checkpoint, device
    )
    if market_model.variant != "market_residual":
        raise ValueError("--market-checkpoint must use variant=market_residual")
    if unanchored_model.variant == "market_residual":
        raise ValueError("--unanchored-checkpoint must not use variant=market_residual")
    market_features = checkpoint_features(market_checkpoint)
    unanchored_features = checkpoint_features(unanchored_checkpoint)
    frame = load_active_race(
        args.db, args.race_id, [market_features, unanchored_features]
    )
    versions = frame["derived_racing_features_version"].dropna().astype(str).unique()
    if len(versions) != 1 or frame["derived_racing_features_version"].isna().any():
        raise ValueError(
            "Race contains missing/mixed derived feature versions; run "
            "update_derived_racing_features.py before ranking"
        )
    market_scored = score_frame(market_model, market_checkpoint, frame, device)
    unanchored_scored = score_frame(
        unanchored_model, unanchored_checkpoint, frame, device
    )
    result = combine_rankings(
        market_scored, unanchored_scored,
        args.market_model_weight, args.unanchored_model_weight,
    )
    race = frame.iloc[0]
    print(
        "CORRECTED RACEFORMER RANKINGS\n"
        f"race={args.race_id} {race['race_name']} competition={int(race['competition_id'])} "
        f"start={race['start_time_iso']} active_runners={len(frame)}\n"
        f"feature_version={versions[0]} market_model={args.market_checkpoint.resolve()}\n"
        f"unanchored_model={args.unanchored_checkpoint.resolve()}\n"
        f"consensus_weights=market:{args.market_model_weight:g},"
        f"unanchored:{args.unanchored_model_weight:g}"
    )
    shown = result[[
        "consensus_rank", "runner_number", "runner_name", "fluc2", "market_rank",
        "market_model_probability", "market_model_rank", "unanchored_probability",
        "unanchored_rank", "model_rank_disagreement", "consensus_score",
    ]]
    print(shown.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if args.output_csv:
        output = args.output_csv.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"saved={output}")


if __name__ == "__main__":
    main()
