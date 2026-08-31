#!/usr/bin/env python3
"""Backtest saved winner-ranker checkpoints and rank models on shared cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from predict_moe_winner_ranker_feature_map import build_model_from_checkpoint_config
from src.config import DEFAULT_DB
from src.model.race_moe import build_race_winner_model
from src.race_moe_data import (
    load_finished_winner_rows,
    numeric_matrix,
    race_indices,
)
from src.race_moe_evaluation import evaluate_model
from src.raceformer_preprocessing import transform_raceformer

SUPPORTED_TYPES = {"race_winner_moe", "race_winner_moe_feature_map"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir", type=Path, default=Path("outputs"),
        help="Directory recursively searched for .pt checkpoints.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="Recorded checkpoint partition used for the backtest.",
    )
    parser.add_argument("--races-per-batch", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def cohort_key(race_ids: list[int]) -> str:
    """Return an order-sensitive ID for one exact chronological race cohort."""
    digest = hashlib.sha256(",".join(map(str, race_ids)).encode("ascii")).hexdigest()
    return digest[:12]


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type == "race_winner_moe":
        model = build_race_winner_model(checkpoint["model_config"])
    elif checkpoint_type == "race_winner_moe_feature_map":
        model = build_model_from_checkpoint_config(checkpoint["model_config"])
    else:
        raise ValueError(f"unsupported checkpoint type {checkpoint_type!r}")
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def evaluate_checkpoint(
    path: Path, checkpoint: dict[str, Any], model: torch.nn.Module,
    split: str, database: Path, races_per_batch: int, device: torch.device,
) -> dict[str, Any]:
    partition_ids = checkpoint.get("partition", {}).get(f"{split}_race_ids")
    if not isinstance(partition_ids, list) or not partition_ids:
        raise ValueError(f"checkpoint has no recorded {split} race IDs")
    race_ids = [int(value) for value in partition_ids]
    features = list(checkpoint["raw_feature_columns"])
    frame = load_finished_winner_rows(database, features)
    available_ids = set(frame["race_id"].astype(int))
    missing_ids = sorted(set(race_ids) - available_ids)
    if missing_ids:
        raise ValueError(
            f"database is missing {len(missing_ids)} recorded {split} races"
        )
    part = frame.loc[frame["race_id"].isin(race_ids)].copy()
    part["_cohort_order"] = pd.Categorical(
        part["race_id"], categories=race_ids, ordered=True,
    )
    part = part.sort_values(["_cohort_order", "runner_number"], kind="stable")
    raw = numeric_matrix(part, features)
    race_id_array = part["race_id"].to_numpy(dtype="int64")
    values = transform_raceformer(
        raw, race_id_array, features, checkpoint["zeroed_features"],
        checkpoint["preprocessing"],
    )
    metrics, diagnostics, _ = evaluate_model(
        model,
        values,
        part["is_winner"].to_numpy(dtype="float32"),
        race_id_array,
        race_indices(race_id_array),
        part,
        races_per_batch,
        device,
    )
    return {
        "path": str(path),
        "model": path.stem,
        "checkpoint_type": checkpoint["checkpoint_type"],
        "cohort_id": cohort_key(race_ids),
        "races": len(race_ids),
        "metrics": metrics,
        "router_diagnostics": diagnostics,
    }


def leaderboard(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Rank a comparable cohort by log loss, MRR, then top-1 accuracy."""
    rows = []
    for record in records:
        metrics = record["metrics"]
        rows.append({
            "cohort_id": record["cohort_id"],
            "races": record["races"],
            "model": record["model"],
            "checkpoint_type": record["checkpoint_type"],
            "path": record["path"],
            "top1": metrics["top1_hit_rate"],
            "top2": metrics["top2_containment"],
            "top3": metrics["top3_containment"],
            "mrr": metrics["mrr"],
            "logloss": metrics["race_logloss"],
            "avg_winner_probability": metrics["average_winner_probability"],
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["cohort_id", "logloss", "mrr", "top1", "model"],
        ascending=[True, True, False, False, True],
        kind="stable",
        ignore_index=True,
    )
    result["cohort_rank"] = result.groupby("cohort_id").cumcount() + 1
    return result


def main() -> None:
    args = parse_args()
    if args.races_per_batch < 1:
        raise ValueError("--races-per-batch must be positive")
    models_dir = args.models_dir.resolve()
    if not models_dir.is_dir():
        raise ValueError(f"--models-dir is not a directory: {models_dir}")

    device = torch.device(args.device)
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    paths = sorted(models_dir.rglob("*.pt"))
    if not paths:
        raise ValueError(f"No .pt checkpoints found under {models_dir}")
    for path in paths:
        try:
            raw_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_type = raw_checkpoint.get("checkpoint_type")
            if checkpoint_type not in SUPPORTED_TYPES:
                skipped.append({
                    "path": str(path),
                    "reason": f"unsupported checkpoint type {checkpoint_type!r}",
                })
                continue
            model, checkpoint = load_checkpoint(path, device)
            record = evaluate_checkpoint(
                path, checkpoint, model, args.split, args.db,
                args.races_per_batch, device,
            )
            record["model"] = str(path.relative_to(models_dir).with_suffix(""))
            records.append(record)
        except (KeyError, RuntimeError, ValueError) as error:
            skipped.append({"path": str(path), "reason": str(error)})

    result = leaderboard(records)
    print(
        f"WINNER-RANKER BACKTEST split={args.split} "
        f"evaluated={len(records)} skipped={len(skipped)}"
    )
    if not result.empty:
        print(
            "cohort       races rank model                              type"
            "                         top1     top2     top3      mrr  logloss"
        )
        for row in result.itertuples(index=False):
            print(
                f"{row.cohort_id:<12} {row.races:>5} {row.cohort_rank:>4} "
                f"{row.model:<34} {row.checkpoint_type:<28} "
                f"{row.top1:>7.2%} {row.top2:>8.2%} {row.top3:>8.2%} "
                f"{row.mrr:>8.4f} {row.logloss:>8.4f}"
            )
        for cohort, group in result.groupby("cohort_id", sort=False):
            winner = group.iloc[0]
            qualifier = (
                "strongest"
                if len(group) > 1
                else "only compatible checkpoint"
            )
            print(
                f"COHORT WINNER cohort={cohort} races={winner.races} "
                f"model={winner.model} status={qualifier} "
                f"logloss={winner.logloss:.4f} mrr={winner.mrr:.4f} "
                f"top1={winner.top1:.2%}"
            )
    if skipped:
        print("\nSKIPPED CHECKPOINTS")
        for item in skipped:
            print(f"{item['path']}: {item['reason']}")

    payload = {
        "models_dir": str(models_dir),
        "database": str(args.db.resolve()),
        "split": args.split,
        "leaderboard": result.to_dict(orient="records"),
        "skipped": skipped,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
