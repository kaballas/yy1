#!/usr/bin/env python3
"""Backtest saved winner-ranker checkpoints and rank models on shared cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
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


class PreparedDataset(TypedDict):
    values: np.ndarray
    winner_array: np.ndarray
    race_id_array: np.ndarray
    race_indices: dict[int, np.ndarray]
    part: pd.DataFrame


class LoadedFrame(TypedDict):
    frame: pd.DataFrame
    available_race_ids: frozenset[int]
    date_range_race_ids: dict[tuple[str | None, str | None], list[int]]


FrameCacheKey = tuple[str, tuple[str, ...]]
PreparedCacheKey = tuple[
    str, tuple[str, ...], tuple[str, ...], str, tuple[int, ...],
]


def preprocessing_fingerprint(value: Any) -> str:
    """Return a stable digest for nested preprocessing metadata and arrays."""
    def freeze(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            contiguous = np.ascontiguousarray(item)
            return (
                "ndarray",
                contiguous.dtype.str,
                contiguous.shape,
                hashlib.sha256(contiguous.tobytes()).hexdigest(),
            )
        if isinstance(item, np.generic):
            return freeze(item.item())
        if isinstance(item, dict):
            return (
                "dict",
                tuple((str(key), freeze(item[key])) for key in sorted(item)),
            )
        if isinstance(item, (list, tuple)):
            return (type(item).__name__, tuple(freeze(member) for member in item))
        if item is None or isinstance(item, (bool, int, float, str)):
            return (type(item).__name__, item)
        raise TypeError(
            "Unsupported preprocessing cache-key value "
            f"of type {type(item).__name__}"
        )

    return hashlib.sha256(repr(freeze(value)).encode("utf-8")).hexdigest()


def iso_date(value: str) -> str:
    """Validate and normalize a command-line ISO calendar date."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir", type=Path, default=Path("outputs"),
        help="Directory recursively searched for .pt checkpoints.",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
        help="Recorded checkpoint partition used when no date filter is supplied.",
    )
    parser.add_argument(
        "--date", type=iso_date,
        help=(
            "Backtest all complete DB races whose start_time_iso calendar "
            "date matches YYYY-MM-DD; cannot be combined with --from-date "
            "or --to-date and ignores --split."
        ),
    )
    parser.add_argument(
        "--from-date", type=iso_date,
        help=(
            "Backtest complete DB races on or after this YYYY-MM-DD calendar "
            "date; may be combined with --to-date and ignores --split."
        ),
    )
    parser.add_argument(
        "--to-date", type=iso_date,
        help=(
            "Backtest complete DB races on or before this YYYY-MM-DD calendar "
            "date; may be combined with --from-date and ignores --split."
        ),
    )
    parser.add_argument(
        "--race-number", type=int,
        help=(
            "Limit the selected date range or checkpoint split to races whose "
            "race_number DB column equals this positive integer."
        ),
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


def race_ids_on_date(frame: pd.DataFrame, backtest_date: str) -> list[int]:
    """Return every complete DB race on one start_time_iso date."""
    return race_ids_in_date_range(frame, backtest_date, backtest_date)


def race_ids_in_date_range(
    frame: pd.DataFrame,
    from_date: str | None,
    to_date: str | None,
) -> list[int]:
    """Return complete DB races in an inclusive calendar-date range."""
    calendar_dates = frame["start_time_iso"].astype("string").str[:10]
    keep = pd.Series(True, index=frame.index)
    if from_date is not None:
        keep &= calendar_dates >= from_date
    if to_date is not None:
        keep &= calendar_dates <= to_date
    return list(map(
        int,
        frame.loc[keep, "race_id"].drop_duplicates(),
    ))


def validate_date_filters(
    backtest_date: str | None,
    from_date: str | None,
    to_date: str | None,
) -> None:
    """Reject ambiguous or reversed command-line date filters."""
    if backtest_date is not None and (from_date is not None or to_date is not None):
        raise ValueError("--date cannot be combined with --from-date or --to-date")
    if from_date is not None and to_date is not None and from_date > to_date:
        raise ValueError("--from-date cannot be after --to-date")


def filter_race_ids_by_number(
    frame: pd.DataFrame,
    race_ids: list[int],
    race_number: int | None,
) -> list[int]:
    """Keep selected race IDs having the requested database race number."""
    if race_number is None:
        return race_ids
    if race_number < 1:
        raise ValueError("--race-number must be positive")
    if "race_number" not in frame:
        raise ValueError("database race_runners table has no race_number column")
    numbers = pd.to_numeric(frame["race_number"], errors="coerce")
    matching_ids = set(frame.loc[numbers == race_number, "race_id"].astype(int))
    return [race_id for race_id in race_ids if race_id in matching_ids]


def build_model_from_checkpoint(
    checkpoint: dict[str, Any], device: torch.device,
) -> torch.nn.Module:
    """Build an inference model from an already-deserialized checkpoint."""
    checkpoint_type = checkpoint.get("checkpoint_type")
    if checkpoint_type == "race_winner_moe":
        model = build_race_winner_model(checkpoint["model_config"])
    elif checkpoint_type == "race_winner_moe_feature_map":
        model = build_model_from_checkpoint_config(checkpoint["model_config"])
    else:
        raise ValueError(f"unsupported checkpoint type {checkpoint_type!r}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def prepare_checkpoint_data(
    database: Path,
    checkpoint: dict[str, Any],
    frame: pd.DataFrame,
    race_ids: list[int],
    prepared_cache: dict[PreparedCacheKey, PreparedDataset],
) -> PreparedDataset:
    """Prepare and cache the immutable arrays shared by compatible models."""
    features = tuple(checkpoint["raw_feature_columns"])
    key: PreparedCacheKey = (
        str(database.resolve()),
        features,
        tuple(checkpoint["zeroed_features"]),
        preprocessing_fingerprint(checkpoint["preprocessing"]),
        tuple(race_ids),
    )
    cached = prepared_cache.get(key)
    if cached is not None:
        return cached

    part = frame.loc[frame["race_id"].isin(race_ids)].copy()
    part["_cohort_order"] = pd.Categorical(
        part["race_id"], categories=race_ids, ordered=True,
    )
    part = part.sort_values(["_cohort_order", "runner_number"], kind="stable")
    race_id_array = part["race_id"].to_numpy(dtype="int64", copy=False)
    values = transform_raceformer(
        numeric_matrix(part, features),
        race_id_array,
        features,
        checkpoint["zeroed_features"],
        checkpoint["preprocessing"],
    )
    prepared: PreparedDataset = {
        "values": values,
        "winner_array": part["is_winner"].to_numpy(dtype="float32", copy=False),
        "race_id_array": race_id_array,
        "race_indices": race_indices(race_id_array),
        "part": part,
    }
    prepared_cache[key] = prepared
    return prepared


def evaluate_checkpoint(
    path: Path, checkpoint: dict[str, Any], model: torch.nn.Module,
    split: str, database: Path, races_per_batch: int, device: torch.device,
    backtest_date: str | None = None,
    frame_cache: dict[FrameCacheKey, LoadedFrame] | None = None,
    prepared_cache: dict[PreparedCacheKey, PreparedDataset] | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    race_number: int | None = None,
) -> dict[str, Any]:
    validate_date_filters(backtest_date, from_date, to_date)
    features = list(checkpoint["raw_feature_columns"])
    if frame_cache is None:
        frame_cache = {}
    if prepared_cache is None:
        prepared_cache = {}
    frame_key = (str(database.resolve()), tuple(sorted(set(features))))
    if frame_key not in frame_cache:
        frame = load_finished_winner_rows(database, frame_key[1])
        frame_cache[frame_key] = {
            "frame": frame,
            "available_race_ids": frozenset(
                frame["race_id"].to_numpy(dtype="int64", copy=False)
            ),
            "date_range_race_ids": {},
        }
    loaded = frame_cache[frame_key]
    frame = loaded["frame"]
    if backtest_date is not None or from_date is not None or to_date is not None:
        range_start = backtest_date if backtest_date is not None else from_date
        range_end = backtest_date if backtest_date is not None else to_date
        range_key = (range_start, range_end)
        if range_key not in loaded["date_range_race_ids"]:
            loaded["date_range_race_ids"][range_key] = race_ids_in_date_range(
                frame, range_start, range_end,
            )
        race_ids = loaded["date_range_race_ids"][range_key]
        if not race_ids:
            requested_range = (
                backtest_date if backtest_date is not None
                else f"{from_date or 'earliest'} through {to_date or 'latest'}"
            )
            raise ValueError(
                f"database has no complete finished races in {requested_range}"
            )
        scope = "date" if backtest_date is not None else "date_range"
    else:
        partition_ids = checkpoint.get("partition", {}).get(f"{split}_race_ids")
        if not isinstance(partition_ids, list) or not partition_ids:
            raise ValueError(f"checkpoint has no recorded {split} race IDs")
        race_ids = [int(value) for value in partition_ids]
        missing_ids = sorted(set(race_ids) - loaded["available_race_ids"])
        if missing_ids:
            raise ValueError(
                f"database is missing {len(missing_ids)} recorded {split} races"
            )
        scope = split
    race_ids = filter_race_ids_by_number(frame, race_ids, race_number)
    if not race_ids:
        raise ValueError(
            f"selected cohort has no complete finished race number {race_number}"
        )
    prepared = prepare_checkpoint_data(
        database, checkpoint, frame, race_ids, prepared_cache,
    )
    metrics, diagnostics, _ = evaluate_model(
        model,
        prepared["values"],
        prepared["winner_array"],
        prepared["race_id_array"],
        prepared["race_indices"],
        prepared["part"],
        races_per_batch,
        device,
    )
    return {
        "path": str(path),
        "model": path.stem,
        "checkpoint_type": checkpoint["checkpoint_type"],
        "cohort_id": cohort_key(race_ids),
        "races": len(race_ids),
        "date": backtest_date,
        "from_date": from_date,
        "to_date": to_date,
        "race_number": race_number,
        "scope": scope,
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
            "scope": record.get("scope"),
            "date": record.get("date"),
            "from_date": record.get("from_date"),
            "to_date": record.get("to_date"),
            "race_number": record.get("race_number"),
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
    validate_date_filters(args.date, args.from_date, args.to_date)
    if args.race_number is not None and args.race_number < 1:
        raise ValueError("--race-number must be positive")
    if args.races_per_batch < 1:
        raise ValueError("--races-per-batch must be positive")
    models_dir = args.models_dir.resolve()
    if not models_dir.is_dir():
        raise ValueError(f"--models-dir is not a directory: {models_dir}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    frame_cache: dict[FrameCacheKey, LoadedFrame] = {}
    prepared_cache: dict[PreparedCacheKey, PreparedDataset] = {}
    paths = sorted(models_dir.rglob("*.pt"))
    if not paths:
        raise ValueError(f"No .pt checkpoints found under {models_dir}")
    for path in paths:
        model: torch.nn.Module | None = None
        checkpoint: dict[str, Any] | None = None
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_type = checkpoint.get("checkpoint_type")
            if checkpoint_type not in SUPPORTED_TYPES:
                skipped.append({
                    "path": str(path),
                    "reason": f"unsupported checkpoint type {checkpoint_type!r}",
                })
                continue
            model = build_model_from_checkpoint(checkpoint, device)
            record = evaluate_checkpoint(
                path, checkpoint, model, args.split, args.db,
                args.races_per_batch, device, args.date,
                frame_cache, prepared_cache,
                from_date=args.from_date, to_date=args.to_date,
                race_number=args.race_number,
            )
            record["model"] = str(path.relative_to(models_dir).with_suffix(""))
            records.append(record)
        except (KeyError, RuntimeError, ValueError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
        finally:
            del model
            del checkpoint

    result = leaderboard(records)
    has_date_range = args.from_date is not None or args.to_date is not None
    scope = "date" if args.date else "date_range" if has_date_range else args.split
    if args.date:
        date_text = f" date={args.date}"
    elif has_date_range:
        date_text = (
            f" from_date={args.from_date or 'earliest'}"
            f" to_date={args.to_date or 'latest'}"
        )
    else:
        date_text = ""
    race_number_text = (
        f" race_number={args.race_number}"
        if args.race_number is not None else ""
    )
    print(
        f"WINNER-RANKER BACKTEST scope={scope}{date_text}{race_number_text} "
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
        "scope": scope,
        "split": None if args.date or has_date_range else args.split,
        "date": args.date,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "race_number": args.race_number,
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
