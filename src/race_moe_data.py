"""Data contracts shared by winner-MLP and winner-MoE training/inference."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch

from src.database import quote_identifier


DIAGNOSTIC_COLUMNS = (
    "distance_m", "class_name", "field_size", "active_field_size",
    "track_status", "career_starts", "runner_number", "runner_name",
)

# Intentionally conservative: a market-blind experiment must reject indirect
# bookmaker fields as well as the obvious live prices.
MARKET_FEATURE_RE = re.compile(
    r"(?:market|price|fluc|odds|bookmaker|favourite|favorite|starting_price|"
    r"implied_prob|steam|sp_rank|rank_minus_market|market_rank)", re.IGNORECASE,
)

CURRENT_RACE_MARKET_DERIVED_FEATURES = {
    "race_consensus_score", "race_consensus_rank",
    "race_overlay_score", "race_overlay_rank",
    "race_signal_agreement_score", "race_signal_agreement_rank",
}

IDENTIFIER_FEATURES = {
    "race_id", "competition_id", "selection_id", "runner_number",
}


def is_market_feature(name: str) -> bool:
    compact = name.replace(" ", "_")
    lowered = compact.lower()
    return (
        lowered in CURRENT_RACE_MARKET_DERIVED_FEATURES
        or bool(MARKET_FEATURE_RE.search(compact))
        or lowered in {
        "marketwinprice", "marketplaceprice", "open_price", "starting_price",
        }
    )


def market_blind_features(
    features: Sequence[str], *, include_market: bool = False,
) -> tuple[list[str], list[str]]:
    excluded = [
        name for name in features
        if name.lower() in IDENTIFIER_FEATURES
        or (not include_market and is_market_feature(name))
    ]
    retained = [name for name in features if name not in set(excluded)]
    if not retained:
        raise ValueError("Market filtering removed every configured feature")
    return retained, excluded


def load_finished_winner_rows(db: Path, features: Sequence[str]) -> pd.DataFrame:
    requested = list(dict.fromkeys([
        "race_id", "start_time_iso", "competition_id", "is_winner",
        "finish_place", *DIAGNOSTIC_COLUMNS, *features,
    ]))
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        available = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(race_runners)")
        }
        missing = [name for name in features if name not in available]
        if missing:
            raise ValueError("Configured DB features are missing: " + ", ".join(missing))
        selected = [name for name in requested if name in available]
        frame = pd.read_sql_query(
            "SELECT " + ", ".join(map(quote_identifier, selected))
            + " FROM race_runners WHERE status = 'finished' "
            + "ORDER BY start_time_iso, race_id, runner_number",
            connection,
        )
    finally:
        connection.close()
    if frame.empty:
        raise ValueError("No finished race rows found")
    keep_ids: list[int] = []
    for race_id, race in frame.groupby("race_id", sort=False):
        winner = pd.to_numeric(race["is_winner"], errors="coerce")
        if len(race) >= 4 and winner.notna().all() and int(winner.sum()) == 1:
            keep_ids.append(int(race_id))
    result = frame.loc[frame["race_id"].isin(keep_ids)].copy()
    if result.empty:
        raise ValueError("No complete races with exactly one winner found")
    return result


def chronological_race_ids(
    frame: pd.DataFrame, validation_races: int, test_races: int,
) -> tuple[list[int], list[int], list[int]]:
    if validation_races < 1 or test_races < 1:
        raise ValueError("validation_races and test_races must be positive")
    ordered = list(map(int, frame["race_id"].drop_duplicates()))
    holdout = validation_races + test_races
    if len(ordered) <= holdout:
        raise ValueError(
            f"Need more than {holdout} complete races; found {len(ordered)}"
        )
    return ordered[:-holdout], ordered[-holdout:-test_races], ordered[-test_races:]


def numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    matrix = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    # A feature may legitimately be absent throughout a future partition. The
    # training-fitted median handles that case; the trainer separately rejects
    # columns with no numerical training coverage.
    return matrix


def race_indices(race_ids: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(race_id): np.flatnonzero(race_ids == race_id)
        for race_id in dict.fromkeys(map(int, race_ids))
    }


def batches(
    indices: dict[int, np.ndarray], races_per_batch: int,
    rng: np.random.Generator | None = None,
) -> Iterator[list[np.ndarray]]:
    ids = np.asarray(list(indices), dtype=np.int64)
    if rng is not None:
        ids = rng.permutation(ids)
    for start in range(0, len(ids), races_per_batch):
        yield [indices[int(value)] for value in ids[start:start + races_per_batch]]


def pad_batch(
    x: np.ndarray, y: np.ndarray, groups: list[np.ndarray], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(map(len, groups))
    bx = np.zeros((len(groups), width, x.shape[1]), dtype=np.float32)
    by = np.zeros((len(groups), width), dtype=np.float32)
    valid = np.zeros((len(groups), width), dtype=bool)
    for batch_index, rows in enumerate(groups):
        count = len(rows)
        bx[batch_index, :count] = x[rows]
        by[batch_index, :count] = y[rows]
        valid[batch_index, :count] = True
    return (
        torch.from_numpy(bx).to(device), torch.from_numpy(by).to(device),
        torch.from_numpy(valid).to(device),
    )
