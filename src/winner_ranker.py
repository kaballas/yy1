"""Leakage-safe helpers for chronological, race-grouped winner ranking.

This module deliberately keeps current-race market inputs separate from form
inputs.  A form model never receives current prices.  A market-aware model gets
only two transparent current-market transforms and is not anchored to either of
them, so it remains free to reorder the field.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.database import quote_identifier


OUTCOME_OR_CONTROL_COLUMNS = {
    "winner_index",
    "is_trainable",
    "selection_id",
    "finish_place",
    "result_code",
    "status",
    "sp_starting_price",
    "runner_mask",
    "rank_label",
    "top3_mask",
    "is_winner",
    "is_validation",
}

IDENTIFIER_COLUMNS = {
    "race_id",
    "competition_id",  # 999 was assigned after results; never model this ID.
    "selection_id",
    "runner_number",
}

CURRENT_MARKET_EXACT = {
    "open_price",
    "fluc1",
    "fluc2",
    "sp_starting_price",
    "open_price_rank",
    "fluc1_price_rank",
    "fluc2_price_rank",
    "market_steam_rank",
    "race_consensus_score",
    "race_consensus_rank",
    "race_overlay_score",
    "race_overlay_rank",
    "race_signal_agreement_score",
    "race_signal_agreement_rank",
}

CURRENT_MARKET_PREFIXES = (
    "market_open_",
    "market_fluc1_",
    "market_total_",
    "market_price_",
    "market_implied_prob_",
)

MARKET_ENGINEERED_FEATURES = (
    "current_market_log_price",
    "current_market_rank_pct",
)


def is_current_market_feature(name: str) -> bool:
    """Return true only for information from the target race's market."""
    if name.startswith("historical_") or name.startswith("recent_"):
        return False
    return name in CURRENT_MARKET_EXACT or name.startswith(CURRENT_MARKET_PREFIXES)


def database_numeric_columns(database: Path) -> list[str]:
    """Return numeric race_runners columns in stable schema order."""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute('PRAGMA table_info("race_runners")').fetchall()
    return [
        str(row[1]) for row in rows
        if str(row[2]).upper() in {"INTEGER", "REAL", "NUMERIC"}
    ]


def load_training_rows(database: Path, numeric_columns: list[str]) -> pd.DataFrame:
    """Load active finished runners without relying on outcome-defined views."""
    metadata = [
        "race_id", "start_time_iso", "competition_id", "competition_name",
        "race_number", "race_name", "runner_number", "runner_name", "fluc2",
        "is_winner", "derived_racing_features_version",
    ]
    requested = list(dict.fromkeys([*metadata, *numeric_columns]))
    selected = ", ".join(quote_identifier(column) for column in requested)
    sql = (
        f"SELECT {selected} FROM race_runners "
        "WHERE status = 'finished' AND runner_mask = 1 "
        "AND is_winner IN (0, 1) "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(sql, connection)


def eligible_races(frame: pd.DataFrame, minimum_runners: int = 4) -> pd.DataFrame:
    """Return chronologically ordered races with exactly one labelled winner."""
    parsed = pd.to_datetime(frame["start_time_iso"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("start_time_iso contains invalid values")
    work = frame.assign(_start_time=parsed)
    races = work.groupby("race_id", as_index=False).agg(
        start_time=("_start_time", "min"),
        runners=("race_id", "size"),
        winners=("is_winner", "sum"),
    )
    return races.loc[
        (races["runners"] >= minimum_runners) & (races["winners"] == 1)
    ].sort_values(["start_time", "race_id"], kind="stable", ignore_index=True)


def chronological_race_split(
    races: pd.DataFrame, validation_races: int, test_races: int
) -> tuple[list[int], list[int], list[int]]:
    """Split whole races chronologically; the final cohort is never tuned on."""
    if validation_races < 1 or test_races < 1:
        raise ValueError("validation_races and test_races must be positive")
    if len(races) <= validation_races + test_races:
        raise ValueError(
            f"Need more than {validation_races + test_races} eligible races; "
            f"found {len(races)}"
        )
    ids = races["race_id"].astype(int).tolist()
    train_end = len(ids) - validation_races - test_races
    validation_end = len(ids) - test_races
    return ids[:train_end], ids[train_end:validation_end], ids[validation_end:]


def select_form_features(
    training: pd.DataFrame,
    numeric_columns: Iterable[str],
    minimum_coverage: float = 0.20,
) -> tuple[list[str], dict[str, str]]:
    """Select numeric pre-race form inputs using training rows only.

    Exact duplicate columns are removed deterministically.  Race-constant
    context such as distance is retained because trees can use it to condition
    runner-varying form, while identifiers and all target-race prices/results
    are excluded explicitly.
    """
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    selected: list[str] = []
    duplicate_of: dict[str, str] = {}
    fingerprints: dict[bytes, str] = {}
    excluded = OUTCOME_OR_CONTROL_COLUMNS | IDENTIFIER_COLUMNS
    for feature in numeric_columns:
        if feature in excluded or is_current_market_feature(feature):
            continue
        values = pd.to_numeric(training[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if float(values.notna().mean()) < minimum_coverage:
            continue
        if int(values.nunique(dropna=True)) <= 1:
            continue
        hashed = pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes()
        fingerprint = hashlib.sha256(hashed).digest()
        if fingerprint in fingerprints:
            duplicate_of[feature] = fingerprints[fingerprint]
            continue
        fingerprints[fingerprint] = feature
        selected.append(feature)
    return selected, duplicate_of


def rows_for_races(frame: pd.DataFrame, race_ids: Iterable[int]) -> pd.DataFrame:
    """Return whole races in chronological/runner order."""
    wanted = set(map(int, race_ids))
    return frame.loc[frame["race_id"].isin(wanted)].sort_values(
        ["start_time_iso", "race_id", "runner_number"], kind="stable"
    ).reset_index(drop=True)


def form_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Create the numeric form matrix, preserving missing values for XGBoost."""
    return frame.loc[:, features].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def current_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create two transparent target-market features with correct direction."""
    price = pd.to_numeric(frame["fluc2"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    price = price.where(price > 0)
    rank = price.groupby(frame["race_id"], sort=False).rank(
        method="first", ascending=True
    )
    valid = price.notna().groupby(frame["race_id"], sort=False).transform("sum")
    denominator = (valid - 1).clip(lower=1)
    percentile = 1.0 - ((rank - 1.0) / denominator)
    percentile = percentile.where(price.notna())
    return pd.DataFrame({
        "current_market_log_price": np.log(price),
        "current_market_rank_pct": percentile,
    }, index=frame.index)


def market_aware_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Append current market context without anchoring the model's score."""
    return pd.concat(
        [form_matrix(frame, features), current_market_features(frame)], axis=1
    )


def model_feature_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build a model matrix in the manifest's exact feature order.

    Most names are database columns. Current-market engineered names are
    calculated on demand and may appear in any configured model group.
    """
    engineered_names = set(MARKET_ENGINEERED_FEATURES)
    engineered = (
        current_market_features(frame)
        if any(feature in engineered_names for feature in features)
        else None
    )
    columns: dict[str, pd.Series] = {}
    for feature in features:
        if feature in engineered_names:
            assert engineered is not None
            columns[feature] = engineered[feature]
        elif feature in frame:
            columns[feature] = frame[feature]
        else:
            raise ValueError(f"Configured model feature is unavailable: {feature}")
    return pd.DataFrame(columns, index=frame.index).apply(
        pd.to_numeric, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Return XGBoost query-group sizes for already race-sorted rows."""
    return frame.groupby("race_id", sort=False).size().to_numpy(dtype=np.uint32)


def rank_percentiles(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalize arbitrary scores within each race: one=best, zero=worst."""
    work = pd.DataFrame({
        "race_id": np.asarray(race_ids),
        "score": np.asarray(scores, dtype=np.float64),
    })
    rank = work.groupby("race_id", sort=False)["score"].rank(
        method="first", ascending=False
    )
    count = work.groupby("race_id", sort=False)["race_id"].transform("size")
    return ((count - rank) / (count - 1).clip(lower=1)).to_numpy(dtype=np.float64)


def market_scores(frame: pd.DataFrame) -> np.ndarray:
    """Return higher-is-better market scores; invalid prices rank last."""
    price = pd.to_numeric(frame["fluc2"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(price) & (price > 0)
    score = np.full(len(frame), -1.0e12, dtype=np.float64)
    score[valid] = -np.log(price[valid])
    return score


def winner_metrics(
    targets: np.ndarray, scores: np.ndarray, race_ids: np.ndarray
) -> dict[str, float]:
    """Calculate equal-race winner ranking metrics."""
    y = np.asarray(targets, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(race_ids)
    if not (y.shape == score.shape == ids.shape) or not len(y):
        raise ValueError("targets, scores, and race_ids must be non-empty/equal")
    if not np.isfinite(score).all():
        raise ValueError("winner scores must be finite")
    ranks: list[int] = []
    losses: list[float] = []
    for race_id in pd.unique(ids):
        positions = np.flatnonzero(ids == race_id)
        race_y = y[positions]
        if int(race_y.sum()) != 1:
            raise ValueError(f"race_id {race_id} does not have exactly one winner")
        race_scores = score[positions]
        order = np.argsort(-race_scores, kind="stable")
        winner = int(np.flatnonzero(race_y == 1)[0])
        ranks.append(int(np.flatnonzero(order == winner)[0]) + 1)
        shifted = race_scores - race_scores.max()
        losses.append(float(-(shifted[winner] - np.log(np.exp(shifted).sum()))))
    rank_array = np.asarray(ranks, dtype=np.float64)
    return {
        "top1_hit_rate": float(np.mean(rank_array == 1)),
        "top3_hit_rate": float(np.mean(rank_array <= 3)),
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_winner_rank": float(np.mean(rank_array)),
        "race_logloss": float(np.mean(losses)),
        "races": float(len(ranks)),
    }


def ensemble_rank_scores(
    models: list[Any], matrix: pd.DataFrame, race_ids: np.ndarray
) -> np.ndarray:
    """Average member rank percentiles, avoiding incompatible raw scales."""
    if not models:
        raise ValueError("At least one model is required")
    members = [
        rank_percentiles(model.predict(matrix), race_ids) for model in models
    ]
    return np.mean(np.stack(members, axis=0), axis=0)


def blend_scores(
    form: np.ndarray,
    market_aware: np.ndarray,
    market: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    """Blend within-race percentiles using explicit non-negative weights."""
    values = {
        "form": np.asarray(form, dtype=np.float64),
        "market_aware": np.asarray(market_aware, dtype=np.float64),
        "market": np.asarray(market, dtype=np.float64),
    }
    unknown = set(weights) - set(values)
    if unknown:
        raise ValueError(f"Unknown blend components: {sorted(unknown)}")
    total = float(sum(weights.values()))
    if total <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("Blend weights must be non-negative with a positive sum")
    return sum(weights.get(name, 0.0) * value for name, value in values.items()) / total


def blend_named_scores(
    scores: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Blend any dynamically named model scores using configured weights."""
    unknown = set(weights) - set(scores)
    if unknown:
        raise ValueError(f"Unknown dynamic blend components: {sorted(unknown)}")
    if any(float(weight) < 0 for weight in weights.values()):
        raise ValueError("Dynamic blend weights must be non-negative")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Dynamic blend weights must have a positive sum")
    return sum(
        float(weight) * np.asarray(scores[name], dtype=np.float64)
        for name, weight in weights.items()
    ) / total


def select_blend_weights(
    targets: np.ndarray,
    race_ids: np.ndarray,
    form: np.ndarray,
    market_aware: np.ndarray,
    market: np.ndarray,
    step: float = 0.05,
) -> tuple[dict[str, float], dict[str, float]]:
    """Select a winner-first blend on validation only.

    Top-one hit rate is primary, then MRR, then mean winner rank.  A final
    deterministic preference for more form weight breaks truly equal metrics.
    """
    if not 0 < step <= 1 or not np.isclose(round(1.0 / step) * step, 1.0):
        raise ValueError("step must divide 1.0 exactly")
    units = int(round(1.0 / step))
    best_key: tuple[float, ...] | None = None
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    for form_units in range(units + 1):
        for aware_units in range(units - form_units + 1):
            market_units = units - form_units - aware_units
            weights = {
                "form": form_units / units,
                "market_aware": aware_units / units,
                "market": market_units / units,
            }
            metrics = winner_metrics(
                targets,
                blend_scores(form, market_aware, market, weights),
                race_ids,
            )
            key = (
                metrics["top1_hit_rate"], metrics["mrr"],
                -metrics["mean_winner_rank"], weights["form"],
                weights["market_aware"],
            )
            if best_key is None or key > best_key:
                best_key, best_weights, best_metrics = key, weights, metrics
    assert best_weights is not None and best_metrics is not None
    return best_weights, best_metrics


def market_deviation_metrics(
    frame: pd.DataFrame, challenger_name: str
) -> dict[str, float]:
    """Summarize whether a challenger makes useful top-pick market changes.

    ``frame`` must contain one winner per race plus ``market_rank`` and the
    requested challenger rank.  Corrected and damaged choices are paired race
    counts, making the net effect much easier to audit than aggregate accuracy.
    """
    challenger_rank = f"{challenger_name}_rank"
    required = {"race_id", "is_winner", "market_rank", challenger_rank}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing deviation columns: " + ", ".join(missing))
    market_top = frame.loc[frame["market_rank"] == 1, [
        "race_id", "is_winner",
    ]].set_index("race_id")
    challenger_top = frame.loc[frame[challenger_rank] == 1, [
        "race_id", "is_winner",
    ]].set_index("race_id")
    if not market_top.index.equals(challenger_top.index):
        raise ValueError("Market and challenger top picks do not cover the same races")
    # Compare the selected runner identity for each race, not just the scores.
    challenger_rows = frame.loc[frame[challenger_rank] == 1].set_index("race_id")
    market_rows = frame.loc[frame["market_rank"] == 1].set_index("race_id")
    identity_column = "runner_number" if "runner_number" in frame else None
    if identity_column is None:
        raise ValueError("Deviation metrics require runner_number")
    changed_mask = (
        challenger_rows[identity_column] != market_rows[identity_column]
    )
    corrected = (
        changed_mask & (challenger_rows["is_winner"] == 1)
        & (market_rows["is_winner"] == 0)
    )
    damaged = (
        changed_mask & (challenger_rows["is_winner"] == 0)
        & (market_rows["is_winner"] == 1)
    )
    races = len(market_rows)
    return {
        "races": float(races),
        "top_pick_changes": float(changed_mask.sum()),
        "top_pick_change_rate": float(changed_mask.mean()),
        "market_losses_corrected": float(corrected.sum()),
        "market_wins_damaged": float(damaged.sum()),
        "net_winners_gained": float(corrected.sum() - damaged.sum()),
    }
