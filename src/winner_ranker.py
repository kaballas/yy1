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
    "finish_rank_minus_market_rank",
    "margin_rank_minus_market_rank",
    "distance_speed_rank_minus_market_rank",
    "jockey_rank_minus_market_rank",
    "career_rank_minus_market_rank",
    "prize_money_rank_minus_market_rank",
    "finish_market_rank_abs_gap",
    "margin_market_rank_abs_gap",
    "distance_speed_market_rank_abs_gap",
    "jockey_market_rank_abs_gap",
    "career_market_rank_abs_gap",
    "prize_money_market_rank_abs_gap",
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


def uses_current_market_features(features: Iterable[str]) -> bool:
    """Return whether a model consumes target-race market information."""
    return any(
        feature in MARKET_ENGINEERED_FEATURES or is_current_market_feature(feature)
        for feature in features
    )


def current_market_free_model_labels(
    model_features: dict[str, list[str]],
) -> list[str]:
    """Return model labels whose inputs exclude every target-race market feature."""
    return [
        label
        for label, features in model_features.items()
        if not uses_current_market_features(features)
    ]


def database_numeric_columns(database: Path) -> list[str]:
    """Return numeric race_runners columns in stable schema order."""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute('PRAGMA table_info("race_runners")').fetchall()
    return [
        str(row[1]) for row in rows
        if str(row[2]).upper() in {"INTEGER", "REAL", "NUMERIC"}
    ]


DEFAULT_NATIVE_CATEGORICAL_FEATURES = [
    "country", "class_name", "grade", "tempo", "track_status",
    "runner_country", "sex", "colour", "blinkers",
    "expected_settling_position",
    *[
        f"recent_{index}_{suffix}"
        for index in range(1, 7)
        for suffix in ("track_name", "track_status", "class")
    ],
]


def database_text_columns(database: Path) -> list[str]:
    """Return text race_runners columns in stable schema order."""
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute('PRAGMA table_info("race_runners")').fetchall()
    return [
        str(row[1]) for row in rows
        if any(token in str(row[2]).upper() for token in ("CHAR", "CLOB", "TEXT"))
    ]


def load_training_rows(
    database: Path,
    numeric_columns: list[str],
    competition_id: int | None = None,
    categorical_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Load active finished runners, optionally for one competition."""
    if competition_id is not None and competition_id < 1:
        raise ValueError("competition_id must be positive")
    metadata = [
        "race_id", "start_time_iso", "competition_id", "competition_name",
        "race_number", "race_name", "runner_number", "runner_name", "fluc2",
        "status", "rank_label", "is_winner", "derived_racing_features_version",
    ]
    requested = list(dict.fromkeys([
        *metadata, *numeric_columns, *categorical_columns,
    ]))
    selected = ", ".join(quote_identifier(column) for column in requested)
    conditions = [
        "status = 'finished'",
        "runner_mask = 1",
        "is_winner IN (0, 1)",
    ]
    parameters: list[int] = []
    if competition_id is not None:
        conditions.append("competition_id = ?")
        parameters.append(int(competition_id))
    sql = (
        f"SELECT {selected} FROM race_runners "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY start_time_iso, race_id, runner_number"
    )
    #print(sql)
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        return pd.read_sql_query(sql, connection, params=parameters)


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


def select_categorical_features(
    training: pd.DataFrame,
    requested: Iterable[str],
    minimum_coverage: float = 0.20,
) -> list[str]:
    """Keep available, populated, non-constant categorical inputs."""
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    selected: list[str] = []
    for feature in requested:
        if feature not in training:
            continue
        values = training[feature].astype("string").str.strip().replace("", pd.NA)
        if float(values.notna().mean()) < minimum_coverage:
            continue
        if int(values.nunique(dropna=True)) <= 1:
            continue
        selected.append(feature)
    return selected


def categorical_levels(
    frame: pd.DataFrame, features: Iterable[str]
) -> dict[str, list[str]]:
    """Return deterministic category vocabularies, including missing values."""
    result: dict[str, list[str]] = {}
    for feature in features:
        values = frame[feature].astype("string").str.strip().fillna("__MISSING__")
        result[feature] = sorted(set(values.astype(str)))
    return result


def prepare_categorical_features(
    frame: pd.DataFrame,
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Apply saved category vocabularies; unseen values become missing."""
    result = frame.copy()
    for feature, categories in levels.items():
        if feature not in result:
            continue
        values = result[feature].astype("string").str.strip().fillna("__MISSING__")
        known = values.where(values.isin(categories), pd.NA)
        result[feature] = pd.Categorical(known, categories=categories)
    return result


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
    matrix = pd.DataFrame(columns, index=frame.index)
    for feature in matrix:
        values = matrix[feature]
        if isinstance(values.dtype, pd.CategoricalDtype):
            matrix[feature] = values
        elif (
            pd.api.types.is_object_dtype(values.dtype)
            or pd.api.types.is_string_dtype(values.dtype)
        ):
            # XGBoost >=3.1 stores category names in JSON models and safely
            # recodes pandas categories at prediction time.
            matrix[feature] = values.astype("string").str.strip().fillna("__MISSING__").astype(
                "category"
            )
        else:
            matrix[feature] = pd.to_numeric(values, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return matrix


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """Return XGBoost query-group sizes for already race-sorted rows."""
    return frame.groupby("race_id", sort=False).size().to_numpy(dtype=np.uint32)


RANKING_TARGETS = (
    "winner",
    "finish_order",
    "margin_aware_finish_order",
)


def finishing_relevance(frame: pd.DataFrame) -> np.ndarray:
    """Return a bounded full-order relevance label for every active runner.

    ``finish_place`` is preferred. Some resulted rows retain only a numeric
    ``rank_label`` (typically a non-finisher placed behind the active field),
    so that value is used as a deterministic fallback and clipped to last.
    """
    if "finish_place" not in frame:
        raise ValueError("finish_order target requires finish_place")
    place = pd.to_numeric(frame["finish_place"], errors="coerce")
    if "rank_label" in frame:
        place = place.fillna(pd.to_numeric(frame["rank_label"], errors="coerce"))
    if place.isna().any():
        raise ValueError(
            "finish_order target has unresolved finish places for "
            f"{int(place.isna().sum()):,} runners"
        )
    if (place < 1).any():
        raise ValueError("finish_order target requires finish places >= 1")
    field_size = frame.groupby("race_id", sort=False)["race_id"].transform("size")
    if (field_size < 2).any():
        raise ValueError("finish_order target requires at least two runners per race")
    bounded_place = np.minimum(
        place.to_numpy(dtype=np.float64),
        field_size.to_numpy(dtype=np.float64),
    )
    relevance = 1.0 - (
        (bounded_place - 1.0) /
        (field_size.to_numpy(dtype=np.float64) - 1.0)
    )
    return np.clip(relevance, 0.0, 1.0)


def ranking_targets(
    frame: pd.DataFrame,
    target: str,
    beaten_margin_column: str = "beaten_margin",
) -> np.ndarray:
    """Build one of the frozen-supervision experiment's three targets."""
    if target not in RANKING_TARGETS:
        raise ValueError(f"Unknown ranking target: {target}")
    if target == "winner":
        return frame["is_winner"].to_numpy(dtype=np.float64)

    finish_score = finishing_relevance(frame)
    if target == "finish_order":
        return finish_score

    if beaten_margin_column not in frame:
        raise ValueError(
            "margin_aware_finish_order requires a current-race beaten-margin "
            f"column; database column {beaten_margin_column!r} does not exist"
        )
    margin = pd.to_numeric(frame[beaten_margin_column], errors="coerce")
    # A winner's stored winning margin is not a beaten margin.
    margin = margin.mask(frame["is_winner"].astype(bool), 0.0)
    if margin.isna().any() or (margin < 0).any():
        raise ValueError(
            "margin_aware_finish_order requires finite non-negative beaten "
            "margins for every runner"
        )
    margin_score = np.exp(-margin.to_numpy(dtype=np.float64) / 5.0)
    return 0.75 * finish_score + 0.25 * margin_score


def validate_ranker_groups(
    frame: pd.DataFrame,
    targets: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> dict[str, int | float]:
    """Fail fast when rows cannot safely be passed to ``XGBRanker.fit``.

    XGBoost receives group *sizes*, not race identifiers, so a race appearing
    in two non-contiguous blocks would silently create incorrect ranking
    queries. Winner models additionally require exactly one positive label in
    every race.
    """
    if frame.empty:
        raise ValueError("ranker frame must not be empty")
    if "race_id" not in frame or frame["race_id"].isna().any():
        raise ValueError("ranker race_id values must be present and non-null")

    race_ids = frame["race_id"].to_numpy()
    block_starts = np.r_[True, race_ids[1:] != race_ids[:-1]]
    block_ids = race_ids[block_starts]
    if len(block_ids) != len(pd.unique(block_ids)):
        raise ValueError("ranker races must occupy one contiguous row block each")

    expected = group_sizes(frame)
    supplied = expected if groups is None else np.asarray(groups)
    if supplied.ndim != 1 or not np.issubdtype(supplied.dtype, np.number):
        raise ValueError("ranker groups must be a one-dimensional numeric array")
    if not np.isfinite(supplied).all() or np.any(supplied <= 0):
        raise ValueError("ranker group sizes must be finite positive values")
    if not np.equal(supplied, np.floor(supplied)).all():
        raise ValueError("ranker group sizes must be integers")
    if int(supplied.sum()) != len(frame):
        raise ValueError(
            "ranker group sizes do not sum to row count: "
            f"groups={int(supplied.sum()):,} rows={len(frame):,}"
        )
    if not np.array_equal(supplied.astype(np.uint64), expected.astype(np.uint64)):
        raise ValueError("ranker group sizes do not match contiguous race blocks")

    label_values = frame["is_winner"].to_numpy() if targets is None else targets
    y = np.asarray(label_values)
    if y.ndim != 1 or len(y) != len(frame):
        raise ValueError("ranker targets must be one-dimensional and match row count")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("winner-ranker targets must contain only zero and one")
    winner_counts = pd.Series(y).groupby(frame["race_id"], sort=False).sum()
    invalid = winner_counts[winner_counts != 1]
    if not invalid.empty:
        raise ValueError(
            "winner-ranker races must contain exactly one winner; invalid_races="
            f"{len(invalid):,}"
        )

    sizes = expected.astype(np.int64)
    return {
        "rows": int(len(frame)),
        "races": int(len(sizes)),
        "minimum_runners": int(sizes.min()),
        "median_runners": float(np.median(sizes)),
        "maximum_runners": int(sizes.max()),
        "singleton_races": int(np.sum(sizes == 1)),
        # One positive versus every negative is the available pair count for
        # this binary winner target before XGBoost's pair sampler is applied.
        "winner_loser_pairs": int(np.sum(sizes - 1)),
    }


def winner_race_report(
    frame: pd.DataFrame, targets: np.ndarray, scores: np.ndarray
) -> pd.DataFrame:
    """Return one auditable result row per race, including random baselines."""
    y = np.asarray(targets, dtype=np.int64)
    prediction = np.asarray(scores, dtype=np.float64)
    validate_ranker_groups(frame, y)
    if prediction.ndim != 1 or len(prediction) != len(frame):
        raise ValueError("ranker scores must be one-dimensional and match row count")
    if not np.isfinite(prediction).all():
        raise ValueError("ranker scores must be finite")

    rows: list[dict[str, Any]] = []
    for race_id, positions in pd.Series(
        np.arange(len(frame)), index=frame.index
    ).groupby(frame["race_id"], sort=False):
        indices = positions.to_numpy(dtype=np.int64)
        race_y = y[indices]
        race_scores = prediction[indices]
        winner = int(np.flatnonzero(race_y == 1)[0])
        winner_rank = float(pd.Series(race_scores).rank(
            method="average", ascending=False
        ).iloc[winner])
        size = len(indices)
        row: dict[str, Any] = {
            "race_id": race_id,
            "runners": size,
            "top1_hit": int(winner_rank == 1),
            "top3_hit": int(winner_rank <= 3),
            "winner_rank": winner_rank,
            "reciprocal_rank": 1.0 / winner_rank,
            "random_top1_expected": 1.0 / size,
            "random_top3_expected": min(3, size) / size,
        }
        for column in (
            "start_time_iso", "competition_id", "competition_name",
            "race_number", "race_name",
        ):
            if column in frame:
                row[column] = frame.iloc[indices[0]][column]
        rows.append(row)
    return pd.DataFrame(rows)


def winner_field_size_slices(report: pd.DataFrame) -> pd.DataFrame:
    """Summarize rank quality by field size without weighting large races more."""
    if report.empty:
        raise ValueError("winner race report must not be empty")
    work = report.copy()
    work["field_size_bucket"] = pd.cut(
        work["runners"],
        [0, 6, 9, 12, 16, np.inf],
        labels=["1-6", "7-9", "10-12", "13-16", "17+"],
        right=True,
    )
    return work.groupby("field_size_bucket", sort=True, observed=True).agg(
        races=("race_id", "size"),
        top1_hit_rate=("top1_hit", "mean"),
        top3_hit_rate=("top3_hit", "mean"),
        mrr=("reciprocal_rank", "mean"),
        mean_winner_rank=("winner_rank", "mean"),
        random_top1_expected=("random_top1_expected", "mean"),
        random_top3_expected=("random_top3_expected", "mean"),
    ).reset_index()


def xgb_ensemble_feature_importance(
    models: list[Any], label: str
) -> pd.DataFrame:
    """Expose gain, cover, split count, and total gain for every ensemble member."""
    records: list[dict[str, Any]] = []
    for member, model in enumerate(models, start=1):
        booster = model.get_booster()
        for importance_type in ("gain", "cover", "weight", "total_gain"):
            values = booster.get_score(importance_type=importance_type)
            for feature, value in values.items():
                records.append({
                    "model": label,
                    "member": member,
                    "importance_type": importance_type,
                    "feature": feature,
                    "value": float(value),
                })
    return pd.DataFrame.from_records(records, columns=[
        "model", "member", "importance_type", "feature", "value",
    ])


def rank_percentiles(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    """Normalize scores within each race, assigning equal values to ties."""
    work = pd.DataFrame({
        "race_id": np.asarray(race_ids),
        "score": np.asarray(scores, dtype=np.float64),
    })
    rank = work.groupby("race_id", sort=False)["score"].rank(
        method="average", ascending=False
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
    ranks: list[float] = []
    losses: list[float] = []
    for race_id in pd.unique(ids):
        positions = np.flatnonzero(ids == race_id)
        race_y = y[positions]
        if int(race_y.sum()) != 1:
            raise ValueError(f"race_id {race_id} does not have exactly one winner")
        race_scores = score[positions]
        winner = int(np.flatnonzero(race_y == 1)[0])
        ranks.append(float(pd.Series(race_scores).rank(
            method="average", ascending=False
        ).iloc[winner]))
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
