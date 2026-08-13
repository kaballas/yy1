"""Leakage-safe sectional, class, and entity-history racing features."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd


HISTORY_FEATURE_NAMES = (
    "sectional_last600_seconds_weighted_3",
    "sectional_last600_seconds_weighted_6",
    "sectional_last600_best_6",
    "sectional_last600_std_6",
    "sectional_last600_trend",
    "sectional_closing_speed_ratio_weighted_3",
    "sectional_closing_speed_ratio_weighted_6",
    "current_class_level",
    "form_class_level_weighted_3",
    "form_class_level_weighted_6",
    "class_change_vs_recent_3",
    "class_drop_from_recent_best",
)

ENTITY_FEATURE_NAMES = tuple(
    f"{prefix}_{suffix}"
    for prefix in ("jockey", "trainer", "jockey_trainer")
    for suffix in (
        "history_starts", "history_top3_excess", "recent_top3_excess",
    )
)

ADVANCED_FEATURE_NAMES = HISTORY_FEATURE_NAMES + ENTITY_FEATURE_NAMES

_GRADE_LEVELS = {"ONE": 125.0, "TWO": 120.0, "THREE": 115.0, "LR": 110.0}


def _weighted_mean(values: np.ndarray, count: int) -> np.ndarray:
    selected = values[:, :count]
    weights = 1.0 / np.arange(1, count + 1, dtype=np.float64)
    valid = np.isfinite(selected)
    denominator = (valid * weights).sum(axis=1)
    numerator = np.where(valid, selected, 0.0).dot(weights)
    return np.divide(
        numerator, denominator,
        out=np.full(len(values), np.nan), where=denominator > 0,
    )


def _duration_seconds(values: pd.Series) -> np.ndarray:
    text = values.astype("string").str.strip()
    colon = text.str.extract(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")
    colon_minutes = pd.to_numeric(colon[0], errors="coerce").fillna(0.0)
    colon_seconds = pd.to_numeric(colon[1], errors="coerce")
    result = colon_minutes.to_numpy(dtype=np.float64) * 60.0 + colon_seconds.to_numpy(
        dtype=np.float64
    )
    dot = text.str.extract(r"^(\d+)\.(\d{2})\.(\d{2})$")
    dot_minutes = pd.to_numeric(dot[0], errors="coerce")
    dot_seconds = pd.to_numeric(dot[1], errors="coerce")
    dot_hundredths = pd.to_numeric(dot[2], errors="coerce")
    dot_valid = dot_minutes.notna() & dot_seconds.notna() & dot_hundredths.notna()
    result[dot_valid.to_numpy()] = (
        dot_minutes[dot_valid].to_numpy(dtype=np.float64) * 60.0
        + dot_seconds[dot_valid].to_numpy(dtype=np.float64)
        + dot_hundredths[dot_valid].to_numpy(dtype=np.float64) / 100.0
    )
    valid = colon_seconds.notna() | dot_valid
    result[~valid.to_numpy()] = np.nan
    return result


def _duration_matrix(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return np.column_stack([
        _duration_seconds(frame[f"recent_{run}_{stem}"])
        for run in range(1, 7)
    ])


def parse_class_level(value: object, grade: object = None) -> float:
    """Map common Australasian class labels onto an approximate rating scale."""
    grade_text = "" if pd.isna(grade) else str(grade).strip().upper()
    if grade_text in _GRADE_LEVELS:
        return _GRADE_LEVELS[grade_text]
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if not text or "BARRIER TRIAL" in text:
        return np.nan
    if re.search(r"\b(?:GROUP\s*1|G1)\b", text):
        return 125.0
    if re.search(r"\b(?:GROUP\s*2|G2)\b", text):
        return 120.0
    if re.search(r"\b(?:GROUP\s*3|G3)\b", text):
        return 115.0
    if re.search(r"\b(?:LISTED|LR)\b", text):
        return 110.0
    benchmark = re.search(r"\b(?:BM|BENCHMARK|RST)\s*([0-9]{2,3})", text)
    if benchmark:
        return float(benchmark.group(1))
    handicap_rating = re.search(r"\bHCP\s*\(([0-9]{2,3})\)", text)
    if handicap_rating:
        return float(handicap_rating.group(1))
    class_number = re.search(r"\b(?:CLS|CLASS|C)\s*([1-6])\b", text)
    if class_number:
        return 52.0 + 4.0 * float(class_number.group(1))
    if re.search(r"\b(?:MDN|MAIDEN)\b", text):
        return 50.0
    if re.search(r"\bOPEN\b", text):
        return 100.0
    return np.nan


def derive_sectional_class_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive features solely from the current race card and stored prior runs."""
    required = {"race_name", "grade", "distance_m"}
    required.update(
        f"recent_{run}_{stem}"
        for run in range(1, 7)
        for stem in ("last600", "time", "distance_m", "class")
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Cannot derive advanced history features; missing: " + ", ".join(missing))

    last600 = _duration_matrix(frame, "last600")
    # A 600 m sectional outside this range is a malformed/full-race duration,
    # not a plausible thoroughbred closing split.
    last600[(last600 < 25.0) | (last600 > 60.0)] = np.nan
    race_time = _duration_matrix(frame, "time")
    recent_distance = np.column_stack([
        pd.to_numeric(frame[f"recent_{run}_distance_m"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for run in range(1, 7)
    ])
    average_speed = np.divide(
        recent_distance, race_time,
        out=np.full(last600.shape, np.nan),
        where=np.isfinite(recent_distance) & np.isfinite(race_time) & (race_time > 0),
    )
    closing_speed = np.divide(
        600.0, last600,
        out=np.full(last600.shape, np.nan),
        where=np.isfinite(last600) & (last600 > 0),
    )
    closing_ratio = np.divide(
        closing_speed, average_speed,
        out=np.full(last600.shape, np.nan),
        where=np.isfinite(closing_speed) & np.isfinite(average_speed) & (average_speed > 0),
    )
    closing_ratio[(closing_ratio < 0.5) | (closing_ratio > 2.0)] = np.nan

    valid_last600 = np.isfinite(last600)
    with np.errstate(invalid="ignore"):
        best_last600 = np.min(np.where(valid_last600, last600, np.inf), axis=1)
    best_last600[~valid_last600.any(axis=1)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last600_std = np.nanstd(last600, axis=1, ddof=1)
    last600_std[valid_last600.sum(axis=1) < 2] = np.nan
    older = _weighted_mean(np.column_stack([last600[:, 1:], np.full(len(frame), np.nan)]), 5)
    trend = older - last600[:, 0]
    trend[~np.isfinite(last600[:, 0]) | ~np.isfinite(older)] = np.nan

    current_class = np.asarray([
        parse_class_level(name, grade)
        for name, grade in zip(frame["race_name"], frame["grade"])
    ], dtype=np.float64)
    recent_class = np.column_stack([
        np.asarray([
            parse_class_level(value)
            for value in frame[f"recent_{run}_class"]
        ], dtype=np.float64)
        for run in range(1, 7)
    ])
    class_3 = _weighted_mean(recent_class, 3)
    class_6 = _weighted_mean(recent_class, 6)
    with np.errstate(invalid="ignore"):
        best_class = np.max(
            np.where(np.isfinite(recent_class), recent_class, -np.inf), axis=1
        )
    best_class[~np.isfinite(recent_class).any(axis=1)] = np.nan

    result = pd.DataFrame(index=frame.index)
    result["sectional_last600_seconds_weighted_3"] = _weighted_mean(last600, 3)
    result["sectional_last600_seconds_weighted_6"] = _weighted_mean(last600, 6)
    result["sectional_last600_best_6"] = best_last600
    result["sectional_last600_std_6"] = last600_std
    result["sectional_last600_trend"] = trend
    result["sectional_closing_speed_ratio_weighted_3"] = _weighted_mean(
        closing_ratio, 3
    )
    result["sectional_closing_speed_ratio_weighted_6"] = _weighted_mean(
        closing_ratio, 6
    )
    result["current_class_level"] = current_class
    result["form_class_level_weighted_3"] = class_3
    result["form_class_level_weighted_6"] = class_6
    result["class_change_vs_recent_3"] = current_class - class_3
    result["class_drop_from_recent_best"] = best_class - current_class
    return result.astype(np.float32)


def _normalize_entity(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.mask(normalized.eq(""))


def _causal_entity_features(
    frame: pd.DataFrame,
    entity: pd.Series,
    prefix: str,
    recent_span: int,
) -> pd.DataFrame:
    """Calculate entity statistics using only strictly earlier start times."""
    result = pd.DataFrame(index=frame.index)
    names = [
        f"{prefix}_history_starts",
        f"{prefix}_history_top3_excess",
        f"{prefix}_recent_top3_excess",
    ]
    for name in names:
        result[name] = np.nan

    start_time = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    field_size = pd.to_numeric(
        frame["active_field_size"], errors="coerce"
    ).fillna(pd.to_numeric(frame["field_size"], errors="coerce"))
    top3 = pd.to_numeric(frame["top3_mask"], errors="coerce")
    eligible = (
        entity.notna() & start_time.notna() & frame["status"].eq("finished")
        & pd.to_numeric(frame["runner_mask"], errors="coerce").eq(1)
        & top3.isin([0, 1]) & field_size.gt(0)
    )
    expected = np.minimum(3.0, field_size) / field_size
    history = pd.DataFrame({
        "entity": entity[eligible],
        "time": start_time[eligible],
        "starts": 1.0,
        "residual": top3[eligible].to_numpy(dtype=np.float64)
        - expected[eligible].to_numpy(dtype=np.float64),
    })
    if history.empty:
        return result
    events = history.groupby(["entity", "time"], as_index=False, sort=True).agg(
        starts=("starts", "sum"), residual=("residual", "sum")
    )
    events = events.sort_values(["entity", "time"], kind="stable")
    events["post_starts"] = events.groupby("entity", sort=False)["starts"].cumsum()
    events["post_residual"] = events.groupby("entity", sort=False)["residual"].cumsum()
    events["event_excess"] = events["residual"] / events["starts"]
    events["post_recent"] = events.groupby("entity", sort=False)[
        "event_excess"
    ].transform(lambda values: values.ewm(span=recent_span, adjust=False).mean())

    targets = pd.DataFrame({
        "entity": entity,
        "time": start_time,
        "row": np.arange(len(frame), dtype=np.int64),
    }).dropna(subset=["entity", "time"])
    event_groups = {
        entity_name: group
        for entity_name, group in events.groupby("entity", sort=False)
    }
    for entity_name, target_group in targets.groupby("entity", sort=False):
        event_group = event_groups.get(entity_name)
        if event_group is None:
            continue
        event_ns = event_group["time"].astype("int64").to_numpy()
        target_ns = target_group["time"].astype("int64").to_numpy()
        prior_index = np.searchsorted(event_ns, target_ns, side="left") - 1
        usable = prior_index >= 0
        if not usable.any():
            continue
        rows = target_group["row"].to_numpy(dtype=np.int64)[usable]
        indices = prior_index[usable]
        prior_starts = event_group["post_starts"].to_numpy(dtype=np.float64)[indices]
        prior_residual = event_group["post_residual"].to_numpy(dtype=np.float64)[indices]
        recent = event_group["post_recent"].to_numpy(dtype=np.float64)[indices]
        reliability = prior_starts / (prior_starts + 10.0)
        result.iloc[rows, result.columns.get_loc(names[0])] = prior_starts
        result.iloc[rows, result.columns.get_loc(names[1])] = (
            prior_residual / (prior_starts + 20.0)
        )
        result.iloc[rows, result.columns.get_loc(names[2])] = recent * reliability
    return result.astype(np.float32)


def derive_entity_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return causal jockey, trainer, and partnership history features."""
    required = {
        "start_time_iso", "status", "runner_mask", "top3_mask",
        "active_field_size", "field_size", "jockey", "trainer",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Cannot derive entity history features; missing: " + ", ".join(missing))
    jockey = _normalize_entity(frame["jockey"])
    trainer = _normalize_entity(frame["trainer"])
    partnership = (jockey + "\x1f" + trainer).where(jockey.notna() & trainer.notna())
    return pd.concat([
        _causal_entity_features(frame, jockey, "jockey", 50),
        _causal_entity_features(frame, trainer, "trainer", 100),
        _causal_entity_features(frame, partnership, "jockey_trainer", 30),
    ], axis=1)
