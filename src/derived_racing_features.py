"""Leakage-safe derived features built only from stored pre-race history."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings


DERIVED_FEATURE_NAMES = (
    "form_margin_weighted_3",
    "form_margin_weighted_6",
    "form_margin_scaled_weighted_3",
    "form_margin_scaled_weighted_6",
    "form_margin_std_3",
    "form_margin_std_6",
    "form_margin_trend",
    "form_finish_quality_weighted_3",
    "form_finish_quality_weighted_6",
    "form_field_size_weighted_3",
    "form_field_size_weighted_6",
    "form_barrier_percentile_weighted_3",
    "form_barrier_percentile_weighted_6",
    "form_market_expected_quality_weighted_3",
    "form_market_expected_quality_weighted_6",
    "form_market_overperformance_weighted_3",
    "form_market_overperformance_weighted_6",
    "form_same_distance_margin_weighted",
    "form_same_distance_finish_quality_weighted",
    "form_recent_distance_change_ratio",
)


def _matrix(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return np.column_stack([
        pd.to_numeric(frame[f"recent_{run}_{stem}"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for run in range(1, 7)
    ])


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


def _row_std(values: np.ndarray, count: int) -> np.ndarray:
    selected = values[:, :count]
    valid_count = np.isfinite(selected).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanstd(selected, axis=1, ddof=1)
    result[valid_count < 2] = np.nan
    return result


def derive_racing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return derived columns without reading any current or future outcome."""
    required = {
        "distance_m",
        *(f"recent_{run}_{stem}" for run in range(1, 7) for stem in (
            "place", "margin", "total_runners", "barrier", "starting_price",
            "distance_m",
        )),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Cannot derive racing features; missing: " + ", ".join(missing))

    margin = _matrix(frame, "margin")
    place = _matrix(frame, "place")
    field = _matrix(frame, "total_runners")
    barrier = _matrix(frame, "barrier")
    price = _matrix(frame, "starting_price")
    distance = _matrix(frame, "distance_m")

    valid_finish = (
        np.isfinite(place) & np.isfinite(field) & (field > 1) & (place >= 1)
    )
    finish_quality = np.full(place.shape, np.nan, dtype=np.float64)
    finish_quality[valid_finish] = np.clip(
        1.0 - (place[valid_finish] - 1.0) / (field[valid_finish] - 1.0),
        0.0, 1.0,
    )

    valid_barrier = (
        np.isfinite(barrier) & np.isfinite(field) & (field > 1) & (barrier >= 1)
    )
    barrier_percentile = np.full(barrier.shape, np.nan, dtype=np.float64)
    barrier_percentile[valid_barrier] = np.clip(
        (barrier[valid_barrier] - 1.0) / (field[valid_barrier] - 1.0),
        0.0, 1.0,
    )

    implied = np.divide(
        1.0, price, out=np.full(price.shape, np.nan),
        where=np.isfinite(price) & (price > 1.0),
    )
    uniform = np.divide(
        1.0, field, out=np.full(field.shape, np.nan),
        where=np.isfinite(field) & (field > 1.0),
    )
    expected_quality = np.divide(
        implied, implied + uniform,
        out=np.full(price.shape, np.nan),
        where=np.isfinite(implied) & np.isfinite(uniform),
    )
    overperformance = finish_quality - expected_quality

    margin_scaled = np.arcsinh(margin / 5.0)
    recent_margin = margin[:, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        older_margin = np.nanmean(margin[:, 1:4], axis=1)
    margin_trend = older_margin - recent_margin

    current_distance = pd.to_numeric(
        frame["distance_m"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    distance_difference = np.abs(distance - current_distance[:, None])
    distance_match = (
        np.isfinite(distance_difference)
        & np.isfinite(current_distance[:, None])
        & (distance_difference <= np.maximum(200.0, current_distance[:, None] * 0.15))
    )
    same_distance_margin = np.where(distance_match, margin, np.nan)
    same_distance_finish = np.where(distance_match, finish_quality, np.nan)
    recent_distance_change = np.divide(
        np.abs(distance[:, 0] - current_distance), current_distance,
        out=np.full(len(frame), np.nan),
        where=np.isfinite(distance[:, 0]) & np.isfinite(current_distance)
        & (current_distance > 0),
    )

    result = pd.DataFrame(index=frame.index)
    result["form_margin_weighted_3"] = _weighted_mean(margin, 3)
    result["form_margin_weighted_6"] = _weighted_mean(margin, 6)
    result["form_margin_scaled_weighted_3"] = _weighted_mean(margin_scaled, 3)
    result["form_margin_scaled_weighted_6"] = _weighted_mean(margin_scaled, 6)
    result["form_margin_std_3"] = _row_std(margin, 3)
    result["form_margin_std_6"] = _row_std(margin, 6)
    result["form_margin_trend"] = margin_trend
    result["form_finish_quality_weighted_3"] = _weighted_mean(finish_quality, 3)
    result["form_finish_quality_weighted_6"] = _weighted_mean(finish_quality, 6)
    result["form_field_size_weighted_3"] = _weighted_mean(field, 3)
    result["form_field_size_weighted_6"] = _weighted_mean(field, 6)
    result["form_barrier_percentile_weighted_3"] = _weighted_mean(
        barrier_percentile, 3
    )
    result["form_barrier_percentile_weighted_6"] = _weighted_mean(
        barrier_percentile, 6
    )
    result["form_market_expected_quality_weighted_3"] = _weighted_mean(
        expected_quality, 3
    )
    result["form_market_expected_quality_weighted_6"] = _weighted_mean(
        expected_quality, 6
    )
    result["form_market_overperformance_weighted_3"] = _weighted_mean(
        overperformance, 3
    )
    result["form_market_overperformance_weighted_6"] = _weighted_mean(
        overperformance, 6
    )
    result["form_same_distance_margin_weighted"] = _weighted_mean(
        same_distance_margin, 6
    )
    result["form_same_distance_finish_quality_weighted"] = _weighted_mean(
        same_distance_finish, 6
    )
    result["form_recent_distance_change_ratio"] = recent_distance_change
    return result.astype(np.float32)
