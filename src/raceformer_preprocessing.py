"""Versioned robust preprocessing for current-race RaceFormer models."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from src.preprocessing import transform, zero_feature_columns


LOG1P_FEATURES = {
    "career_starts", "career_wins", "career_seconds", "career_thirds",
    "distance_starts", "distance_wins", "track_starts", "track_wins",
    "horse_jockey_starts", "horse_jockey_wins", "recent_1_starting_price",
    "recent_2_starting_price", "recent_3_starting_price",
    "recent_3_total_runners",
    "recent_same_distance_runs", "recent_same_track_runs",
    "recent_same_condition_runs", "recent_days_since_last_run",
    "recent_best_margin",
    "open_price", "fluc1", "fluc2",
}

RELATIVE_FEATURES = {
    "draw_number", "weight_kg", "career_starts", "career_wins",
    "career_seconds", "career_thirds", "trackFamiliarity",
    "distanceFamiliarity", "last3_win_percentage", "last3_place_percentage",
    "win_percentage", "place_percentage", "distance_starts", "distance_wins",
    "track_starts", "track_wins", "horse_jockey_starts", "horse_jockey_wins",
    "recent_1_starting_price", "recent_2_starting_price",
    "recent_3_starting_price", "recent_avg_place", "recent_best_place",
    "recent_3_total_runners",
    "recent_same_distance_runs", "recent_same_track_runs",
    "recent_same_condition_runs", "recent_days_since_last_run",
    "recent_best_margin", "form_barrier_percentile_weighted_6",
    "sectional_last600_best_6", "trainer_recent_top3_excess",
    "open_price", "fluc1", "fluc2",
}

LAYOFF_BUCKETS = (
    ("recent_days_30_plus", 30.0),
    ("recent_days_60_plus", 60.0),
    ("recent_days_90_plus", 90.0),
    ("recent_days_180_plus", 180.0),
    ("recent_days_365_plus", 365.0),
)
LAYOFF_MISSING_FEATURE = "recent_days_since_last_run_missing"
EXCLUSIVE_LAYOFF_BUCKETS = (
    "recent_days_30_59",
    "recent_days_60_89",
    "recent_days_90_179",
    "recent_days_180_364",
    "recent_days_365_plus",
)


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress long tails without producing NaNs for unexpected negative values."""
    return np.sign(values) * np.log1p(np.abs(values))


def _base_values(
    raw: np.ndarray, feature_columns: Sequence[str], log1p_features: Sequence[str]
) -> np.ndarray:
    result = np.asarray(raw, dtype=np.float32).copy()
    selected = set(log1p_features)
    for index, feature in enumerate(feature_columns):
        if feature in selected:
            result[:, index] = _signed_log1p(result[:, index])
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return average zero-based ranks, leaving missing values as NaN."""
    result = np.full(len(values), np.nan, dtype=np.float32)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return result
    ordered = valid[np.argsort(values[valid], kind="stable")]
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        result[ordered[start:end]] = (start + end - 1) / 2.0
        start = end
    return result


def race_percentiles(
    raw: np.ndarray,
    race_ids: np.ndarray,
    feature_columns: Sequence[str],
    relative_features: Sequence[str],
) -> np.ndarray:
    """Build centred [-1, 1] within-race percentile features with neutral missing values."""
    selected = [feature_columns.index(name) for name in relative_features]
    result = np.zeros((len(raw), len(selected)), dtype=np.float32)
    for race_id in dict.fromkeys(map(int, race_ids)):
        rows = np.flatnonzero(race_ids == race_id)
        for output_index, input_index in enumerate(selected):
            ranks = _average_ranks(raw[rows, input_index])
            valid = np.isfinite(ranks)
            count = int(valid.sum())
            if count > 1:
                result[rows[valid], output_index] = (
                    2.0 * ranks[valid] / (count - 1.0) - 1.0
                )
            # A constant/singleton/missing feature is deliberately neutral.
    return result


def fit_raceformer_preprocessor(
    raw: np.ndarray,
    feature_columns: Sequence[str],
    *,
    clip: float = 5.0,
    layoff_buckets: bool = False,
    layoff_bucket_mode: str | None = None,
) -> dict[str, Any]:
    if clip <= 0:
        raise ValueError("clip must be positive")
    if layoff_bucket_mode is None:
        layoff_bucket_mode = "cumulative" if layoff_buckets else "none"
    if layoff_bucket_mode not in {"none", "cumulative", "exclusive"}:
        raise ValueError("layoff_bucket_mode must be none, cumulative, or exclusive")
    log1p_features = [name for name in feature_columns if name in LOG1P_FEATURES]
    relative_features = [name for name in feature_columns if name in RELATIVE_FEATURES]
    base = _base_values(raw, feature_columns, log1p_features)
    median = np.nanmedian(base, axis=0).astype(np.float32)
    median = np.nan_to_num(median)
    filled = np.where(np.isnan(base), median, base)
    # 1.4826 * MAD estimates standard deviation for a normal distribution while
    # remaining resistant to the price/count tails that motivated this contract.
    scale = (
        1.4826 * np.median(np.abs(filled - median), axis=0)
    ).astype(np.float32)
    fallback = np.std(filled, axis=0).astype(np.float32)
    scale = np.where(scale < 1e-6, fallback, scale).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return {
        "version": 3,
        "log1p_features": log1p_features,
        "relative_features": relative_features,
        "relative_suffix": "__race_percentile",
        "clip": float(clip),
        "layoff_bucket_features": (
            [name for name, _ in LAYOFF_BUCKETS] + [LAYOFF_MISSING_FEATURE]
            if layoff_bucket_mode == "cumulative" else
            [*EXCLUSIVE_LAYOFF_BUCKETS, LAYOFF_MISSING_FEATURE]
            if layoff_bucket_mode == "exclusive" else []
        ),
        "layoff_bucket_mode": layoff_bucket_mode,
        "median": median,
        "scale": scale,
    }


def model_feature_columns(
    feature_columns: Sequence[str], contract: dict[str, Any] | None
) -> list[str]:
    if not contract or int(contract.get("version", 1)) < 2:
        return list(feature_columns)
    suffix = str(contract.get("relative_suffix", "__race_percentile"))
    return [
        *feature_columns,
        *(f"{name}{suffix}" for name in contract.get("relative_features", [])),
        *contract.get("layoff_bucket_features", []),
    ]


def layoff_bucket_values(
    raw: np.ndarray, feature_columns: Sequence[str], contract: dict[str, Any]
) -> np.ndarray:
    names = list(contract.get("layoff_bucket_features", []))
    if not names:
        return np.empty((len(raw), 0), dtype=np.float32)
    if "recent_days_since_last_run" not in feature_columns:
        raise ValueError("Layoff buckets require recent_days_since_last_run")
    days = raw[:, feature_columns.index("recent_days_since_last_run")]
    missing = ~np.isfinite(days)
    columns = []
    thresholds = dict(LAYOFF_BUCKETS)
    for name in names:
        if name == LAYOFF_MISSING_FEATURE:
            columns.append(missing.astype(np.float32))
        elif name == "recent_days_30_59":
            columns.append((~missing & (days >= 30) & (days < 60)).astype(np.float32))
        elif name == "recent_days_60_89":
            columns.append((~missing & (days >= 60) & (days < 90)).astype(np.float32))
        elif name == "recent_days_90_179":
            columns.append((~missing & (days >= 90) & (days < 180)).astype(np.float32))
        elif name == "recent_days_180_364":
            columns.append((~missing & (days >= 180) & (days < 365)).astype(np.float32))
        elif name == "recent_days_365_plus":
            columns.append((~missing & (days >= 365)).astype(np.float32))
        else:
            threshold = thresholds[name]
            columns.append((~missing & (days >= threshold)).astype(np.float32))
    return np.column_stack(columns).astype(np.float32)


def transform_raceformer(
    raw: np.ndarray,
    race_ids: np.ndarray,
    feature_columns: Sequence[str],
    zero_features: Sequence[str],
    contract: dict[str, Any] | None,
    *,
    legacy_median: np.ndarray | None = None,
    legacy_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a saved v2 contract or the exact legacy v1 transformation."""
    if not contract or int(contract.get("version", 1)) < 2:
        if legacy_median is None or legacy_scale is None:
            raise ValueError("Legacy RaceFormer preprocessing statistics are missing")
        return zero_feature_columns(
            transform(raw, legacy_median, legacy_scale),
            list(feature_columns), list(zero_features),
        )

    log1p_features = list(contract["log1p_features"])
    relative_features = list(contract["relative_features"])
    base = transform(
        _base_values(raw, feature_columns, log1p_features),
        np.asarray(contract["median"], dtype=np.float32),
        np.asarray(contract["scale"], dtype=np.float32),
    )
    base = np.clip(base, -float(contract["clip"]), float(contract["clip"]))
    base = zero_feature_columns(base, list(feature_columns), list(zero_features))
    relative = race_percentiles(raw, race_ids, feature_columns, relative_features)
    zeroed = set(zero_features)
    for index, feature in enumerate(relative_features):
        if feature in zeroed:
            relative[:, index] = 0.0
    layoff = layoff_bucket_values(raw, feature_columns, contract)
    return np.concatenate((base, relative, layoff), axis=1).astype(np.float32)


def raceformer_base_diagnostics(
    raw: np.ndarray,
    feature_columns: Sequence[str],
    contract: dict[str, Any] | None,
    *,
    legacy_median: np.ndarray | None = None,
    legacy_scale: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Expose each base-feature preprocessing stage for debugging."""
    if contract and int(contract.get("version", 1)) >= 2:
        transformed = _base_values(
            raw, feature_columns, list(contract["log1p_features"])
        )
        median = np.asarray(contract["median"], dtype=np.float32)
        scale = np.asarray(contract["scale"], dtype=np.float32)
        clip = float(contract["clip"])
    else:
        if legacy_median is None or legacy_scale is None:
            raise ValueError("Legacy RaceFormer preprocessing statistics are missing")
        transformed = np.asarray(raw, dtype=np.float32).copy()
        median = np.asarray(legacy_median, dtype=np.float32)
        scale = np.asarray(legacy_scale, dtype=np.float32)
        clip = float("inf")
    missing = np.isnan(transformed)
    filled = np.where(missing, median, transformed)
    unclipped = ((filled - median) / scale).astype(np.float32)
    clipped = np.clip(unclipped, -clip, clip).astype(np.float32)
    return {
        "raw": np.asarray(raw, dtype=np.float32),
        "transformed": transformed,
        "training_median": median,
        "training_scale": scale,
        "unclipped_standardized": unclipped,
        "clipped_standardized": clipped,
        "was_missing": missing,
        "was_clipped": np.abs(unclipped) > clip,
        "clip": np.asarray(clip, dtype=np.float32),
    }
