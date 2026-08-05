"""TabFM training preprocessing helpers."""

from __future__ import annotations

import numpy as np


def fit_preprocessor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the original median and robust-scale preprocessing statistics."""
    median = np.nanmedian(x, axis=0).astype(np.float32)
    median = np.nan_to_num(median)
    filled = np.where(np.isnan(x), median, x)
    scale = np.std(filled, axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return median, scale


def transform(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply fitted median imputation and scaling."""
    return ((np.where(np.isnan(x), median, x) - median) / scale).astype(np.float32)


def zero_feature_columns(
    x: np.ndarray, feature_columns: list[str], zero_features: list[str]
) -> np.ndarray:
    """Neutralize selected standardized feature columns."""
    missing = sorted(set(zero_features) - set(feature_columns))
    if missing:
        raise ValueError(
            "Cannot zero features absent from the model: " + ", ".join(missing)
        )
    if zero_features:
        indices = [feature_columns.index(column) for column in zero_features]
        x[:, indices] = 0.0
    return x
