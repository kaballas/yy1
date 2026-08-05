"""TabFM training dataset helpers."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def load_feature_columns(path: Path) -> list[str]:
    """Load and validate the ordered feature manifest."""
    payload = json.loads(path.read_text())
    features = payload.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"{path} must contain a non-empty 'features' list")
    if any(not isinstance(column, str) or not column for column in features):
        raise ValueError(f"{path} contains an invalid feature name")
    if len(features) != len(set(features)):
        raise ValueError(f"{path} contains duplicate features")
    return features


def chronological_masks(
    race_ids: np.ndarray, times: np.ndarray, valid_frac: float
) -> tuple[np.ndarray, np.ndarray, str]:
    """Build chronological training and validation masks by whole race."""
    races = {}
    for race_id, time in zip(race_ids, times):
        races[int(race_id)] = time
    ordered = sorted(races, key=lambda race_id: (races[race_id], race_id))
    split = min(max(1, round(len(ordered) * (1.0 - valid_frac))), len(ordered) - 1)
    valid_races = set(ordered[split:])
    valid = np.fromiter((int(race_id) in valid_races for race_id in race_ids), dtype=bool)
    return ~valid, valid, races[ordered[split]].isoformat()
