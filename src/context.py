"""TabFM training context helpers."""

from __future__ import annotations

import json
from pathlib import Path


def load_context_race_ids(path: Path) -> list[int]:
    """Load and validate ordered context race IDs from a JSON manifest."""
    payload = json.loads(path.read_text())
    race_ids = payload.get("race_ids")
    if not isinstance(race_ids, list) or not race_ids:
        raise ValueError(f"{path} must contain a non-empty 'race_ids' list")
    if any(not isinstance(race_id, int) or race_id <= 0 for race_id in race_ids):
        raise ValueError(f"{path} contains an invalid race_id")
    if len(race_ids) != len(set(race_ids)):
        raise ValueError(f"{path} contains duplicate race_ids")
    return race_ids


def context_race_id_changes(
    source_race_ids: list[int], next_race_ids: list[int]
) -> tuple[list[int], list[int]]:
    """Return context races added to and removed from a fine-tune run."""
    source = set(map(int, source_race_ids))
    next_context = set(map(int, next_race_ids))
    return sorted(next_context - source), sorted(source - next_context)
