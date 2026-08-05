"""Fail-closed validation for a materialized split-v2 training snapshot."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_and_validate_runtime_manifest(path: Path, db_path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split_version") != "tabfm_split_v2":
        raise ValueError("split manifest is not tabfm_split_v2")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != {"training", "development", "final_holdout"}:
        raise ValueError("split manifest must contain training, development and final_holdout partitions")
    sets = {name: {int(value) for value in values} for name, values in partitions.items()}
    if any(not values for values in sets.values()):
        raise ValueError("split-v2 partitions cannot be empty")
    if sets["training"] & sets["development"] or sets["training"] & sets["final_holdout"] or sets["development"] & sets["final_holdout"]:
        raise ValueError("split-v2 partitions overlap")
    context = {int(value) for value in payload.get("fixed_context_race_ids", [])}
    if not context or not context <= sets["training"]:
        raise ValueError("fixed context must be a non-empty subset of training")
    expected = sets["training"] | sets["development"] | sets["final_holdout"]
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT race_id, MIN(start_time_iso), MAX(start_time_iso) "
            "FROM race_runners WHERE race_id IN (%s) GROUP BY race_id"
            % ",".join("?" for _ in sorted(expected)),
            sorted(expected),
        ).fetchall()
    finally:
        connection.close()
    times = {int(row[0]): (row[1], row[2]) for row in rows}
    if set(times) != expected or any(start is None or start != end for start, end in times.values()):
        raise ValueError("split-v2 manifest does not resolve to complete, timestamped races")
    latest_training = max(times[race_id][0] for race_id in sets["training"])
    earliest_development = min(times[race_id][0] for race_id in sets["development"])
    earliest_holdout = min(times[race_id][0] for race_id in sets["final_holdout"])
    if not latest_training < earliest_development < earliest_holdout:
        raise ValueError("split-v2 partitions are not strictly chronological")
    return payload | {"manifest_sha256": manifest_sha256(path)}
