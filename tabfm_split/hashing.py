"""Canonical JSON and domain-separated SHA-256 helpers for split-v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Return one stable UTF-8 JSON representation for JSON-compatible data."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def domain_hash(domain: str, value: Any) -> str:
    if not domain or "\x00" in domain:
        raise ValueError("Hash domain must be a non-empty string without NUL")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def hash_eligibility_policy(canonical_policy: Mapping[str, Any]) -> str:
    return domain_hash("tabfm_split_v2/eligibility_policy/v1", canonical_policy)


def hash_eligible_race_ids(ordered_race_ids: Sequence[int]) -> str:
    race_ids = [int(race_id) for race_id in ordered_race_ids]
    if len(race_ids) != len(set(race_ids)):
        raise ValueError("Eligible race IDs must be unique")
    return domain_hash("tabfm_split_v2/eligible_race_ids/v1", race_ids)


def hash_dataset(
    canonical_policy: Mapping[str, Any],
    ordered_race_records: Sequence[Mapping[str, Any]],
) -> str:
    """Hash policy plus ordered race-level records, not mutable runner row order."""

    return domain_hash(
        "tabfm_split_v2/dataset/v1",
        {
            "eligibility_policy": dict(canonical_policy),
            "eligible_races": [dict(record) for record in ordered_race_records],
        },
    )
