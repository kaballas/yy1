"""Fail-closed internal validation for eligibility manifest inputs."""

from __future__ import annotations

from typing import Any

from .hashing import hash_dataset, hash_eligibility_policy, hash_eligible_race_ids
from .manifest import ELIGIBILITY_SCHEMA_VERSION


def validate_eligibility_manifest_input(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION:
        raise ValueError("Unsupported eligibility manifest schema_version")
    if payload.get("split_version") != "tabfm_split_v2":
        raise ValueError("Eligibility manifest must use tabfm_split_v2")

    policy = payload.get("eligibility_policy")
    race_ids = payload.get("eligible_race_ids")
    records = payload.get("eligible_races")
    if not isinstance(policy, dict) or not isinstance(race_ids, list) or not isinstance(records, list):
        raise ValueError("Eligibility manifest is missing policy, race IDs, or records")
    if payload.get("eligible_race_count") != len(race_ids) or len(records) != len(race_ids):
        raise ValueError("Eligibility manifest race counts differ")
    if [record.get("race_id") for record in records] != race_ids:
        raise ValueError("Eligibility records and ordered race IDs differ")

    expected = {
        "eligibility_policy_hash": hash_eligibility_policy(policy),
        "eligible_race_ids_hash": hash_eligible_race_ids(race_ids),
        "dataset_hash": hash_dataset(policy, records),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Eligibility manifest {field} mismatch")
