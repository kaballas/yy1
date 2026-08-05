"""Manifest-input serialization for the split-v2 eligibility stage."""

from __future__ import annotations

from typing import Any

from .eligibility import EligibilityResult


ELIGIBILITY_SCHEMA_VERSION = "tabfm_eligibility_v1"


def build_eligibility_manifest_input(result: EligibilityResult) -> dict[str, Any]:
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "split_version": "tabfm_split_v2",
        "eligibility_policy": result.policy.canonical_dict(),
        "eligibility_policy_hash": result.eligibility_policy_hash,
        "eligible_race_count": len(result.records),
        "eligible_race_ids": list(result.ordered_race_ids),
        "eligible_race_ids_hash": result.eligible_race_ids_hash,
        "eligible_races": [record.canonical_dict() for record in result.records],
        "dataset_hash": result.dataset_hash,
    }
