"""TabFM split-v2 dataset identity primitives."""

from .eligibility import (
    EligibilityPolicy,
    EligibilityResult,
    RaceEligibilityRecord,
    extract_eligible_races,
)
from .manifest import build_eligibility_manifest_input

__all__ = [
    "EligibilityPolicy",
    "EligibilityResult",
    "RaceEligibilityRecord",
    "build_eligibility_manifest_input",
    "extract_eligible_races",
]
