"""Authoritative race-level eligibility extraction for TabFM split-v2."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import hash_dataset, hash_eligibility_policy, hash_eligible_race_ids


DEFAULT_SNAPSHOT_CUTOFF = "2026-07-27T00:00:00+10:00"


def canonical_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Timestamp must be a non-empty ISO-8601 string")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timestamp must include a UTC offset: {value!r}")
    utc = parsed.astimezone(timezone.utc)
    return utc.isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class EligibilityPolicy:
    minimum_race_number: int = 6
    minimum_runner_count: int = 5
    requires_complete_runner_rows: bool = True
    requires_complete_labels: bool = True
    requires_race_timestamp: bool = True
    excluded_race_types: tuple[str, ...] = ()
    database_snapshot_cutoff: str = DEFAULT_SNAPSHOT_CUTOFF

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_race_number, int) or isinstance(
            self.minimum_race_number, bool
        ) or self.minimum_race_number < 1:
            raise ValueError("minimum_race_number must be a positive integer")
        if not isinstance(self.minimum_runner_count, int) or isinstance(
            self.minimum_runner_count, bool
        ) or self.minimum_runner_count < 1:
            raise ValueError("minimum_runner_count must be a positive integer")
        for field_name in (
            "requires_complete_runner_rows",
            "requires_complete_labels",
            "requires_race_timestamp",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if not isinstance(self.excluded_race_types, tuple) or any(
            not isinstance(value, str) for value in self.excluded_race_types
        ):
            raise ValueError("excluded_race_types must contain strings")
        canonical_timestamp(self.database_snapshot_cutoff)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "EligibilityPolicy":
        expected = {
            "minimum_race_number",
            "minimum_runner_count",
            "requires_complete_runner_rows",
            "requires_complete_labels",
            "requires_race_timestamp",
            "excluded_race_types",
            "database_snapshot_cutoff",
        }
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"Unknown eligibility policy fields: {sorted(unknown)}")
        values = dict(payload)
        if "excluded_race_types" in values:
            raw_types = values["excluded_race_types"]
            if not isinstance(raw_types, (list, tuple)):
                raise ValueError("excluded_race_types must be a list")
            values["excluded_race_types"] = tuple(raw_types)
        return cls(**values)

    def canonical_dict(self) -> dict[str, Any]:
        excluded = sorted(
            {
                str(race_type).strip()
                for race_type in self.excluded_race_types
                if str(race_type).strip()
            }
        )
        return {
            "database_snapshot_cutoff": canonical_timestamp(
                self.database_snapshot_cutoff
            ),
            "excluded_race_types": excluded,
            "minimum_race_number": int(self.minimum_race_number),
            "minimum_runner_count": int(self.minimum_runner_count),
            "requires_complete_labels": bool(self.requires_complete_labels),
            "requires_complete_runner_rows": bool(
                self.requires_complete_runner_rows
            ),
            "requires_race_timestamp": bool(self.requires_race_timestamp),
        }


@dataclass(frozen=True)
class RaceEligibilityRecord:
    race_id: int
    race_time: str
    race_number: int
    meeting_id: int | None
    race_type: str | None
    runner_count: int

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "race_id": self.race_id,
            "race_number": self.race_number,
            "race_time": self.race_time,
            "race_type": self.race_type,
            "runner_count": self.runner_count,
        }


@dataclass(frozen=True)
class EligibilityResult:
    policy: EligibilityPolicy
    records: tuple[RaceEligibilityRecord, ...]
    eligibility_policy_hash: str
    eligible_race_ids_hash: str
    dataset_hash: str

    @property
    def ordered_race_ids(self) -> tuple[int, ...]:
        return tuple(record.race_id for record in self.records)


_REQUIRED_COLUMNS = {
    "race_id",
    "race_number",
    "start_time_iso",
    "competition_id",
    "class_name",
    "selection_id",
    "runner_number",
    "is_trainable",
    "top3_mask",
}


_RACE_LEVEL_SQL = """
SELECT
    race_id,
    MIN(race_number) AS min_race_number,
    MAX(race_number) AS max_race_number,
    MIN(start_time_iso) AS min_start_time,
    MAX(start_time_iso) AS max_start_time,
    MIN(competition_id) AS min_meeting_id,
    MAX(competition_id) AS max_meeting_id,
    MIN(class_name) AS min_race_type,
    MAX(class_name) AS max_race_type,
    COUNT(*) AS runner_count,
    COUNT(selection_id) AS non_null_selection_count,
    COUNT(DISTINCT selection_id) AS unique_selection_count,
    COUNT(runner_number) AS non_null_runner_number_count,
    COUNT(DISTINCT runner_number) AS unique_runner_number_count,
    MIN(is_trainable) AS min_is_trainable,
    MAX(is_trainable) AS max_is_trainable,
    COUNT(top3_mask) AS labelled_count,
    SUM(CASE WHEN top3_mask = 1 THEN 1 ELSE 0 END) AS top3_count,
    SUM(CASE WHEN top3_mask = 0 THEN 1 ELSE 0 END) AS non_top3_count,
    SUM(CASE WHEN top3_mask IS NOT NULL AND top3_mask NOT IN (0, 1)
             THEN 1 ELSE 0 END) AS invalid_label_count
FROM race_runners
WHERE race_id IS NOT NULL
GROUP BY race_id
"""


def _table_columns(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute('PRAGMA table_info("race_runners")').fetchall()
    }


def _consistent(minimum: Any, maximum: Any) -> bool:
    return minimum is not None and minimum == maximum


def _eligible_record(
    row: sqlite3.Row,
    policy: EligibilityPolicy,
    cutoff: datetime,
) -> RaceEligibilityRecord | None:
    runner_count = int(row["runner_count"])
    if runner_count < policy.minimum_runner_count:
        return None
    if not _consistent(row["min_race_number"], row["max_race_number"]):
        return None
    race_number = int(row["min_race_number"])
    if race_number < policy.minimum_race_number:
        return None

    timestamp_consistent = _consistent(
        row["min_start_time"], row["max_start_time"]
    )
    if policy.requires_race_timestamp and not timestamp_consistent:
        return None
    if not timestamp_consistent:
        return None
    try:
        race_time = canonical_timestamp(str(row["min_start_time"]))
        parsed_time = datetime.fromisoformat(race_time.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed_time >= cutoff:
        return None

    if not _consistent(row["min_meeting_id"], row["max_meeting_id"]):
        return None
    meeting_id = int(row["min_meeting_id"])
    race_type = (
        str(row["min_race_type"])
        if _consistent(row["min_race_type"], row["max_race_type"])
        else None
    )
    if policy.requires_complete_runner_rows and race_type is None:
        return None
    excluded_types = set(policy.canonical_dict()["excluded_race_types"])
    if race_type in excluded_types:
        return None

    if policy.requires_complete_runner_rows:
        complete_identifiers = (
            int(row["non_null_selection_count"]) == runner_count
            and int(row["unique_selection_count"]) == runner_count
            and int(row["non_null_runner_number_count"]) == runner_count
            and int(row["unique_runner_number_count"]) == runner_count
        )
        completely_trainable = (
            row["min_is_trainable"] == 1 and row["max_is_trainable"] == 1
        )
        if not complete_identifiers or not completely_trainable:
            return None

    if policy.requires_complete_labels:
        labels_complete = (
            int(row["labelled_count"]) == runner_count
            and int(row["invalid_label_count"]) == 0
            and int(row["top3_count"]) == 3
            and int(row["non_top3_count"]) == runner_count - 3
        )
        if not labels_complete:
            return None

    return RaceEligibilityRecord(
        race_id=int(row["race_id"]),
        race_time=race_time,
        race_number=race_number,
        meeting_id=meeting_id,
        race_type=race_type,
        runner_count=runner_count,
    )


def extract_eligible_races(
    db_path: Path | str,
    policy: EligibilityPolicy,
) -> EligibilityResult:
    """Execute one race-level aggregate query and return deterministic identity."""

    path = Path(db_path)
    uri = f"file:{path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        columns = _table_columns(connection)
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"race_runners is missing required columns: {missing}")
        # This is the single data query. Eligibility is applied to its race-level rows.
        aggregate_rows = connection.execute(_RACE_LEVEL_SQL).fetchall()

    cutoff_text = policy.canonical_dict()["database_snapshot_cutoff"]
    cutoff = datetime.fromisoformat(cutoff_text.replace("Z", "+00:00"))
    records = [
        record
        for row in aggregate_rows
        if (record := _eligible_record(row, policy, cutoff)) is not None
    ]
    records.sort(key=lambda record: (record.race_time, record.race_id))

    canonical_policy = policy.canonical_dict()
    record_payloads = [record.canonical_dict() for record in records]
    ordered_race_ids = [record.race_id for record in records]
    return EligibilityResult(
        policy=policy,
        records=tuple(records),
        eligibility_policy_hash=hash_eligibility_policy(canonical_policy),
        eligible_race_ids_hash=hash_eligible_race_ids(ordered_race_ids),
        dataset_hash=hash_dataset(canonical_policy, record_payloads),
    )
