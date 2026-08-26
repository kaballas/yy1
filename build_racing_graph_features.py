#!/usr/bin/env python3
"""Build leakage-safe chronological node2vec features for race runners.

The graph for a snapshot contains only finished, active runner records whose
start time is strictly earlier than the snapshot. Target rows are assigned the
latest snapshot strictly earlier than their own start time. Current target-race
relationships are never added to the graph used to score that target.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from src.advanced_racing_features import race_relative_runner_mask


ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "db" / "race_runners.sqlite"
DEFAULT_TABLE = "graph_features"

IDENTITY_COLUMNS = (
    "runner_name",
    "runner_country",
    "jockey",
    "trainer",
    "sire",
    "dam",
    "competition_name",
    "distance_m",
    "track_status",
    *(f"recent_{slot}_{name}" for slot in range(1, 7) for name in (
        "date", "class", "jockey", "track_name", "distance_m", "place",
        "track_status",
    )),
)

RECENT_RUN_SLOTS = tuple(range(1, 7))

NODE_COLUMNS = (
    "graph_horse_node",
    "graph_jockey_node",
    "graph_trainer_node",
    "graph_sire_node",
    "graph_dam_node",
    "graph_venue_node",
    "graph_distance_node",
    "graph_track_status_node",
    *(f"graph_recent_{slot}_{name}_node" for slot in RECENT_RUN_SLOTS for name in (
        "run", "jockey", "venue", "distance", "place", "track_status", "class",
    )),
)

SIMILARITY_PAIRS = {
    "graph_horse_jockey_similarity": (
        "graph_horse_node", "graph_jockey_node"
    ),
    "graph_horse_trainer_similarity": (
        "graph_horse_node", "graph_trainer_node"
    ),
    "graph_horse_sire_similarity": ("graph_horse_node", "graph_sire_node"),
    "graph_horse_dam_similarity": ("graph_horse_node", "graph_dam_node"),
    "graph_horse_venue_similarity": ("graph_horse_node", "graph_venue_node"),
    "graph_jockey_trainer_similarity": (
        "graph_jockey_node", "graph_trainer_node"
    ),
    "graph_sire_trainer_similarity": (
        "graph_sire_node", "graph_trainer_node"
    ),
    "graph_sire_jockey_similarity": (
        "graph_sire_node", "graph_jockey_node"
    ),
    "graph_dam_trainer_similarity": (
        "graph_dam_node", "graph_trainer_node"
    ),
    "graph_trainer_venue_similarity": (
        "graph_trainer_node", "graph_venue_node"
    ),
    "graph_jockey_venue_similarity": (
        "graph_jockey_node", "graph_venue_node"
    ),
    "graph_horse_distance_similarity": (
        "graph_horse_node", "graph_distance_node"
    ),
    "graph_horse_track_status_similarity": (
        "graph_horse_node", "graph_track_status_node"
    ),
    "graph_jockey_distance_similarity": (
        "graph_jockey_node", "graph_distance_node"
    ),
    "graph_trainer_distance_similarity": (
        "graph_trainer_node", "graph_distance_node"
    ),
    "graph_jockey_track_status_similarity": (
        "graph_jockey_node", "graph_track_status_node"
    ),
    "graph_trainer_track_status_similarity": (
        "graph_trainer_node", "graph_track_status_node"
    ),
    "graph_horse_recent_run_similarity": (
        "graph_horse_node", "graph_recent_1_run_node"
    ),
    "graph_jockey_recent_jockey_similarity": (
        "graph_jockey_node", "graph_recent_1_jockey_node"
    ),
    "graph_venue_recent_venue_similarity": (
        "graph_venue_node", "graph_recent_1_venue_node"
    ),
    "graph_distance_recent_distance_similarity": (
        "graph_distance_node", "graph_recent_1_distance_node"
    ),
    "graph_track_status_recent_track_status_similarity": (
        "graph_track_status_node", "graph_recent_1_track_status_node"
    ),
    "graph_horse_recent_class_similarity": (
        "graph_horse_node", "graph_recent_1_class_node"
    ),
    "graph_horse_recent_place_similarity": (
        "graph_horse_node", "graph_recent_1_place_node"
    ),
}

# Experiment 1 deliberately gives node2vec only the primitive relationships.
# The additional sire/trainer/jockey/venue cosine features must reflect learned
# multi-hop neighborhoods rather than direct edges inserted for the target pair.
REPEATED_GRAPH_EDGE_PAIRS = (
    ("graph_horse_node", "graph_jockey_node"),
    ("graph_horse_node", "graph_trainer_node"),
    ("graph_horse_node", "graph_venue_node"),
    ("graph_jockey_node", "graph_trainer_node"),
)

# Pedigree is an invariant relationship, not a count of historical runs. These
# edges occur once and deliberately do not receive temporal decay.
STATIC_GRAPH_EDGE_PAIRS = (
    ("graph_horse_node", "graph_sire_node"),
    ("graph_horse_node", "graph_dam_node"),
)

RECENT_RUN_CONTEXT_NODE_NAMES = (
    "jockey", "venue", "distance", "place", "track_status", "class",
)

CURRENT_RUN_CONTEXT_NODE_COLUMNS = (
    "graph_horse_node",
    "graph_jockey_node",
    "graph_venue_node",
    "graph_distance_node",
    "graph_track_status_node",
)

AVAILABILITY_FEATURES = {
    "graph_horse_embedding_available": "graph_horse_node",
    "graph_jockey_embedding_available": "graph_jockey_node",
    "graph_trainer_embedding_available": "graph_trainer_node",
    "graph_sire_embedding_available": "graph_sire_node",
    "graph_dam_embedding_available": "graph_dam_node",
    "graph_venue_embedding_available": "graph_venue_node",
    "graph_distance_embedding_available": "graph_distance_node",
    "graph_track_status_embedding_available": "graph_track_status_node",
    "graph_recent_run_embedding_available": "graph_recent_1_run_node",
    "graph_recent_jockey_embedding_available": "graph_recent_1_jockey_node",
    "graph_recent_venue_embedding_available": "graph_recent_1_venue_node",
    "graph_recent_distance_embedding_available": "graph_recent_1_distance_node",
    "graph_recent_place_embedding_available": "graph_recent_1_place_node",
    "graph_recent_track_status_embedding_available": "graph_recent_1_track_status_node",
    "graph_recent_class_embedding_available": "graph_recent_1_class_node",
}

RACE_RELATIVE_SUFFIXES = ("rank_in_race", "minus_race_mean")
GRAPH_FEATURE_NAMES = (
    *SIMILARITY_PAIRS,
    *(f"{name}_{suffix}" for name in SIMILARITY_PAIRS for suffix in RACE_RELATIVE_SUFFIXES),
    *AVAILABILITY_FEATURES,
)

VENUE_ALIASES = {
    "belmont": "belmont park",
    "darwin": "fannie bay",
    "alice springs": "pioneer park",
    "devonport": "devonport synthetic",
    "riccarton": "riccarton park",
    "murray bridge": "murray bridge gh",
}

RACE_TIMEZONES = {
    "australia": "Australia/Sydney",
    "new zealand": "Pacific/Auckland",
}


@dataclass(frozen=True)
class SnapshotAssignment:
    timestamp: pd.Timestamp
    target_indices: np.ndarray


@dataclass(frozen=True)
class GraphSummary:
    nodes: int
    edges: int
    history_rows: int


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def validate_identifier(value: str, description: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe {description}: {value!r}")
    return value


def normalize_identity(value: object) -> str | None:
    """Return a stable Unicode/case/whitespace-normalized identity."""
    if value is None or pd.isna(value):
        return None
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = re.sub(r"\s+", " ", normalized.strip()).casefold()
    return normalized or None


def typed_node(kind: str, value: object) -> str | None:
    normalized = normalize_identity(value)
    return f"{kind}:{normalized}" if normalized is not None else None


def distance_node(value: object) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return f"distance:{int(round(numeric))}m"


def place_node(value: object) -> str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return f"place:{int(round(numeric))}"


def track_status_node(value: object) -> str | None:
    normalized = normalize_identity(value)
    if normalized is None:
        return None
    match = re.search(
        r"\b(good|soft|heavy|firm|synthetic|yielding|dead|slow|fast|sloppy)\b",
        normalized,
    )
    going = match.group(1) if match else normalized
    return f"track_status:{going}"


def venue_node(value: object) -> str | None:
    normalized = normalize_identity(value)
    if normalized is None:
        return None
    normalized = VENUE_ALIASES.get(normalized, normalized)
    return f"venue:{normalized}"


def parse_recent_dates(values: pd.Series) -> pd.Series:
    """Parse feed DD/MM/YYYY dates while preserving explicit ISO dates."""
    text = values.astype("string").str.strip()
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)", na=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    parsed.loc[iso] = pd.to_datetime(
        text.loc[iso], format="mixed", utc=True, errors="coerce"
    )
    parsed.loc[~iso] = pd.to_datetime(
        text.loc[~iso], format="mixed", dayfirst=True, utc=True, errors="coerce"
    )
    return parsed


def recent_run_node(
    horse_node: object,
    date_value: object,
    venue: object,
    distance: object,
) -> str | None:
    if horse_node is None or pd.isna(horse_node):
        return None
    if isinstance(date_value, pd.Timestamp):
        date = date_value
    else:
        date = parse_recent_dates(pd.Series([date_value])).iloc[0]
    if pd.isna(date):
        return None
    venue_value = str(venue) if venue is not None and not pd.isna(venue) else "unknown"
    distance_value = (
        str(distance) if distance is not None and not pd.isna(distance) else "unknown"
    )
    return f"run:{horse_node}|{date.date().isoformat()}|{venue_value}|{distance_value}"


def local_race_time(value: pd.Timestamp, country: object) -> pd.Timestamp:
    """Return a race timestamp in the feed's local calendar-date timezone."""
    timezone = RACE_TIMEZONES.get(normalize_identity(country))
    return value.tz_convert(timezone) if timezone is not None else value


def add_node_identities(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach typed graph node IDs without mutating source identity columns."""
    result = frame.copy()
    country = result["runner_country"].map(normalize_identity).fillna("unknown")
    horse = result["runner_name"].map(normalize_identity)
    result["graph_horse_node"] = np.where(
        horse.notna(), "horse:" + country + ":" + horse.fillna(""), None
    )
    for node_type, column in (
        ("jockey", "jockey"),
        ("trainer", "trainer"),
        ("sire", "sire"),
        ("dam", "dam"),
    ):
        result[f"graph_{node_type}_node"] = result[column].map(
            lambda value, kind=node_type: typed_node(kind, value)
        )
    result["graph_venue_node"] = result["competition_name"].map(venue_node)
    result["graph_distance_node"] = result["distance_m"].map(distance_node)
    result["graph_track_status_node"] = result["track_status"].map(track_status_node)
    for slot in RECENT_RUN_SLOTS:
        prefix = f"recent_{slot}"
        graph_prefix = f"graph_recent_{slot}"
        result[f"_recent_{slot}_time"] = parse_recent_dates(
            result[f"{prefix}_date"]
        )
        result[f"{graph_prefix}_jockey_node"] = result[f"{prefix}_jockey"].map(
            lambda value: typed_node("jockey", value)
        )
        result[f"{graph_prefix}_venue_node"] = result[f"{prefix}_track_name"].map(
            venue_node
        )
        result[f"{graph_prefix}_distance_node"] = result[
            f"{prefix}_distance_m"
        ].map(distance_node)
        result[f"{graph_prefix}_place_node"] = result[f"{prefix}_place"].map(
            place_node
        )
        result[f"{graph_prefix}_track_status_node"] = result[
            f"{prefix}_track_status"
        ].map(track_status_node)
        result[f"{graph_prefix}_class_node"] = result[f"{prefix}_class"].map(
            lambda value: typed_node("class", value)
        )
        event_dates = result[f"_recent_{slot}_time"]
        event_venues = result[f"{graph_prefix}_venue_node"].fillna("unknown")
        event_distances = result[f"{graph_prefix}_distance_node"].fillna("unknown")
        valid_event = result["graph_horse_node"].notna() & event_dates.notna()
        run_nodes = (
            "run:"
            + result["graph_horse_node"].fillna("").astype(str)
            + "|"
            + event_dates.dt.strftime("%Y-%m-%d").fillna("")
            + "|"
            + event_venues.astype(str)
            + "|"
            + event_distances.astype(str)
        )
        result[f"{graph_prefix}_run_node"] = run_nodes.where(valid_event, None)
    return result


def parse_utc(value: str, description: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    if pd.isna(parsed):
        raise ValueError(f"Invalid {description}: {value!r}")
    return parsed


def snapshot_boundaries(
    target_times: pd.Series,
    frequency: str,
    explicit_snapshots: Sequence[str] = (),
) -> list[pd.Timestamp]:
    """Return ordered snapshot timestamps covering the target population."""
    if explicit_snapshots:
        snapshots = sorted({parse_utc(value, "snapshot") for value in explicit_snapshots})
        return snapshots
    if target_times.empty:
        return []
    earliest = target_times.min()
    latest = target_times.max()
    if frequency == "monthly":
        start = earliest.tz_localize(None).to_period("M").start_time.tz_localize("UTC")
        if earliest == start:
            start -= pd.offsets.MonthBegin(1)
        values = pd.date_range(start, latest, freq="MS", tz="UTC")
    elif frequency == "weekly":
        start = earliest.normalize() - pd.Timedelta(days=earliest.weekday())
        if earliest == start:
            start -= pd.Timedelta(days=7)
        values = pd.date_range(start, latest, freq="7D", tz="UTC")
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unsupported snapshot frequency: {frequency}")
    return list(values)


def assign_snapshots_strictly_before(
    target_times: pd.Series, snapshots: Sequence[pd.Timestamp]
) -> tuple[list[SnapshotAssignment], np.ndarray]:
    """Assign each target the latest snapshot strictly before its start time."""
    if not snapshots:
        return [], np.arange(len(target_times), dtype=np.int64)
    snapshot_ns = np.array([value.value for value in snapshots], dtype=np.int64)
    # Pandas 3 can preserve second/microsecond input resolution while
    # Timestamp.value is always nanoseconds. Normalize both sides explicitly.
    time_ns = (
        target_times.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    )
    positions = np.searchsorted(snapshot_ns, time_ns, side="left") - 1
    unassigned = np.flatnonzero(positions < 0)
    assignments = [
        SnapshotAssignment(
            timestamp=snapshot,
            target_indices=np.flatnonzero(positions == position),
        )
        for position, snapshot in enumerate(snapshots)
        if np.any(positions == position)
    ]
    return assignments, unassigned


def load_race_rows(database: Path, table: str) -> pd.DataFrame:
    table = validate_identifier(table, "source table")
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    columns = (
        "rowid AS source_rowid",
        "race_id",
        "runner_number",
        "runner_name",
        "runner_country",
        "jockey",
        "trainer",
        "sire",
        "dam",
        "competition_name",
        "country",
        "competition_id",
        "start_time_iso",
        "status",
        "runner_mask",
        "source_betting_status",
        "active_field_size",
        "career_starts",
        "distance_m",
        "track_status",
        *(f"recent_{slot}_{name}" for slot in RECENT_RUN_SLOTS for name in (
            "date", "class", "jockey", "track_name", "distance_m", "place",
            "track_status",
        )),
    )
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        schema = {
            str(row[1])
            for row in connection.execute(
                f"PRAGMA table_info({quote_identifier(table)})"
            )
        }
        required = {
            value.split(" AS ")[0] for value in columns if value != "rowid AS source_rowid"
        }
        missing = sorted(required - schema)
        if missing:
            raise ValueError("Database is missing graph inputs: " + ", ".join(missing))
        selected = ", ".join(
            value if " AS " in value else quote_identifier(value) for value in columns
        )
        frame = pd.read_sql_query(
            f"SELECT {selected} FROM {quote_identifier(table)} "
            "ORDER BY start_time_iso, race_id, runner_number",
            connection,
        )
    times = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    if times.isna().any():
        examples = frame.loc[times.isna(), "start_time_iso"].head(3).tolist()
        raise ValueError(f"Invalid start_time_iso values: {examples}")
    frame["_start_time"] = times
    return add_node_identities(frame)


def filter_targets(
    frame: pd.DataFrame,
    competition_ids: Sequence[int],
    statuses: Sequence[str],
    target_from: pd.Timestamp | None,
    target_to: pd.Timestamp | None,
    limit_target_races: int | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if competition_ids:
        mask &= frame["competition_id"].isin(competition_ids)
    if statuses:
        wanted = {value.strip().casefold() for value in statuses}
        mask &= frame["status"].astype("string").str.strip().str.casefold().isin(wanted)
    if target_from is not None:
        mask &= frame["_start_time"].ge(target_from)
    if target_to is not None:
        mask &= frame["_start_time"].lt(target_to)
    target = frame.loc[mask].copy()
    if limit_target_races is not None:
        race_ids = target["race_id"].drop_duplicates().head(limit_target_races)
        target = target.loc[target["race_id"].isin(race_ids)].copy()
    return target.reset_index(drop=True)


def historical_rows(frame: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    status = frame["status"].astype("string").str.strip().str.casefold()
    active = pd.to_numeric(frame["runner_mask"], errors="coerce").eq(1)
    return frame.loc[status.eq("finished") & active & frame["_start_time"].lt(snapshot)]


def graph_edges(
    history: pd.DataFrame,
    snapshot: pd.Timestamp,
    half_life_days: float | None,
    edge_weight_mode: str = "raw",
) -> dict[tuple[str, str], float]:
    """Aggregate typed undirected relationship edges from historical rows."""
    if edge_weight_mode not in {"raw", "log", "binary"}:
        raise ValueError(f"Unknown edge weight mode: {edge_weight_mode}")
    repeated_edges: dict[tuple[str, str], float] = defaultdict(float)
    static_edges: set[tuple[str, str]] = set()
    run_event_edges: set[tuple[str, str]] = set()
    recent_date_columns = [f"_recent_{slot}_time" for slot in RECENT_RUN_SLOTS]
    selected_columns = [*NODE_COLUMNS, "country", *recent_date_columns]
    selected = history.loc[:, [*selected_columns, "_start_time"]]
    for row in selected.itertuples(index=False, name=None):
        values = dict(zip(selected_columns, row[:-1]))
        start_time = row[-1]
        weight = 1.0
        if half_life_days is not None:
            age_days = max(0.0, (snapshot - start_time).total_seconds() / 86400.0)
            weight = math.exp(-math.log(2.0) * age_days / half_life_days)
        for left_column, right_column in REPEATED_GRAPH_EDGE_PAIRS:
            left, right = values[left_column], values[right_column]
            if left is None or right is None or pd.isna(left) or pd.isna(right):
                continue
            key = (left, right) if left < right else (right, left)
            if key[0] != key[1]:
                repeated_edges[key] += weight

        for left_column, right_column in STATIC_GRAPH_EDGE_PAIRS:
            left, right = values[left_column], values[right_column]
            if left is None or right is None or pd.isna(left) or pd.isna(right):
                continue
            key = (left, right) if left < right else (right, left)
            if key[0] != key[1]:
                static_edges.add(key)

        # Materialize the row's own finished performance. Without this edge set,
        # a target's latest run can never already exist in a strictly earlier
        # graph: the target is the first later row that exposes it as recent_1.
        current_run_node = recent_run_node(
            values["graph_horse_node"],
            local_race_time(start_time, values["country"]),
            values["graph_venue_node"],
            values["graph_distance_node"],
        )
        if current_run_node is not None:
            for column in CURRENT_RUN_CONTEXT_NODE_COLUMNS:
                context_node = values[column]
                if context_node is None or pd.isna(context_node):
                    continue
                key = (
                    (current_run_node, context_node)
                    if current_run_node < context_node
                    else (context_node, current_run_node)
                )
                if key[0] != key[1]:
                    run_event_edges.add(key)

        # A historical performance can recur in recent_1..recent_6 across
        # several later database rows. The set keeps those facts deduplicated
        # while adding context not present on the performance's own row.
        for slot in RECENT_RUN_SLOTS:
            event_time = values[f"_recent_{slot}_time"]
            if pd.isna(event_time) or event_time >= start_time or event_time >= snapshot:
                continue
            run_node = values[f"graph_recent_{slot}_run_node"]
            if run_node is None or pd.isna(run_node):
                continue
            context_nodes = [values["graph_horse_node"]]
            context_nodes.extend(
                values[f"graph_recent_{slot}_{name}_node"]
                for name in RECENT_RUN_CONTEXT_NODE_NAMES
            )
            for context_node in context_nodes:
                if context_node is None or pd.isna(context_node):
                    continue
                key = (
                    (run_node, context_node)
                    if run_node < context_node
                    else (context_node, run_node)
                )
                if key[0] != key[1]:
                    run_event_edges.add(key)

    if edge_weight_mode == "log":
        edges = {key: math.log1p(weight) for key, weight in repeated_edges.items()}
    elif edge_weight_mode == "binary":
        edges = {key: 1.0 for key in repeated_edges}
    else:
        edges = dict(repeated_edges)
    # Pedigree remains exactly one regardless of repeated-edge mode or decay.
    edges.update({key: 1.0 for key in static_edges})
    # Run-context edges are deduplicated historical facts, not interaction counts.
    edges.update({key: 1.0 for key in run_event_edges})
    return edges


def repeated_edge_weight_statistics(
    edges: Mapping[tuple[str, str], float],
) -> dict[str, float | int]:
    weights = np.asarray(
        [
            weight
            for (left, right), weight in edges.items()
            if frozenset((left.split(":", 1)[0], right.split(":", 1)[0]))
            in {
                frozenset(("horse", "jockey")),
                frozenset(("horse", "trainer")),
                frozenset(("horse", "venue")),
                frozenset(("jockey", "trainer")),
            }
        ],
        dtype=np.float64,
    )
    if not len(weights):
        return {
            "count": 0, "min": math.nan, "median": math.nan,
            "mean": math.nan, "p95": math.nan, "max": math.nan,
        }
    return {
        "count": int(len(weights)),
        "min": float(weights.min()),
        "median": float(np.median(weights)),
        "mean": float(weights.mean()),
        "p95": float(np.quantile(weights, 0.95)),
        "max": float(weights.max()),
    }


def adjacency_from_edges(
    edges: Mapping[tuple[str, str], float]
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)
    for (left, right), weight in edges.items():
        if not math.isfinite(weight) or weight <= 0:
            continue
        adjacency[left][right] = adjacency[left].get(right, 0.0) + weight
        adjacency[right][left] = adjacency[right].get(left, 0.0) + weight
    return dict(adjacency)


def _weighted_choice(
    candidates: Sequence[str], weights: np.ndarray, rng: np.random.Generator
) -> str:
    total = float(weights.sum())
    probabilities = weights / total if total > 0 else None
    return candidates[int(rng.choice(len(candidates), p=probabilities))]


def first_order_samplers(
    adjacency: Mapping[str, Mapping[str, float]],
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """Cache exact weighted samplers for the p=q=1 node2vec special case."""
    samplers: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for node, neighbors in adjacency.items():
        # Mapping insertion order can originate in set-backed edge aggregation.
        # Canonical ordering makes --seed reproducible across Python processes.
        candidates = tuple(sorted(neighbors))
        weights = np.fromiter(
            (neighbors[candidate] for candidate in candidates), dtype=np.float64
        )
        samplers[node] = (candidates, np.cumsum(weights))
    return samplers


def _cached_weighted_choice(
    sampler: tuple[tuple[str, ...], np.ndarray], rng: np.random.Generator
) -> str:
    candidates, cumulative = sampler
    threshold = float(rng.random()) * float(cumulative[-1])
    position = int(np.searchsorted(cumulative, threshold, side="right"))
    return candidates[min(position, len(candidates) - 1)]


def biased_walk(
    adjacency: Mapping[str, Mapping[str, float]],
    start: str,
    walk_length: int,
    p: float,
    q: float,
    rng: np.random.Generator,
    cached_samplers: Mapping[
        str, tuple[tuple[str, ...], np.ndarray]
    ] | None = None,
) -> list[str]:
    """Generate one second-order node2vec walk."""
    walk = [start]
    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = adjacency.get(current, {})
        if not neighbors:
            break
        if cached_samplers is not None:
            walk.append(_cached_weighted_choice(cached_samplers[current], rng))
            continue
        candidates = tuple(sorted(neighbors))
        weights = np.fromiter((neighbors[node] for node in candidates), dtype=float)
        if len(walk) > 1:
            previous = walk[-2]
            previous_neighbors = adjacency.get(previous, {})
            bias = np.fromiter(
                (
                    1.0 / p
                    if node == previous
                    else 1.0
                    if node in previous_neighbors
                    else 1.0 / q
                    for node in candidates
                ),
                dtype=float,
            )
            weights *= bias
        walk.append(_weighted_choice(candidates, weights, rng))
    return walk


class WalkCorpus:
    """Re-iterable, bounded-memory node2vec walk corpus for Gensim."""

    def __init__(
        self,
        adjacency: Mapping[str, Mapping[str, float]],
        walk_length: int,
        walks_per_node: int,
        p: float,
        q: float,
        seed: int,
        progress_label: str = "node2vec",
        progress_every_nodes: int = 5000,
    ) -> None:
        self.adjacency = adjacency
        self.nodes = np.array(sorted(adjacency), dtype=object)
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.seed = seed
        self.progress_label = progress_label
        self.progress_every_nodes = progress_every_nodes
        self._iterations = 0
        self.cached_samplers: dict[
            str, tuple[tuple[str, ...], np.ndarray]
        ] | None = None
        if p == 1.0 and q == 1.0:
            started = time.perf_counter()
            print(
                f"{self.progress_label} sampler_cache started mode=p1_q1",
                flush=True,
            )
            self.cached_samplers = first_order_samplers(adjacency)
            print(
                f"{self.progress_label} sampler_cache complete "
                f"nodes={len(self.cached_samplers):,} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    @property
    def total_examples(self) -> int:
        return len(self.nodes) * self.walks_per_node

    def __iter__(self) -> Iterator[list[str]]:
        self._iterations += 1
        phase = "vocabulary" if self._iterations == 1 else f"training_pass_{self._iterations - 1}"
        pass_started = time.perf_counter()
        print(
            f"{self.progress_label} walks phase={phase} started "
            f"walks={self.total_examples:,} nodes={len(self.nodes):,}",
            flush=True,
        )
        rng = np.random.default_rng(self.seed)
        for walk_round in range(1, self.walks_per_node + 1):
            round_started = time.perf_counter()
            order = rng.permutation(self.nodes)
            for completed, node in enumerate(order, 1):
                yield biased_walk(
                    self.adjacency,
                    str(node),
                    self.walk_length,
                    self.p,
                    self.q,
                    rng,
                    self.cached_samplers,
                )
                if (
                    completed % self.progress_every_nodes == 0
                    or completed == len(self.nodes)
                ):
                    overall = (walk_round - 1) * len(self.nodes) + completed
                    print(
                        f"{self.progress_label} walks phase={phase} "
                        f"round={walk_round}/{self.walks_per_node} "
                        f"nodes={completed:,}/{len(self.nodes):,} "
                        f"progress={overall / self.total_examples:.1%} "
                        f"round_elapsed={time.perf_counter() - round_started:.1f}s",
                        flush=True,
                    )
        print(
            f"{self.progress_label} walks phase={phase} complete "
            f"elapsed={time.perf_counter() - pass_started:.1f}s",
            flush=True,
        )


def train_node2vec(
    adjacency: Mapping[str, Mapping[str, float]],
    dimensions: int,
    walk_length: int,
    walks_per_node: int,
    window: int,
    epochs: int,
    p: float,
    q: float,
    negative: int,
    workers: int,
    seed: int,
    progress_label: str = "node2vec",
    progress_every_nodes: int = 5000,
) -> dict[str, np.ndarray]:
    """Train skip-gram embeddings from biased walks and L2-normalize them."""
    if not adjacency:
        return {}
    try:
        from gensim.models import Word2Vec
        from gensim.models.callbacks import CallbackAny2Vec
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Gensim is required for node2vec training. Install it in the active "
            "environment with: pip install 'gensim>=4.4,<5'"
        ) from exc
    class EpochProgress(CallbackAny2Vec):
        def __init__(self) -> None:
            self.epoch = 0
            self.previous_loss = 0.0
            self.epoch_started = 0.0
            self.training_started = 0.0

        def on_train_begin(self, model: Word2Vec) -> None:
            self.training_started = time.perf_counter()
            print(
                f"{progress_label} training started epochs={epochs} "
                f"vocabulary={len(model.wv):,}",
                flush=True,
            )

        def on_epoch_begin(self, model: Word2Vec) -> None:
            self.epoch_started = time.perf_counter()
            print(
                f"{progress_label} epoch={self.epoch + 1}/{epochs} started",
                flush=True,
            )

        def on_epoch_end(self, model: Word2Vec) -> None:
            cumulative_loss = float(model.get_latest_training_loss())
            epoch_loss = cumulative_loss - self.previous_loss
            self.previous_loss = cumulative_loss
            print(
                f"{progress_label} epoch={self.epoch + 1}/{epochs} complete "
                f"epoch_loss={epoch_loss:.6g} cumulative_loss={cumulative_loss:.6g} "
                f"epoch_elapsed={time.perf_counter() - self.epoch_started:.1f}s "
                f"total_elapsed={time.perf_counter() - self.training_started:.1f}s",
                flush=True,
            )
            self.epoch += 1

    corpus = WalkCorpus(
        adjacency, walk_length, walks_per_node, p, q, seed,
        progress_label=progress_label,
        progress_every_nodes=progress_every_nodes,
    )
    print(
        f"{progress_label} vocabulary_build started dimensions={dimensions} "
        f"walk_length={walk_length} walks_per_node={walks_per_node}",
        flush=True,
    )
    vocabulary_started = time.perf_counter()
    model = Word2Vec(
        vector_size=dimensions,
        window=window,
        min_count=1,
        sg=1,
        negative=negative,
        workers=workers,
        seed=seed,
    )
    model.build_vocab(corpus_iterable=corpus)
    print(
        f"{progress_label} vocabulary_build complete vocabulary={len(model.wv):,} "
        f"elapsed={time.perf_counter() - vocabulary_started:.1f}s",
        flush=True,
    )
    model.train(
        corpus_iterable=corpus,
        total_examples=corpus.total_examples,
        epochs=epochs,
        compute_loss=True,
        callbacks=[EpochProgress()],
    )
    embeddings: dict[str, np.ndarray] = {}
    for node in model.wv.index_to_key:
        vector = np.asarray(model.wv[node], dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0 and math.isfinite(norm):
            embeddings[node] = vector / norm
    return embeddings


def cosine_or_nan(
    left: object, right: object, embeddings: Mapping[str, np.ndarray]
) -> float:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return math.nan
    left_vector = embeddings.get(str(left))
    right_vector = embeddings.get(str(right))
    if left_vector is None or right_vector is None:
        return math.nan
    return float(np.clip(np.dot(left_vector, right_vector), -1.0, 1.0))


def calculate_graph_features(
    target: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    result = pd.DataFrame(index=target.index)
    for feature, node_column in AVAILABILITY_FEATURES.items():
        result[feature] = target[node_column].map(
            lambda node: float(node is not None and not pd.isna(node) and str(node) in embeddings)
        )
    for feature, (left, right) in SIMILARITY_PAIRS.items():
        result[feature] = [
            cosine_or_nan(left_node, right_node, embeddings)
            for left_node, right_node in zip(target[left], target[right])
        ]

    eligible = race_relative_runner_mask(target)
    race_id = target["race_id"]
    for feature in SIMILARITY_PAIRS:
        values = pd.to_numeric(result[feature], errors="coerce").where(eligible)
        grouped = values.groupby(race_id, sort=False, dropna=False)
        result[f"{feature}_rank_in_race"] = grouped.rank(
            method="min", ascending=False
        )
        result[f"{feature}_minus_race_mean"] = values - grouped.transform("mean")
    return result.loc[:, GRAPH_FEATURE_NAMES].astype(np.float32)


def output_columns() -> tuple[str, ...]:
    return (
        "source_rowid",
        "race_id",
        "runner_number",
        "runner_name",
        "runner_country",
        "snapshot_date",
        *GRAPH_FEATURE_NAMES,
    )


def ensure_output_available(
    database: Path, output_table: str, replace: bool
) -> None:
    output_table = validate_identifier(output_table, "output table")
    with sqlite3.connect(database) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (output_table,),
        ).fetchone()
    if exists and not replace:
        raise ValueError(
            f"Output table {output_table!r} already exists; pass --replace to rebuild it"
        )


def write_graph_feature_table(
    database: Path,
    output_table: str,
    frame: pd.DataFrame,
    replace: bool,
) -> None:
    """Atomically install the separately keyed graph-feature table."""
    output_table = validate_identifier(output_table, "output table")
    temporary = validate_identifier(
        f"{output_table}_building_{os.getpid()}", "temporary table"
    )
    metadata_types = {
        "source_rowid": "INTEGER PRIMARY KEY",
        "race_id": "INTEGER NOT NULL",
        "runner_number": "INTEGER",
        "runner_name": "TEXT",
        "runner_country": "TEXT",
        "snapshot_date": "TEXT NOT NULL",
    }
    definitions = [
        f"{quote_identifier(column)} {metadata_types.get(column, 'REAL')}"
        for column in output_columns()
    ]
    values = frame.loc[:, output_columns()].to_numpy(dtype=object, copy=True)
    values[pd.isna(values)] = None
    uri = str(database.resolve())
    with sqlite3.connect(uri) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                f"CREATE TABLE {quote_identifier(temporary)} "
                f"({', '.join(definitions)})"
            )
            columns_sql = ", ".join(quote_identifier(name) for name in output_columns())
            placeholders = ", ".join("?" for _ in output_columns())
            connection.executemany(
                f"INSERT INTO {quote_identifier(temporary)} ({columns_sql}) "
                f"VALUES ({placeholders})",
                map(tuple, values),
            )
            connection.execute(
                f"CREATE INDEX {quote_identifier(temporary + '_race_idx')} "
                f"ON {quote_identifier(temporary)} (race_id, runner_number)"
            )
            connection.execute(
                f"CREATE INDEX {quote_identifier(temporary + '_snapshot_idx')} "
                f"ON {quote_identifier(temporary)} (snapshot_date)"
            )
            existing = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (output_table,),
            ).fetchone()
            if existing:
                if not replace:
                    raise ValueError(
                        f"Output table {output_table!r} already exists; pass --replace"
                    )
                connection.execute(f"DROP TABLE {quote_identifier(output_table)}")
            connection.execute(
                f"ALTER TABLE {quote_identifier(temporary)} "
                f"RENAME TO {quote_identifier(output_table)}"
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def validate_hyperparameters(args: argparse.Namespace) -> None:
    positive = (
        "dimensions",
        "walk_length",
        "walks_per_node",
        "window",
        "epochs",
        "negative",
        "workers",
        "progress_every_nodes",
        "p",
        "q",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.half_life_days is not None and args.half_life_days <= 0:
        raise ValueError("--half-life-days must be positive")
    if args.limit_target_races is not None and args.limit_target_races <= 0:
        raise ValueError("--limit-target-races must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chronological typed-relationship node2vec features."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--source-table", default="race_runners")
    parser.add_argument("--output-table", default=DEFAULT_TABLE)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--snapshot-frequency", choices=("monthly", "weekly"), default="monthly"
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        help="Explicit UTC snapshot timestamp; repeat to provide a custom schedule.",
    )
    parser.add_argument("--competition-id", type=int, action="append", default=[])
    parser.add_argument("--status", action="append", default=[])
    parser.add_argument("--target-from", help="Inclusive target timestamp/date.")
    parser.add_argument("--target-to", help="Exclusive target timestamp/date.")
    parser.add_argument(
        "--limit-target-races",
        type=int,
        help="Development-only limit to the earliest N filtered target races.",
    )
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--walks-per-node", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--edge-weight-mode",
        choices=("raw", "log", "binary"),
        default="raw",
        help=(
            "Transform accumulated repeated relationship weights: raw keeps "
            "counts, log uses log1p(weight), binary uses 1. Pedigree stays 1."
        ),
    )
    parser.add_argument(
        "--progress-every-nodes",
        type=int,
        default=5000,
        help="Report walk-generation progress every N nodes (default: 5000).",
    )
    parser.add_argument(
        "--half-life-days",
        type=float,
        help="Optional exponential edge-count decay half-life; default is no decay.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build graph summaries but skip node2vec training and database writes.",
    )
    args = parser.parse_args()
    validate_hyperparameters(args)
    return args


def main() -> int:
    args = parse_args()
    database = args.db.resolve()
    validate_identifier(args.source_table, "source table")
    validate_identifier(args.output_table, "output table")
    if not args.dry_run:
        ensure_output_available(database, args.output_table, args.replace)

    all_rows = load_race_rows(database, args.source_table)
    target_from = parse_utc(args.target_from, "target-from") if args.target_from else None
    target_to = parse_utc(args.target_to, "target-to") if args.target_to else None
    target = filter_targets(
        all_rows,
        args.competition_id,
        args.status,
        target_from,
        target_to,
        args.limit_target_races,
    )
    if target.empty:
        raise ValueError("Target filters selected no race rows")
    snapshots = snapshot_boundaries(
        target["_start_time"], args.snapshot_frequency, args.snapshot
    )
    assignments, unassigned = assign_snapshots_strictly_before(
        target["_start_time"], snapshots
    )
    print(
        f"database={database} source_rows={len(all_rows):,} "
        f"target_rows={len(target):,} target_races={target['race_id'].nunique():,} "
        f"snapshots_with_targets={len(assignments):,} unassigned_rows={len(unassigned):,}"
    )
    if len(unassigned):
        first = target.iloc[unassigned]["_start_time"].min()
        raise ValueError(
            f"{len(unassigned):,} target rows have no strictly earlier snapshot; "
            f"earliest unassigned target={first.isoformat()}. Add an earlier "
            "--snapshot timestamp."
        )

    outputs: list[pd.DataFrame] = []
    for number, assignment in enumerate(assignments, 1):
        snapshot = assignment.timestamp
        batch = target.iloc[assignment.target_indices].copy().reset_index(drop=True)
        history = historical_rows(all_rows, snapshot)
        edges = graph_edges(
            history, snapshot, args.half_life_days, args.edge_weight_mode
        )
        adjacency = adjacency_from_edges(edges)
        repeated_stats = repeated_edge_weight_statistics(edges)
        run_nodes = sum(node.startswith("run:") for node in adjacency)
        run_context_edges = sum(
            left.startswith("run:") or right.startswith("run:")
            for left, right in edges
        )
        summary = GraphSummary(len(adjacency), len(edges), len(history))
        print(
            f"snapshot[{number}/{len(assignments)}]={snapshot.isoformat()} "
            f"history_rows={summary.history_rows:,} nodes={summary.nodes:,} "
            f"edges={summary.edges:,} target_rows={len(batch):,} "
            f"target_races={batch['race_id'].nunique():,}",
            flush=True,
        )
        print(
            f"snapshot_repeated_edge_weights mode={args.edge_weight_mode} "
            f"count={repeated_stats['count']:,} min={repeated_stats['min']:.4f} "
            f"median={repeated_stats['median']:.4f} "
            f"mean={repeated_stats['mean']:.4f} p95={repeated_stats['p95']:.4f} "
            f"max={repeated_stats['max']:.4f}",
            flush=True,
        )
        print(
            f"snapshot_historical_run_graph run_nodes={run_nodes:,} "
            f"run_context_edges={run_context_edges:,} deduplicated=yes",
            flush=True,
        )
        if args.dry_run:
            continue
        embeddings = train_node2vec(
            adjacency,
            dimensions=args.dimensions,
            walk_length=args.walk_length,
            walks_per_node=args.walks_per_node,
            window=args.window,
            epochs=args.epochs,
            p=args.p,
            q=args.q,
            negative=args.negative,
            workers=args.workers,
            seed=args.seed + number - 1,
            progress_label=f"snapshot[{number}/{len(assignments)}] node2vec",
            progress_every_nodes=args.progress_every_nodes,
        )
        features = calculate_graph_features(batch, embeddings)
        metadata = batch.loc[
            :, ["source_rowid", "race_id", "runner_number", "runner_name", "runner_country"]
        ].copy()
        metadata["snapshot_date"] = snapshot.isoformat()
        output = pd.concat([metadata, features], axis=1)
        outputs.append(output.loc[:, output_columns()])
        available = int(features["graph_horse_embedding_available"].sum())
        print(
            f"snapshot_embeddings={len(embeddings):,} "
            f"horse_embeddings_available={available:,}/{len(batch):,}",
            flush=True,
        )

    if args.dry_run:
        print("dry_run=yes database_modified=no")
        return 0
    if not outputs:
        raise ValueError("No target rows had a strictly earlier graph snapshot")
    combined = pd.concat(outputs, ignore_index=True)
    write_graph_feature_table(database, args.output_table, combined, args.replace)
    print(
        f"output_table={args.output_table} rows_written={len(combined):,} "
        f"features_written={len(GRAPH_FEATURE_NAMES):,} database_modified=yes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
