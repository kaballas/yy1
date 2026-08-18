#!/usr/bin/env python3
"""Rank finished-race runners independently by every usable numeric feature."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_DB


EXCLUDED_COLUMNS = {
    "id", "race_id", "race_number", "race_name", "competition_id",
    "winner_index", "is_trainable", "selection_id", "runner_number",
    "runner_name", "finish_place", "result", "result_code", "status",
    "source_betting_status", "sp_starting_price", "runner_mask", "rank_label",
    "top3_mask", "is_winner", "place", "is_validation",
}
NUMERIC_TYPE_TOKENS = ("INT", "REAL", "FLOA", "DOUB", "NUM", "DEC")


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


@dataclass(frozen=True)
class FeatureResult:
    feature: str
    direction: str
    total_top3_hits: int
    possible_top3_hits: int
    top3_capture_rate: float
    races_with_3_of_3: int
    races_with_2plus_of_3: int
    races_tested: int
    winner_hits: int
    winner_races_tested: int
    winner_hit_rate: float
    winner_rank_total: int
    mean_winner_rank: float
    winner_top3_hits: int
    winner_top3_rate: float


def parse_race_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError("race IDs must be comma-separated integers")
    try:
        race_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "race IDs must be comma-separated integers"
        ) from exc
    if any(race_id < 1 for race_id in race_ids):
        raise argparse.ArgumentTypeError("race IDs must be positive")
    return list(dict.fromkeys(race_ids))


def parse_competition_ids(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "competition IDs must be comma-separated integers"
        )
    try:
        competition_ids = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "competition IDs must be comma-separated integers"
        ) from exc
    if any(competition_id < 1 for competition_id in competition_ids):
        raise argparse.ArgumentTypeError("competition IDs must be positive")
    return list(dict.fromkeys(competition_ids))


def validate_competition_scope(
    competition_ids: list[int] | None, allow_competition_999: bool
) -> str | None:
    if competition_ids and 999 in competition_ids:
        if not allow_competition_999:
            raise ValueError(
                "competition_id=999 spans multiple venues; pass "
                "--allow-competition-999 to evaluate it explicitly as one entity"
            )
        if competition_ids != [999]:
            raise ValueError(
                "--allow-competition-999 requires competition 999 to be selected "
                "alone so it remains its own entity"
            )
        return "competition_999_mode=derived_market_miss_entity"
    if allow_competition_999:
        raise ValueError("--allow-competition-999 requires --competition-id 999")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--race-id", type=int)
    selection.add_argument("--race-ids", type=parse_race_ids)
    selection.add_argument("--all-races", action="store_true")
    parser.add_argument(
        "--competition-id",
        type=parse_competition_ids,
        metavar="ID[,ID...]",
        help="Limit evaluation to one or more competition IDs.",
    )
    parser.add_argument(
        "--allow-competition-999",
        action="store_true",
        help=(
            "Treat the derived v_market_top3_complete_misses race cohort as the "
            "standalone competition-999 entity. This is broader than filtering "
            "the current race_runners.competition_id column for literal 999."
        ),
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--top-features", type=int, default=50)
    parser.add_argument(
        "--minimum-races",
        type=int,
        default=1,
        help="Only report features tested in at least this many usable races.",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=64,
        help="Numeric columns loaded per batch in --all-races mode.",
    )
    parser.add_argument(
        "--detail",
        action="append",
        metavar="FEATURE",
        help=(
            "For a single race, print the complete ranking for this feature. "
            "May be supplied more than once."
        ),
    )
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def is_numeric_declared_type(declared_type: str) -> bool:
    upper = declared_type.upper()
    return any(token in upper for token in NUMERIC_TYPE_TOKENS)


def exclusion_reason(column: str, declared_type: str) -> str | None:
    """Return why a database column cannot be an independent feature."""
    lowered = column.casefold()
    if lowered in EXCLUDED_COLUMNS:
        return "identifier, control, or current-race outcome column"
    if lowered == "id" or lowered.endswith("_id") or lowered.startswith("id_"):
        return "identifier column"
    if lowered.endswith("_mask"):
        return "mask/target column"
    if re.search(r"(^|_)result($|_)", lowered):
        return "current-race result column"
    if lowered.startswith("finish_"):
        return "current-race finishing outcome column"
    if not is_numeric_declared_type(declared_type):
        return "non-numeric declared type"
    return None


def database_schema(connection: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = connection.execute('PRAGMA table_info("race_runners")').fetchall()
    if not rows:
        raise ValueError("Database has no race_runners table")
    return [(str(row[1]), str(row[2])) for row in rows]


def candidate_features(schema: list[tuple[str, str]]) -> list[str]:
    return [
        name for name, declared_type in schema
        if exclusion_reason(name, declared_type) is None
    ]


def load_finished_runners(
    database: Path,
    features: list[str],
    race_ids: list[int] | None,
    competition_ids: list[int] | None = None,
    competition_999_entity: bool = False,
) -> pd.DataFrame:
    """Load active finished runners read-only, preserving race-local identity."""
    required = ["race_id", "runner_number", "runner_name", "top3_mask", "is_winner"]
    columns = list(dict.fromkeys([*required, *features]))
    selected = ", ".join(quote_identifier(column) for column in columns)
    conditions = ['runner_mask = 1', "status = 'finished'"]
    parameters: list[Any] = []
    if race_ids is not None:
        placeholders = ",".join("?" for _ in race_ids)
        conditions.append(f"race_id IN ({placeholders})")
        parameters.extend(race_ids)
    if competition_999_entity:
        conditions.append(
            'race_id IN (SELECT race_id FROM "v_market_top3_complete_misses")'
        )
    elif competition_ids is not None:
        placeholders = ",".join("?" for _ in competition_ids)
        conditions.append(f"competition_id IN ({placeholders})")
        parameters.extend(competition_ids)
    sql = (
        f'SELECT {selected} FROM "race_runners" '
        f"WHERE {' AND '.join(conditions)} ORDER BY race_id, runner_number"
    )
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        frame = pd.read_sql_query(sql, connection, params=parameters)
    if frame.empty:
        scope = "requested races" if race_ids is not None else "database"
        raise ValueError(f"No active finished runners found for {scope}")
    return frame


def usable_races(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Keep complete top-3 races; return IDs rejected for invalid labels/size."""
    top3 = pd.to_numeric(frame["top3_mask"], errors="coerce")
    winner = pd.to_numeric(frame["is_winner"], errors="coerce")
    working = frame.copy()
    working["top3_mask"] = top3
    working["is_winner"] = winner
    valid_ids: list[int] = []
    invalid_ids: list[int] = []
    for race_id, race in working.groupby("race_id", sort=False):
        valid = (
            len(race) >= 3
            and race["top3_mask"].isin([0, 1]).all()
            and int(race["top3_mask"].sum()) == 3
            and race["is_winner"].isin([0, 1]).all()
            and int(race["is_winner"].sum()) == 1
        )
        (valid_ids if valid else invalid_ids).append(int(race_id))
    if not valid_ids:
        raise ValueError(
            "No usable races: each race needs at least three runners, exactly "
            "three top3_mask=1 rows, and exactly one is_winner=1 row"
        )
    return (
        working.loc[working["race_id"].isin(valid_ids)].reset_index(drop=True),
        invalid_ids,
    )


def ranked_indices(values: np.ndarray, direction: str) -> np.ndarray:
    """Stable numeric ordering with missing/non-finite values always last."""
    numeric = np.asarray(values, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(numeric))
    missing = np.flatnonzero(~np.isfinite(numeric))
    key = numeric[finite] if direction == "ASC" else -numeric[finite]
    return np.concatenate([finite[np.argsort(key, kind="stable")], missing])


def direction_metrics(
    frame: pd.DataFrame, feature: str, direction: str
) -> dict[str, int | float]:
    total_hits = 0
    races_3 = 0
    races_2plus = 0
    races_tested = 0
    winner_hits = 0
    winner_races = 0
    winner_rank_total = 0
    winner_top3_hits = 0
    for _, race in frame.groupby("race_id", sort=False):
        values = pd.to_numeric(race[feature], errors="coerce").to_numpy(float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) < 2 or len(np.unique(finite_values)) < 2:
            continue
        order = ranked_indices(values, direction)
        selected = order[:3]
        targets = race["top3_mask"].to_numpy(dtype=np.int64)
        winners = race["is_winner"].to_numpy(dtype=np.int64)
        hits = int(targets[selected].sum())
        total_hits += hits
        races_3 += hits == 3
        races_2plus += hits >= 2
        races_tested += 1
        winner_hits += int(winners[order[0]] == 1)
        winner_races += 1
        winner_position = int(np.flatnonzero(winners[order] == 1)[0]) + 1
        winner_rank_total += winner_position
        winner_top3_hits += winner_position <= 3
    possible = 3 * races_tested
    return {
        "total_top3_hits": total_hits,
        "possible_top3_hits": possible,
        "top3_capture_rate": total_hits / possible if possible else np.nan,
        "races_with_3_of_3": races_3,
        "races_with_2plus_of_3": races_2plus,
        "races_tested": races_tested,
        "winner_hits": winner_hits,
        "winner_races_tested": winner_races,
        "winner_hit_rate": winner_hits / winner_races if winner_races else np.nan,
        "winner_rank_total": winner_rank_total,
        "mean_winner_rank": (
            winner_rank_total / winner_races if winner_races else np.nan
        ),
        "winner_top3_hits": winner_top3_hits,
        "winner_top3_rate": (
            winner_top3_hits / winner_races if winner_races else np.nan
        ),
    }


def evaluate_features(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Evaluate ASC and DESC independently and retain the top-3-optimal direction."""
    feature_count = len(features)
    full_matrix = frame.loc[:, features].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    all_targets = frame["top3_mask"].to_numpy(dtype=np.int64)
    all_winners = frame["is_winner"].to_numpy(dtype=np.int64)
    race_ids = frame["race_id"].to_numpy()
    boundaries = np.flatnonzero(race_ids[1:] != race_ids[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(frame)]))
    accumulators = {
        direction: {
            "total_top3_hits": np.zeros(feature_count, dtype=np.int64),
            "races_with_3_of_3": np.zeros(feature_count, dtype=np.int64),
            "races_with_2plus_of_3": np.zeros(feature_count, dtype=np.int64),
            "races_tested": np.zeros(feature_count, dtype=np.int64),
            "winner_hits": np.zeros(feature_count, dtype=np.int64),
            "winner_rank_total": np.zeros(feature_count, dtype=np.int64),
            "winner_top3_hits": np.zeros(feature_count, dtype=np.int64),
        }
        for direction in ("ASC", "DESC")
    }
    for start, end in zip(starts, ends):
        matrix = full_matrix[start:end]
        finite = np.isfinite(matrix)
        finite_count = finite.sum(axis=0)
        minimum = np.where(finite, matrix, np.inf).min(axis=0)
        maximum = np.where(finite, matrix, -np.inf).max(axis=0)
        valid = (finite_count >= 2) & (maximum > minimum)
        if not valid.any():
            continue
        targets = all_targets[start:end]
        winners = all_winners[start:end]
        for direction in ("ASC", "DESC"):
            keys = matrix if direction == "ASC" else -matrix
            keys = np.where(finite, keys, np.inf)
            order = np.argsort(keys, axis=0, kind="stable")
            hits = targets[order[:3, :]].sum(axis=0)
            winner_hit = winners[order[0, :]] == 1
            winner_row = int(np.flatnonzero(winners == 1)[0])
            winner_rank = np.argmax(order == winner_row, axis=0) + 1
            accumulator = accumulators[direction]
            accumulator["total_top3_hits"][valid] += hits[valid]
            accumulator["races_with_3_of_3"][valid] += hits[valid] == 3
            accumulator["races_with_2plus_of_3"][valid] += hits[valid] >= 2
            accumulator["races_tested"][valid] += 1
            accumulator["winner_hits"][valid] += winner_hit[valid]
            accumulator["winner_rank_total"][valid] += winner_rank[valid]
            accumulator["winner_top3_hits"][valid] += winner_rank[valid] <= 3

    rows: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        by_direction: dict[str, dict[str, int | float]] = {}
        for direction in ("ASC", "DESC"):
            accumulator = accumulators[direction]
            races = int(accumulator["races_tested"][feature_index])
            hits = int(accumulator["total_top3_hits"][feature_index])
            winner_hits = int(accumulator["winner_hits"][feature_index])
            winner_rank_total = int(
                accumulator["winner_rank_total"][feature_index]
            )
            winner_top3_hits = int(
                accumulator["winner_top3_hits"][feature_index]
            )
            by_direction[direction] = {
                "total_top3_hits": hits,
                "possible_top3_hits": 3 * races,
                "top3_capture_rate": hits / (3 * races) if races else np.nan,
                "races_with_3_of_3": int(
                    accumulator["races_with_3_of_3"][feature_index]
                ),
                "races_with_2plus_of_3": int(
                    accumulator["races_with_2plus_of_3"][feature_index]
                ),
                "races_tested": races,
                "winner_hits": winner_hits,
                "winner_races_tested": races,
                "winner_hit_rate": winner_hits / races if races else np.nan,
                "winner_rank_total": winner_rank_total,
                "mean_winner_rank": (
                    winner_rank_total / races if races else np.nan
                ),
                "winner_top3_hits": winner_top3_hits,
                "winner_top3_rate": winner_top3_hits / races if races else np.nan,
            }
        ascending = by_direction["ASC"]
        descending = by_direction["DESC"]
        if not ascending["races_tested"]:
            continue
        # Never use winner performance as a direction tie-breaker.
        best_direction, best = (
            ("DESC", descending)
            if descending["top3_capture_rate"] > ascending["top3_capture_rate"]
            else ("ASC", ascending)
        )
        rows.append({"feature": feature, "direction": best_direction, **best})
    if not rows:
        return pd.DataFrame(columns=[
            "feature", "direction", "total_top3_hits", "possible_top3_hits",
            "top3_capture_rate", "races_with_3_of_3",
            "races_with_2plus_of_3", "races_tested", "winner_hits",
            "winner_races_tested", "winner_hit_rate",
            "winner_rank_total", "mean_winner_rank", "winner_top3_hits",
            "winner_top3_rate",
        ])
    return pd.DataFrame(rows).sort_values(
        [
            "top3_capture_rate", "races_with_3_of_3",
            "races_with_2plus_of_3", "races_tested", "feature",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
        ignore_index=True,
    )


def sort_feature_results(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(
        [
            "top3_capture_rate", "races_with_3_of_3",
            "races_with_2plus_of_3", "races_tested", "feature",
        ],
        ascending=[False, False, False, False, True],
        kind="stable",
        ignore_index=True,
    )


def filter_results_by_minimum_races(
    results: pd.DataFrame, minimum_races: int
) -> pd.DataFrame:
    if minimum_races < 1:
        raise ValueError("minimum-races must be positive")
    filtered = results.loc[
        results["races_tested"] >= minimum_races
    ].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"No feature was testable in at least {minimum_races} races"
        )
    return filtered


def print_leaderboard(results: pd.DataFrame, limit: int) -> None:
    shown = results.head(limit).copy()
    maximum_races = int(results["races_tested"].max())
    print("=" * 80)
    print("FEATURE HINTING LEADERBOARD")
    print("=" * 80)
    legend = (
        "READING THE TABLE\n"
        "  Dir ASC: lowest feature values are ranked first.\n"
        "  Dir DESC: highest feature values are ranked first.\n"
    )
    if maximum_races == 1:
        legend += (
            "  Top3 Hits: actual Top-3 finishers in the feature's first 3.\n"
            "  Winner Rank: actual winner's position in the feature ranking.\n"
            "  Leaderboard order uses Top3 Hits, not Winner Rank."
        )
    else:
        legend += (
            "  Top3 Capture: actual Top-3 finishers present in the feature's first 3.\n"
            "  Winner #1: races where the feature ranked the actual winner first.\n"
            "  Avg Win Rank: actual winner's average position in the feature ranking.\n"
            "  Direction is selected using Top3 Capture only, never Winner #1."
        )
    print(legend)
    if maximum_races == 1:
        print(
            "\nWARNING SINGLE-RACE HINDSIGHT: ASC/DESC was selected after seeing "
            "this race's result. With hundreds of features, chance 100% captures "
            "are expected and are not predictive evidence. Raw, rank, gap, mean, "
            "median, percentile, and z-score variants of one source are also one "
            "signal family, not independent confirmations. Use --all-races or a "
            "large --competition-id cohort to estimate reusable directions.\n"
        )
    if maximum_races == 1:
        print(
            f"{'Rank':<5} {'Feature':<44} {'Dir':<5} "
            f"{'Top3 Hits':>10} {'Winner Rank':>12}"
        )
        for index, row in shown.iterrows():
            hits = (
                f"{int(row['total_top3_hits'])}/"
                f"{int(row['possible_top3_hits'])}"
            )
            winner_rank = f"{int(row['winner_rank_total'])}"
            print(
                f"{index + 1:<5} {str(row['feature']):<44} "
                f"{str(row['direction']):<5} {hits:>10} {winner_rank:>12}"
            )
    else:
        print(
            f"{'Rank':<5} {'Feature':<44} {'Dir':<5} "
            f"{'Top3 Capture':>12} {'Winner #1':>10} {'Avg Win Rank':>12} "
            f"{'Races':>7}"
        )
        for index, row in shown.iterrows():
            print(
                f"{index + 1:<5} {str(row['feature']):<44} "
                f"{str(row['direction']):<5} "
                f"{row['top3_capture_rate']:>11.2%} "
                f"{row['winner_hit_rate']:>9.2%} "
                f"{row['mean_winner_rank']:>12.2f} "
                f"{int(row['races_tested']):>7,}"
            )


def print_race_detail(
    race: pd.DataFrame, feature: str, direction: str
) -> None:
    values = pd.to_numeric(race[feature], errors="coerce").to_numpy(float)
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < 2 or len(np.unique(finite_values)) < 2:
        print(f"\nFEATURE {feature!r} has no varying usable values in this race.")
        return
    order = ranked_indices(values, direction)
    print(f"\nRACE {int(race.iloc[0]['race_id'])}")
    print(f"FEATURE: {feature}")
    print(f"DIRECTION: {direction}\n")
    print(
        f"{'Rank':<5} {'Runner':<30} {'Value':>14} "
        f"{'top3_mask':>10} {'is_winner':>10}"
    )
    for rank, position in enumerate(order, start=1):
        row = race.iloc[int(position)]
        value = values[int(position)]
        rendered = "NULL" if not np.isfinite(value) else f"{value:.6g}"
        runner = f"#{int(row['runner_number'])} {row['runner_name']}"
        print(
            f"{rank:<5} {runner:<30} {rendered:>14} "
            f"{int(row['top3_mask']):>10} {int(row['is_winner']):>10}"
        )
    hits = int(race.iloc[order[:3]]["top3_mask"].sum())
    winner_rank = int(np.flatnonzero(
        race.iloc[order]["is_winner"].to_numpy(dtype=int) == 1
    )[0]) + 1
    print(f"\nTop-3 capture: {hits}/3 = {hits / 3:.2%}")
    print(f"Actual winner's feature rank: {winner_rank}/{len(race)}")


def winner_rank_one_payload(
    results: pd.DataFrame, frame: pd.DataFrame
) -> dict[str, Any]:
    """Build a JSON-safe list of features that ranked every tested winner first."""
    selected = results.loc[
        results["winner_rank_total"] == results["winner_races_tested"]
    ]
    race_ids = sorted(frame["race_id"].astype(int).unique().tolist())
    payload: dict[str, Any] = {
        #"race_ids": race_ids,
        "criterion": "actual winner ranked #1 in every race tested for the feature",
        "feature_count": len(selected),
        "feature_names": selected["feature"].astype(str).tolist(),
        "features": [
            {
                "feature": str(row.feature),
                "direction": str(row.direction),
                "top3_hits": int(row.total_top3_hits),
                "possible_top3_hits": int(row.possible_top3_hits),
                "top3_capture_rate": float(row.top3_capture_rate),
                "races_tested": int(row.races_tested),
            }
            for row in selected.itertuples(index=False)
        ],
    }
    if len(race_ids) == 1:
        labels = pd.to_numeric(frame["is_winner"], errors="coerce")
        winner = frame.loc[labels.eq(1)].iloc[0]
        payload["winner"] = {
            "runner_number": int(winner["runner_number"]),
            "runner_name": str(winner["runner_name"]),
        }
    return {"winner_rank_1_features": payload}


def main() -> None:
    args = parse_args()
    if args.race_id is not None and args.race_id < 1:
        raise ValueError("race-id must be positive")
    if args.top_features < 1:
        raise ValueError("top-features must be positive")
    if args.minimum_races < 1:
        raise ValueError("minimum-races must be positive")
    if args.feature_batch_size < 1:
        raise ValueError("feature-batch-size must be positive")
    requested_ids = (
        None if args.all_races
        else [args.race_id] if args.race_id is not None
        else args.race_ids
    )
    if args.detail and (requested_ids is None or len(requested_ids) != 1):
        raise ValueError("--detail requires exactly one --race-id")
    database = args.db.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    with sqlite3.connect(
        f"file:{database}?mode=ro", uri=True
    ) as connection:
        schema = database_schema(connection)
    competition_scope = validate_competition_scope(
        args.competition_id, args.allow_competition_999
    )
    competition_999_entity = bool(
        args.allow_competition_999 and args.competition_id == [999]
    )
    features = candidate_features(schema)
    if args.all_races:
        frame, invalid_ids = usable_races(
            load_finished_runners(
                database, [], None, args.competition_id,
                competition_999_entity=competition_999_entity,
            )
        )
        valid_ids = set(frame["race_id"].astype(int))
        result_parts: list[pd.DataFrame] = []
        for start in range(0, len(features), args.feature_batch_size):
            batch = features[start:start + args.feature_batch_size]
            batch_frame = load_finished_runners(
                database, batch, None, args.competition_id,
                competition_999_entity=competition_999_entity,
            )
            batch_frame = batch_frame.loc[
                batch_frame["race_id"].isin(valid_ids)
            ].reset_index(drop=True)
            result_parts.append(evaluate_features(batch_frame, batch))
        results = sort_feature_results(pd.concat(result_parts, ignore_index=True))
    else:
        frame, invalid_ids = usable_races(
            load_finished_runners(
                database, features, requested_ids, args.competition_id,
                competition_999_entity=competition_999_entity,
            )
        )
        results = evaluate_features(frame, features)
    if results.empty:
        raise ValueError("No numeric feature varied within any usable race")
    unfiltered_feature_count = len(results)
    results = filter_results_by_minimum_races(results, args.minimum_races)
    if requested_ids is not None:
        found = set(frame["race_id"].astype(int)) | set(invalid_ids)
        missing = sorted(set(requested_ids) - found)
        if missing:
            raise ValueError("Requested finished races not found: " + ", ".join(map(str, missing)))
    print(
        f"database={database}\n"
        f"competition_ids={args.competition_id or 'all'}\n"
        f"usable_races={frame['race_id'].nunique():,} runners={len(frame):,} "
        f"numeric_candidates={len(features):,} "
        f"ranked_features={len(results):,}/{unfiltered_feature_count:,} "
        f"minimum_races={args.minimum_races:,}\n"
        f"skipped_invalid_races={len(invalid_ids):,}"
    )
    if competition_scope:
        print(competition_scope)
    print_leaderboard(results, args.top_features)
    if args.output_csv:
        output = args.output_csv.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output, index=False)
        print(f"\nsaved={output} rows={len(results):,}")
    if args.detail:
        race = frame.loc[frame["race_id"] == requested_ids[0]].reset_index(drop=True)
        result_by_feature = results.set_index("feature")
        for feature in args.detail:
            if feature not in features:
                raise ValueError(f"Feature is unavailable or excluded: {feature}")
            if feature not in result_by_feature.index:
                print(f"\nFEATURE {feature!r} has no varying usable values.")
                continue
            direction = str(result_by_feature.loc[feature, "direction"])
            print_race_detail(race, feature, direction)
    print("\nWINNER-RANK-1 FEATURES JSON")
    print(json.dumps(
        winner_rank_one_payload(results, frame), indent=2, sort_keys=False
    ))


if __name__ == "__main__":
    main()
