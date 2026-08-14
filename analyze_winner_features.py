#!/usr/bin/env python3
"""Audit pre-race, non-market top-3 features on a future race holdout.

The diagnostic model is a runner-level XGBoost top-3 classifier, but feature
importance is measured with both runner AUC and race-level top-3 metrics.
Runner-varying features are shuffled within races. Race-constant features are
shuffled as whole race blocks, preserving a valid shared context for each race.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
except ImportError as exc:  # pragma: no cover - useful CLI failure
    raise SystemExit(
        "Missing dependency. Install numpy, pandas, and scikit-learn in the "
        "active environment.\nExample: pip install numpy pandas scikit-learn xgboost"
    ) from exc

try:
    from xgboost import XGBClassifier
except ImportError:  # Keep metric/selection helpers importable in tests.
    XGBClassifier = None  # type: ignore[assignment,misc]


OUTCOME_LEAKAGE = {
    "is_winner",
    "finish_place",
    "winner_index",
    "rank_label",
    "result_code",
    "top3_mask",
    "runner_mask",
    "is_trainable",
    "is_validation",
}

IDENTIFIERS = {
    "race_id",
    "selection_id",
    "runner_number",
    "competition_id",
    "race_number",
    "feature_schema_version",
}

# Current-race market columns are excluded. Historical starting prices and
# historical market-derived form remain eligible because they were known before
# the current race.
CURRENT_MARKET_EXACT = {
    "sp_starting_price",
    "open_price",
    "fluc1",
    "fluc2",
    "open_price_rank",
    "fluc1_price_rank",
    "fluc2_price_rank",
    "market_steam_rank",
    "race_consensus_score",
    "race_consensus_rank",
    "race_overlay_score",
    "race_overlay_rank",
    "race_signal_agreement_score",
    "race_signal_agreement_rank",
}

METRIC_COLUMNS = {
    "auc_drop_mean",
    "top3_hit_drop_mean",
    "top3_mrr_drop_mean",
}


def parse_competition_ids(value: str) -> list[int]:
    """Parse one competition ID or a comma-separated list of IDs."""
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
    return list(dict.fromkeys(competition_ids))


def normalize_competition_ids(competition_id: int | list[int]) -> list[int]:
    """Normalize the public query helpers while retaining single-ID calls."""
    return [competition_id] if isinstance(competition_id, int) else competition_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find important non-market top-3 features using later races as "
            "a chronological validation cohort."
        )
    )
    parser.add_argument("--db", default="db/race_runners.sqlite")
    parser.add_argument(
        "--competition-id",
        type=parse_competition_ids,
        default=[580],
        help="One competition ID or comma-separated IDs (default: 580).",
    )
    parser.add_argument(
        "--validation-races",
        type=int,
        help=(
            "Number of latest races used for validation. By default, use up to "
            "300 races or 25%% of the eligible cohort when it is smaller."
        ),
    )
    parser.add_argument(
        "--validation-folds",
        type=int,
        default=3,
        help="Expanding chronological validation folds (default: 3).",
    )
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--minimum-observations", type=int, default=10)
    parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.5,
        help=(
            "Minimum non-null fraction in the earliest training fold; combined "
            "with --minimum-observations using the stricter threshold "
            "(default: 0.5)."
        ),
    )
    parser.add_argument(
        "--permutation-scope",
        choices=("auto", "within-race", "race-block", "global"),
        default="auto",
        help=(
            "auto shuffles runner-varying features within races and race-constant "
            "features between whole races (default: auto)."
        ),
    )
    parser.add_argument(
        "--sort-by", choices=sorted(METRIC_COLUMNS), default="top3_hit_drop_mean",
        help="Primary importance column (default: top3_hit_drop_mean).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--allow-outcome-conditioned-cohort", action="store_true",
        help=(
            "Allow an explicitly diagnostic cohort where the market favourite "
            "never wins. Such a cohort must not drive production feature selection."
        ),
    )
    parser.add_argument(
        "--features-json",
        default="",
        help=(
            "Optional manifest whose zeroed_features list will be updated so the "
            "reported top features remain active. Disabled by default."
        ),
    )
    return parser.parse_args()


def is_current_market_feature(name: str) -> bool:
    return name in CURRENT_MARKET_EXACT or name.startswith("market_")


def select_features(
    training_df: pd.DataFrame,
    minimum_observations: int,
    minimum_coverage: float = 0.0,
) -> list[str]:
    """Select usable columns using training rows only.

    Looking at validation coverage or variance while selecting columns leaks
    information about the future cohort into the experiment.
    """
    numeric = [
        column
        for column in training_df.columns
        if pd.api.types.is_numeric_dtype(training_df[column])
    ]
    excluded = OUTCOME_LEAKAGE | IDENTIFIERS
    required_observations = max(
        minimum_observations,
        int(np.ceil(len(training_df) * minimum_coverage)),
    )
    return [
        column
        for column in numeric
        if column not in excluded
        and not is_current_market_feature(column)
        and int(training_df[column].notna().sum()) >= required_observations
        and int(training_df[column].nunique(dropna=True)) > 1
    ]


def load_finished_runners(
    database: Path, competition_id: int | list[int]
) -> pd.DataFrame:
    if not database.is_file():
        raise SystemExit(f"Database does not exist: {database}")
    competition_ids = normalize_competition_ids(competition_id)
    if not competition_ids:
        raise ValueError("At least one competition ID is required")
    placeholders = ", ".join("?" for _ in competition_ids)
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(
            f"""
            SELECT *
            FROM race_runners
            WHERE top3_mask IN (0, 1)
              AND runner_mask = 1
              AND status = 'finished'
              AND competition_id IN ({placeholders})
            ORDER BY start_time_iso, race_id, runner_number
            """,
            connection,
            params=tuple(competition_ids),
        )


def eligible_race_table(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return ordered races with at least four runners and three top-3 labels."""
    parsed_time = pd.to_datetime(df["start_time_iso"], errors="coerce", utc=True)
    if parsed_time.isna().any():
        examples = df.loc[parsed_time.isna(), "start_time_iso"].head(3).tolist()
        raise SystemExit(f"Invalid start_time_iso value(s): {examples}")
    working = df.assign(_parsed_start_time=parsed_time)
    races = (
        working.groupby("race_id", as_index=False)
        .agg(
            start_time=("_parsed_start_time", "min"),
            runners=("race_id", "size"),
            top3=("top3_mask", "sum"),
        )
    )
    valid = (races["runners"] >= 4) & (races["top3"] == 3)
    skipped = int((~valid).sum())
    races = races.loc[valid].sort_values(
        ["start_time", "race_id"], kind="stable", ignore_index=True
    )
    return races, skipped


def resolve_validation_races(total_races: int, requested: int | None) -> int:
    """Resolve an explicit holdout size or choose a useful adaptive default."""
    if requested is not None:
        return requested
    return min(300, max(10, total_races // 4))


def temporal_validation_folds(
    races: pd.DataFrame, validation_races: int, fold_count: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding chronological train/validation race-id folds."""
    if validation_races >= len(races):
        raise ValueError("validation_races must leave at least one training race")
    if fold_count > validation_races:
        raise ValueError("validation_folds cannot exceed validation_races")

    race_ids = races["race_id"].to_numpy(copy=True)
    validation_start = len(race_ids) - validation_races
    validation_positions = np.arange(validation_start, len(race_ids))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for positions in np.array_split(validation_positions, fold_count):
        fold_start = int(positions[0])
        folds.append((race_ids[:fold_start], race_ids[positions]))
    return folds


def random_top3_metrics(race_ids: np.ndarray) -> dict[str, float]:
    """Return exact expected metrics for selecting and ranking runners randomly."""
    race_ids = np.asarray(race_ids)
    if not len(race_ids):
        raise ValueError("random top3 metrics require non-empty rows")
    race_sizes = np.asarray(
        [np.count_nonzero(race_ids == race_id) for race_id in pd.unique(race_ids)],
        dtype=np.int64,
    )
    if np.any(race_sizes < 3):
        raise ValueError("random top3 metrics require at least three runners per race")
    expected_mrr = [
        float(np.reciprocal(np.arange(1, size + 1, dtype=np.float64)).mean())
        for size in race_sizes
    ]
    return {
        "auc": 0.5,
        "top3_hit_rate": float(np.mean(3.0 / race_sizes)),
        "top3_mrr": float(np.mean(expected_mrr)),
    }


def aggregate_fold_metrics(
    metrics: list[dict[str, float]],
    row_counts: list[int],
    race_counts: list[int],
) -> dict[str, float]:
    """Aggregate fold metrics without comparing score scales across models."""
    if not metrics or not (
        len(metrics) == len(row_counts) == len(race_counts)
    ):
        raise ValueError("metrics and fold counts must have equal non-zero lengths")
    return {
        "auc": float(
            np.average([item["auc"] for item in metrics], weights=row_counts)
        ),
        "top3_hit_rate": float(
            np.average(
                [item["top3_hit_rate"] for item in metrics], weights=race_counts
            )
        ),
        "top3_mrr": float(
            np.average(
                [item["top3_mrr"] for item in metrics], weights=race_counts
            )
        ),
    }


def numeric_heuristic_scores(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    column: str,
    higher_is_better: bool = True,
) -> np.ndarray | None:
    """Create finite validation scores using only a training-fold fill value."""
    if column not in training_df or column not in validation_df:
        return None
    training_values = pd.to_numeric(training_df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    validation_values = pd.to_numeric(
        validation_df[column], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    if not training_values.notna().any():
        return None
    fill_value = float(training_values.median())
    scores = validation_values.fillna(fill_value).to_numpy(dtype=np.float64)
    return scores if higher_is_better else -scores


def outcome_conditioned_market_cohort(
    df: pd.DataFrame, minimum_races: int = 100
) -> tuple[bool, int, int]:
    """Detect the known post-result market-miss cohort failure mode.

    A genuine market can have a poor favourite strike rate, but zero favourite
    winners over hundreds of races signals that membership was selected after
    results.  This check prevents competition_id=999 from silently being treated
    as a live, pre-race competition again.
    """
    if "fluc2" not in df or "is_winner" not in df or "race_id" not in df:
        return False, 0, 0
    valid = df.copy()
    valid["_price"] = pd.to_numeric(valid["fluc2"], errors="coerce")
    valid = valid.loc[np.isfinite(valid["_price"]) & (valid["_price"] > 0)]
    total = 0
    favourite_wins = 0
    for _, race in valid.groupby("race_id", sort=False):
        if int(pd.to_numeric(race["is_winner"], errors="coerce").fillna(0).sum()) != 1:
            continue
        minimum = float(race["_price"].min())
        total += 1
        favourite_wins += int(
            ((race["_price"] == minimum) & (race["is_winner"] == 1)).any()
        )
    return total >= minimum_races and favourite_wins == 0, total, favourite_wins


def top3_metrics(
    targets: np.ndarray, margins: np.ndarray, race_ids: np.ndarray
) -> dict[str, float]:
    """Calculate runner AUC and equal-weighted race-level top-3 metrics."""
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(margins, dtype=np.float64)
    race_ids = np.asarray(race_ids)
    if not (targets.shape == scores.shape == race_ids.shape):
        raise ValueError("targets, scores, and race_ids must have equal shapes")
    if not len(targets) or not np.isfinite(scores).all():
        raise ValueError("top3 metrics require finite scores and non-empty rows")

    race_hits: list[float] = []
    reciprocal_ranks: list[float] = []
    for race_id in pd.unique(race_ids):
        positions = np.flatnonzero(race_ids == race_id)
        race_targets = targets[positions]
        if int(race_targets.sum()) != 3:
            raise ValueError(
                f"race_id {race_id} does not have exactly 3 top3 runners"
            )
        race_scores = scores[positions]
        # Stable ties follow database runner order instead of varying randomly.
        order = np.argsort(-race_scores, kind="stable")
        predicted_top3 = order[:3]
        race_hits.append(float(race_targets[predicted_top3].sum()) / 3.0)

        actual_positions = np.flatnonzero(race_targets == 1)
        ranks = [
            int(np.flatnonzero(order == position)[0]) + 1
            for position in actual_positions
        ]
        reciprocal_ranks.append(
            float(np.mean([1.0 / rank for rank in ranks]))
        )

    return {
        "auc": float(roc_auc_score(targets, scores)),
        "top3_hit_rate": float(np.mean(race_hits)),
        "top3_mrr": float(np.mean(reciprocal_ranks)),
    }


def permute_feature(
    original: np.ndarray,
    race_ids: np.ndarray,
    rng: np.random.Generator,
    scope: str,
) -> np.ndarray:
    """Permute a feature globally, within races, or between whole races."""
    original = np.asarray(original)
    if scope == "global":
        return rng.permutation(original)
    if scope == "race-block":
        unique_races = pd.unique(race_ids)
        race_values: list[Any] = []
        for race_id in unique_races:
            values = original[race_ids == race_id]
            if not pd.Series(values).nunique(dropna=False) == 1:
                raise ValueError(
                    "race-block permutation requires a feature that is constant "
                    f"within every race; feature varies in race_id {race_id}"
                )
            race_values.append(values[0])
        donor_values = rng.permutation(np.asarray(race_values, dtype=original.dtype))
        shuffled = original.copy()
        for race_id, donor_value in zip(unique_races, donor_values):
            shuffled[race_ids == race_id] = donor_value
        return shuffled
    if scope != "within-race":
        raise ValueError(f"Unknown permutation scope: {scope}")
    shuffled = original.copy()
    for race_id in pd.unique(race_ids):
        positions = np.flatnonzero(race_ids == race_id)
        shuffled[positions] = rng.permutation(original[positions])
    return shuffled


def feature_permutation_scope(
    values: np.ndarray, race_ids: np.ndarray, requested_scope: str
) -> str:
    """Resolve auto to race-block only when every race has one feature value."""
    if requested_scope != "auto":
        return requested_scope
    for race_id in pd.unique(race_ids):
        if pd.Series(values[race_ids == race_id]).nunique(dropna=False) > 1:
            return "within-race"
    return "race-block"


def activate_top_manifest_features(manifest_path: Path, top_features: list[str]) -> list[str]:
    """Atomically remove reported top features from a manifest's zero bucket."""
    if not manifest_path.is_file():
        raise SystemExit(f"Feature manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    features = payload.get("features")
    zeroed = payload.get("zeroed_features")
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise SystemExit(f"{manifest_path} must contain a string features list")
    if not isinstance(zeroed, list) or not all(isinstance(x, str) for x in zeroed):
        raise SystemExit(f"{manifest_path} must contain a string zeroed_features list")

    feature_set = set(features)
    selected = [feature for feature in top_features if feature in feature_set]
    selected_set = set(selected)
    payload["zeroed_features"] = [feature for feature in zeroed if feature not in selected_set]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return selected


def quote_sqlite_identifier(identifier: str) -> str:
    """Quote a SQLite identifier, including embedded double quotes."""
    return '"' + identifier.replace('"', '""') + '"'


def top_features_select_sql(
    top_features: list[str], competition_id: int | list[int]
) -> str:
    """Build a copy/pasteable query for the displayed feature columns."""
    competition_ids = normalize_competition_ids(competition_id)
    if not competition_ids:
        raise ValueError("At least one competition ID is required")
    selected_columns = ["race_id", "selection_id", "runner_number", *top_features]
    columns = ",\n    ".join(
        quote_sqlite_identifier(column) for column in selected_columns
    )
    competition_filter = (
        f'= {competition_ids[0]}'
        if len(competition_ids) == 1
        else "IN (" + ", ".join(str(value) for value in competition_ids) + ")"
    )
    return (
        "SELECT\n"
        f"    {columns}\n"
        "FROM \"race_runners\"\n"
        "WHERE \"top3_mask\" IN (0, 1)\n"
        "  AND \"runner_mask\" = 1\n"
        "  AND \"status\" = 'finished'\n"
        f"  AND \"competition_id\" {competition_filter}\n"
        "ORDER BY \"start_time_iso\", \"race_id\", \"runner_number\";"
    )


def summarize_permutations(
    baseline: dict[str, float], permutations: list[dict[str, float]]
) -> dict[str, float]:
    """Express every importance so positive values mean performance worsened."""
    definitions = {
        "auc_drop": ("auc", -1.0),
        "top3_hit_drop": ("top3_hit_rate", -1.0),
        "top3_mrr_drop": ("top3_mrr", -1.0),
    }
    result: dict[str, float] = {}
    for output_name, (metric_name, lower_is_worse_sign) in definitions.items():
        values = np.asarray(
            [
                lower_is_worse_sign
                * (metrics[metric_name] - baseline[metric_name])
                for metrics in permutations
            ],
            dtype=np.float64,
        )
        result[f"{output_name}_mean"] = float(values.mean())
        result[f"{output_name}_sd"] = float(values.std())
    return result


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "validation-folds": args.validation_folds,
        "top-features": args.top_features,
        "permutation-repeats": args.permutation_repeats,
        "minimum-observations": args.minimum_observations,
        "jobs": args.jobs,
    }
    if args.validation_races is not None:
        positive["validation-races"] = args.validation_races
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise SystemExit("These arguments must be positive: " + ", ".join(invalid))
    if not 0.0 <= args.minimum_coverage <= 1.0:
        raise SystemExit("--minimum-coverage must be between 0 and 1")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if XGBClassifier is None:
        raise SystemExit(
            "Missing dependency xgboost in the active environment.\n"
            "Example: pip install xgboost"
        )

    df = load_finished_runners(Path(args.db).resolve(), args.competition_id)
    if df.empty:
        raise SystemExit(
            "No finished active runners found for competition_id="
            + ",".join(map(str, args.competition_id))
        )
    races, skipped_races = eligible_race_table(df)
    conditioned_competitions: list[tuple[int, int, int]] = []
    for competition_id, competition_df in df.groupby("competition_id", sort=False):
        conditioned, market_races, favourite_wins = outcome_conditioned_market_cohort(
            competition_df
        )
        if conditioned:
            conditioned_competitions.append(
                (int(competition_id), market_races, favourite_wins)
            )
    if conditioned_competitions and not args.allow_outcome_conditioned_cohort:
        details = ", ".join(
            f"competition_id={competition_id}: favourite won {favourite_wins} "
            f"of {market_races} races"
            for competition_id, market_races, favourite_wins
            in conditioned_competitions
        )
        raise SystemExit(
            f"Refusing outcome-conditioned cohort(s): {details}. "
            "In this database competition_id=999 was assigned after results to "
            "market-miss races. Use train_winner_ranker_pipeline.py for production "
            "training, or pass --allow-outcome-conditioned-cohort only for an "
            "explicitly labelled diagnostic audit."
        )
    if conditioned_competitions:
        print(
            "WARNING outcome-conditioned cohort enabled: results are diagnostic "
            "and must not drive production feature selection; competition_ids="
            + ",".join(str(item[0]) for item in conditioned_competitions)
        )

    validation_race_count = resolve_validation_races(
        len(races), args.validation_races
    )
    if len(races) <= validation_race_count:
        raise SystemExit(
            f"Need more than {validation_race_count} eligible completed races; "
            f"found {len(races)}"
        )
    if args.validation_folds > validation_race_count:
        raise SystemExit(
            f"--validation-folds={args.validation_folds} exceeds the "
            f"{validation_race_count} validation races"
        )
    folds = temporal_validation_folds(
        races, validation_race_count, args.validation_folds
    )
    eligible_ids = set(races["race_id"])
    df = df.loc[df["race_id"].isin(eligible_ids)].copy()

    earliest_training_ids = set(folds[0][0])
    earliest_training_df = df.loc[df["race_id"].isin(earliest_training_ids)]
    minimum_feature_observations = max(
        args.minimum_observations,
        int(np.ceil(len(earliest_training_df) * args.minimum_coverage)),
    )
    features = select_features(
        earliest_training_df,
        args.minimum_observations,
        args.minimum_coverage,
    )
    if not features:
        raise SystemExit(
            "No eligible numeric features; reduce --minimum-observations or "
            "--minimum-coverage, or inspect the database schema"
        )

    fold_artifacts: list[dict[str, Any]] = []
    for fold_index, (training_ids, validation_ids) in enumerate(folds, start=1):
        training_mask = df["race_id"].isin(set(training_ids))
        validation_mask = df["race_id"].isin(set(validation_ids))
        training_df = df.loc[training_mask]
        validation_df = df.loc[validation_mask]
        x_train = training_df.loc[:, features].replace([np.inf, -np.inf], np.nan)
        x_validation = (
            validation_df.loc[:, features]
            .replace([np.inf, -np.inf], np.nan)
            .reset_index(drop=True)
        )
        y_train = training_df["top3_mask"].astype(np.int64)
        y_validation = (
            validation_df.loc[:, "top3_mask"]
            .astype(np.int64)
            .reset_index(drop=True)
        )
        validation_race_ids = validation_df.loc[:, "race_id"].to_numpy(
            dtype=np.int64, copy=True
        )
        heuristic_scores = numeric_heuristic_scores(
            training_df, validation_df, "win_percentage"
        )

        model = XGBClassifier(
            n_estimators=450,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=8,
            reg_lambda=3,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=args.jobs,
            random_state=args.seed + fold_index - 1,
        )
        model.fit(x_train, y_train)
        baseline_margin = model.predict(x_validation, output_margin=True)
        fold_artifacts.append(
            {
                "fold": fold_index,
                "model": model,
                "x_validation": x_validation,
                "targets": y_validation.to_numpy(),
                "race_ids": validation_race_ids,
                "baseline_margin": baseline_margin,
                "training_rows": len(x_train),
                "training_races": len(training_ids),
                "validation_rows": len(x_validation),
                "validation_races": len(validation_ids),
                "metrics": top3_metrics(
                    y_validation.to_numpy(), baseline_margin, validation_race_ids
                ),
                "heuristic_metrics": (
                    top3_metrics(
                        y_validation.to_numpy(),
                        heuristic_scores,
                        validation_race_ids,
                    )
                    if heuristic_scores is not None
                    else None
                ),
            }
        )

    pooled_targets = np.concatenate(
        [artifact["targets"] for artifact in fold_artifacts]
    )
    pooled_race_ids = np.concatenate(
        [artifact["race_ids"] for artifact in fold_artifacts]
    )
    fold_row_counts = [artifact["validation_rows"] for artifact in fold_artifacts]
    fold_race_counts = [artifact["validation_races"] for artifact in fold_artifacts]
    baseline = aggregate_fold_metrics(
        [artifact["metrics"] for artifact in fold_artifacts],
        fold_row_counts,
        fold_race_counts,
    )
    random_baseline = random_top3_metrics(pooled_race_ids)
    heuristic_baseline = None
    if all(
        artifact["heuristic_metrics"] is not None
        for artifact in fold_artifacts
    ):
        heuristic_baseline = aggregate_fold_metrics(
            [artifact["heuristic_metrics"] for artifact in fold_artifacts],
            fold_row_counts,
            fold_race_counts,
        )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for feature in features:
        original = np.concatenate(
            [
                artifact["x_validation"][feature].to_numpy(copy=True)
                for artifact in fold_artifacts
            ]
        )
        permutation_scope = feature_permutation_scope(
            original, pooled_race_ids, args.permutation_scope
        )
        permutation_metrics: list[dict[str, float]] = []
        for _ in range(args.permutation_repeats):
            permuted_fold_metrics: list[dict[str, float]] = []
            for artifact in fold_artifacts:
                x_validation = artifact["x_validation"]
                fold_race_ids = artifact["race_ids"]
                fold_original = x_validation[feature].to_numpy(copy=True)
                shuffled = x_validation.copy()
                shuffled[feature] = permute_feature(
                    fold_original, fold_race_ids, rng, permutation_scope
                )
                margin = artifact["model"].predict(shuffled, output_margin=True)
                permuted_fold_metrics.append(
                    top3_metrics(
                        artifact["targets"], margin, artifact["race_ids"]
                    )
                )
            permutation_metrics.append(
                aggregate_fold_metrics(
                    permuted_fold_metrics,
                    fold_row_counts,
                    fold_race_counts,
                )
            )
        rows.append(
            {
                "feature": feature,
                "permutation_scope": permutation_scope,
                **summarize_permutations(baseline, permutation_metrics),
            }
        )

    importance_order = [
        args.sort_by,
        "top3_hit_drop_mean",
        "top3_mrr_drop_mean",
        "auc_drop_mean",
    ]
    importance_order = list(dict.fromkeys(importance_order))
    result = pd.DataFrame(rows).sort_values(
        [*importance_order, "feature"],
        ascending=[False] * len(importance_order) + [True],
        ignore_index=True,
    )
    shown = result.head(args.top_features)

    if args.features_json:
        manifest_path = Path(args.features_json).resolve()
        activated = activate_top_manifest_features(
            manifest_path, shown["feature"].astype(str).tolist()
        )
        missing = [feature for feature in shown["feature"] if feature not in activated]
        print(
            f"updated_manifest={manifest_path} activated_top_features={len(activated)}"
        )
        if missing:
            print(
                "WARNING top analyzed features absent from manifest: "
                + ", ".join(missing)
            )

    print("TOP-3 FEATURE IMPORTANCE")
    competition_label = (
        "competition_id=" if len(args.competition_id) == 1 else "competition_ids="
    ) + ",".join(map(str, args.competition_id))
    print(
        f"{competition_label} analysis_rows={len(df):,} "
        f"top3_runners={int(df['top3_mask'].sum()):,} races={len(races):,} "
        f"skipped_invalid_races={skipped_races:,}"
    )
    first_fold = fold_artifacts[0]
    last_fold = fold_artifacts[-1]
    print(
        f"temporal_folds={len(fold_artifacts)} "
        f"earliest_train_rows={first_fold['training_rows']:,} "
        f"earliest_training_races={first_fold['training_races']:,} "
        f"latest_training_races={last_fold['training_races']:,} "
        f"validation_rows={len(pooled_targets):,} "
        f"validation_races={validation_race_count:,} features={len(features)}"
    )
    if first_fold["training_races"] < validation_race_count:
        print(
            "WARNING validation cohort contains more races than training; "
            "use a smaller --validation-races value for a more stable audit"
        )
    for artifact in fold_artifacts:
        metrics = artifact["metrics"]
        print(
            f"fold={artifact['fold']} "
            f"train_races={artifact['training_races']:,} "
            f"validation_races={artifact['validation_races']:,} "
            f"auc={metrics['auc']:.5f} "
            f"top3_hit_rate={metrics['top3_hit_rate']:.5f} "
            f"top3_mrr={metrics['top3_mrr']:.5f}"
        )
    print(
        f"validation_auc={baseline['auc']:.5f} "
        f"top3_hit_rate={baseline['top3_hit_rate']:.5f} "
        f"top3_mrr={baseline['top3_mrr']:.5f}"
    )
    print(
        f"random_expected_auc={random_baseline['auc']:.5f} "
        f"random_expected_top3_hit_rate={random_baseline['top3_hit_rate']:.5f} "
        f"random_expected_top3_mrr={random_baseline['top3_mrr']:.5f} "
        f"top3_hit_lift_vs_random="
        f"{baseline['top3_hit_rate'] - random_baseline['top3_hit_rate']:.5f}"
    )
    if heuristic_baseline is not None:
        print(
            f"win_percentage_baseline_auc={heuristic_baseline['auc']:.5f} "
            f"win_percentage_baseline_top3_hit_rate="
            f"{heuristic_baseline['top3_hit_rate']:.5f} "
            f"win_percentage_baseline_top3_mrr="
            f"{heuristic_baseline['top3_mrr']:.5f} "
            f"model_top3_hit_lift_vs_win_percentage="
            f"{baseline['top3_hit_rate'] - heuristic_baseline['top3_hit_rate']:.5f}"
        )
    print(
        f"repeats={args.permutation_repeats} "
        f"permutation_scope={args.permutation_scope} sort_by={args.sort_by} "
        f"minimum_feature_observations={minimum_feature_observations:,} "
        f"minimum_coverage={args.minimum_coverage:.3f} "
        "feature_selection=train_only current_market_features=excluded "
        "historical_derived_features=eligible outcome_leakage=excluded"
    )
    print(shown.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    if args.output_csv:
        output = Path(args.output_csv).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"saved={output} rows={len(result)}")

    print("\nSQL TO SELECT TOP FEATURES")
    print(
        top_features_select_sql(
            shown["feature"].astype(str).tolist(), args.competition_id
        )
    )


if __name__ == "__main__":
    main()
