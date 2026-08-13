#!/usr/bin/env python3
"""Audit pre-race, non-market features on a future race holdout.

The diagnostic model is a runner-level XGBoost winner classifier, but feature
importance is measured with both runner AUC and race-level winner metrics.
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
    "top1_drop_mean",
    "mrr_drop_mean",
    "race_logloss_increase_mean",
    "winner_rank_increase_mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find important non-market winner features using later races as "
            "a chronological validation cohort."
        )
    )
    parser.add_argument("--db", default="db/race_runners.sqlite")
    parser.add_argument("--competition-id", type=int, default=999)
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--minimum-observations", type=int, default=1000)
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
        "--sort-by", choices=sorted(METRIC_COLUMNS), default="mrr_drop_mean",
        help="Primary importance column (default: mrr_drop_mean).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--features-json",
        default=str(Path(__file__).resolve().with_name("tabfm_features.json")),
        help=(
            "Manifest whose zeroed_features list will be updated so the reported "
            "top features remain active. Pass an empty value to disable."
        ),
    )
    return parser.parse_args()


def is_current_market_feature(name: str) -> bool:
    return name in CURRENT_MARKET_EXACT or name.startswith("market_")


def select_features(
    training_df: pd.DataFrame, minimum_observations: int
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
    return [
        column
        for column in numeric
        if column not in excluded
        and not is_current_market_feature(column)
        and int(training_df[column].notna().sum()) >= minimum_observations
        and int(training_df[column].nunique(dropna=True)) > 1
    ]


def load_finished_runners(
    database: Path, competition_id: int
) -> pd.DataFrame:
    if not database.is_file():
        raise SystemExit(f"Database does not exist: {database}")
    with sqlite3.connect(database) as connection:
        return pd.read_sql_query(
            """
            SELECT *
            FROM race_runners
            WHERE is_winner IN (0, 1)
              AND runner_mask = 1
              AND status = 'finished'
              AND competition_id = ?
            ORDER BY start_time_iso, race_id, runner_number
            """,
            connection,
            params=(competition_id,),
        )


def eligible_race_table(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return chronologically ordered races having exactly one winner."""
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
            winners=("is_winner", "sum"),
        )
    )
    valid = (races["runners"] >= 2) & (races["winners"] == 1)
    skipped = int((~valid).sum())
    races = races.loc[valid].sort_values(
        ["start_time", "race_id"], kind="stable", ignore_index=True
    )
    return races, skipped


def winner_metrics(
    targets: np.ndarray, margins: np.ndarray, race_ids: np.ndarray
) -> dict[str, float]:
    """Calculate ranking metrics and race softmax log loss from raw margins."""
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(margins, dtype=np.float64)
    race_ids = np.asarray(race_ids)
    if not (targets.shape == scores.shape == race_ids.shape):
        raise ValueError("targets, scores, and race_ids must have equal shapes")
    if not len(targets) or not np.isfinite(scores).all():
        raise ValueError("winner metrics require finite scores and non-empty rows")

    reciprocal_ranks: list[float] = []
    winner_ranks: list[int] = []
    race_log_losses: list[float] = []
    for race_id in pd.unique(race_ids):
        positions = np.flatnonzero(race_ids == race_id)
        race_targets = targets[positions]
        if int(race_targets.sum()) != 1:
            raise ValueError(f"race_id {race_id} does not have exactly one winner")
        race_scores = scores[positions]
        # Stable ties follow database runner order instead of varying randomly.
        order = np.argsort(-race_scores, kind="stable")
        winner_position = int(np.flatnonzero(race_targets == 1)[0])
        winner_rank = int(np.flatnonzero(order == winner_position)[0]) + 1
        winner_ranks.append(winner_rank)
        reciprocal_ranks.append(1.0 / winner_rank)

        shifted_logits = race_scores - np.max(race_scores)
        log_normalizer = float(np.log(np.exp(shifted_logits).sum()))
        race_log_losses.append(
            -(float(shifted_logits[winner_position]) - log_normalizer)
        )

    return {
        "auc": float(roc_auc_score(targets, scores)),
        "top1_hit_rate": float(np.mean(np.asarray(winner_ranks) == 1)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "mean_winner_rank": float(np.mean(winner_ranks)),
        "race_logloss": float(np.mean(race_log_losses)),
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


def summarize_permutations(
    baseline: dict[str, float], permutations: list[dict[str, float]]
) -> dict[str, float]:
    """Express every importance so positive values mean performance worsened."""
    definitions = {
        "auc_drop": ("auc", -1.0),
        "top1_drop": ("top1_hit_rate", -1.0),
        "mrr_drop": ("mrr", -1.0),
        "race_logloss_increase": ("race_logloss", 1.0),
        "winner_rank_increase": ("mean_winner_rank", 1.0),
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
        "validation-races": args.validation_races,
        "top-features": args.top_features,
        "permutation-repeats": args.permutation_repeats,
        "minimum-observations": args.minimum_observations,
        "jobs": args.jobs,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise SystemExit("These arguments must be positive: " + ", ".join(invalid))


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
            f"No finished active runners found for competition_id={args.competition_id}"
        )
    races, skipped_races = eligible_race_table(df)
    if len(races) <= args.validation_races:
        raise SystemExit(
            f"Need more than {args.validation_races} eligible completed races; "
            f"found {len(races)}"
        )

    validation_ids = set(races["race_id"].iloc[-args.validation_races :])
    eligible_ids = set(races["race_id"])
    training_race_count = len(races) - args.validation_races
    df = df.loc[df["race_id"].isin(eligible_ids)].copy()
    validation_mask = df["race_id"].isin(validation_ids)
    training_mask = ~validation_mask
    training_df = df.loc[training_mask]
    features = select_features(training_df, args.minimum_observations)
    if not features:
        raise SystemExit(
            "No eligible numeric features; reduce --minimum-observations or "
            "inspect the database schema"
        )

    x_train = training_df.loc[:, features].replace([np.inf, -np.inf], np.nan)
    x_validation = (
        df.loc[validation_mask, features]
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )
    y_train = training_df["is_winner"].astype(np.int64)
    y_validation = (
        df.loc[validation_mask, "is_winner"].astype(np.int64).reset_index(drop=True)
    )
    validation_race_ids = (
        df.loc[validation_mask, "race_id"].to_numpy(dtype=np.int64, copy=True)
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
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    baseline_margin = model.predict(x_validation, output_margin=True)
    baseline = winner_metrics(
        y_validation.to_numpy(), baseline_margin, validation_race_ids
    )

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for feature in features:
        original = x_validation[feature].to_numpy(copy=True)
        permutation_scope = feature_permutation_scope(
            original, validation_race_ids, args.permutation_scope
        )
        shuffled = x_validation.copy()
        permutation_metrics: list[dict[str, float]] = []
        for _ in range(args.permutation_repeats):
            shuffled[feature] = permute_feature(
                original, validation_race_ids, rng, permutation_scope
            )
            margin = model.predict(shuffled, output_margin=True)
            permutation_metrics.append(
                winner_metrics(
                    y_validation.to_numpy(), margin, validation_race_ids
                )
            )
        rows.append(
            {
                "feature": feature,
                "permutation_scope": permutation_scope,
                **summarize_permutations(baseline, permutation_metrics),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        [args.sort_by, "feature"], ascending=[False, True], ignore_index=True
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

    print("WINNER FEATURE IMPORTANCE")
    print(
        f"competition_id={args.competition_id} analysis_rows={len(df):,} "
        f"winners={int(df['is_winner'].sum()):,} races={len(races):,} "
        f"skipped_invalid_races={skipped_races:,}"
    )
    print(
        f"train_rows={len(x_train):,} training_races={training_race_count:,} "
        f"validation_rows={len(x_validation):,} "
        f"validation_races={args.validation_races:,} features={len(features)}"
    )
    if training_race_count < args.validation_races:
        print(
            "WARNING validation cohort contains more races than training; "
            "use a smaller --validation-races value for a more stable audit"
        )
    print(
        f"validation_auc={baseline['auc']:.5f} "
        f"top1={baseline['top1_hit_rate']:.5f} mrr={baseline['mrr']:.5f} "
        f"winner_rank={baseline['mean_winner_rank']:.5f} "
        f"race_logloss={baseline['race_logloss']:.5f}"
    )
    print(
        f"repeats={args.permutation_repeats} "
        f"permutation_scope={args.permutation_scope} sort_by={args.sort_by} "
        "feature_selection=train_only current_market_features=excluded "
        "outcome_leakage=excluded"
    )
    print(shown.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    if args.output_csv:
        output = Path(args.output_csv).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"saved={output} rows={len(result)}")


if __name__ == "__main__":
    main()
