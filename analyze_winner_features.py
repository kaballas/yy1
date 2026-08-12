#!/usr/bin/env python3
"""Rank pre-race features associated with winners on a chronological holdout.

This deliberately excludes result leakage and current-race market prices. Feature
importance is the decrease in validation ROC AUC after independently shuffling a
column. A larger positive decrease means the fitted model relied on that feature
more heavily on unseen, later races.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - useful CLI failure
    raise SystemExit(
        "Missing dependency. Install pandas, scikit-learn, and xgboost in the "
        "active environment.\nExample: pip install pandas scikit-learn xgboost"
    ) from exc


OUTCOME_LEAKAGE = {
    "is_winner",
    "finish_place",
    "winner_index",
    "top3_mask",
    "runner_mask",
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

# sp_starting_price is only known after betting closes. The other fields describe
# the current race's market and are excluded so this report seeks non-market form
# signals. Historical starting-price features remain eligible.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find important non-market winner features using later races as validation."
    )
    parser.add_argument("--db", default="db/race_runners.sqlite")
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--top-features", type=int, default=10)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--minimum-observations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output-csv")
    return parser.parse_args()


def is_current_market_feature(name: str) -> bool:
    return name in CURRENT_MARKET_EXACT or name.startswith("market_")


def select_features(df: pd.DataFrame, minimum_observations: int) -> list[str]:
    numeric = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    excluded = OUTCOME_LEAKAGE | IDENTIFIERS
    return [
        column
        for column in numeric
        if column not in excluded
        and not is_current_market_feature(column)
        and int(df[column].notna().sum()) >= minimum_observations
        and int(df[column].nunique(dropna=True)) > 1
    ]


def load_finished_runners(database: Path) -> tuple[pd.DataFrame, int]:
    if not database.is_file():
        raise SystemExit(f"Database does not exist: {database}")
    with sqlite3.connect(database) as connection:
        # Run the requested winner query explicitly and report its row count.
        winner_count = int(
            pd.read_sql_query(
                "SELECT COUNT(*) AS count FROM race_runners WHERE is_winner = 1",
                connection,
            ).iloc[0]["count"]
        )
        frame = pd.read_sql_query(
            """
            SELECT *
            FROM race_runners
            WHERE is_winner IN (0, 1)
              AND runner_mask = 1
              AND status = 'finished'
            ORDER BY start_time_iso, race_id, runner_number
            """,
            connection,
        )
    return frame, winner_count


def main() -> None:
    args = parse_args()
    if args.validation_races < 1 or args.top_features < 1:
        raise SystemExit("--validation-races and --top-features must be positive")
    if args.permutation_repeats < 1:
        raise SystemExit("--permutation-repeats must be positive")

    df, winner_query_rows = load_finished_runners(Path(args.db).resolve())
    race_times = df.groupby("race_id")["start_time_iso"].min().sort_values()
    if len(race_times) <= args.validation_races:
        raise SystemExit(
            f"Need more than {args.validation_races} completed races; found {len(race_times)}"
        )

    features = select_features(df, args.minimum_observations)
    validation_ids = set(race_times.index[-args.validation_races :])
    validation_mask = df["race_id"].isin(validation_ids)
    training_mask = ~validation_mask

    x_train = df.loc[training_mask, features].replace([np.inf, -np.inf], np.nan)
    x_validation = df.loc[validation_mask, features].replace(
        [np.inf, -np.inf], np.nan
    )
    y_train = df.loc[training_mask, "is_winner"]
    y_validation = df.loc[validation_mask, "is_winner"]

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
        n_jobs=args.jobs,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    baseline_probability = model.predict_proba(x_validation)[:, 1]
    baseline_auc = float(roc_auc_score(y_validation, baseline_probability))

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float | str]] = []
    for feature in features:
        drops = []
        original = x_validation[feature].to_numpy(copy=True)
        for _ in range(args.permutation_repeats):
            shuffled = x_validation.copy()
            shuffled[feature] = rng.permutation(original)
            probability = model.predict_proba(shuffled)[:, 1]
            drops.append(baseline_auc - roc_auc_score(y_validation, probability))
        rows.append(
            {
                "feature": feature,
                "auc_drop_mean": float(np.mean(drops)),
                "auc_drop_sd": float(np.std(drops)),
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["auc_drop_mean", "feature"], ascending=[False, True]
    )
    shown = result.head(args.top_features)

    print("WINNER FEATURE IMPORTANCE")
    print(
        f"winner_query_rows={winner_query_rows:,} analysis_rows={len(df):,} "
        f"races={df['race_id'].nunique():,}"
    )
    print(
        f"train_rows={len(x_train):,} validation_rows={len(x_validation):,} "
        f"validation_races={args.validation_races:,} features={len(features)}"
    )
    print(
        f"validation_auc={baseline_auc:.5f} repeats={args.permutation_repeats} "
        "current_market_features=excluded outcome_leakage=excluded"
    )
    print(shown.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    if args.output_csv:
        output = Path(args.output_csv).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"saved={output} rows={len(result)}")


if __name__ == "__main__":
    main()
