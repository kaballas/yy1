#!/usr/bin/env python3
"""Rank one active race with dynamically configured model groups.

Every model uses the exact ordered feature list stored in the bundle. Output
explicitly reports whether the requested model or blend uses current-market
features. Tuned rankings can select the strongest historical OOF strategy for
the target race's competition and race number.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:  # pragma: no cover - CLI environment failure
    raise SystemExit("xgboost is required: pip install xgboost") from exc

from src.config import DEFAULT_DB
from src.database import quote_identifier
from src.winner_ranker import (
    MARKET_ENGINEERED_FEATURES,
    blend_named_scores,
    blend_scores,
    ensemble_rank_scores,
    is_current_market_feature,
    market_scores,
    model_feature_matrix,
    rank_percentiles,
)
from src.advanced_racing_features import race_relative_runner_mask
from backtest_all_finished_winner_blends import (
    artifact_strategies,
    best_backtest_strategy,
    filter_complete_races,
    load_predictions,
)


METADATA = [
    "race_id", "start_time_iso", "competition_id", "competition_name",
    "race_number", "race_name", "runner_number", "runner_name", "runner_mask",
    "status", "source_betting_status", "active_field_size", "fluc2",
    "derived_racing_features_version",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--bundle", type=Path,
        default=Path("outputs/winner_ranker/winner_ranker_bundle.json"),
    )
    parser.add_argument(
        "--ranking",
        default="deployment",
        help=(
            "Ranking to display: deployment, tuned, market, or any model-group "
            "name from the bundle (for example form, market_aware, or fun). "
            "Tuned uses the best matching competition/race-number OOF strategy "
            "when all-finished predictions are available."
        ),
    )
    parser.add_argument(
        "--blend-config",
        type=Path,
        default=Path("outputs/winner_ranker/form_market_aware_blend.json"),
        help=(
            "Blend strategies and fallback weights used by --ranking tuned."
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Saved all-finished OOF predictions used to choose the best tuned "
            "strategy for the target competition and race number. Defaults to "
            "all_finished_oof_predictions.csv beside --blend-config."
        ),
    )
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("objective") != "single_winner_ranking":
        raise ValueError(f"Unsupported winner bundle: {path}")
    return payload


def load_active_race(database: Path, race_id: int, features: list[str]) -> pd.DataFrame:
    requested = list(dict.fromkeys([*METADATA, *features]))
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        schema = {
            str(row[1]) for row in connection.execute('PRAGMA table_info("race_runners")')
        }
        missing = sorted(set(requested) - schema)
        if missing:
            raise ValueError("Database is missing winner-ranker inputs: " + ", ".join(missing))
        selected = ", ".join(quote_identifier(column) for column in requested)
        frame = pd.read_sql_query(
            f"SELECT {selected} FROM race_runners "
            "WHERE race_id = ? ORDER BY runner_number",
            connection,
            params=(race_id,),
        )
    eligible = race_relative_runner_mask(frame) if not frame.empty else pd.Series(
        False, index=frame.index
    )
    frame = frame.loc[eligible].reset_index(drop=True)
    if frame.empty:
        raise ValueError(
            f"Race {race_id} does not exist or has no verified complete active field"
        )
    return frame


def load_models(paths: list[str]) -> list[XGBRanker]:
    models: list[XGBRanker] = []
    for path in paths:
        model = XGBRanker()
        model.load_model(path)
        models.append(model)
    if not models:
        raise ValueError("Bundle contains no models")
    return models


def ranked_output(
    frame: pd.DataFrame,
    scores: dict[str, np.ndarray],
    ranking: str,
) -> pd.DataFrame:
    if "market" not in scores:
        raise ValueError("Ranking output requires the market benchmark score")
    if ranking not in scores:
        raise ValueError(f"Ranking {ranking!r} was not calculated")
    output = frame[[
        "runner_number", "runner_name", "fluc2",
    ]].copy()
    for name, values in scores.items():
        output[f"{name}_score"] = values
        output[f"{name}_rank"] = pd.Series(values).rank(
            method="first", ascending=False
        ).astype(int)
    comparison = "form" if "form" in scores else "deployment"
    output["market_to_form_upgrade"] = (
        output["market_rank"] - output[f"{comparison}_rank"]
    )
    output["contrarian_top3"] = (
        (output[f"{comparison}_rank"] <= 3) & (output["market_rank"] > 3)
    ).astype(int)
    output = output.sort_values(
        [f"{ranking}_rank", "runner_number"], kind="stable", ignore_index=True
    )
    output.insert(0, "display_rank", np.arange(1, len(output) + 1))
    return output


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle)
    legacy_form_features = list(bundle["form_features"])
    configured_model_features = bundle.get("model_features", {})
    model_features = {
        label: list(features)
        for label, features in configured_model_features.items()
    }
    available_models = bundle.get("models", {})
    if "form" in available_models:
        model_features.setdefault("form", legacy_form_features)
    if "market_aware" in available_models:
        model_features.setdefault(
            "market_aware",
            [*legacy_form_features, *MARKET_ENGINEERED_FEATURES],
        )
    database_features = list(dict.fromkeys([
        *(
            feature
            for features in model_features.values()
            for feature in features
            if feature not in MARKET_ENGINEERED_FEATURES
        ),
    ]))
    frame = load_active_race(args.db, args.race_id, database_features)
    versions = sorted(
        str(value) for value in frame["derived_racing_features_version"].dropna().unique()
    )
    expected = list(bundle.get("derived_feature_versions", []))
    if len(versions) != 1 or frame["derived_racing_features_version"].isna().any():
        raise ValueError(
            "Race has missing/mixed derived feature versions; run the feature updater"
        )
    if expected and versions[0] not in expected:
        raise ValueError(
            f"Race feature version {versions[0]!r} was not used for training; "
            "rerun train_winner_ranker_pipeline.py"
        )

    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    scores: dict[str, np.ndarray] = {}
    for label, configured_features in model_features.items():
        paths = list(bundle.get("models", {}).get(label, []))
        if not paths:
            continue
        models = load_models(paths)
        scores[label] = ensemble_rank_scores(
            models, model_feature_matrix(frame, configured_features), race_ids
        )
    market_score = rank_percentiles(market_scores(frame), race_ids)
    scores["market"] = market_score
    deployment_model = str(bundle.get("deployment_default", "form"))
    if deployment_model not in scores:
        raise ValueError(f"Deployment model is unavailable: {deployment_model}")
    scores["deployment"] = scores[deployment_model]
    scores["selected"] = scores[deployment_model]
    diagnostic_weights: dict[str, float] | None = None
    cohort_strategy: str | None = None
    cohort_strategy_metrics: dict[str, Any] | None = None
    cohort_strategy_fallback: str | None = None
    if args.ranking in {"tuned", "benchmark"}:
        if args.ranking == "tuned":
            config_path = args.blend_config.resolve()
            if not config_path.is_file():
                raise ValueError(
                    f"Blend recommendation does not exist: {config_path}; run "
                    "backtest_winner_blend.py first"
                )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            configured_weights = dict(config.get("selected_weights", {}))
            predictions_path = args.predictions or (
                config_path.parent / "all_finished_oof_predictions.csv"
            )
            if predictions_path.resolve().is_file():
                oof_model_labels, strategies = artifact_strategies(bundle, config)
                predictions = load_predictions(
                    predictions_path, oof_model_labels
                )
                try:
                    cohort = filter_complete_races(
                        predictions,
                        int(frame.iloc[0]["competition_id"]),
                        None,
                        None,
                        int(frame.iloc[0]["race_number"]),
                    )
                except ValueError as exc:
                    if "No complete OOF races match" not in str(exc):
                        raise
                    diagnostic_weights = configured_weights
                    cohort_strategy_fallback = str(exc)
                else:
                    (
                        cohort_strategy,
                        diagnostic_weights,
                        cohort_strategy_metrics,
                    ) = best_backtest_strategy(
                        cohort, oof_model_labels, strategies
                    )
            else:
                diagnostic_weights = configured_weights
                cohort_strategy_fallback = (
                    f"OOF predictions do not exist: {predictions_path.resolve()}"
                )
            if not diagnostic_weights:
                raise ValueError("Blend recommendation contains no usable weights")
            scores["tuned"] = blend_named_scores(scores, diagnostic_weights)
        elif args.ranking == "benchmark":
            if "form" not in scores or "market_aware" not in scores:
                raise ValueError(
                    "Legacy benchmark ranking requires form and market_aware models"
                )
            diagnostic_weights = bundle.get("benchmark_blend_weights")
            if not diagnostic_weights:
                raise ValueError("Bundle has no validation-selected benchmark blend")
            scores["benchmark"] = blend_scores(
                scores["form"], scores["market_aware"], market_score,
                dict(diagnostic_weights),
            )
    output = ranked_output(frame, scores, args.ranking)

    race = frame.iloc[0]
    displayed_model = (
        deployment_model
        if args.ranking in {"deployment", "selected"}
        else args.ranking
    )
    displayed_features = model_features.get(displayed_model, [])
    if args.ranking == "tuned" and diagnostic_weights is not None:
        market_used = (
            float(diagnostic_weights.get("market", 0.0)) > 0
            or any(
                float(diagnostic_weights.get(label, 0.0)) > 0
                and any(
                    feature in MARKET_ENGINEERED_FEATURES
                    or is_current_market_feature(feature)
                    for feature in model_features.get(label, [])
                )
                for label in model_features
            )
        )
    else:
        market_used = args.ranking in {"market", "benchmark"} or any(
            feature in MARKET_ENGINEERED_FEATURES
            or is_current_market_feature(feature)
            for feature in displayed_features
        )
    heading = (
        "WINNER RANKER\n"
        f"race={args.race_id} {race['race_name']} venue={race['competition_name']} "
        f"start={race['start_time_iso']} active_runners={len(frame)}\n"
        f"display_ranking={args.ranking} "
        f"current_market_used_in_display_ranking={'yes' if market_used else 'no'}\n"
        f"deployment_model={deployment_model}\n"
    )
    if diagnostic_weights is not None:
        weight_label = (
            "tuned_blend_weights" if args.ranking == "tuned"
            else "diagnostic_benchmark_weights"
        )
        heading += (
            weight_label + "="
            + json.dumps(diagnostic_weights, sort_keys=True)
            + "\n"
        )
    if args.ranking == "tuned":
        heading += (
            f"target_competition_id={int(race['competition_id'])} "
            f"target_race_number={int(race['race_number'])}\n"
        )
        if cohort_strategy is not None and cohort_strategy_metrics is not None:
            heading += (
                f"cohort_best_strategy={cohort_strategy} "
                f"historical_oof_races="
                f"{int(cohort_strategy_metrics['races'])} "
                f"top1_hit_rate="
                f"{float(cohort_strategy_metrics['top1_hit_rate']):.5f} "
                f"top3_hit_rate="
                f"{float(cohort_strategy_metrics['top3_hit_rate']):.5f} "
                f"mrr={float(cohort_strategy_metrics['mrr']):.5f}\n"
            )
        elif cohort_strategy_fallback is not None:
            heading += (
                "cohort_best_strategy=config_selected_fallback "
                f"reason={cohort_strategy_fallback}\n"
            )
    heading += (
        "market_to_form_upgrade: positive means the form model promotes the runner\n"
        "contrarian_top3: form top three while outside market top three"
    )
    print(heading)
    dynamic_model_rank_columns = [
        f"{label}_rank"
        for label in model_features
        if label in scores
    ]
    columns = [
        "display_rank", "runner_number", "runner_name", "fluc2",
        f"{args.ranking}_rank",
        *dynamic_model_rank_columns, "market_rank",
        "tuned_rank", "market_aware_rank", "benchmark_rank", "market_to_form_upgrade",
        "contrarian_top3",
    ]
    visible_columns = [
        column for column in dict.fromkeys(columns) if column in output.columns
    ]
    print(output.loc[:, visible_columns].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))
    if int(race["competition_id"]) == 999:
        print(
            "WARNING competition_id=999 is a post-result market-miss label, not a "
            "live competition identifier. Rankings remain valid because the model "
            "never uses competition_id."
        )
    if (
        str(bundle.get("training_scope", "")) == "all_eligible_finished_races"
        and str(race.get("status", "")).strip().casefold() == "finished"
    ):
        print(
            "WARNING this is a finished race and the selected bundle was refit "
            "on all eligible finished races. This ranking may be in-sample and "
            "must not be reported as a held-out backtest result."
        )
    if args.output_csv:
        path = args.output_csv.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        print(f"saved={path}")


if __name__ == "__main__":
    main()
