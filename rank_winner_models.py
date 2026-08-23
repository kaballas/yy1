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
import os
import sqlite3
import sys
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
    uses_current_market_features,
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

MIN_EXACT_COHORT_RACES = 30


def select_historical_cohort(
    predictions: pd.DataFrame,
    competition_id: int,
    race_number: int,
    minimum_exact_races: int = MIN_EXACT_COHORT_RACES,
) -> tuple[pd.DataFrame, str, int]:
    """Prefer exact history, broadening small or absent exact cohorts."""
    try:
        exact = filter_complete_races(
            predictions, competition_id, None, None, race_number
        )
    except ValueError as exc:
        if "No complete OOF races match" not in str(exc):
            raise
        exact_races = 0
    else:
        exact_races = int(exact["race_id"].nunique())
        if exact_races >= minimum_exact_races:
            return exact, "competition_id+race_number", exact_races

    competition = filter_complete_races(
        predictions, competition_id, None, None, None
    )
    competition_races = int(competition["race_id"].nunique())
    if competition_races < minimum_exact_races:
        raise ValueError(
            "No complete OOF races match the minimum cohort size for "
            "competition_id "
            f"{competition_id}: found {competition_races} races; "
            f"minimum={minimum_exact_races}"
        )
    return competition, "competition_id", exact_races


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
            "Ranking to display: deployment, tuned, consensus, own, market_free, "
            "market, or any "
            "model-group name from the bundle (for example form, market_aware, "
            "or fun). "
            "Tuned uses the best matching competition/race-number OOF strategy "
            "when all-finished predictions are available."
        ),
    )
    parser.add_argument(
        "--race-models-manifest",
        type=Path,
        help=(
            "Load the original per-race models and exact input features directly "
            "from models_test/per_race_models_manifest.json."
        ),
    )
    parser.add_argument(
        "--model-display-limit",
        type=int,
        default=10,
        help=(
            "Maximum individual model columns/results to print when using a "
            "race-model manifest (default: 10). All models still contribute to "
            "consensus. Use 0 to display every model."
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
    parser.add_argument(
        "--market-free-blend-config",
        type=Path,
        help=(
            "Saved current-market-free blend. Defaults to market_free_blend.json "
            "beside --blend-config."
        ),
    )
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if payload.get("objective") != "single_winner_ranking":
        raise ValueError(f"Unsupported winner bundle: {path}")
    return payload


def load_original_race_models(
    manifest_path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load original model paths and features without retraining substitutions."""
    path = manifest_path.resolve()
    if not path.is_file():
        raise ValueError(f"Race-model manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_sets: dict[str, list[str]] = {}
    model_paths: dict[str, list[str]] = {}
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("name", ""))
        configured_paths = item.get("models")
        if not isinstance(configured_paths, list) or not configured_paths:
            configured_paths = [item.get("model", "")]
        model_paths_for_entry = [
            Path(str(configured_path)) for configured_path in configured_paths
        ]
        details = item.get("details")
        features = details.get("input_features") if isinstance(details, dict) else None
        sidecar_path = Path(str(item.get("features_file", "")))
        if not features and sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            features = sidecar.get("input_features")
        if (
            not label
            or not model_paths_for_entry
            or not all(path.is_file() for path in model_paths_for_entry)
            or not features
        ):
            continue
        if label in feature_sets:
            raise ValueError(f"Race-model manifest has duplicate model name: {label}")
        feature_sets[label] = list(map(str, features))
        model_paths[label] = [
            str(model_path.resolve()) for model_path in model_paths_for_entry
        ]
    if not feature_sets:
        raise ValueError(f"Race-model manifest contains no usable models: {path}")
    return feature_sets, model_paths


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
    score_columns: dict[str, np.ndarray | pd.Series] = {}
    for name, values in scores.items():
        score_columns[f"{name}_score"] = np.asarray(values)
        score_columns[f"{name}_rank"] = pd.Series(values).rank(
            method="average", ascending=False
        ).to_numpy(dtype=np.float64)
    output = pd.concat(
        [output.reset_index(drop=True), pd.DataFrame(score_columns)], axis=1
    )
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


def terminal_display_table(
    output: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Shorten rank headers for terminal display without changing artifacts."""
    table = output.loc[:, columns].copy()
    return table.rename(columns={
        column: column.removesuffix("_rank")
        for column in table.columns
        if column.endswith("_rank")
    })


def terminal_table_text(
    output: pd.DataFrame,
    columns: list[str],
    *,
    color: bool | None = None,
) -> str:
    """Render the table with rank-one cells red on color terminals."""
    table = terminal_display_table(output, columns)
    rank_columns = {
        column.removesuffix("_rank")
        for column in columns
        if column.endswith("_rank")
    }
    use_color = (
        sys.stdout.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM", "") != "dumb"
        if color is None else color
    )
    sentinel = "¤"
    formatters = {
        column: (
            lambda value: sentinel
            if pd.notna(value) and float(value) == 1.0
            else (
                str(int(value))
                if pd.notna(value) and float(value).is_integer()
                else f"{float(value):.1f}"
            )
        )
        for column in rank_columns
        if column in table
    }
    rendered = table.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        formatters=formatters,
    )
    replacement = "\033[31m1\033[0m" if use_color else "1"
    return rendered.replace(sentinel, replacement)


def number_one_summary(
    output: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Summarize how many displayed rankings select each runner first."""
    rank_columns = [
        column for column in columns
        if column.endswith("_rank") and column != "display_rank" and column in output
    ]
    summary = output[[
        "display_rank", "runner_number", "runner_name", "fluc2",
    ]].copy()
    if not rank_columns:
        summary["number_ones"] = 0
        summary["picked_first_by"] = ""
    else:
        first = output.loc[:, rank_columns].eq(1)
        summary["number_ones"] = first.sum(axis=1).astype(int)
        labels = [column.removesuffix("_rank") for column in rank_columns]
        summary["picked_first_by"] = [
            ",".join(label for label, selected in zip(labels, row) if selected)
            for row in first.to_numpy(dtype=bool)
        ]
    return (
        summary.loc[summary["number_ones"] > 0]
        .sort_values(
            ["number_ones", "display_rank", "runner_number"],
            ascending=[False, True, True],
            kind="stable",
        )
        .drop(columns="display_rank")
        .reset_index(drop=True)
    )


def model_rank_total_summary(
    output: pd.DataFrame, model_rank_columns: list[str]
) -> pd.DataFrame:
    """Aggregate every configured model rank into a Borda-style consensus."""
    rank_columns = list(dict.fromkeys(
        column for column in model_rank_columns if column in output.columns
    ))
    if not rank_columns:
        return pd.DataFrame(columns=[
            "consensus_rank", "runner_number", "runner_name", "fluc2",
            "models_counted", "model_rank_total", "average_model_rank",
            "best_model_rank", "worst_model_rank", "number_ones",
        ])
    ranks = output.loc[:, rank_columns].apply(pd.to_numeric, errors="raise")
    summary = output[["runner_number", "runner_name", "fluc2"]].copy()
    summary["models_counted"] = len(rank_columns)
    summary["model_rank_total"] = ranks.sum(axis=1)
    summary["average_model_rank"] = ranks.mean(axis=1)
    summary["best_model_rank"] = ranks.min(axis=1)
    summary["worst_model_rank"] = ranks.max(axis=1)
    summary["number_ones"] = ranks.eq(1).sum(axis=1).astype(int)
    summary = summary.sort_values(
        [
            "model_rank_total", "average_model_rank", "best_model_rank",
            "worst_model_rank", "number_ones", "runner_number",
        ],
        ascending=[True, True, True, True, False, True],
        kind="stable",
        ignore_index=True,
    )
    summary.insert(0, "consensus_rank", np.arange(1, len(summary) + 1))
    return summary


def number_one_rank_summary(rank_totals: pd.DataFrame) -> pd.DataFrame:
    """Rank runners by first-place votes across all configured models."""
    columns = [
        "number_one_rank", "runner_number", "runner_name", "fluc2",
        "number_ones", "number_one_pct", "number_one_vote_share_pct",
        "models_with_unique_top", "models_counted",
        "average_model_rank", "consensus_rank",
    ]
    if rank_totals.empty:
        return pd.DataFrame(columns=columns)
    summary = rank_totals.copy()
    denominator = pd.to_numeric(summary["models_counted"], errors="raise")
    summary["number_one_pct"] = np.where(
        denominator > 0,
        pd.to_numeric(summary["number_ones"], errors="raise")
        / denominator * 100.0,
        0.0,
    )
    unique_top_votes = int(summary["number_ones"].sum())
    summary["models_with_unique_top"] = unique_top_votes
    summary["number_one_vote_share_pct"] = (
        pd.to_numeric(summary["number_ones"], errors="raise")
        / unique_top_votes * 100.0
        if unique_top_votes else 0.0
    )
    summary = summary.sort_values(
        ["number_ones", "average_model_rank", "consensus_rank", "runner_number"],
        ascending=[False, True, True, True],
        kind="stable",
        ignore_index=True,
    )
    summary.insert(0, "number_one_rank", np.arange(1, len(summary) + 1))
    return summary.loc[:, columns]


def model_prediction_tie_diagnostics(
    scores: dict[str, np.ndarray], model_labels: list[str]
) -> dict[str, int]:
    """Count constant, uniquely led, and top-tied model predictions."""
    constant = 0
    unique_top = 0
    top_tied = 0
    for label in model_labels:
        if label not in scores:
            continue
        values = np.asarray(scores[label], dtype=np.float64)
        if np.isclose(values.max(), values.min(), rtol=0.0, atol=1e-12):
            constant += 1
        top_count = int(np.isclose(
            values, values.max(), rtol=0.0, atol=1e-12
        ).sum())
        if top_count == 1:
            unique_top += 1
        else:
            top_tied += 1
    return {
        "models_constant": constant,
        "models_unique_top": unique_top,
        "models_tied_top": top_tied,
    }


def consensus_representative_model_columns(
    output: pd.DataFrame,
    model_rank_columns: list[str],
    limit: int,
) -> list[str]:
    """Select model rankings closest to the all-model consensus ranking."""
    columns = list(dict.fromkeys(
        column for column in model_rank_columns if column in output.columns
    ))
    if limit < 0:
        raise ValueError("--model-display-limit must be zero or greater")
    if limit == 0 or len(columns) <= limit:
        return columns
    if "consensus_rank" not in output.columns:
        return columns[:limit]
    consensus = pd.to_numeric(output["consensus_rank"], errors="raise")
    distances = []
    for column in columns:
        ranks = pd.to_numeric(output[column], errors="raise")
        distances.append((float((ranks - consensus).abs().mean()), column))
    distances.sort(key=lambda item: (item[0], item[1]))
    return [column for _, column in distances[:limit]]


def load_finished_actual_winner(
    database: Path, frame: pd.DataFrame
) -> pd.Series | None:
    """Load outcome identity only after the race is known to be finished."""
    status = frame["status"].astype("string").str.strip().str.casefold()
    if not status.eq("finished").all():
        return None
    race_ids = frame["race_id"].drop_duplicates()
    if len(race_ids) != 1:
        raise ValueError("Ranking frame must contain exactly one race")
    with sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    ) as connection:
        schema = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("race_runners")')
        }
        if "is_winner" not in schema:
            raise ValueError("Finished race database is missing is_winner")
        winners = pd.read_sql_query(
            'SELECT runner_number, runner_name FROM "race_runners" '
            "WHERE race_id = ? AND runner_mask = 1 AND status = 'finished' "
            "AND is_winner = 1 ORDER BY runner_number",
            connection,
            params=(int(race_ids.iloc[0]),),
        )
    if len(winners) != 1:
        raise ValueError("Finished race must contain exactly one is_winner=1 row")
    winner_number = int(winners.iloc[0]["runner_number"])
    if winner_number not in set(frame["runner_number"].astype(int)):
        raise ValueError("Actual winner is not present in the active ranking field")
    return winners.iloc[0]


def completed_winner_model_results(
    winner: pd.Series | None,
    output: pd.DataFrame,
    model_features: dict[str, list[str]],
) -> tuple[pd.Series, pd.DataFrame] | None:
    """Return the actual winner and every configured model's rank/features."""
    if winner is None:
        return None
    winner_number = int(winner["runner_number"])
    output_winner = output.loc[output["runner_number"] == winner_number]
    if len(output_winner) != 1:
        raise ValueError("Actual winner is missing or duplicated in ranking output")
    ranked = output_winner.iloc[0]
    rows: list[dict[str, Any]] = []
    for label, features in model_features.items():
        rank_column = f"{label}_rank"
        if rank_column not in output:
            continue
        winner_rank = float(ranked[rank_column])
        rows.append({
            "model": label,
            "winner_rank": winner_rank,
            "winner_correct": winner_rank == 1,
            "feature_count": len(features),
            "features": json.dumps(features),
        })
    results = pd.DataFrame(rows).sort_values(
        ["winner_rank", "model"], kind="stable", ignore_index=True
    )
    return winner, results


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
    using_original_race_models = args.race_models_manifest is not None
    if using_original_race_models:
        model_features, available_models = load_original_race_models(
            args.race_models_manifest
        )
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
        paths = list(available_models.get(label, []))
        if not paths:
            continue
        models = load_models(paths)
        scores[label] = ensemble_rank_scores(
            models, model_feature_matrix(frame, configured_features), race_ids
        )
    market_score = rank_percentiles(market_scores(frame), race_ids)
    scores["market"] = market_score
    own_model = f"race_{args.race_id}"
    deployment_model = (
        own_model
        if using_original_race_models and own_model in scores
        else str(bundle.get("deployment_default", "form"))
    )
    if using_original_race_models and deployment_model not in scores:
        deployment_model = next(iter(model_features))
    if deployment_model not in scores:
        raise ValueError(f"Deployment model is unavailable: {deployment_model}")
    scores["deployment"] = scores[deployment_model]
    scores["selected"] = scores[deployment_model]
    if using_original_race_models:
        scores["consensus"] = np.mean(
            np.column_stack([scores[label] for label in model_features]), axis=1
        )
        if own_model in scores:
            scores["own"] = scores[own_model]
        elif args.ranking == "own":
            raise ValueError(
                f"No original model was trained on race {args.race_id}"
            )
        if args.ranking == "tuned":
            raise ValueError(
                "--ranking tuned uses retrained bundle/OOF strategies; with "
                "--race-models-manifest use --ranking own, consensus, or a "
                "specific race_<id> model"
            )
    diagnostic_weights: dict[str, float] | None = None
    cohort_strategy: str | None = None
    cohort_strategy_metrics: dict[str, Any] | None = None
    cohort_strategy_fallback: str | None = None
    cohort_scope: str | None = None
    exact_cohort_races: int | None = None
    if args.ranking in {"tuned", "market_free", "benchmark"}:
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
                    cohort, cohort_scope, exact_cohort_races = (
                        select_historical_cohort(
                            predictions,
                            int(frame.iloc[0]["competition_id"]),
                            int(frame.iloc[0]["race_number"]),
                        )
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
        elif args.ranking == "market_free":
            config_path = (
                args.market_free_blend_config
                or args.blend_config.parent / "market_free_blend.json"
            ).resolve()
            if not config_path.is_file():
                raise ValueError(
                    f"Current-market-free blend does not exist: {config_path}; run "
                    "build_market_free_winner_blend.py first"
                )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("target_race_market_inputs") is not False:
                raise ValueError(
                    "Current-market-free blend is not marked as market-free"
                )
            diagnostic_weights = dict(config.get("selected_weights", {}))
            if not diagnostic_weights:
                raise ValueError(
                    "Current-market-free blend contains no usable weights"
                )
            market_components = [
                label
                for label, weight in diagnostic_weights.items()
                if float(weight) > 0
                and (
                    label == "market"
                    or label not in model_features
                    or uses_current_market_features(model_features[label])
                )
            ]
            if market_components:
                raise ValueError(
                    "Current-market-free blend uses market components: "
                    + ", ".join(sorted(market_components))
                )
            scores["market_free"] = blend_named_scores(scores, diagnostic_weights)
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

    all_model_rank_columns = [
        f"{label}_rank"
        for label in model_features
        if label in scores
    ]
    displayed_model_rank_columns = all_model_rank_columns
    if using_original_race_models:
        displayed_model_rank_columns = consensus_representative_model_columns(
            output, all_model_rank_columns, args.model_display_limit
        )
    displayed_model_labels = {
        column.removesuffix("_rank")
        for column in displayed_model_rank_columns
    }
    tie_diagnostics = model_prediction_tie_diagnostics(
        scores,
        [column.removesuffix("_rank") for column in all_model_rank_columns],
    )

    race = frame.iloc[0]
    displayed_model = (
        deployment_model
        if args.ranking in {"deployment", "selected", "own"}
        else args.ranking
    )
    displayed_features = model_features.get(displayed_model, [])
    if args.ranking in {"tuned", "market_free"} and diagnostic_weights is not None:
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
    elif args.ranking == "consensus" and using_original_race_models:
        market_used = any(
            feature in MARKET_ENGINEERED_FEATURES
            or is_current_market_feature(feature)
            for features in model_features.values()
            for feature in features
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
    if using_original_race_models:
        heading += (
            f"models_loaded={len(all_model_rank_columns)} "
            f"models_displayed={len(displayed_model_rank_columns)} "
            "display_selection=closest_to_all_model_consensus\n"
            f"models_unique_top={tie_diagnostics['models_unique_top']} "
            f"models_tied_top={tie_diagnostics['models_tied_top']} "
            f"models_constant={tie_diagnostics['models_constant']}\n"
        )
    if diagnostic_weights is not None:
        weight_label = {
            "tuned": "tuned_blend_weights",
            "market_free": "market_free_blend_weights",
        }.get(args.ranking, "diagnostic_benchmark_weights")
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
                f"cohort_scope={cohort_scope} "
                f"exact_race_number_oof_races={exact_cohort_races} "
                f"minimum_exact_cohort_races={MIN_EXACT_COHORT_RACES}\n"
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
    columns = [
        "display_rank", "runner_number", "runner_name", "fluc2",
        f"{args.ranking}_rank",
        *displayed_model_rank_columns, "market_rank",
        "tuned_rank", "market_aware_rank", "benchmark_rank", "market_to_form_upgrade",
        "contrarian_top3",
    ]
    visible_columns = [
        column for column in dict.fromkeys(columns) if column in output.columns
    ]
    print(terminal_table_text(output, visible_columns))
    number_ones = number_one_summary(output, visible_columns)
    print("\nNUMBER-ONE TOTALS")
    if number_ones.empty:
        print("No displayed ranking selected a runner first")
    else:
        print(number_ones.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ))
    rank_totals = model_rank_total_summary(output, all_model_rank_columns)
    print("\nMODEL RANK TOTALS (lower is better)")
    if rank_totals.empty:
        print("No model rank columns are available")
    else:
        print(rank_totals.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ))
    number_one_ranking = number_one_rank_summary(rank_totals)
    print("\nALL-MODEL NUMBER-ONE RANKING (higher votes is better)")
    if number_one_ranking.empty:
        print("No model first-place votes are available")
    else:
        print(number_one_ranking.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ))
    actual_winner = load_finished_actual_winner(args.db, frame)
    result_model_features = model_features
    if using_original_race_models:
        result_model_features = {
            label: features
            for label, features in model_features.items()
            if label in displayed_model_labels
        }
    completed_results = completed_winner_model_results(
        actual_winner, output, result_model_features
    )
    if completed_results is not None:
        actual_winner, model_results = completed_results
        print("\nACTUAL WINNER MODEL RESULTS")
        print(
            f"winner={int(actual_winner['runner_number'])} "
            f"{actual_winner['runner_name']}"
        )
        print(model_results.loc[:, [
            "model", "winner_rank", "winner_correct", "feature_count",
        ]].to_string(index=False))
        correct = model_results.loc[model_results["winner_correct"]]
        print("\nWINNER-CORRECT MODEL FEATURES")
        if correct.empty:
            print("No configured model ranked the actual winner #1")
        else:
            for row in correct.itertuples(index=False):
                print(
                    f"model={row.model} feature_count={row.feature_count}\n"
                    f"features={row.features}"
                )
        strategy_columns = [
            column for column in (f"{args.ranking}_rank", "market_rank")
            if column in output
        ]
        ranked_winner = output.loc[
            output["runner_number"] == int(actual_winner["runner_number"])
        ].iloc[0]
        print("\nACTUAL WINNER STRATEGY RANKS")
        for column in dict.fromkeys(strategy_columns):
            value = float(ranked_winner[column])
            rendered = str(int(value)) if value.is_integer() else f"{value:.1f}"
            print(f"{column.removesuffix('_rank')}={rendered}")
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
    if (
        using_original_race_models
        and str(race.get("status", "")).strip().casefold() == "finished"
    ):
        print(
            "WARNING original per-race models are deliberately in-sample on "
            "their own training races; use these results for feature/model-wars "
            "analysis only."
        )
    if args.output_csv:
        path = args.output_csv.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(path, index=False)
        print(f"saved={path}")


if __name__ == "__main__":
    main()
