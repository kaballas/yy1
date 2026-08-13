#!/usr/bin/env python3
"""Rank one active race with a current-market-free deployment model.

Current prices are displayed as a benchmark, but they do not affect the
default ranking. Market-aware and blended benchmark rankings are diagnostic
choices that must be requested explicitly.
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
    blend_scores,
    ensemble_rank_scores,
    form_matrix,
    is_current_market_feature,
    market_aware_matrix,
    market_scores,
    rank_percentiles,
)
from src.advanced_racing_features import race_relative_runner_mask


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
        choices=(
            "deployment", "form", "tuned", "market_aware", "benchmark", "market",
            "selected",
        ),
        default="deployment",
        help=(
            "Ranking to display. 'deployment' (default), 'form', and legacy "
            "'selected' are current-market-free. Other choices are explicit "
            "market diagnostics."
        ),
    )
    parser.add_argument(
        "--blend-config",
        type=Path,
        default=Path("outputs/winner_ranker/form_market_aware_blend.json"),
        help="Validation-selected form/market-aware weights used by --ranking tuned.",
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
    if "form" not in scores or "market" not in scores:
        raise ValueError("Ranking output requires form and market benchmark scores")
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
    output["market_to_form_upgrade"] = output["market_rank"] - output["form_rank"]
    output["contrarian_top3"] = (
        (output["form_rank"] <= 3) & (output["market_rank"] > 3)
    ).astype(int)
    output = output.sort_values(
        [f"{ranking}_rank", "runner_number"], kind="stable", ignore_index=True
    )
    output.insert(0, "display_rank", np.arange(1, len(output) + 1))
    return output


def main() -> None:
    args = parse_args()
    bundle = load_bundle(args.bundle)
    features = list(bundle["form_features"])
    forbidden = [feature for feature in features if is_current_market_feature(feature)]
    if forbidden:
        raise ValueError(
            "Refusing deployment bundle with current-race market form inputs: "
            + ", ".join(forbidden)
        )
    frame = load_active_race(args.db, args.race_id, features)
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
    form_models = load_models(list(bundle["models"]["form"]))
    form_score = ensemble_rank_scores(form_models, form_matrix(frame, features), race_ids)
    market_score = rank_percentiles(market_scores(frame), race_ids)
    # Deployment is hard-wired to the form ensemble. It cannot silently inherit
    # market weights from an older bundle. ``selected`` remains as a legacy alias.
    scores: dict[str, np.ndarray] = {
        "form": form_score,
        "deployment": form_score,
        "selected": form_score,
        "market": market_score,
    }
    diagnostic_weights: dict[str, float] | None = None
    if args.ranking in {"tuned", "market_aware", "benchmark"}:
        aware_paths = list(bundle.get("models", {}).get("market_aware", []))
        if not aware_paths:
            raise ValueError(
                "This bundle has no market-aware diagnostic models. Retrain with "
                "--include-market-aware-benchmark."
            )
        aware_models = load_models(aware_paths)
        aware_score = ensemble_rank_scores(
            aware_models, market_aware_matrix(frame, features), race_ids
        )
        scores["market_aware"] = aware_score
        if args.ranking == "tuned":
            config_path = args.blend_config.resolve()
            if not config_path.is_file():
                raise ValueError(
                    f"Blend recommendation does not exist: {config_path}; run "
                    "backtest_winner_blend.py first"
                )
            config = json.loads(config_path.read_text(encoding="utf-8"))
            diagnostic_weights = dict(config.get("selected_weights", {}))
            if float(diagnostic_weights.get("market", 0.0)) != 0.0:
                raise ValueError("Tuned form/market-aware blend must have raw market weight zero")
            scores["tuned"] = blend_scores(
                form_score, aware_score, market_score, diagnostic_weights
            )
        elif args.ranking == "benchmark":
            diagnostic_weights = bundle.get("benchmark_blend_weights")
            if not diagnostic_weights:
                raise ValueError("Bundle has no validation-selected benchmark blend")
            scores["benchmark"] = blend_scores(
                form_score, aware_score, market_score,
                dict(diagnostic_weights),
            )
    output = ranked_output(frame, scores, args.ranking)

    race = frame.iloc[0]
    market_used = args.ranking in {
        "market", "tuned", "market_aware", "benchmark",
    }
    heading = (
        "WINNER RANKER\n"
        f"race={args.race_id} {race['race_name']} venue={race['competition_name']} "
        f"start={race['start_time_iso']} active_runners={len(frame)}\n"
        f"display_ranking={args.ranking} "
        f"current_market_used_in_display_ranking={'yes' if market_used else 'no'}\n"
        "deployment_weights={\"form\": 1.0, \"market\": 0.0, "
        "\"market_aware\": 0.0}\n"
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
    heading += (
        "market_to_form_upgrade: positive means the form model promotes the runner\n"
        "contrarian_top3: form top three while outside market top three"
    )
    print(heading)
    columns = [
        "display_rank", "runner_number", "runner_name", "fluc2",
        f"{args.ranking}_score", "form_rank", "market_rank",
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
