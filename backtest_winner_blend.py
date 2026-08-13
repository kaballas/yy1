#!/usr/bin/env python3
"""Tune a form/market-aware winner blend without using raw market scores.

Weights are selected only on the chronological validation predictions. The
chosen weight is then evaluated once on the later sealed test predictions.
Raw market ranking is reported as a benchmark but is never part of the blend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.winner_ranker import market_deviation_metrics, winner_metrics


FORM_SCORE_ALIASES = ("form_score", "form_deployment_score")
MARKET_AWARE_SCORE_ALIASES = (
    "market_aware_score",
    "market_aware_benchmark_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-predictions",
        type=Path,
        default=Path("outputs/winner_ranker/validation_predictions.csv"),
    )
    parser.add_argument(
        "--test-predictions",
        type=Path,
        default=Path("outputs/winner_ranker/test_predictions.csv"),
    )
    parser.add_argument(
        "--objective",
        choices=("top1", "mrr", "top3", "composite"),
        default="top1",
        help="Validation-only metric used to select the weight.",
    )
    parser.add_argument(
        "--weight-step",
        type=float,
        default=0.001,
        help="Form-weight grid interval; 0.001 evaluates 1,001 mixtures.",
    )
    parser.add_argument(
        "--minimum-form-weight",
        type=float,
        default=0.0,
        help="Optional lower bound on form weight.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/winner_ranker/form_market_aware_blend.json"),
    )
    parser.add_argument(
        "--sweep-csv",
        type=Path,
        default=Path("outputs/winner_ranker/form_market_aware_weight_sweep.csv"),
    )
    return parser.parse_args()


def _find_column(frame: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
    for name in aliases:
        if name in frame.columns:
            return name
    raise ValueError(
        f"Predictions contain no {label} score; expected one of {list(aliases)}"
    )


def load_prediction_cohort(path: Path) -> tuple[pd.DataFrame, str, str]:
    """Load and validate one complete-race prediction cohort."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Prediction file does not exist: {resolved}")
    frame = pd.read_csv(resolved)
    required = {"race_id", "runner_number", "is_winner"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Prediction file is missing: " + ", ".join(missing))
    form_column = _find_column(frame, FORM_SCORE_ALIASES, "form")
    aware_column = _find_column(
        frame, MARKET_AWARE_SCORE_ALIASES, "market-aware"
    )
    if frame.duplicated(["race_id", "runner_number"]).any():
        raise ValueError("Prediction file has duplicate race/runner rows")
    for column in ("race_id", "runner_number", "is_winner", form_column, aware_column):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame["is_winner"].isin([0, 1]).all():
        raise ValueError("is_winner must contain only zero or one")
    if frame[[form_column, aware_column]].isna().any().any():
        raise ValueError("Form and market-aware scores must be complete and numeric")
    if not np.isfinite(frame[[form_column, aware_column]].to_numpy()).all():
        raise ValueError("Form and market-aware scores must be finite")
    winners = frame.groupby("race_id", sort=False)["is_winner"].sum()
    if not (winners == 1).all():
        bad = winners.loc[winners != 1].index.astype(str).tolist()[:10]
        raise ValueError("Every race must have exactly one winner; bad=" + ",".join(bad))
    sort_columns = [
        column for column in ("start_time_iso", "race_id", "runner_number")
        if column in frame.columns
    ]
    frame = frame.sort_values(sort_columns, kind="stable", ignore_index=True)
    return frame, form_column, aware_column


def validate_holdout_order(validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Require disjoint cohorts and, when available, strict chronology."""
    overlap = set(validation["race_id"]) & set(test["race_id"])
    if overlap:
        raise ValueError(
            "Validation and test cohorts overlap; refusing leakage-prone tuning"
        )
    if "start_time_iso" in validation and "start_time_iso" in test:
        validation_time = pd.to_datetime(
            validation["start_time_iso"], errors="coerce", utc=True
        )
        test_time = pd.to_datetime(test["start_time_iso"], errors="coerce", utc=True)
        if validation_time.isna().any() or test_time.isna().any():
            raise ValueError("Prediction cohorts contain invalid start_time_iso values")
        if validation_time.max() >= test_time.min():
            raise ValueError(
                "Test races must occur strictly after all validation races"
            )


def candidate_form_weights(step: float, minimum_form_weight: float) -> np.ndarray:
    """Return a deterministic grid including both allowed endpoints."""
    if not 0.0 < step <= 1.0:
        raise ValueError("weight-step must be in (0, 1]")
    if not 0.0 <= minimum_form_weight <= 1.0:
        raise ValueError("minimum-form-weight must be in [0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0, atol=1e-12):
        raise ValueError("weight-step must divide 1.0 exactly")
    weights = np.linspace(0.0, 1.0, units + 1, dtype=np.float64)
    return weights[weights >= minimum_form_weight - 1e-12]


def winner_ranks_for_weights(
    frame: pd.DataFrame,
    form_column: str,
    aware_column: str,
    form_weights: np.ndarray,
) -> np.ndarray:
    """Calculate winner ranks for every weight efficiently and deterministically.

    Score ties use row order, matching the stable ranking used elsewhere. Each
    race is handled independently so large and small fields have equal influence.
    """
    weights = np.asarray(form_weights, dtype=np.float64)
    ranks = np.empty((len(weights), frame["race_id"].nunique()), dtype=np.int32)
    for race_index, (_, race) in enumerate(frame.groupby("race_id", sort=False)):
        form = race[form_column].to_numpy(dtype=np.float64)
        aware = race[aware_column].to_numpy(dtype=np.float64)
        target = race["is_winner"].to_numpy(dtype=np.int64)
        winner_position = int(np.flatnonzero(target == 1)[0])
        score = aware[:, None] + (form - aware)[:, None] * weights[None, :]
        winner_score = score[winner_position]
        positions = np.arange(len(race))[:, None]
        better = (score > winner_score) | (
            (score == winner_score) & (positions < winner_position)
        )
        ranks[:, race_index] = 1 + better.sum(axis=0)
    return ranks


def metrics_from_ranks(ranks: np.ndarray) -> pd.DataFrame:
    rank = np.asarray(ranks, dtype=np.float64)
    return pd.DataFrame({
        "top1_hit_rate": np.mean(rank == 1, axis=1),
        "top3_hit_rate": np.mean(rank <= 3, axis=1),
        "mrr": np.mean(1.0 / rank, axis=1),
        "mean_winner_rank": np.mean(rank, axis=1),
    })


def _objective_values(metrics: pd.DataFrame, objective: str) -> np.ndarray:
    if objective == "top1":
        return metrics["top1_hit_rate"].to_numpy()
    if objective == "top3":
        return metrics["top3_hit_rate"].to_numpy()
    if objective == "mrr":
        return metrics["mrr"].to_numpy()
    # A transparent equal-scale summary. Top-one receives the largest share
    # because the task is winner selection, while MRR rewards useful ordering.
    return (
        0.50 * metrics["top1_hit_rate"].to_numpy()
        + 0.30 * metrics["mrr"].to_numpy()
        + 0.20 * metrics["top3_hit_rate"].to_numpy()
    )


def select_form_weight(
    frame: pd.DataFrame,
    form_column: str,
    aware_column: str,
    weights: np.ndarray,
    objective: str,
) -> tuple[float, pd.DataFrame]:
    """Select a weight on one tuning cohort; never inspect the test cohort."""
    ranks = winner_ranks_for_weights(frame, form_column, aware_column, weights)
    sweep = metrics_from_ranks(ranks)
    sweep.insert(0, "market_aware_weight", 1.0 - weights)
    sweep.insert(0, "form_weight", weights)
    sweep["objective_value"] = _objective_values(sweep, objective)
    # Deterministic lexicographic selection. For equal ranking metrics, prefer
    # the centre of the tied plateau because it is less sensitive to tiny score
    # changes than choosing an edge. Remaining ties prefer more form weight.
    best_objective = float(sweep["objective_value"].max())
    candidates = sweep.loc[np.isclose(
        sweep["objective_value"], best_objective, rtol=0.0, atol=1e-15
    )].copy()
    for column, ascending in (
        ("mrr", False),
        ("top1_hit_rate", False),
        ("top3_hit_rate", False),
        ("mean_winner_rank", True),
    ):
        best = candidates[column].min() if ascending else candidates[column].max()
        candidates = candidates.loc[np.isclose(
            candidates[column], best, rtol=0.0, atol=1e-15
        )]
    plateau_centre = float(candidates["form_weight"].median())
    chosen_index = (
        (candidates["form_weight"] - plateau_centre).abs()
    ).sort_values(kind="stable").index[0]
    return float(sweep.loc[chosen_index, "form_weight"]), sweep


def blend_two_scores(
    frame: pd.DataFrame,
    form_column: str,
    aware_column: str,
    form_weight: float,
) -> np.ndarray:
    form = frame[form_column].to_numpy(dtype=np.float64)
    aware = frame[aware_column].to_numpy(dtype=np.float64)
    return form_weight * form + (1.0 - form_weight) * aware


def cohort_metrics(
    frame: pd.DataFrame,
    form_column: str,
    aware_column: str,
    selected_form_weight: float,
) -> tuple[dict[str, dict[str, float]], dict[str, float] | None]:
    target = frame["is_winner"].to_numpy(dtype=np.int64)
    race_ids = frame["race_id"].to_numpy()
    form = frame[form_column].to_numpy(dtype=np.float64)
    aware = frame[aware_column].to_numpy(dtype=np.float64)
    selected = blend_two_scores(
        frame, form_column, aware_column, selected_form_weight
    )
    scores = {
        "form_only": form,
        "market_aware_only": aware,
        "equal_blend": 0.5 * form + 0.5 * aware,
        "tuned_blend": selected,
    }
    if "market_score" in frame.columns:
        scores["raw_market_benchmark"] = pd.to_numeric(
            frame["market_score"], errors="coerce"
        ).to_numpy(dtype=np.float64)
    metrics = {
        name: winner_metrics(target, score, race_ids)
        for name, score in scores.items()
    }
    deviation: dict[str, float] | None = None
    if "market_rank" in frame.columns:
        audit = frame[[
            "race_id", "runner_number", "is_winner", "market_rank",
        ]].copy()
        audit["tuned_blend_rank"] = (
            pd.Series(selected).groupby(audit["race_id"], sort=False).rank(
                method="first", ascending=False
            ).astype(int)
        )
        deviation = market_deviation_metrics(audit, "tuned_blend")
    return metrics, deviation


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    validation, validation_form, validation_aware = load_prediction_cohort(
        args.validation_predictions
    )
    test, test_form, test_aware = load_prediction_cohort(args.test_predictions)
    validate_holdout_order(validation, test)
    weights = candidate_form_weights(args.weight_step, args.minimum_form_weight)
    selected_form_weight, sweep = select_form_weight(
        validation, validation_form, validation_aware, weights, args.objective
    )
    selected_weights = {
        "form": selected_form_weight,
        "market_aware": 1.0 - selected_form_weight,
        "market": 0.0,
    }
    validation_metrics, validation_deviation = cohort_metrics(
        validation, validation_form, validation_aware, selected_form_weight
    )
    test_metrics, test_deviation = cohort_metrics(
        test, test_form, test_aware, selected_form_weight
    )

    print("FORM / MARKET-AWARE BLEND BACKTEST")
    print(
        f"tuning_cohort=validation validation_races={validation['race_id'].nunique():,} "
        f"sealed_test_races={test['race_id'].nunique():,} objective={args.objective} "
        f"weight_step={args.weight_step:g}"
    )
    print(
        "selected_weights=" + json.dumps(selected_weights, sort_keys=True)
        + " raw_market_weight=0.0"
    )
    for cohort, metrics in (
        ("VALIDATION (WEIGHT SELECTION)", validation_metrics),
        ("SEALED TEST (NO RETUNING)", test_metrics),
    ):
        print(cohort)
        table = pd.DataFrame(metrics).T[[
            "top1_hit_rate", "top3_hit_rate", "mrr",
            "mean_winner_rank", "race_logloss",
        ]]
        print(table.to_string(float_format=lambda value: f"{value:.5f}"))
    if test_deviation is not None:
        print(
            "SEALED TEST MARKET DEVIATION\n"
            + pd.Series(test_deviation).to_string(
                float_format=lambda value: f"{value:.5f}"
            )
        )

    output = {
        "schema_version": 1,
        "blend": "form_plus_market_aware_no_raw_market",
        "raw_market_weight_fixed": 0.0,
        "selection_cohort": "validation",
        "objective": args.objective,
        "weight_step": args.weight_step,
        "minimum_form_weight": args.minimum_form_weight,
        "selected_weights": selected_weights,
        "validation_predictions": str(args.validation_predictions.resolve()),
        "test_predictions": str(args.test_predictions.resolve()),
        "validation_metrics": validation_metrics,
        "sealed_test_metrics": test_metrics,
        "validation_market_deviation": validation_deviation,
        "sealed_test_market_deviation": test_deviation,
    }
    json_path = args.output_json.resolve()
    sweep_path = args.sweep_csv.resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sweep.to_csv(sweep_path, index=False)
    print(f"saved_recommendation={json_path}")
    print(f"saved_validation_sweep={sweep_path} rows={len(sweep):,}")
    print(
        "rank_command=python rank_winner_models.py --race-id RACE_ID "
        f"--ranking tuned --blend-config {json_path}"
    )


if __name__ == "__main__":
    main()
