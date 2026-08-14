#!/usr/bin/env python3
"""Backtest artifact-defined winner blends on saved out-of-fold predictions.

The base-model scores are out of fold, but blend weights may have been selected
on this same OOF cohort. Results are therefore blend-selection diagnostics, not
a sealed future test. The fitted bundle models are deliberately not replayed on
finished races because they were refit on all eligible finished races.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Winner-ranker bundle containing deployment and tuned weights.",
    )
    parser.add_argument(
        "--blend-config",
        type=Path,
        required=True,
        help="All-finished blend recommendation to evaluate.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Saved OOF predictions. Defaults to all_finished_oof_predictions.csv "
            "beside --blend-config."
        ),
    )
    parser.add_argument("--competition-id", type=int)
    parser.add_argument("--from-date", help="Inclusive UTC date/time filter.")
    parser.add_argument("--to-date", help="Inclusive UTC date/time filter.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional race-level selections, returns, and profits.",
    )
    return parser.parse_args()


def blend_named_scores(
    scores: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Match the production normalized, non-negative named-score blend."""
    unknown = set(weights) - set(scores)
    if unknown:
        raise ValueError(f"Unknown blend components: {sorted(unknown)}")
    values = np.asarray(list(weights.values()), dtype=np.float64)
    total = float(values.sum())
    if not np.isfinite(values).all() or np.any(values < 0) or total <= 0:
        raise ValueError("Blend weights must be finite, non-negative, and non-zero")
    return sum(
        float(weight) * np.asarray(scores[name], dtype=np.float64)
        for name, weight in weights.items()
    ) / total


def winner_metrics(
    targets: np.ndarray, scores: np.ndarray, race_ids: np.ndarray
) -> dict[str, float]:
    """Calculate the same equal-race metrics used by the winner ranker."""
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    race_ids = np.asarray(race_ids)
    if not (
        targets.shape == scores.shape == race_ids.shape
    ) or not len(targets) or not np.isfinite(scores).all():
        raise ValueError("Winner metric inputs must be finite, non-empty, and equal")
    ranks: list[int] = []
    losses: list[float] = []
    for race_id in pd.unique(race_ids):
        positions = np.flatnonzero(race_ids == race_id)
        race_targets = targets[positions]
        if int(race_targets.sum()) != 1:
            raise ValueError(f"race_id {race_id} does not have exactly one winner")
        race_scores = scores[positions]
        order = np.argsort(-race_scores, kind="stable")
        winner = int(np.flatnonzero(race_targets == 1)[0])
        ranks.append(int(np.flatnonzero(order == winner)[0]) + 1)
        shifted = race_scores - race_scores.max()
        losses.append(
            float(-(shifted[winner] - np.log(np.exp(shifted).sum())))
        )
    rank = np.asarray(ranks, dtype=np.float64)
    return {
        "top1_hit_rate": float(np.mean(rank == 1)),
        "top3_hit_rate": float(np.mean(rank <= 3)),
        "mrr": float(np.mean(1.0 / rank)),
        "mean_winner_rank": float(np.mean(rank)),
        "race_logloss": float(np.mean(losses)),
        "races": float(len(rank)),
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return payload


def artifact_strategies(
    bundle: dict[str, Any], blend: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Collect every distinct, relevant blend without silently reconciling it."""
    model_labels = [str(label) for label in blend.get("model_labels", [])]
    if not model_labels:
        model_labels = [str(label) for label in bundle.get("models", {})]
    if not model_labels:
        raise ValueError("Artifacts contain no model labels")

    candidates = {
        "config_selected": blend.get("selected_weights"),
        "bundle_selected": bundle.get("selected_blend_weights"),
        "bundle_all_finished_tuned": bundle.get(
            "all_finished_tuned_blend_weights"
        ),
        "bundle_deployment": bundle.get("deployment_blend_weights"),
    }
    strategies: dict[str, dict[str, float]] = {}
    allowed = {*model_labels, "market"}
    for name, raw_weights in candidates.items():
        if not isinstance(raw_weights, dict) or not raw_weights:
            continue
        unknown = sorted(set(raw_weights) - allowed)
        if unknown:
            raise ValueError(f"{name} has unknown components: {unknown}")
        weights = {
            label: float(raw_weights.get(label, 0.0))
            for label in [*model_labels, "market"]
        }
        values = np.asarray(list(weights.values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError(f"{name} weights must be finite, non-negative, and non-zero")
        strategies[name] = weights

    equal_weight = 1.0 / len(model_labels)
    strategies["equal_model_blend"] = {
        **{label: equal_weight for label in model_labels},
        "market": 0.0,
    }
    for label in model_labels:
        strategies[f"{label}_only"] = {
            **{candidate: float(candidate == label) for candidate in model_labels},
            "market": 0.0,
        }
    strategies["raw_market_benchmark"] = {
        **{label: 0.0 for label in model_labels},
        "market": 1.0,
    }
    return model_labels, strategies


def load_predictions(path: Path, model_labels: list[str]) -> pd.DataFrame:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"OOF predictions do not exist: {resolved}")
    frame = pd.read_csv(resolved)
    required = {
        "race_id",
        "runner_number",
        "is_winner",
        "fluc2",
        "market_score",
        *(f"{label}_score" for label in model_labels),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("OOF predictions are missing: " + ", ".join(missing))
    if frame.duplicated(["race_id", "runner_number"]).any():
        raise ValueError("OOF predictions contain duplicate race/runner rows")

    numeric = [
        "race_id", "runner_number", "is_winner", "fluc2", "market_score",
        *(f"{label}_score" for label in model_labels),
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    score_columns = ["market_score", *(f"{label}_score" for label in model_labels)]
    if not np.isfinite(frame[score_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("All OOF model and market scores must be finite")
    if not frame["is_winner"].isin([0, 1]).all():
        raise ValueError("is_winner must contain only zero and one")
    winners = frame.groupby("race_id", sort=False)["is_winner"].sum()
    if not (winners == 1).all():
        bad = winners.loc[winners != 1].index.astype(str).tolist()[:10]
        raise ValueError("Every OOF race must have one winner; bad=" + ",".join(bad))
    sort_columns = [
        column
        for column in ("start_time_iso", "race_id", "runner_number")
        if column in frame
    ]
    return frame.sort_values(sort_columns, kind="stable", ignore_index=True)


def filter_complete_races(
    frame: pd.DataFrame,
    competition_id: int | None,
    from_date: str | None,
    to_date: str | None,
) -> pd.DataFrame:
    race_rows = frame.groupby("race_id", sort=False).head(1).copy()
    keep = pd.Series(True, index=race_rows.index)
    if competition_id is not None:
        if "competition_id" not in race_rows:
            raise ValueError("Predictions have no competition_id column")
        keep &= race_rows["competition_id"] == competition_id
    if from_date is not None or to_date is not None:
        if "start_time_iso" not in race_rows:
            raise ValueError("Predictions have no start_time_iso column")
        times = pd.to_datetime(race_rows["start_time_iso"], errors="coerce", utc=True)
        if times.isna().any():
            raise ValueError("Predictions contain invalid start_time_iso values")
        if from_date is not None:
            start = pd.to_datetime(from_date, errors="raise", utc=True)
            keep &= times >= start
        if to_date is not None:
            end = pd.to_datetime(to_date, errors="raise", utc=True)
            keep &= times <= end
    race_ids = set(race_rows.loc[keep, "race_id"])
    filtered = frame.loc[frame["race_id"].isin(race_ids)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No complete OOF races match the requested filters")
    return filtered


def strategy_scores(
    frame: pd.DataFrame,
    model_labels: list[str],
    weights: dict[str, float],
) -> np.ndarray:
    available = {
        label: frame[f"{label}_score"].to_numpy(dtype=np.float64)
        for label in model_labels
    }
    available["market"] = frame["market_score"].to_numpy(dtype=np.float64)
    return blend_named_scores(available, weights)


def race_selections(
    frame: pd.DataFrame, strategy: str, scores: np.ndarray
) -> pd.DataFrame:
    working = frame.copy()
    working["strategy"] = strategy
    working["strategy_score"] = scores
    working["strategy_rank"] = working.groupby("race_id", sort=False)[
        "strategy_score"
    ].rank(method="first", ascending=False).astype(int)
    selected = working.loc[working["strategy_rank"] == 1].copy()
    price = pd.to_numeric(selected["fluc2"], errors="coerce")
    selected["priced_selection"] = np.isfinite(price) & (price > 1.0)
    selected["stake"] = selected["priced_selection"].astype(float)
    selected["return"] = np.where(
        selected["priced_selection"] & (selected["is_winner"] == 1), price, 0.0
    )
    selected["profit"] = selected["return"] - selected["stake"]
    return selected


def backtest_summary(
    frame: pd.DataFrame,
    model_labels: list[str],
    strategies: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = frame["is_winner"].to_numpy(dtype=np.int64)
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    summary_rows: list[dict[str, Any]] = []
    selection_frames: list[pd.DataFrame] = []
    for name, weights in strategies.items():
        scores = strategy_scores(frame, model_labels, weights)
        metrics = winner_metrics(targets, scores, race_ids)
        selections = race_selections(frame, name, scores)
        priced = selections.loc[selections["priced_selection"]]
        stake = float(priced["stake"].sum())
        profit = float(priced["profit"].sum())
        summary_rows.append({
            "strategy": name,
            "weight_sum": float(sum(weights.values())),
            **metrics,
            "priced_races": int(len(priced)),
            "missing_price_races": int(len(selections) - len(priced)),
            "flat_win_profit": profit,
            "flat_win_roi": profit / stake if stake else np.nan,
        })
        selection_frames.append(selections)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["top1_hit_rate", "mrr", "strategy"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    return summary, pd.concat(selection_frames, ignore_index=True)


def fold_summary(
    frame: pd.DataFrame,
    model_labels: list[str],
    strategies: dict[str, dict[str, float]],
) -> pd.DataFrame | None:
    if "crossfit_fold" not in frame:
        return None
    rows: list[pd.DataFrame] = []
    for fold, fold_frame in frame.groupby("crossfit_fold", sort=True):
        summary, _ = backtest_summary(fold_frame, model_labels, strategies)
        summary.insert(1, "crossfit_fold", fold)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    bundle = load_json(args.bundle, "Bundle")
    blend = load_json(args.blend_config, "Blend config")
    if bundle.get("objective") != "single_winner_ranking":
        raise ValueError("Bundle is not a single-winner ranking artifact")
    model_labels, strategies = artifact_strategies(bundle, blend)
    predictions_path = args.predictions or (
        args.blend_config.parent / "all_finished_oof_predictions.csv"
    )
    frame = filter_complete_races(
        load_predictions(predictions_path, model_labels),
        args.competition_id,
        args.from_date,
        args.to_date,
    )
    summary, selections = backtest_summary(frame, model_labels, strategies)
    folds = fold_summary(frame, model_labels, strategies)

    print("ALL-FINISHED WINNER BLEND OOF BACKTEST")
    print(
        f"bundle={args.bundle.resolve()}\n"
        f"blend_config={args.blend_config.resolve()}\n"
        f"predictions={predictions_path.resolve()}\n"
        f"rows={len(frame):,} races={frame['race_id'].nunique():,} "
        f"models={','.join(model_labels)}"
    )
    print(
        "WARNING base scores are out-of-fold, but configured blend weights were "
        "selected on this cohort. This is not a sealed future backtest."
    )
    config_weights = strategies.get("config_selected")
    bundle_weights = strategies.get("bundle_all_finished_tuned")
    if config_weights != bundle_weights:
        print(
            "WARNING blend config and bundle tuned weights differ; both are shown "
            "as separate strategies."
        )
    non_unit = {
        name: sum(weights.values())
        for name, weights in strategies.items()
        if not np.isclose(sum(weights.values()), 1.0)
    }
    if non_unit:
        print(
            "NOTE non-unit weights are normalized by the production blend helper: "
            + json.dumps(non_unit, sort_keys=True)
        )

    columns = [
        "strategy", "weight_sum", "races", "top1_hit_rate", "top3_hit_rate",
        "mrr", "mean_winner_rank", "race_logloss", "priced_races",
        "flat_win_profit", "flat_win_roi",
    ]
    print("\nOVERALL")
    print(summary.loc[:, columns].to_string(
        index=False, float_format=lambda value: f"{value:.5f}"
    ))
    if folds is not None:
        focus = folds.loc[
            folds["strategy"].isin([
                "config_selected", "bundle_selected",
                "bundle_all_finished_tuned", "bundle_deployment",
                "raw_market_benchmark",
            ]),
            [
                "strategy", "crossfit_fold", "races", "top1_hit_rate",
                "top3_hit_rate", "mrr", "flat_win_roi",
            ],
        ]
        print("\nBY CROSSFIT FOLD")
        print(focus.to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    if args.output_csv:
        output = args.output_csv.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        selections.to_csv(output, index=False)
        print(f"saved={output} rows={len(selections):,}")


if __name__ == "__main__":
    main()
