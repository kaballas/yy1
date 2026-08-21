#!/usr/bin/env python3
"""Backtest artifact-defined winner blends on saved out-of-fold predictions.

The base-model scores are out of fold, but blend weights may have been selected
on this same OOF cohort. Results are therefore blend-selection diagnostics, not
a sealed future test. The fitted bundle models are deliberately not replayed on
finished races because they were refit on all eligible finished races.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_competition_ids(value: str) -> list[int]:
    """Parse one competition ID or a comma-separated list without duplicates."""
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


def parse_race_numbers(value: str) -> list[int]:
    """Parse one race number or a comma-separated list without duplicates."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "race numbers must be comma-separated integers"
        )
    try:
        race_numbers = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "race numbers must be comma-separated integers"
        ) from exc
    if any(race_number < 1 for race_number in race_numbers):
        raise argparse.ArgumentTypeError("race numbers must be positive integers")
    return list(dict.fromkeys(race_numbers))


def parse_model_labels(value: str) -> list[str]:
    """Parse a non-empty, comma-separated model shortlist."""
    labels = [part.strip() for part in value.split(",")]
    if not labels or any(not label for label in labels):
        raise argparse.ArgumentTypeError(
            "model labels must be non-empty comma-separated names"
        )
    return list(dict.fromkeys(labels))


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
    parser.add_argument(
        "--competition-id",
        type=parse_competition_ids,
        metavar="ID[,ID...]",
        help="Limit the backtest to one or more comma-separated competition IDs.",
    )
    parser.add_argument(
        "--race-number",
        type=parse_race_numbers,
        metavar="NUMBER[,NUMBER...]",
        help="Limit the backtest to one or more comma-separated race numbers.",
    )
    parser.add_argument("--from-date", help="Inclusive UTC date/time filter.")
    parser.add_argument("--to-date", help="Inclusive UTC date/time filter.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional race-level selections, returns, and profits.",
    )
    parser.add_argument(
        "--top-strategies",
        type=int,
        default=5,
        help="Number of leading strategies shown in the concise report.",
    )
    parser.add_argument(
        "--show-all-strategies",
        action="store_true",
        help="Print the full weight matrix and every strategy result.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Run this many Optuna blend-weight trials; zero disables tuning.",
    )
    parser.add_argument("--optuna-jobs", type=int, default=1)
    parser.add_argument("--optuna-seed", type=int, default=42)
    parser.add_argument(
        "--optuna-storage",
        type=Path,
        help=(
            "Persistent SQLite study path. Defaults to winner_blend_optuna.db "
            "beside --blend-config."
        ),
    )
    parser.add_argument(
        "--optuna-study-name",
        help="Study name; defaults to a name derived from the search shortlist.",
    )
    parser.add_argument(
        "--optuna-models",
        type=parse_model_labels,
        metavar="MODEL[,MODEL...]",
        help=(
            "Tune only this model shortlist and assign all other models zero "
            "weight. Useful for focused sparse searches."
        ),
    )
    parser.add_argument(
        "--optuna-include-market",
        action="store_true",
        help="Allow raw market score to receive weight (disabled by default).",
    )
    parser.add_argument(
        "--optuna-output",
        type=Path,
        help="Best-weight JSON path; defaults beside --blend-config.",
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
    competition_id: int | list[int] | None,
    from_date: str | None,
    to_date: str | None,
    race_number: int | list[int] | None = None,
) -> pd.DataFrame:
    race_rows = frame.groupby("race_id", sort=False).head(1).copy()
    keep = pd.Series(True, index=race_rows.index)
    if competition_id is not None:
        if "competition_id" not in race_rows:
            raise ValueError("Predictions have no competition_id column")
        competition_ids = (
            [competition_id] if isinstance(competition_id, int) else competition_id
        )
        keep &= race_rows["competition_id"].isin(competition_ids)
    if race_number is not None:
        if "race_number" not in race_rows:
            raise ValueError("Predictions have no race_number column")
        race_numbers = [race_number] if isinstance(race_number, int) else race_number
        keep &= race_rows["race_number"].isin(race_numbers)
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


def optuna_trial_weights(
    trial: Any,
    model_labels: list[str],
    include_market: bool = False,
) -> dict[str, float]:
    """Suggest continuous non-negative weights and normalize to the simplex."""
    components = [*model_labels, *(["market"] if include_market else [])]
    raw = {
        component: float(trial.suggest_float(f"raw_{component}", 0.0, 1.0))
        for component in components
    }
    total = float(sum(raw.values()))
    if not np.isfinite(total) or total <= 0:
        # This has probability zero for continuous samplers, but keeps custom and
        # test samplers from producing an invalid blend.
        raw = {component: 1.0 for component in components}
        total = float(len(components))
    weights = {component: value / total for component, value in raw.items()}
    if not include_market:
        weights["market"] = 0.0
    return weights


def optuna_cohort_fingerprint(
    frame: pd.DataFrame, model_labels: list[str]
) -> str:
    """Identify the exact OOF cohort and scores used by a persistent study."""
    columns = [
        "race_id", "runner_number", "is_winner", "market_score",
        *(f"{label}_score" for label in model_labels),
    ]
    hashes = pd.util.hash_pandas_object(frame.loc[:, columns], index=False).values
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def optuna_baseline_parameters(
    model_labels: list[str],
    include_market: bool = False,
    pair_steps: int = 0,
) -> list[dict[str, float]]:
    """Return simplex corners, equal blend, and optional pairwise mixtures."""
    if pair_steps < 0:
        raise ValueError("pair_steps must be non-negative")
    components = [*model_labels, *(["market"] if include_market else [])]
    corners = [{
        f"raw_{component}": float(component == selected)
        for component in components
    } for selected in components]
    baselines = [*corners, {
        f"raw_{component}": 1.0 for component in components
    }]
    if pair_steps:
        for left_index, left in enumerate(components):
            for right in components[left_index + 1:]:
                for step in range(1, pair_steps + 1):
                    left_weight = step / (pair_steps + 1)
                    baselines.append({
                        f"raw_{component}": (
                            left_weight if component == left
                            else 1.0 - left_weight if component == right
                            else 0.0
                        )
                        for component in components
                    })
    return baselines


def tune_optuna_blend(
    frame: pd.DataFrame,
    model_labels: list[str],
    *,
    trials: int,
    jobs: int,
    seed: int,
    storage_path: Path,
    study_name: str,
    include_market: bool = False,
    pair_steps: int = 0,
) -> tuple[dict[str, float], dict[str, Any], Any]:
    """Tune blend weights on OOF races and safely resume a persistent study."""
    if trials < 1:
        raise ValueError("optuna-trials must be positive")
    if jobs < 1:
        raise ValueError("optuna-jobs must be positive")
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna tuning requires optuna: pip install -r requirements.txt"
        ) from exc

    resolved_storage = storage_path.resolve()
    resolved_storage.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{resolved_storage}"
    effective_jobs = 1
    if jobs != effective_jobs:
        print(
            "NOTE SQLite-backed Optuna studies run with effective_jobs=1 to "
            "avoid concurrent trial-claim/completion races; "
            f"requested_jobs={jobs}",
            flush=True,
        )
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        constant_liar=False,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    fingerprint = optuna_cohort_fingerprint(frame, model_labels)
    definition = {
        "cohort_fingerprint": fingerprint,
        "model_labels": model_labels,
        "include_market": include_market,
        "pair_steps": pair_steps,
        "primary_objective": "top1_hit_rate",
    }
    previous = study.user_attrs.get("winner_blend_definition")
    if previous is None and study.trials:
        raise ValueError(
            "Existing Optuna study has trials but no winner-blend definition; "
            "choose another --optuna-study-name or storage file"
        )
    if previous is not None and previous != definition:
        raise ValueError(
            "Persistent Optuna study was created for a different cohort or search "
            "space; choose another --optuna-study-name or storage file"
        )
    study.set_user_attr("winner_blend_definition", definition)
    baseline_key = "winner_blend_baselines_enqueued"
    if not study.user_attrs.get(baseline_key, False):
        # Independent U(0, 1) raw weights overwhelmingly produce dense blends in
        # a high-dimensional simplex. Seed its important corners so TPE observes
        # every individual model and can learn toward sparse solutions.
        for parameters in optuna_baseline_parameters(
            model_labels, include_market, pair_steps
        ):
            study.enqueue_trial(parameters)
        study.set_user_attr(baseline_key, True)
    targets = frame["is_winner"].to_numpy(dtype=np.int64)
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)

    def objective(trial: Any) -> float:
        weights = optuna_trial_weights(trial, model_labels, include_market)
        metrics = winner_metrics(
            targets, strategy_scores(frame, model_labels, weights), race_ids
        )
        for name, value in metrics.items():
            trial.set_user_attr(name, float(value))
        trial.set_user_attr("normalized_weights", weights)
        return float(metrics["top1_hit_rate"])

    study.optimize(objective, n_trials=trials, n_jobs=effective_jobs)
    best = study.best_trial
    weights = {
        str(name): float(value)
        for name, value in best.user_attrs["normalized_weights"].items()
    }
    metrics = {
        name: float(best.user_attrs[name])
        for name in (
            "top1_hit_rate", "top3_hit_rate", "mrr", "mean_winner_rank",
            "race_logloss", "races",
        )
    }
    return weights, metrics, study


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


def best_backtest_strategy(
    frame: pd.DataFrame,
    model_labels: list[str],
    strategies: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Select the same first-ranked strategy displayed in the OVERALL table."""
    summary, _ = backtest_summary(frame, model_labels, strategies)
    best = summary.iloc[0]
    name = str(best["strategy"])
    return name, dict(strategies[name]), best.to_dict()


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


def blend_weights_table(
    model_labels: list[str], strategies: dict[str, dict[str, float]]
) -> pd.DataFrame:
    """Show the normalized component weights actually used by each strategy."""
    components = [*model_labels, "market"]
    rows: list[dict[str, Any]] = []
    for name, weights in strategies.items():
        configured_sum = float(sum(weights.values()))
        rows.append({
            "strategy": name,
            **{
                component: float(weights.get(component, 0.0)) / configured_sum
                for component in components
            },
            "configured_sum": configured_sum,
        })
    return pd.DataFrame(rows, columns=["strategy", *components, "configured_sum"])


def main() -> None:
    args = parse_args()
    if args.top_strategies < 1:
        raise ValueError("--top-strategies must be positive")
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
        args.race_number,
    )
    optuna_result: tuple[dict[str, float], dict[str, Any], Any] | None = None
    if args.optuna_trials:
        optimized_labels = args.optuna_models or model_labels
        unknown_optuna_labels = sorted(set(optimized_labels) - set(model_labels))
        if unknown_optuna_labels:
            raise ValueError(
                "Unknown --optuna-models: " + ", ".join(unknown_optuna_labels)
            )
        sparse_search = args.optuna_models is not None
        study_name = args.optuna_study_name
        if study_name is None:
            study_name = "winner_blend_weights"
            if sparse_search:
                study_name += "_sparse_" + "_".join(optimized_labels)
        storage_path = args.optuna_storage or (
            args.blend_config.parent / "winner_blend_optuna.db"
        )
        optuna_result = tune_optuna_blend(
            frame,
            optimized_labels,
            trials=args.optuna_trials,
            jobs=args.optuna_jobs,
            seed=args.optuna_seed,
            storage_path=storage_path,
            study_name=study_name,
            include_market=args.optuna_include_market,
            pair_steps=9 if sparse_search else 0,
        )
        optuna_weights, optuna_metrics, study = optuna_result
        strategies["optuna_best"] = optuna_weights
        default_output_name = "winner_blend_optuna_best.json"
        if sparse_search:
            default_output_name = (
                "winner_blend_optuna_best_sparse_"
                + "_".join(optimized_labels)
                + ".json"
            )
        optuna_output = args.optuna_output or (
            args.blend_config.parent / default_output_name
        )
        resolved_output = optuna_output.resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        resolved_output.write_text(json.dumps({
            "schema_version": 1,
            "selection_scope": "saved_out_of_fold_predictions",
            "sealed_test_used": False,
            "study_name": study.study_name,
            "storage": str(storage_path.resolve()),
            "best_trial_number": study.best_trial.number,
            "best_value": study.best_value,
            "completed_trials": len(study.trials),
            "model_labels": model_labels,
            "optimized_model_labels": optimized_labels,
            "include_market": args.optuna_include_market,
            "selected_weights": optuna_weights,
            "oof_metrics": optuna_metrics,
            "cohort_fingerprint": optuna_cohort_fingerprint(
                frame, optimized_labels
            ),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "OPTUNA BLEND TUNING COMPLETE\n"
            f"study={study.study_name} trials={len(study.trials)} "
            f"best_trial={study.best_trial.number} "
            f"top1={study.best_value:.6f}\n"
            f"selected_weights={json.dumps(optuna_weights, sort_keys=True)}\n"
            f"saved_optuna_best={resolved_output}",
            flush=True,
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
    market_row = summary.loc[summary["strategy"].eq("raw_market_benchmark")]
    market_top1 = (
        float(market_row["top1_hit_rate"].iloc[0]) if not market_row.empty else np.nan
    )
    market_roi = (
        float(market_row["flat_win_roi"].iloc[0]) if not market_row.empty else np.nan
    )
    core_names = {
        "config_selected", "bundle_selected", "bundle_all_finished_tuned",
        "bundle_deployment", "raw_market_benchmark", "optuna_best",
    }
    leading_names = summary.head(args.top_strategies)["strategy"].tolist()
    shown_names = set(leading_names) | core_names
    shown = summary.loc[summary["strategy"].isin(shown_names)].copy()
    shown["top1_vs_market"] = shown["top1_hit_rate"] - market_top1
    shown["roi_vs_market"] = shown["flat_win_roi"] - market_roi
    concise_columns = [
        "strategy", "races", "top1_hit_rate", "top1_vs_market",
        "top3_hit_rate", "mrr", "flat_win_roi", "roi_vs_market",
    ]
    print("\nDECISION SUMMARY (ranked by top-1, deltas versus raw market)")
    print(shown.loc[:, concise_columns].to_string(
        index=False, float_format=lambda value: f"{value:.5f}"
    ))
    best = summary.iloc[0]
    print(
        f"recommendation=best_oof_strategy strategy={best['strategy']} "
        f"top1={best['top1_hit_rate']:.2%} top3={best['top3_hit_rate']:.2%} "
        f"roi={best['flat_win_roi']:.2%}"
    )
    for name in shown["strategy"]:
        weights = strategies[str(name)]
        nonzero = {
            key: round(float(value) / sum(weights.values()), 6)
            for key, value in weights.items() if float(value) > 0
        }
        if len(nonzero) > 1 or str(name) in core_names:
            print(f"weights[{name}]={json.dumps(nonzero, sort_keys=True)}")

    if args.show_all_strategies:
        print("\nALL BLEND WEIGHTS (normalized values used)")
        print(blend_weights_table(model_labels, strategies).to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        ))
        print("\nALL STRATEGIES")
        print(summary.loc[:, columns].to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        ))
    if folds is not None:
        fold_names = {
            str(best["strategy"]), "config_selected", "bundle_deployment",
            "raw_market_benchmark", "optuna_best",
        }
        focus = folds.loc[
            folds["strategy"].isin(fold_names),
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
