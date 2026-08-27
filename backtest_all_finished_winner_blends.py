#!/usr/bin/env python3
"""Evaluate artifact-defined winner blends on OOF or database predictions.

By default, base-model scores come from saved out-of-fold predictions, but blend
weights may have been selected on that same cohort. ``--predict-db`` instead
runs the fitted bundle over eligible finished database races; those predictions
are in-sample when the bundle was trained on the same races. Neither mode is a
sealed future test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_DB


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


def parse_date(value: str) -> str:
    """Validate and retain an ISO calendar date."""
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYY-MM-DD") from exc


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
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Race database used for --predict-db and --date bundle inference.",
    )
    parser.add_argument(
        "--predict-db",
        action="store_true",
        help=(
            "Run the fitted bundle models over eligible finished database races "
            "instead of reading saved OOF predictions. Competition/date/race "
            "filters are applied before inference. Results may be in-sample."
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
        "--date",
        type=parse_date,
        help="Limit the backtest to this UTC date from start_time_iso (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--train-per-race",
        action="store_true",
        help=(
            "With --date, train and self-validate one analysis model for every "
            "race, saving each model and its exact feature metadata."
        ),
    )
    parser.add_argument(
        "--models-test-dir",
        type=Path,
        default=Path("models_test"),
        help="Output directory for --train-per-race models and diagnostics.",
    )
    parser.add_argument(
        "--per-race-estimators",
        type=int,
        default=300,
        help="Trees fitted by each deliberately overfit per-race model.",
    )
    parser.add_argument(
        "--per-race-feature-manifest",
        type=Path,
        default=Path("winner_ranker_features.json"),
        help=(
            "Feature candidates for --train-per-race (default: "
            "winner_ranker_features.json)."
        ),
    )
    parser.add_argument(
        "--per-race-feature-models",
        nargs="+",
        metavar="MODEL",
        help=(
            "Use only these model groups from --per-race-feature-manifest "
            "(for example: --per-race-feature-models top3). Comma-separated "
            "names are also accepted."
        ),
    )
    parser.add_argument(
        "--per-race-feature-trials",
        type=int,
        default=12,
        help="Maximum nested feature subsets tested per race (default: 12).",
    )
    parser.add_argument(
        "--per-race-select-blend",
        action="store_true",
        help=(
            "After selecting a feature subset, test an equal-weight ensemble of "
            "rankers with different tree settings and save the best in-sample "
            "rank/margin result."
        ),
    )
    parser.add_argument(
        "--per-race-blend-members",
        type=int,
        default=4,
        help="Maximum model variants considered by --per-race-select-blend.",
    )
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
    ranks: list[float] = []
    losses: list[float] = []
    for race_id in pd.unique(race_ids):
        positions = np.flatnonzero(race_ids == race_id)
        race_targets = targets[positions]
        if int(race_targets.sum()) != 1:
            raise ValueError(f"race_id {race_id} does not have exactly one winner")
        race_scores = scores[positions]
        winner = int(np.flatnonzero(race_targets == 1)[0])
        ranks.append(float(pd.Series(race_scores).rank(
            method="average", ascending=False
        ).iloc[winner]))
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
    if "status" not in frame:
        warnings.warn(
            "Legacy all-finished OOF predictions have no status column; treating "
            "all rows as status=finished because the generating training query "
            "only selects finished races",
            RuntimeWarning,
            stacklevel=2,
        )
        frame["status"] = "finished"
    frame = frame.loc[frame["status"].eq("finished")].copy()
    if frame.empty:
        raise ValueError("OOF predictions contain no rows with status=finished")

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


def load_finished_date_rows(
    database: Path,
    exact_date: str,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load eligible finished race rows for a UTC date without model inference."""
    from src.winner_ranker import (
        database_numeric_columns,
        eligible_races,
        load_training_rows,
        rows_for_races,
    )
    resolved_database = database.resolve()
    if not resolved_database.is_file():
        raise ValueError(f"Database does not exist: {resolved_database}")
    numeric_columns = database_numeric_columns(resolved_database)
    all_finished = load_training_rows(
        resolved_database,
        numeric_columns,
        categorical_columns=categorical_columns or [],
    )
    times = pd.to_datetime(
        all_finished["start_time_iso"], errors="coerce", utc=True
    )
    if times.isna().any():
        raise ValueError("Database contains invalid start_time_iso values")
    requested_date = pd.Timestamp(exact_date, tz="UTC")
    dated = all_finished.loc[times.dt.normalize().eq(requested_date)].copy()
    races = eligible_races(dated)
    if races.empty:
        raise ValueError(
            f"No eligible status=finished database races exist on {exact_date}"
        )
    return rows_for_races(dated, races["race_id"].astype(int).tolist())


def load_finished_database_rows(
    database: Path,
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load every eligible finished race currently present in the database."""
    from src.winner_ranker import (
        database_numeric_columns,
        eligible_races,
        load_training_rows,
        rows_for_races,
    )
    resolved_database = database.resolve()
    if not resolved_database.is_file():
        raise ValueError(f"Database does not exist: {resolved_database}")
    numeric_columns = database_numeric_columns(resolved_database)
    all_finished = load_training_rows(
        resolved_database,
        numeric_columns,
        categorical_columns=categorical_columns or [],
    )
    races = eligible_races(all_finished)
    if races.empty:
        raise ValueError("Database contains no eligible finished races")
    return rows_for_races(all_finished, races["race_id"].astype(int).tolist())


def score_finished_rows_from_bundle(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    model_labels: list[str],
) -> pd.DataFrame:
    """Score supplied eligible finished race rows with fitted bundle models."""
    try:
        from xgboost import XGBRanker
    except ImportError as exc:  # pragma: no cover - CLI environment failure
        raise SystemExit("xgboost is required: pip install xgboost") from exc
    from src.winner_ranker import (
        ensemble_rank_scores,
        market_scores,
        model_feature_matrix,
        prepare_categorical_features,
        rank_percentiles,
    )
    categorical_features = [
        str(feature) for feature in bundle.get("categorical_features", [])
    ]
    levels = {
        str(feature): [str(value) for value in values]
        for feature, values in bundle.get("categorical_levels", {}).items()
        if isinstance(values, list)
    }
    frame = prepare_categorical_features(frame, levels)
    output = frame.copy()
    race_ids = frame["race_id"].to_numpy(dtype=np.int64)
    configured_features = dict(bundle.get("model_features", {}))
    configured_models = dict(bundle.get("models", {}))
    for label in model_labels:
        features = list(configured_features.get(label, []))
        paths = list(configured_models.get(label, []))
        if not features or not paths:
            raise ValueError(f"Bundle cannot score model {label}: missing features/models")
        models: list[Any] = []
        for path in paths:
            model = XGBRanker()
            model.load_model(path)
            models.append(model)
        matrix = model_feature_matrix(frame, features)
        expected_categorical = set(categorical_features) & set(features)
        for feature in expected_categorical:
            if not isinstance(matrix[feature].dtype, pd.CategoricalDtype):
                raise ValueError(
                    f"Bundle model {label} expects categorical feature {feature}, "
                    "but its backtest matrix is not categorical"
                )
        for member, model in enumerate(models, start=1):
            feature_types = list(model.feature_types or [])
            if expected_categorical and len(feature_types) != len(features):
                raise ValueError(
                    f"Bundle model {label} member {member} has incompatible "
                    "feature-type metadata"
                )
            mismatched = [
                feature
                for index, feature in enumerate(features)
                if expected_categorical
                and ((feature in expected_categorical) != (feature_types[index] == "c"))
            ]
            if mismatched:
                raise ValueError(
                    f"Bundle model {label} member {member} categorical metadata "
                    "mismatch: " + ", ".join(mismatched)
                )
        score = ensemble_rank_scores(
            models, matrix, race_ids
        )
        output[f"{label}_score"] = score
        output[f"{label}_rank"] = pd.Series(score, index=output.index).groupby(
            output["race_id"], sort=False
        ).rank(method="first", ascending=False).astype("Int64")
    market_score = rank_percentiles(market_scores(frame), race_ids)
    output["market_score"] = market_score
    output["market_rank"] = pd.Series(
        market_score, index=output.index
    ).groupby(
        output["race_id"], sort=False
    ).rank(method="first", ascending=False).astype("Int64")
    categorical_by_model = {
        label: [
            feature for feature in configured_features.get(label, [])
            if feature in categorical_features
        ]
        for label in model_labels
    }
    categorical_by_model = {
        label: features
        for label, features in categorical_by_model.items()
        if features
    }
    print(
        "native_categorical_by_model="
        + json.dumps(categorical_by_model, sort_keys=True),
        flush=True,
    )
    return output


def score_finished_date_from_bundle(
    database: Path,
    bundle: dict[str, Any],
    model_labels: list[str],
    exact_date: str,
) -> pd.DataFrame:
    """Score all eligible finished races on one UTC date with fitted models."""
    return score_finished_rows_from_bundle(
        load_finished_date_rows(
            database,
            exact_date,
            [str(feature) for feature in bundle.get("categorical_features", [])],
        ),
        bundle,
        model_labels,
    )


def load_per_race_candidate_features(
    manifest_path: Path,
    requested_models: list[str] | None = None,
) -> list[str]:
    """Load the ordered union of feature groups from a winner feature manifest."""
    resolved = manifest_path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Per-race feature manifest does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    groups = payload.get("models")
    if not isinstance(groups, dict) or not groups:
        raise ValueError(
            f"Per-race feature manifest has no model feature groups: {resolved}"
        )
    requested = list(dict.fromkeys(requested_models or groups.keys()))
    unknown = sorted(set(requested) - set(groups))
    if unknown:
        raise ValueError(
            "Requested per-race feature models are absent from the manifest: "
            + ", ".join(unknown)
        )
    candidates: list[str] = []
    for label in requested:
        configured = groups[label]
        if not isinstance(configured, dict):
            raise ValueError(f"Feature group {label!r} must be a JSON object")
        features = configured.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"Feature group {label!r} has no features")
        if any(not isinstance(feature, str) or not feature for feature in features):
            raise ValueError(f"Feature group {label!r} has invalid feature names")
        candidates.extend(features)
    return list(dict.fromkeys(candidates))


def normalize_per_race_feature_models(
    requested: list[str] | None,
) -> list[str] | None:
    """Accept space-separated and comma-separated manifest group names."""
    if requested is None:
        return None
    labels: list[str] = []
    for value in requested:
        parts = [part.strip() for part in value.split(",")]
        if any(not part for part in parts):
            raise ValueError("--per-race-feature-models contains an empty name")
        labels.extend(parts)
    return list(dict.fromkeys(labels))


def per_race_feature_subsets(
    matrix: pd.DataFrame,
    targets: np.ndarray,
    maximum_trials: int,
) -> tuple[list[list[str]], list[dict[str, float | str]]]:
    """Build nested subsets ordered by in-race winner separation."""
    if maximum_trials < 1:
        raise ValueError("--per-race-feature-trials must be positive")
    winner_positions = np.flatnonzero(np.asarray(targets, dtype=np.int64) == 1)
    if len(winner_positions) != 1:
        raise ValueError("Per-race feature search requires exactly one winner")
    winner_index = int(winner_positions[0])
    priorities: list[dict[str, float | str]] = []
    for feature in matrix.columns:
        values = pd.to_numeric(matrix[feature], errors="coerce")
        losers = values.drop(index=values.index[winner_index]).dropna()
        winner_value = values.iloc[winner_index]
        if pd.isna(winner_value) or losers.empty:
            separation = -1.0
        else:
            scale = float(losers.std(ddof=0))
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = max(float((losers.max() - losers.min())), 1.0)
            separation = abs(float(winner_value) - float(losers.median())) / scale
        priorities.append({"feature": str(feature), "winner_separation": separation})
    priorities.sort(
        key=lambda item: (-float(item["winner_separation"]), str(item["feature"]))
    )
    ordered = [str(item["feature"]) for item in priorities]
    if not ordered:
        return [], priorities
    if maximum_trials == 1:
        sizes = [len(ordered)]
    else:
        sizes = sorted(set(
            int(round(value))
            for value in np.geomspace(1, len(ordered), num=maximum_trials)
        ))
        if len(ordered) not in sizes:
            sizes.append(len(ordered))
        if len(sizes) > maximum_trials:
            sizes = sizes[-maximum_trials:]
    return [ordered[:size] for size in sizes], priorities


def winner_rank_and_margin(
    scores: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    """Return the winner's tie-aware rank and lead over the best rival."""
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    winner_index = int(np.flatnonzero(labels == 1)[0])
    winner_rank = float(pd.Series(values).rank(
        method="average", ascending=False
    ).iloc[winner_index])
    rivals = values[labels != 1]
    margin = float(values[winner_index] - rivals.max())
    return winner_rank, margin


def train_per_race_analysis_models(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    output_dir: Path,
    estimators: int,
    feature_manifest: Path,
    feature_models: list[str] | None,
    feature_trials: int,
    select_blend: bool,
    blend_members: int,
) -> pd.DataFrame:
    """Train, self-validate, and save one diagnostic model for every race."""
    if estimators < 1:
        raise ValueError("--per-race-estimators must be positive")
    if blend_members < 1:
        raise ValueError("--per-race-blend-members must be positive")
    try:
        from xgboost import XGBRanker
    except ImportError as exc:  # pragma: no cover - CLI environment failure
        raise SystemExit("xgboost is required: pip install xgboost") from exc
    from src.winner_ranker import (
        IDENTIFIER_COLUMNS,
        MARKET_ENGINEERED_FEATURES,
        OUTCOME_OR_CONTROL_COLUMNS,
        model_feature_matrix,
        rank_percentiles,
    )

    manifest_candidates = load_per_race_candidate_features(
        feature_manifest, feature_models
    )
    forbidden = OUTCOME_OR_CONTROL_COLUMNS | IDENTIFIER_COLUMNS
    candidate_features = [
        feature for feature in manifest_candidates
        if feature not in forbidden
        and (
            feature in frame.columns
            or feature in MARKET_ENGINEERED_FEATURES
        )
    ]
    unavailable_features = [
        feature for feature in manifest_candidates
        if feature not in forbidden
        and feature not in frame.columns
        and feature not in MARKET_ENGINEERED_FEATURES
    ]
    excluded_features = [
        feature for feature in manifest_candidates if feature in forbidden
    ]
    if not candidate_features:
        raise ValueError("Feature manifest contains no available safe candidates")
    print(
        "PER-RACE FEATURE SEARCH START\n"
        f"feature_manifest={feature_manifest.resolve()} "
        f"feature_models={json.dumps(feature_models or ['all'])} "
        f"manifest_candidates={len(manifest_candidates):,} "
        f"available_safe_candidates={len(candidate_features):,} "
        f"unavailable={len(unavailable_features):,} "
        f"excluded_outcome_or_identifier={len(excluded_features):,} "
        f"subset_trials={feature_trials:,} "
        f"blend_search={'yes' if select_blend else 'no'}",
        flush=True,
    )
    resolved_output = output_dir.resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved_output / "per_race_models_manifest.json"
    analysis_path = resolved_output / "per_race_model_analysis.csv"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        existing_manifest = {}
    models_by_race = {
        int(item["trained_on_race_id"]): item
        for item in existing_manifest.get("models", [])
        if isinstance(item, dict) and "trained_on_race_id" in item
    }
    training_dates = set(map(str, existing_manifest.get("training_dates", [])))
    for item in models_by_race.values():
        saved_date = item.get("training_date_utc")
        if not saved_date and isinstance(item.get("details"), dict):
            saved_time = item["details"].get("start_time_iso")
            if saved_time:
                saved_date = str(pd.to_datetime(saved_time, utc=True).date())
        if saved_date:
            training_dates.add(str(saved_date))
    rows: list[dict[str, Any]] = []
    race_total = int(frame["race_id"].nunique())
    for race_position, (race_id, race) in enumerate(
        frame.groupby("race_id", sort=False), start=1
    ):
        matrix = model_feature_matrix(race, candidate_features)
        usable = [
            feature
            for feature in matrix.columns
            if int(matrix[feature].nunique(dropna=True)) > 1
        ]
        coverage = float(matrix.notna().mean().mean()) if matrix.size else 0.0
        race_number = int(race.iloc[0]["race_number"])
        competition_id = int(race.iloc[0]["competition_id"])
        training_date = str(
            pd.to_datetime(race.iloc[0]["start_time_iso"], utc=True).date()
        )
        training_dates.add(training_date)
        targets = race["is_winner"].to_numpy(dtype=np.int64)
        winner_index = int(np.flatnonzero(targets == 1)[0])
        model_name = f"race_{int(race_id)}"
        model_path = resolved_output / f"{model_name}.json"
        features_path = resolved_output / f"{model_name}_features.json"
        self_validation_rank: int | None = None
        self_validation_margin: float | None = None
        gain_by_feature: dict[str, float] = {}
        selected_features: list[str] = []
        selected_model_paths: list[str] = []
        feature_search_trials: list[dict[str, Any]] = []
        blend_search_trials: list[dict[str, Any]] = []
        selected_models: list[Any] = []
        selected_training_parameters: list[dict[str, Any]] = []
        if usable:
            usable_matrix = matrix.loc[:, usable]
            subsets, feature_priority = per_race_feature_subsets(
                usable_matrix, targets, feature_trials
            )
            print(
                f"per_race_search={race_position:,}/{race_total:,} "
                f"race_id={int(race_id)} runners={len(race):,} "
                f"manifest_candidates={len(manifest_candidates):,} "
                f"varying_candidates={len(usable):,} subsets={len(subsets):,}",
                flush=True,
            )
            best_key: tuple[float, int, float] | None = None
            best_model: Any | None = None
            best_scores: np.ndarray | None = None
            base_parameters = {
                "objective": "rank:ndcg",
                "eval_metric": "ndcg@1",
                "n_estimators": estimators,
                "max_depth": 4,
                "learning_rate": 0.10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 0.0,
                "reg_lambda": 0.0,
                "reg_alpha": 0.0,
                "tree_method": "hist",
                "n_jobs": 1,
            }
            for trial_index, subset in enumerate(subsets, start=1):
                parameters = {
                    **base_parameters,
                    "random_state": 42 + int(race_id) + trial_index,
                }
                model = XGBRanker(**parameters)
                model.fit(
                    matrix.loc[:, subset], targets,
                    group=[len(race)], verbose=False,
                )
                scores = np.asarray(
                    model.predict(matrix.loc[:, subset]), dtype=np.float64
                )
                winner_rank, winner_margin = winner_rank_and_margin(scores, targets)
                trial = {
                    "trial": trial_index,
                    "feature_count": len(subset),
                    "features": subset,
                    "winner_rank": winner_rank,
                    "winner_margin": winner_margin,
                    "parameters": parameters,
                }
                feature_search_trials.append(trial)
                key = (winner_rank, len(subset), -winner_margin)
                if best_key is None or key < best_key:
                    best_key = key
                    best_model = model
                    best_scores = scores
                    selected_features = list(subset)
                    selected_training_parameters = [parameters]
            assert best_model is not None and best_scores is not None
            selected_models = [best_model]
            self_validation_rank, self_validation_margin = winner_rank_and_margin(
                best_scores, targets
            )

            if select_blend:
                variant_settings = [
                    (4, 0.10, estimators),
                    (2, 0.05, estimators),
                    (3, 0.20, estimators),
                    (6, 0.05, estimators),
                    (1, 0.20, estimators),
                    (5, 0.10, max(50, estimators // 2)),
                ][:blend_members]
                variant_models = [best_model]
                variant_parameters = [selected_training_parameters[0]]
                variant_scores = [
                    rank_percentiles(
                        best_scores, np.full(len(race), int(race_id), dtype=np.int64)
                    )
                ]
                for variant_index, (depth, learning_rate, trees) in enumerate(
                    variant_settings[1:], start=2
                ):
                    parameters = {
                        **base_parameters,
                        "n_estimators": trees,
                        "max_depth": depth,
                        "learning_rate": learning_rate,
                        "random_state": 4200 + int(race_id) + variant_index,
                    }
                    variant = XGBRanker(**parameters)
                    variant.fit(
                        matrix.loc[:, selected_features], targets,
                        group=[len(race)], verbose=False,
                    )
                    raw_scores = np.asarray(
                        variant.predict(matrix.loc[:, selected_features]),
                        dtype=np.float64,
                    )
                    variant_models.append(variant)
                    variant_parameters.append(parameters)
                    variant_scores.append(rank_percentiles(
                        raw_scores,
                        np.full(len(race), int(race_id), dtype=np.int64),
                    ))
                best_members = (0,)
                single_rank, single_margin = winner_rank_and_margin(
                    variant_scores[0], targets
                )
                best_blend_key = (single_rank, -single_margin, 1)
                blend_search_trials.append({
                    "members": [1], "winner_rank": single_rank,
                    "winner_margin": single_margin,
                })
                for member_count in range(2, len(variant_models) + 1):
                    for member_indexes in combinations(
                        range(len(variant_models)), member_count
                    ):
                        blended = np.mean(
                            np.stack([variant_scores[i] for i in member_indexes]),
                            axis=0,
                        )
                        blend_rank, blend_margin = winner_rank_and_margin(
                            blended, targets
                        )
                        blend_search_trials.append({
                            "members": [i + 1 for i in member_indexes],
                            "winner_rank": blend_rank,
                            "winner_margin": blend_margin,
                        })
                        blend_key = (blend_rank, -blend_margin, member_count)
                        if blend_key < best_blend_key:
                            best_blend_key = blend_key
                            best_members = member_indexes
                            self_validation_rank = blend_rank
                            self_validation_margin = blend_margin
                selected_models = [variant_models[i] for i in best_members]
                selected_training_parameters = [
                    variant_parameters[i] for i in best_members
                ]

            for member_number, selected_model in enumerate(selected_models, start=1):
                selected_path = (
                    model_path
                    if len(selected_models) == 1
                    else resolved_output / f"{model_name}_member_{member_number}.json"
                )
                selected_model_paths.append(str(selected_path))
            gain_totals: dict[str, list[float]] = {}
            for selected_model in selected_models:
                for feature, gain in selected_model.get_booster().get_score(
                    importance_type="gain"
                ).items():
                    gain_totals.setdefault(str(feature), []).append(float(gain))
            gain_by_feature = {
                feature: float(np.mean(gains))
                for feature, gains in gain_totals.items()
            }
            model_details = {
                "analysis_type": "single_race_in_sample_feature_search_ranker",
                "selection_warning": (
                    "Features, model and optional blend were selected and validated "
                    "on this same finished race; results are deliberately in-sample."
                ),
                "race_id": int(race_id),
                "competition_id": competition_id,
                "competition_name": str(race.iloc[0]["competition_name"]),
                "race_number": race_number,
                "race_name": str(race.iloc[0]["race_name"]),
                "start_time_iso": str(race.iloc[0]["start_time_iso"]),
                "training_rows": len(race),
                "winner_runner_number": int(
                    race.iloc[winner_index]["runner_number"]
                ),
                "winner_runner_name": str(race.iloc[winner_index]["runner_name"]),
                "self_validation_winner_rank": self_validation_rank,
                "self_validation_winner_margin": self_validation_margin,
                "feature_manifest": str(feature_manifest.resolve()),
                "feature_models": feature_models,
                "manifest_candidate_count": len(manifest_candidates),
                "available_candidate_count": len(candidate_features),
                "varying_candidate_count": len(usable),
                "input_features": selected_features,
                "split_feature_gain": gain_by_feature,
                "selection_criterion": (
                    "winner_rank, then fewer features, then larger winner margin"
                ),
                "feature_priority": feature_priority,
                "feature_search_trials": feature_search_trials,
                "blend_enabled": select_blend,
                "blend_search_trials": blend_search_trials,
                "ensemble_members": len(selected_models),
                "ensemble_weights": [1.0 / len(selected_models)] * len(selected_models),
                "training_parameters": selected_training_parameters[0],
                "ensemble_training_parameters": selected_training_parameters,
            }
            for selected_model, selected_path in zip(
                selected_models, selected_model_paths
            ):
                selected_model.get_booster().set_attr(
                    analysis_details=json.dumps(model_details, sort_keys=True),
                    analysis_type="single_race_in_sample_feature_search_ranker",
                    race_id=str(int(race_id)),
                    input_features=json.dumps(selected_features),
                    split_feature_gain=json.dumps(gain_by_feature, sort_keys=True),
                    self_validation_winner_rank=str(self_validation_rank),
                )
                selected_model.save_model(selected_path)
            print(
                f"per_race_selected={race_position:,}/{race_total:,} "
                f"race_id={int(race_id)} winner_rank={self_validation_rank} "
                f"features={len(selected_features):,} "
                f"ensemble_members={len(selected_models):,}",
                flush=True,
            )
        else:
            model_details = {
                "analysis_type": "single_race_in_sample_feature_search_ranker",
                "race_id": int(race_id),
                "error": "No runner-varying input features",
            }
        feature_payload = {
            "race_id": int(race_id),
            "competition_id": competition_id,
            "race_number": race_number,
            "model": selected_model_paths[0] if selected_model_paths else str(model_path),
            "models": selected_model_paths,
            "feature_manifest": str(feature_manifest.resolve()),
            "feature_models": feature_models,
            "manifest_candidate_features": manifest_candidates,
            "unavailable_features": unavailable_features,
            "excluded_outcome_or_identifier_features": excluded_features,
            "candidate_features": candidate_features,
            "varying_features": usable,
            "input_features": selected_features,
            "split_feature_gain": gain_by_feature,
            "self_validation_winner_rank": self_validation_rank,
            "self_validation_winner_margin": self_validation_margin,
            "feature_search_trials": feature_search_trials,
            "blend_search_trials": blend_search_trials,
            "model_details": model_details,
        }
        features_path.write_text(
            json.dumps(feature_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        models_by_race[int(race_id)] = {
            "name": model_name,
            "trained_on_race_id": int(race_id),
            "training_date_utc": training_date,
            "model": selected_model_paths[0] if selected_model_paths else str(model_path),
            "models": selected_model_paths,
            "features_file": str(features_path),
            "details": model_details,
        }
        rows.append({
            "race_id": int(race_id),
            "training_date_utc": training_date,
            "competition_id": competition_id,
            "race_number": race_number,
            "runners": len(race),
            "candidate_features": len(candidate_features),
            "varying_features": len(usable),
            "mean_feature_coverage": coverage,
            "nonzero_gain_features": len(gain_by_feature),
            "selected_features": len(selected_features),
            "feature_trials": len(feature_search_trials),
            "ensemble_members": len(selected_models),
            "input_features": json.dumps(selected_features),
            "split_feature_gain": json.dumps(gain_by_feature, sort_keys=True),
            "model_saved": bool(selected_models),
            "model": selected_model_paths[0] if selected_model_paths else str(model_path),
            "features_file": str(features_path),
            "self_validation_winner_rank": self_validation_rank,
        })
    summary = pd.DataFrame(rows)
    if analysis_path.is_file():
        previous_summary = pd.read_csv(analysis_path)
        previous_summary = previous_summary.loc[
            ~previous_summary["race_id"].isin(summary["race_id"])
        ]
        combined_summary = pd.concat(
            [previous_summary, summary], ignore_index=True
        ).sort_values(
            ["training_date_utc", "race_id"], kind="stable", ignore_index=True
        )
    else:
        combined_summary = summary
    combined_summary.to_csv(analysis_path, index=False)
    manifest = {
        "schema_version": 5,
        "feature_manifest": str(feature_manifest.resolve()),
        "feature_models": feature_models,
        "feature_search_trials": feature_trials,
        "blend_search_enabled": select_blend,
        "blend_members_considered": blend_members,
        "training_dates": sorted(training_dates),
        "models": list(models_by_race.values()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary.attrs["manifest_total_models"] = len(models_by_race)
    summary.attrs["manifest_training_dates"] = sorted(training_dates)
    return summary


def filter_complete_races(
    frame: pd.DataFrame,
    competition_id: int | list[int] | None,
    from_date: str | None,
    to_date: str | None,
    race_number: int | list[int] | None = None,
    exact_date: str | None = None,
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
    if exact_date is not None and (from_date is not None or to_date is not None):
        raise ValueError("--date cannot be combined with --from-date or --to-date")
    if exact_date is not None or from_date is not None or to_date is not None:
        if "start_time_iso" not in race_rows:
            raise ValueError("Predictions have no start_time_iso column")
        times = pd.to_datetime(race_rows["start_time_iso"], errors="coerce", utc=True)
        if times.isna().any():
            raise ValueError("Predictions contain invalid start_time_iso values")
        if exact_date is not None:
            requested_date = pd.Timestamp(exact_date, tz="UTC")
            keep &= times.dt.normalize() == requested_date
        if from_date is not None:
            start = pd.to_datetime(from_date, errors="raise", utc=True)
            keep &= times >= start
        if to_date is not None:
            end = pd.to_datetime(to_date, errors="raise", utc=True)
            keep &= times <= end
    race_ids = set(race_rows.loc[keep, "race_id"])
    filtered = frame.loc[frame["race_id"].isin(race_ids)].reset_index(drop=True)
    if filtered.empty:
        if exact_date is not None and "start_time_iso" in frame:
            available = pd.to_datetime(
                frame["start_time_iso"], errors="coerce", utc=True
            ).dropna()
            if not available.empty:
                raise ValueError(
                    f"No complete finished OOF races exist on {exact_date}; "
                    f"available_utc_dates={available.min().date()}.."
                    f"{available.max().date()}"
                )
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


def print_per_race_training_report(
    args: argparse.Namespace,
    frame: pd.DataFrame,
    analysis: pd.DataFrame,
) -> None:
    """Print the standalone per-race feature-search result."""
    trained = int(analysis["model_saved"].sum())
    top3 = int(analysis["self_validation_winner_rank"].le(3).sum())
    print("PER-RACE WINNER FEATURE SEARCH COMPLETE")
    print(
        f"database={args.db.resolve()} date_utc={args.date}\n"
        f"rows={len(frame):,} races={frame['race_id'].nunique():,}\n"
        f"per_race_models_saved={trained:,}/{len(analysis):,} "
        f"self_validation_winner_top3={top3:,}/{len(analysis):,} "
        f"models_test_dir={args.models_test_dir.resolve()}\n"
        f"per_race_feature_manifest={args.per_race_feature_manifest.resolve()} "
        f"feature_models={json.dumps(normalize_per_race_feature_models(args.per_race_feature_models) or ['all'])} "
        f"feature_trials={args.per_race_feature_trials} "
        f"blend_search={'yes' if args.per_race_select_blend else 'no'}\n"
        f"manifest_total_models="
        f"{int(analysis.attrs['manifest_total_models']):,} "
        f"manifest_training_dates="
        f"{json.dumps(analysis.attrs['manifest_training_dates'])}\n"
        f"per_race_analysis_csv="
        f"{(args.models_test_dir.resolve() / 'per_race_model_analysis.csv')}\n"
        "WARNING feature/model selection and validation used the same finished "
        "race; this is deliberately in-sample analysis."
    )


def main() -> None:
    args = parse_args()
    per_race_feature_models = normalize_per_race_feature_models(
        args.per_race_feature_models
    )
    if args.top_strategies < 1:
        raise ValueError("--top-strategies must be positive")
    if args.train_per_race and args.date is None:
        raise ValueError("--train-per-race requires --date")
    if args.train_per_race and args.predict_db:
        raise ValueError("--train-per-race cannot be combined with --predict-db")
    if args.predict_db and args.predictions is not None:
        raise ValueError("--predict-db cannot be combined with --predictions")
    if args.per_race_feature_trials < 1:
        raise ValueError("--per-race-feature-trials must be positive")
    if not 1 <= args.per_race_blend_members <= 6:
        raise ValueError("--per-race-blend-members must be between 1 and 6")
    bundle = load_json(args.bundle, "Bundle")
    blend = load_json(args.blend_config, "Blend config")
    if bundle.get("objective") != "single_winner_ranking":
        raise ValueError("Bundle is not a single-winner ranking artifact")
    if args.train_per_race:
        frame = filter_complete_races(
            load_finished_date_rows(
                args.db,
                args.date,
                [str(feature) for feature in bundle.get("categorical_features", [])],
            ),
            args.competition_id,
            None,
            None,
            args.race_number,
            args.date,
        )
        per_race_analysis = train_per_race_analysis_models(
            frame,
            bundle,
            args.models_test_dir,
            args.per_race_estimators,
            args.per_race_feature_manifest,
            per_race_feature_models,
            args.per_race_feature_trials,
            args.per_race_select_blend,
            args.per_race_blend_members,
        )
        print_per_race_training_report(args, frame, per_race_analysis)
        return
    model_labels, strategies = artifact_strategies(bundle, blend)
    predictions_path = args.predictions or (
        args.blend_config.parent / "all_finished_oof_predictions.csv"
    )
    database_inference = args.predict_db or args.date is not None
    evaluation_mode = "saved_out_of_fold_predictions"
    if database_inference:
        database_rows = (
            load_finished_date_rows(
                args.db,
                args.date,
                [str(feature) for feature in bundle.get("categorical_features", [])],
            )
            if args.date is not None and not args.predict_db
            else load_finished_database_rows(
                args.db,
                [str(feature) for feature in bundle.get("categorical_features", [])],
            )
        )
        database_rows = filter_complete_races(
            database_rows,
            args.competition_id,
            args.from_date,
            args.to_date,
            args.race_number,
            args.date,
        )
        frame = score_finished_rows_from_bundle(
            database_rows, bundle, model_labels
        )
        evaluation_mode = "fitted_bundle_database_inference"
    else:
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

    report_label = (
        "ALL-FINISHED WINNER BUNDLE DATABASE EVALUATION"
        if database_inference
        else "ALL-FINISHED WINNER BLEND OOF BACKTEST"
    )
    print(report_label)
    print(
        f"bundle={args.bundle.resolve()}\n"
        f"blend_config={args.blend_config.resolve()}\n"
        f"evaluation_mode={evaluation_mode}\n"
        + (
            f"database={args.db.resolve()}"
            + (f" date_utc={args.date}" if args.date is not None else "")
            + "\n"
            if database_inference
            else f"predictions={predictions_path.resolve()}\n"
        )
        +
        f"rows={len(frame):,} races={frame['race_id'].nunique():,} "
        f"models={','.join(model_labels)}"
    )
    if database_inference:
        print(
            "WARNING scores come from fitted bundle models, not OOF models. If "
            "the bundle trained on these races, this evaluation is in-sample and "
            "must not be treated as a historical backtest."
        )
    else:
        print(
            "WARNING base scores are out-of-fold, but configured blend weights "
            "were selected on this cohort. This is not a sealed future backtest."
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
        *(f"{label}_only" for label in model_labels),
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
        f"recommendation={'best_database_strategy' if database_inference else 'best_oof_strategy'} "
        f"strategy={best['strategy']} "
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
            *(f"{label}_only" for label in model_labels),
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
