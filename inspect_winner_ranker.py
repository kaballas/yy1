#!/usr/bin/env python3
"""Inspect what one saved XGBoost winner ranker learned for a finished race.

The command reconstructs the model's exact ordered feature matrix, ranks every
runner, and uses XGBoost's native TreeSHAP contributions to explain why the
selected runner was placed above the actual winner (or above the runner-up when
the selection was correct). It also displays the saved cross-fit score when an
all-finished OOF file is available.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from xgboost import DMatrix, XGBRanker
except ImportError as exc:  # pragma: no cover - CLI environment failure
    raise SystemExit("xgboost is required: pip install xgboost") from exc

from rank_winner_models import METADATA, load_bundle, load_models
from src.advanced_racing_features import race_relative_runner_mask
from src.config import DEFAULT_DB
from src.database import quote_identifier
from src.winner_ranker import (
    MARKET_ENGINEERED_FEATURES,
    model_feature_matrix,
    rank_percentiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument(
        "--model",
        help=(
            "Model group from the bundle. When omitted, inspect every available "
            "model group."
        ),
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(
            "outputs/winner_ranker_all_finished/winner_ranker_bundle.json"
        ),
    )
    parser.add_argument(
        "--oof-predictions",
        type=Path,
        help=(
            "Optional saved OOF CSV. By default, use "
            "all_finished_oof_predictions.csv beside the bundle when present."
        ),
    )
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Save the complete selected-versus-comparison SHAP delta table.",
    )
    parser.add_argument(
        "--trees-csv",
        type=Path,
        help="Optionally save every raw tree node for every ensemble member.",
    )
    return parser.parse_args()


def model_features_from_bundle(bundle: dict[str, Any], label: str) -> list[str]:
    configured = bundle.get("model_features", {})
    if label in configured:
        features = list(configured[label])
    elif label == "form":
        features = list(bundle.get("form_features", []))
    elif label == "market_aware":
        features = [
            *list(bundle.get("form_features", [])),
            *MARKET_ENGINEERED_FEATURES,
        ]
    else:
        features = []
    if not features:
        raise ValueError(f"Bundle contains no feature schema for model {label!r}")
    return features


def load_finished_race(
    database: Path, race_id: int, features: list[str]
) -> pd.DataFrame:
    """Load the active resulted field and its winner label for inspection."""
    database_features = [
        feature for feature in features if feature not in MARKET_ENGINEERED_FEATURES
    ]
    requested = list(dict.fromkeys([*METADATA, "is_winner", *database_features]))
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        schema = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("race_runners")')
        }
        missing = sorted(set(requested) - schema)
        if missing:
            raise ValueError(
                "Database is missing winner-ranker inspection inputs: "
                + ", ".join(missing)
            )
        selected = ", ".join(quote_identifier(column) for column in requested)
        frame = pd.read_sql_query(
            f"SELECT {selected} FROM race_runners "
            "WHERE race_id = ? ORDER BY runner_number",
            connection,
            params=(race_id,),
        )
    if frame.empty:
        raise ValueError(f"Race {race_id} does not exist")
    status = frame["status"].astype("string").str.strip().str.casefold()
    if not status.eq("finished").all():
        raise ValueError(
            f"Race {race_id} is not finished; an actual winner is required"
        )
    frame = frame.loc[race_relative_runner_mask(frame)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Race {race_id} has no active finished runners")
    labels = pd.to_numeric(frame["is_winner"], errors="coerce")
    if not labels.isin([0, 1]).all() or int(labels.sum()) != 1:
        raise ValueError(f"Race {race_id} must contain exactly one labelled winner")
    return frame


def ensemble_predictions_and_contributions(
    models: list[XGBRanker], matrix: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ensemble rank score, mean raw margin, and member SHAP arrays."""
    race_ids = np.zeros(len(matrix), dtype=np.int64)
    rank_scores: list[np.ndarray] = []
    raw_margins: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    feature_names = list(matrix.columns)
    data = DMatrix(matrix, feature_names=feature_names)
    for model in models:
        booster = model.get_booster()
        raw = np.asarray(booster.predict(data, output_margin=True), dtype=np.float64)
        shap = np.asarray(booster.predict(data, pred_contribs=True), dtype=np.float64)
        if shap.shape != (len(matrix), len(feature_names) + 1):
            raise ValueError("Unexpected XGBoost contribution matrix shape")
        if not np.allclose(
            shap.sum(axis=1), raw, rtol=1e-5, atol=1e-5, equal_nan=False
        ):
            raise ValueError("XGBoost SHAP contributions do not sum to raw margins")
        rank_scores.append(rank_percentiles(raw, race_ids))
        raw_margins.append(raw)
        contributions.append(shap)
    return (
        np.mean(np.stack(rank_scores), axis=0),
        np.mean(np.stack(raw_margins), axis=0),
        np.stack(contributions),
    )


def comparison_indices(
    targets: np.ndarray, scores: np.ndarray
) -> tuple[int, int, str]:
    """Compare a wrong pick to the winner, or a correct pick to second place."""
    y = np.asarray(targets, dtype=np.int64)
    ranking = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    selected = int(ranking[0])
    winner = int(np.flatnonzero(y == 1)[0])
    if selected != winner:
        return selected, winner, "selected_vs_actual_winner"
    if len(ranking) < 2:
        raise ValueError("A race needs at least two runners for comparison")
    return selected, int(ranking[1]), "correct_winner_vs_model_runner_up"


def contribution_delta_table(
    matrix: pd.DataFrame,
    member_contributions: np.ndarray,
    selected: int,
    comparison: int,
) -> pd.DataFrame:
    """Aggregate member TreeSHAP deltas with stability information."""
    # The last contribution is the constant bias and cancels between runners.
    selected_values = member_contributions[:, selected, :-1]
    comparison_values = member_contributions[:, comparison, :-1]
    deltas = selected_values - comparison_values
    mean_delta = deltas.mean(axis=0)
    return pd.DataFrame({
        "feature": list(matrix.columns),
        "selected_value": matrix.iloc[selected].to_numpy(),
        "comparison_value": matrix.iloc[comparison].to_numpy(),
        "selected_shap_mean": selected_values.mean(axis=0),
        "comparison_shap_mean": comparison_values.mean(axis=0),
        "shap_delta_mean": mean_delta,
        "shap_delta_sd": deltas.std(axis=0),
        "member_sign_agreement": np.mean(
            np.sign(deltas) == np.sign(mean_delta)[None, :], axis=0
        ),
        "absolute_delta": np.abs(mean_delta),
    }).sort_values("absolute_delta", ascending=False, kind="stable", ignore_index=True)


def winner_backing_feature_table(
    matrix: pd.DataFrame,
    member_contributions: np.ndarray,
    winner: int,
    other: int,
) -> pd.DataFrame:
    """List features whose evidence pointed to the actual winner over ``other``.

    ``winner_field_rank`` ranks the winner's raw value within the field in the
    direction the model rewards (sign of the value/SHAP correlation); rank 1
    means ranking the race by this feature alone would have picked the winner.
    """
    feature_shap = member_contributions[:, :, :-1].mean(axis=0)
    winner_delta = feature_shap[winner] - feature_shap[other]
    values = matrix.to_numpy(dtype=np.float64)
    field_ranks = np.full(len(matrix.columns), np.nan)
    solo_correct = np.zeros(len(matrix.columns), dtype=bool)
    for index, feature in enumerate(matrix.columns):
        column = pd.to_numeric(matrix[feature], errors="coerce")
        with warnings.catch_warnings():
            # Constant columns produce an undefined correlation by design.
            warnings.simplefilter("ignore", RuntimeWarning)
            correlation = column.corr(pd.Series(feature_shap[:, index]))
        if not np.isfinite(correlation) or correlation == 0:
            continue
        rank = column.rank(
            ascending=correlation < 0, method="min", na_option="bottom"
        )
        field_ranks[index] = rank.iloc[winner]
        solo_correct[index] = field_ranks[index] == 1
    table = pd.DataFrame({
        "feature": list(matrix.columns),
        "winner_value": values[winner],
        "other_value": values[other],
        "winner_shap_minus_other": winner_delta,
        "winner_field_rank": field_ranks,
        "solo_pick_correct": solo_correct,
    })
    table = table.loc[table["winner_shap_minus_other"] > 0]
    return table.sort_values(
        "winner_shap_minus_other", ascending=False, kind="stable", ignore_index=True
    )


def runner_vs_field_contribution_table(
    matrix: pd.DataFrame,
    member_contributions: np.ndarray,
    runner: int,
) -> pd.DataFrame:
    """Explain one runner's raw margin relative to the average field runner."""
    feature_contributions = member_contributions[:, :, :-1].mean(axis=0)
    runner_shap = feature_contributions[runner]
    field_shap = feature_contributions.mean(axis=0)
    values = matrix.to_numpy(dtype=np.float64)
    runner_values = values[runner]
    return pd.DataFrame({
        "feature": list(matrix.columns),
        "winner_value": runner_values,
        "field_median": matrix.median(axis=0, skipna=True).to_numpy(),
        "field_min": matrix.min(axis=0, skipna=True).to_numpy(),
        "field_max": matrix.max(axis=0, skipna=True).to_numpy(),
        "winner_shap_mean": runner_shap,
        "field_shap_mean": field_shap,
        "winner_vs_field_shap": runner_shap - field_shap,
        "winner_value_missing": ~np.isfinite(runner_values),
    })


def ensemble_member_runner_diagnostics(
    member_contributions: np.ndarray, runner: int
) -> pd.DataFrame:
    """Show how consistently individual ensemble members ranked one runner."""
    raw = member_contributions.sum(axis=2)
    race_ids = np.zeros(raw.shape[1], dtype=np.int64)
    rows = []
    for member_index, member_raw in enumerate(raw, start=1):
        order = np.argsort(-member_raw, kind="stable")
        rank = int(np.flatnonzero(order == runner)[0]) + 1
        score = float(rank_percentiles(member_raw, race_ids)[runner])
        rows.append({
            "member": member_index,
            "winner_rank": rank,
            "field_size": len(member_raw),
            "winner_rank_score": score,
            "winner_raw_margin": float(member_raw[runner]),
        })
    return pd.DataFrame(rows)


def attach_oof_scores(
    output: pd.DataFrame, path: Path, race_id: int, label: str
) -> pd.DataFrame:
    """Attach honest cross-fit score/rank when the matching saved OOF file exists."""
    if not path.is_file():
        return output
    oof = pd.read_csv(path)
    score_column = f"{label}_score"
    required = {"race_id", "runner_number", score_column}
    if not required <= set(oof):
        return output
    race = oof.loc[oof["race_id"] == race_id, ["runner_number", score_column]].copy()
    if race.empty or race["runner_number"].duplicated().any():
        return output
    oof_score_column = f"{label}_oof_score"
    race = race.rename(columns={score_column: oof_score_column})
    race[f"{label}_oof_rank"] = race[oof_score_column].rank(
        method="first", ascending=False
    ).astype(int)
    return output.merge(race, on="runner_number", how="left", validate="one_to_one")


def global_gain_table(models: list[XGBRanker]) -> pd.DataFrame:
    """Average gain importance across members, treating unused features as zero."""
    member_scores = [
        model.get_booster().get_score(importance_type="gain") for model in models
    ]
    features = sorted({feature for scores in member_scores for feature in scores})
    return pd.DataFrame({
        "feature": features,
        "mean_gain": [
            float(np.mean([scores.get(feature, 0.0) for scores in member_scores]))
            for feature in features
        ],
        "members_using_feature": [
            sum(feature in scores for scores in member_scores) for feature in features
        ],
    }).sort_values("mean_gain", ascending=False, kind="stable", ignore_index=True)


def combined_global_gain_table(
    model_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Average gain across model groups, counting absent features as zero."""
    if not model_tables:
        return pd.DataFrame(columns=[
            "feature", "mean_gain_across_models", "models_using_feature",
            "members_using_feature",
        ])
    features = sorted({
        str(feature)
        for table in model_tables.values()
        for feature in table["feature"]
    })
    indexed = {
        label: table.set_index("feature") for label, table in model_tables.items()
    }
    rows = []
    for feature in features:
        gains = []
        models_using = 0
        members_using = 0
        for table in indexed.values():
            if feature in table.index:
                row = table.loc[feature]
                gains.append(float(row["mean_gain"]))
                models_using += 1
                members_using += int(row["members_using_feature"])
            else:
                gains.append(0.0)
        rows.append({
            "feature": feature,
            "mean_gain_across_models": float(np.mean(gains)),
            "models_using_feature": models_using,
            "members_using_feature": members_using,
        })
    return pd.DataFrame(rows).sort_values(
        "mean_gain_across_models", ascending=False, kind="stable", ignore_index=True
    )


def model_output_path(path: Path | None, label: str, multiple: bool) -> Path | None:
    """Keep explicit single-model paths, and avoid collisions for all-model runs."""
    if path is None:
        return None
    resolved = path.resolve()
    if not multiple:
        return resolved
    return resolved.with_name(f"{resolved.stem}_{label}{resolved.suffix}")


def validate_derived_feature_version(
    frame: pd.DataFrame,
    expected_versions: list[str],
    *,
    target: str,
    refresh_hint: str,
) -> None:
    """Warn when the race and bundle were built with different feature versions."""
    versions = sorted(
        str(value)
        for value in frame["derived_racing_features_version"].dropna().unique()
    )
    if len(versions) != 1 or frame["derived_racing_features_version"].isna().any():
        raise ValueError(
            "Race has missing/mixed derived feature versions; run the feature updater"
        )
    if expected_versions and versions[0] not in expected_versions:
        warnings.warn(
            f"Race feature version {versions[0]!r} was not used by this {target}; "
            f"the loaded model bundle may be stale. {refresh_hint}",
            UserWarning,
            stacklevel=2,
        )


def inspect_model(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    bundle_path: Path,
    label: str,
    paths: list[str],
    multiple: bool,
) -> pd.DataFrame:
    features = model_features_from_bundle(bundle, label)
    frame = load_finished_race(args.db, args.race_id, features)
    expected_versions = [
        str(value) for value in bundle.get("derived_feature_versions", [])
    ]
    validate_derived_feature_version(
        frame,
        expected_versions,
        target="bundle",
        refresh_hint=(
            "Rerun train_winner_ranker_pipeline.py to regenerate the bundle before "
            "using it for ranking or inspection."
        ),
    )
    matrix = model_feature_matrix(frame, features)
    models = load_models(paths)
    scores, raw_margin, member_contributions = (
        ensemble_predictions_and_contributions(models, matrix)
    )
    targets = frame["is_winner"].to_numpy(dtype=np.int64)
    selected, comparison, comparison_kind = comparison_indices(targets, scores)
    delta = contribution_delta_table(
        matrix, member_contributions, selected, comparison
    )

    ranking = frame[["runner_number", "runner_name", "fluc2", "is_winner"]].copy()
    ranking[f"{label}_score"] = scores
    ranking[f"{label}_rank"] = pd.Series(scores).rank(
        method="first", ascending=False
    ).astype(int)
    ranking[f"{label}_raw_margin_mean"] = raw_margin
    oof_path = (
        args.oof_predictions.resolve()
        if args.oof_predictions
        else bundle_path.with_name("all_finished_oof_predictions.csv")
    )
    ranking = attach_oof_scores(ranking, oof_path, args.race_id, label)
    ranking = ranking.sort_values(
        f"{label}_rank", kind="stable", ignore_index=True
    )

    selected_row = frame.iloc[selected]
    comparison_row = frame.iloc[comparison]
    winner_row = frame.iloc[int(np.flatnonzero(targets == 1)[0])]
    winner = int(np.flatnonzero(targets == 1)[0])
    race = frame.iloc[0]
    print(
        "XGBOOST WINNER-RANKER INSPECTION\n"
        f"race_id={args.race_id} race={race['race_name']} "
        f"venue={race['competition_name']} start={race['start_time_iso']}\n"
        f"model={label} ensemble_members={len(models)} features={len(features)}\n"
        f"selected={selected_row['runner_number']} {selected_row['runner_name']} "
        f"actual_winner={winner_row['runner_number']} {winner_row['runner_name']}\n"
        f"comparison={comparison_kind}: {selected_row['runner_name']} minus "
        f"{comparison_row['runner_name']}\n"
        "positive SHAP delta supports the selected runner; negative supports "
        "the comparison runner\n"
        "SHAP values explain the mean member raw margin; the final averaged "
        "rank-percentile score is not itself additive",
        flush=True,
    )
    print("\nRUNNER RANKING")
    print(ranking.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nTOP WITHIN-RACE SHAP DIFFERENCES")
    print(delta.head(args.top_features).drop(columns="absolute_delta").to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))

    other = selected if selected != winner else comparison
    backing = winner_backing_feature_table(matrix, member_contributions, winner, other)
    other_row = frame.iloc[other]
    print("\nFEATURES THAT WOULD HAVE PREDICTED THE RACE CORRECTLY")
    print(
        f"winner={winner_row['runner_number']} {winner_row['runner_name']} "
        f"other={other_row['runner_number']} {other_row['runner_name']}\n"
        f"features_backing_winner={len(backing)}/{len(features)} "
        f"solo_pick_correct={int(backing['solo_pick_correct'].sum())} "
        "(winner_field_rank=1 means ranking by that feature alone picks the "
        "winner)"
    )
    if backing.empty:
        print("no features favoured the actual winner over the other runner")
    else:
        print(backing.head(args.top_features).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))
    print("\nGLOBAL GAIN IMPORTANCE")
    gain = global_gain_table(models)
    print(gain.head(args.top_features).to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))

    winner_diagnosis = runner_vs_field_contribution_table(
        matrix, member_contributions, winner
    )
    member_diagnosis = ensemble_member_runner_diagnostics(
        member_contributions, winner
    )
    winner_rank = int(ranking.loc[
        ranking["runner_number"] == winner_row["runner_number"], f"{label}_rank"
    ].iloc[0])
    prices = pd.to_numeric(frame["fluc2"], errors="coerce").to_numpy(float)
    finite_price = np.isfinite(prices) & (prices > 0)
    market_order = np.concatenate([
        np.flatnonzero(finite_price)[
            np.argsort(prices[finite_price], kind="stable")
        ],
        np.flatnonzero(~finite_price),
    ])
    market_rank = int(np.flatnonzero(market_order == winner)[0]) + 1
    winner_raw_delta = float(raw_margin[winner] - np.mean(raw_margin))
    print("\nWHY THE ACTUAL WINNER RANKED LOW")
    print(
        f"winner={winner_row['runner_number']} {winner_row['runner_name']} "
        f"model_rank={winner_rank}/{len(frame)} "
        f"market_rank={market_rank}/{len(frame)} fluc2={prices[winner]:.6g}\n"
        f"winner_raw_margin={raw_margin[winner]:.6f} "
        f"field_raw_margin_mean={np.mean(raw_margin):.6f} "
        f"winner_minus_field={winner_raw_delta:.6f}\n"
        "Negative values below are features pushing the winner below the average "
        "runner; positive values are offsets helping the winner."
    )
    print("\nWINNER RANK BY ENSEMBLE MEMBER")
    print(member_diagnosis.to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    diagnosis_columns = [
        "feature", "winner_value", "field_median", "field_min", "field_max",
        "winner_shap_mean", "field_shap_mean", "winner_vs_field_shap",
        "winner_value_missing",
    ]
    negative = winner_diagnosis.loc[
        winner_diagnosis["winner_vs_field_shap"] < 0
    ].sort_values("winner_vs_field_shap", kind="stable")
    positive = winner_diagnosis.loc[
        winner_diagnosis["winner_vs_field_shap"] > 0
    ].sort_values("winner_vs_field_shap", ascending=False, kind="stable")
    diagnosis_limit = min(args.top_features, 15)
    print("\nSTRONGEST REASONS THE WINNER WAS RATED BELOW THE FIELD")
    print(negative.head(diagnosis_limit)[diagnosis_columns].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    print("\nSTRONGEST FEATURES HELPING THE WINNER")
    print(positive.head(diagnosis_limit)[diagnosis_columns].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    if f"{label}_score" in ranking and f"{label}_oof_rank" in ranking:
        print(
            "\nOOF columns are honest cross-fit predictions. SHAP columns explain "
            "the saved full-history model, which trained on this finished race."
        )
    elif str(bundle.get("training_scope", "")) == "all_eligible_finished_races":
        print(
            "\nWARNING SHAP explains a full-history model that trained on this "
            "finished race; it is an inspection, not held-out evidence."
        )

    path = model_output_path(args.output_csv, label, multiple)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        delta.to_csv(path, index=False)
        print(f"saved_shap_deltas={path}")
    path = model_output_path(args.trees_csv, label, multiple)
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tree_parts = []
        for member, model in enumerate(models, start=1):
            trees = model.get_booster().trees_to_dataframe()
            trees.insert(0, "member", member)
            tree_parts.append(trees)
        pd.concat(tree_parts, ignore_index=True).to_csv(path, index=False)
        print(f"saved_trees={path}")
    return gain


def main() -> None:
    args = parse_args()
    if args.top_features < 1:
        raise ValueError("top-features must be positive")
    bundle_path = args.bundle.resolve()
    bundle = load_bundle(bundle_path)
    available = bundle.get("models", {})
    if args.model is not None:
        labels = [args.model]
    else:
        labels = sorted(label for label, paths in available.items() if paths)
    missing = [label for label in labels if not available.get(label)]
    if missing:
        raise ValueError(
            f"Model {missing[0]!r} is unavailable; choose from "
            + ", ".join(sorted(label for label, paths in available.items() if paths))
        )
    if not labels:
        raise ValueError("Bundle contains no available models")
    multiple = len(labels) > 1
    gain_tables: dict[str, pd.DataFrame] = {}
    for index, label in enumerate(labels):
        if index:
            print("\n" + "=" * 100 + "\n")
        gain_tables[label] = inspect_model(
            args, bundle, bundle_path, label, list(available[label]), multiple
        )
    if multiple:
        print("\n" + "=" * 100)
        print("\nFINAL GLOBAL GAIN IMPORTANCE ACROSS ALL MODELS")
        print(combined_global_gain_table(gain_tables).head(args.top_features).to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        ))


if __name__ == "__main__":
    main()
