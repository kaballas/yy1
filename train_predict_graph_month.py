#!/usr/bin/env python3
"""Tune graph winner models on one date window, refit, and rank one race."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from xgboost import XGBRanker

from evaluate_graph_winner_features import (
    DEFAULT_MANIFEST,
    evaluate_models,
    graph_experiment_feature_sets,
    load_baseline_features,
    load_joined_rows,
    metrics_table,
    quote_identifier,
    selection_eval_metrics,
    train_experiment_ensemble,
    validate_identifier,
)
from src.advanced_racing_features import race_relative_runner_mask
from src.config import DEFAULT_DB
from src.winner_ranker import (
    eligible_races,
    ensemble_rank_scores,
    group_sizes,
    model_feature_matrix,
    rows_for_races,
    validate_ranker_groups,
)
from train_winner_ranker_pipeline import model_parameters


ROOT = Path(__file__).resolve().parent

RECENT_DETAIL_NAMES = (
    "class", "jockey", "track_name", "distance_m", "place", "track_status",
)
RACE_DETAIL_COLUMNS = (
    "runner_country", "jockey", "trainer", "sire", "dam", "distance_m",
    "track_status",
    *(f"recent_{slot}_{name}" for slot in range(1, 7) for name in RECENT_DETAIL_NAMES),
)


def utc_timestamp(value: str, option: str, timezone: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {option}: {value!r}") from exc
    if parsed.tzinfo is None:
        try:
            parsed = parsed.tz_localize(timezone)
        except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(f"Invalid --timezone: {timezone!r}") from exc
    else:
        parsed = parsed.tz_convert("UTC")
    return parsed


def load_target_race(
    database: Path,
    graph_table: str,
    race_id: int,
    required_features: Sequence[str],
) -> pd.DataFrame:
    """Load one finished or complete live race through the source-row graph join."""
    graph_table = validate_identifier(graph_table, "graph table")
    metadata = (
        "race_id", "start_time_iso", "competition_id", "competition_name",
        "race_number", "race_name", "runner_number", "runner_name",
        "career_starts", "status", "runner_mask", "is_winner",
        "source_betting_status", "active_field_size",
    )
    graph_features = [name for name in required_features if name.startswith("graph_")]
    race_features = [
        name
        for name in required_features
        if not name.startswith("graph_")
        and name not in metadata
        and name not in RACE_DETAIL_COLUMNS
    ]
    uri = f"file:{database.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        race_schema = {
            str(row[1])
            for row in connection.execute('PRAGMA table_info("race_runners")')
        }
        graph_schema = {
            str(row[1])
            for row in connection.execute(
                f"PRAGMA table_info({quote_identifier(graph_table)})"
            )
        }
        missing_race = sorted(
            set([*metadata, *RACE_DETAIL_COLUMNS, *race_features]) - race_schema
        )
        missing_graph = sorted(set(graph_features) - graph_schema)
        if missing_race:
            raise ValueError("race_runners is missing: " + ", ".join(missing_race))
        if missing_graph:
            raise ValueError(f"{graph_table} is missing: " + ", ".join(missing_graph))
        selected = [
            "r.rowid AS source_rowid",
            *(f"r.{quote_identifier(name)}" for name in metadata),
            *(f"r.{quote_identifier(name)}" for name in RACE_DETAIL_COLUMNS),
            *(f"r.{quote_identifier(name)}" for name in race_features),
            "g.snapshot_date",
            *(f"g.{quote_identifier(name)}" for name in graph_features),
        ]
        frame = pd.read_sql_query(
            f"SELECT {', '.join(selected)} FROM race_runners AS r "
            f"JOIN {quote_identifier(graph_table)} AS g "
            "ON g.source_rowid = r.rowid WHERE r.race_id = ? "
            "ORDER BY r.runner_number",
            connection,
            params=(race_id,),
        )
    if frame.empty:
        raise ValueError(
            f"Race {race_id} has no rows in {graph_table}; rebuild graph features "
            "after importing this race"
        )
    if frame["source_rowid"].duplicated().any():
        raise ValueError("Graph join produced duplicate target rows")
    start = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    snapshot = pd.to_datetime(frame["snapshot_date"], utc=True, errors="coerce")
    if start.isna().any() or snapshot.isna().any() or snapshot.ge(start).any():
        raise ValueError("Target race has an invalid or non-causal graph snapshot")
    active = race_relative_runner_mask(frame)
    target = frame.loc[active].reset_index(drop=True)
    if target.empty:
        raise ValueError(
            f"Race {race_id} is not finished and does not yet have a verified "
            "complete PRICED/OFF field"
        )
    return target


def display_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "<null>"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def print_target_race_details(target: pd.DataFrame) -> None:
    race = target.iloc[0]
    snapshots = sorted(str(value) for value in target["snapshot_date"].dropna().unique())
    print("\nTARGET RACE DETAILS")
    print(
        f"race_id={int(race['race_id'])} competition_id={race['competition_id']} "
        f"venue={display_value(race['competition_name'])} "
        f"race_number={display_value(race['race_number'])} "
        f"race_name={display_value(race['race_name'])}"
    )
    print(
        f"start={race['start_time_iso']} status={display_value(race['status'])} "
        f"betting_status={display_value(race['source_betting_status'])} "
        f"distance_m={display_value(race['distance_m'])} "
        f"track_status={display_value(race['track_status'])} "
        f"active_runners={len(target)} graph_snapshots={json.dumps(snapshots)}"
    )
    print("\nTARGET RUNNER AND RECENT-FORM DETAILS")
    for row in target.itertuples(index=False):
        values = row._asdict()
        print(
            f"runner={display_value(values['runner_number'])} "
            f"name={display_value(values['runner_name'])} "
            f"country={display_value(values['runner_country'])} "
            f"jockey={display_value(values['jockey'])} "
            f"trainer={display_value(values['trainer'])} "
            f"sire={display_value(values['sire'])} dam={display_value(values['dam'])}"
        )
        for slot in range(1, 7):
            details = " ".join(
                f"{name}={display_value(values[f'recent_{slot}_{name}'])}"
                for name in RECENT_DETAIL_NAMES
            )
            print(f"  recent_{slot}: {details}")


def write_race_input_audit(
    target: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    output_dir: Path,
) -> Path:
    model_features = list(dict.fromkeys(
        feature for features in feature_sets.values() for feature in features
    ))
    columns = list(dict.fromkeys([
        "source_rowid", "race_id", "start_time_iso", "snapshot_date",
        "competition_id", "competition_name", "race_number", "race_name",
        "status", "source_betting_status", "active_field_size", "runner_mask",
        "runner_number", "runner_name", *RACE_DETAIL_COLUMNS, *model_features,
    ]))
    path = output_dir / f"race_{int(target['race_id'].iloc[0])}_input_values.csv"
    target.loc[:, columns].to_csv(path, index=False)
    return path


def exact_xgboost_input_frame(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Return audit metadata plus the exact numeric X matrix in column order."""
    matrix = model_feature_matrix(frame, features)
    metadata_columns = [
        column
        for column in (
            "race_id", "runner_number", "runner_name", "is_winner"
        )
        if column in frame
    ]
    return pd.concat(
        [
            frame.loc[:, metadata_columns].reset_index(drop=True),
            matrix.reset_index(drop=True),
        ],
        axis=1,
    )


def write_selected_xgboost_audits(
    label: str,
    features: list[str],
    tuning: pd.DataFrame,
    validation: pd.DataFrame,
    full_training: pd.DataFrame,
    target: pd.DataFrame,
    selected_trees: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, str]:
    cohorts = {
        "tuning": tuning,
        "validation": validation,
        "full_refit": full_training,
        "prediction": target,
    }
    paths: dict[str, str] = {}
    for cohort_name, cohort in cohorts.items():
        path = output_dir / f"{label}_{cohort_name}_xgboost_input.csv"
        exact_xgboost_input_frame(cohort, features).to_csv(path, index=False)
        paths[f"{cohort_name}_input_csv"] = str(path)

    group_payload = {
        name: group_sizes(cohort).astype(int).tolist()
        for name, cohort in cohorts.items()
    }
    groups_path = output_dir / f"{label}_xgboost_group_sizes.json"
    groups_path.write_text(
        json.dumps(group_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["group_sizes_json"] = str(groups_path)

    parameters = [
        model_parameters(args, args.seed + member * 1009, trees)
        for member, trees in enumerate(selected_trees)
    ]
    parameters_path = output_dir / f"{label}_xgboost_parameters.json"
    parameters_path.write_text(
        json.dumps(parameters, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["parameters_json"] = str(parameters_path)
    return paths


def print_selected_xgboost_details(
    label: str,
    features: list[str],
    full_training: pd.DataFrame,
    target: pd.DataFrame,
    selected_trees: list[int],
    args: argparse.Namespace,
) -> None:
    training_matrix = model_feature_matrix(full_training, features)
    prediction_matrix = model_feature_matrix(target, features)
    groups = group_sizes(full_training).astype(int)
    missing = training_matrix.isna().sum()
    parameters = [
        model_parameters(args, args.seed + member * 1009, trees)
        for member, trees in enumerate(selected_trees)
    ]
    print("\nSELECTED XGBOOST EXACT INPUT DETAILS")
    print(
        f"model={label} training_rows={len(training_matrix):,} "
        f"training_races={len(groups):,} features={len(features):,} "
        f"group_size_min={groups.min()} group_size_median={np.median(groups):g} "
        f"group_size_max={groups.max()}"
    )
    print(f"feature_columns={json.dumps(features)}")
    print(
        "training_missing_values="
        + json.dumps({feature: int(missing[feature]) for feature in features})
    )
    print(f"member_parameters={json.dumps(parameters, sort_keys=True)}")
    target_display = pd.concat(
        [
            target.loc[:, ["runner_number", "runner_name"]].reset_index(drop=True),
            prediction_matrix.reset_index(drop=True),
        ],
        axis=1,
    )
    print("\nSELECTED XGBOOST EXACT PREDICTION MATRIX")
    print(
        target_display.to_string(
            index=False,
            na_rep="NaN",
            float_format=lambda value: f"{value:.6g}",
        )
    )


MODEL_SIMPLICITY_ORDER = {
    "graph_a": 0,
    "graph_b": 1,
    "graph_c": 2,
    "graph_d": 3,
    "graph_e": 4,
    "graph_only": 5,
}


def select_validation_model(
    validation_metrics: dict[str, dict[str, float]],
) -> str:
    if not validation_metrics:
        raise ValueError("No validation model metrics are available")
    unknown = sorted(set(validation_metrics) - set(MODEL_SIMPLICITY_ORDER))
    if unknown:
        raise ValueError("Unknown graph validation models: " + ", ".join(unknown))
    return min(
        validation_metrics,
        key=lambda label: (
            -validation_metrics[label]["top1_hit_rate"],
            -validation_metrics[label]["mrr"],
            MODEL_SIMPLICITY_ORDER[label],
        ),
    )


def expand_validation_models(requested: Sequence[str]) -> list[str]:
    """Keep pure graph-only runs pure; expand hybrid experiments to A-E."""
    requested_unique = list(dict.fromkeys(requested))
    if requested_unique == ["graph_only"]:
        return ["graph_only"]
    hybrid_models = [label for label in MODEL_SIMPLICITY_ORDER if label != "graph_only"]
    if "graph_only" in requested_unique:
        return [*hybrid_models, "graph_only"]
    return hybrid_models


def refit_selected_ensemble(
    args: argparse.Namespace,
    label: str,
    features: list[str],
    training: pd.DataFrame,
    selected_trees: list[int],
    output_dir: Path,
) -> list[XGBRanker]:
    targets = training["is_winner"].to_numpy(dtype=np.int64)
    groups = group_sizes(training)
    validate_ranker_groups(training, targets, groups)
    matrix = model_feature_matrix(training, features)
    members: list[XGBRanker] = []
    for member, trees in enumerate(selected_trees):
        seed = args.seed + member * 1009
        model = XGBRanker(**model_parameters(args, seed, trees))
        model.fit(matrix, targets, group=groups, verbose=False)
        path = output_dir / f"{label}_selected_one_month_seed_{seed}.json"
        model.save_model(path)
        members.append(model)
        print(f"refit_selected={label} member={member + 1} trees={trees} path={path}")
    return members


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--graph-table", default="graph_features")
    parser.add_argument("--features-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-model", default="c2")
    parser.add_argument(
        "--models", nargs="+",
        choices=(
            "graph_a", "graph_b", "graph_c", "graph_d", "graph_e", "graph_only",
        ),
        default=["graph_a", "graph_b", "graph_c", "graph_d", "graph_e"],
        help=(
            "Accepted for command compatibility; model selection always evaluates "
            "graph_a through graph_e, except --models graph_only runs only the "
            "pure 87-column graph model."
        ),
    )
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument(
        "--timezone", default="Australia/Sydney",
        help="Timezone for date boundaries without an explicit offset.",
    )
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument("--validation-races", type=int, default=100)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument(
        "--selection-objective", choices=("top1", "top3", "map"), default="top1"
    )
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print/write target inputs, then exit without training models.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "graph_one_month_prediction",
    )
    args = parser.parse_args()
    for name in (
        "race_id", "validation_races", "minimum_runners", "ensemble_size",
        "max_estimators", "early_stopping_rounds", "jobs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    requested_models = list(dict.fromkeys(args.models))
    args.models = expand_validation_models(requested_models)
    if requested_models != args.models:
        print(
            f"requested_models={','.join(requested_models)} "
            f"expanded_validation_models={','.join(args.models)}",
            flush=True,
        )
    return args


def main() -> int:
    args = parse_args()
    train_start = utc_timestamp(args.train_start, "train-start", args.timezone)
    train_end = utc_timestamp(args.train_end, "train-end", args.timezone)
    if train_end <= train_start:
        raise ValueError("--train-end must be later than --train-start")

    baseline = load_baseline_features(args.features_json.resolve(), args.baseline_model)
    all_feature_sets = graph_experiment_feature_sets(baseline)
    feature_sets = {label: all_feature_sets[label] for label in args.models}
    required = list(dict.fromkeys(
        feature for features in feature_sets.values() for feature in features
    ))
    finished = load_joined_rows(args.db.resolve(), args.graph_table, required)
    starts = pd.to_datetime(finished["start_time_iso"], utc=True, errors="raise")
    window = finished.loc[starts.ge(train_start) & starts.lt(train_end)].copy()
    races = eligible_races(window, args.minimum_runners)
    if len(races) <= args.validation_races:
        raise ValueError(
            f"Training window has {len(races):,} eligible races, which is not more "
            f"than --validation-races={args.validation_races:,}"
        )
    race_ids = races["race_id"].astype(int).tolist()
    tuning_ids = race_ids[:-args.validation_races]
    validation_ids = race_ids[-args.validation_races:]
    tuning = rows_for_races(window, tuning_ids)
    validation = rows_for_races(window, validation_ids)
    full_training = rows_for_races(window, race_ids)
    target = load_target_race(
        args.db.resolve(), args.graph_table, args.race_id, required
    )
    target_start = pd.to_datetime(target["start_time_iso"].iloc[0], utc=True)
    if target_start < train_end:
        raise ValueError(
            f"Target starts at {target_start}; it must be at or after train-end "
            f"{train_end}"
        )
    for cohort in (tuning, validation, full_training):
        validate_ranker_groups(
            cohort,
            cohort["is_winner"].to_numpy(dtype=np.int64),
            group_sizes(cohort),
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print_target_race_details(target)
    race_input_path = write_race_input_audit(target, feature_sets, output_dir)
    print(f"race_input_values={race_input_path}")
    if args.inspect_only:
        print("inspect_only=yes models_trained=no")
        return 0
    print(
        f"training_window=[{train_start}, {train_end}) "
        f"eligible_races={len(race_ids):,} tuning_races={len(tuning_ids):,} "
        f"validation_races={len(validation_ids):,} target_race={args.race_id} "
        f"target_start={target_start} active_runners={len(target):,}"
    )

    evaluation_models: dict[str, list[XGBRanker]] = {}
    selected_trees: dict[str, list[int]] = {}
    for label, features in feature_sets.items():
        members, trees = train_experiment_ensemble(
            args, label, features, tuning, validation, output_dir
        )
        evaluation_models[label] = members
        selected_trees[label] = trees
    _, validation_metrics, _, _ = evaluate_models(
        evaluation_models, feature_sets, validation
    )
    selected_model = select_validation_model(validation_metrics)
    validation_table = metrics_table(validation_metrics)
    print("\nVALIDATION MODEL SELECTION")
    print(validation_table.to_string(float_format=lambda value: f"{value:.5f}"))
    print(
        f"selected_model={selected_model} "
        f"validation_top1={validation_metrics[selected_model]['top1_hit_rate']:.2%} "
        f"validation_mrr={validation_metrics[selected_model]['mrr']:.5f}",
        flush=True,
    )
    print(
        f"selected_model_input_columns="
        f"{json.dumps(feature_sets[selected_model])}",
        flush=True,
    )
    print_selected_xgboost_details(
        selected_model,
        feature_sets[selected_model],
        full_training,
        target,
        selected_trees[selected_model],
        args,
    )
    xgboost_audit_paths = write_selected_xgboost_audits(
        selected_model,
        feature_sets[selected_model],
        tuning,
        validation,
        full_training,
        target,
        selected_trees[selected_model],
        args,
        output_dir,
    )
    print(f"selected_xgboost_input_audits={json.dumps(xgboost_audit_paths)}")
    selected_ensemble = refit_selected_ensemble(
        args,
        selected_model,
        feature_sets[selected_model],
        full_training,
        selected_trees[selected_model],
        output_dir,
    )

    target_race_ids = target["race_id"].to_numpy(dtype=np.int64)
    output = target.loc[:, ["runner_number", "runner_name"]].copy()
    finished_target = (
        target["status"].astype("string").str.casefold().eq("finished").all()
        and pd.to_numeric(target["is_winner"], errors="coerce").eq(1).sum() == 1
    )
    if finished_target:
        output["actual_winner"] = pd.to_numeric(
            target["is_winner"], errors="coerce"
        ).eq(1)
    for label, members in evaluation_models.items():
        scores = ensemble_rank_scores(
            members, model_feature_matrix(target, feature_sets[label]), target_race_ids
        )
        output[f"{label}_score"] = scores
        rank_column = f"{label}_rank"
        output[rank_column] = pd.Series(scores).rank(
            method="average", ascending=False
        ).to_numpy()
    selected_scores = ensemble_rank_scores(
        selected_ensemble,
        model_feature_matrix(target, feature_sets[selected_model]),
        target_race_ids,
    )
    output["selected_score"] = selected_scores
    output["selected_rank"] = pd.Series(selected_scores).rank(
        method="average", ascending=False
    ).to_numpy()
    output = output.sort_values(
        ["selected_rank", "runner_number"], kind="stable"
    ).reset_index(drop=True)
    output.insert(0, "display", np.arange(1, len(output) + 1))
    prediction_path = output_dir / f"race_{args.race_id}_prediction.csv"
    output.to_csv(prediction_path, index=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "database": str(args.db.resolve()),
        "graph_table": args.graph_table,
        "training_window": {"start": str(train_start), "end": str(train_end)},
        "training_races": len(race_ids),
        "validation_races_for_tree_selection": len(validation_ids),
        "target_race_id": args.race_id,
        "target_start": str(target_start),
        "feature_sets": feature_sets,
        "selected_trees": selected_trees,
        "selected_model": selected_model,
        "validation_metrics": validation_metrics,
        "diagnostic_model_scope": (
            "tuning-window models before the selected model is refit on the "
            "complete training window"
        ),
        "selection_metric": selection_eval_metrics(args.selection_objective)[-1],
        "prediction_csv": str(prediction_path),
        "race_input_values_csv": str(race_input_path),
        "selected_xgboost_input_audits": xgboost_audit_paths,
    }
    (output_dir / "one_month_prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nONE-MONTH GRAPH MODEL RANKING")
    print(
        output.drop(columns=[column for column in output if column.endswith("_score")])
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    if finished_target:
        winner = output.loc[output["actual_winner"]].iloc[0]
        winner_ranks = {
            label: float(winner[f"{label}_rank"]) for label in feature_sets
        }
        winner_ranks["selected"] = float(winner["selected_rank"])
        print(
            f"actual_winner={int(winner['runner_number'])} {winner['runner_name']} "
            f"model_ranks={json.dumps(winner_ranks, sort_keys=True)}"
        )
    print(f"prediction={prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
