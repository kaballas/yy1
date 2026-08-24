#!/usr/bin/env python3
"""Chronologically evaluate nested A/B/C/D node2vec winner feature sets."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from xgboost import XGBRanker
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xgboost is required: pip install xgboost") from exc

from build_racing_graph_features import GRAPH_FEATURE_NAMES
from src.config import DEFAULT_DB
from src.winner_ranker import (
    eligible_races,
    ensemble_rank_scores,
    group_sizes,
    is_current_market_feature,
    model_feature_matrix,
    rows_for_races,
    validate_ranker_groups,
    winner_race_report,
)
from train_winner_ranker_pipeline import model_parameters


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "winner_ranker_features.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "graph_winner_feature_experiment"
CPU_THREADS = os.cpu_count() or 1

EXPERIENCED_ABSOLUTE_FEATURES = (
    "graph_horse_jockey_similarity",
    "graph_horse_trainer_similarity",
    "graph_horse_sire_similarity",
    "graph_horse_dam_similarity",
    "graph_horse_venue_similarity",
    "graph_jockey_trainer_similarity",
)

RACE_RELATIVE_SOURCES = (
    "graph_horse_trainer_similarity",
    "graph_horse_jockey_similarity",
    "graph_jockey_trainer_similarity",
    "graph_horse_sire_similarity",
    "graph_horse_venue_similarity",
)

DEBUTANT_RELATIONSHIP_FEATURES = (
    "graph_sire_trainer_similarity",
    "graph_sire_jockey_similarity",
    "graph_dam_trainer_similarity",
    "graph_trainer_venue_similarity",
    "graph_jockey_venue_similarity",
)

AVAILABILITY_FEATURES = (
    "graph_horse_embedding_available",
    "graph_jockey_embedding_available",
    "graph_trainer_embedding_available",
    "graph_sire_embedding_available",
    "graph_dam_embedding_available",
    "graph_venue_embedding_available",
)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def validate_identifier(value: str, description: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"Unsafe {description}: {value!r}")
    return value


def load_baseline_features(manifest: Path, model: str) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    models = payload.get("models") if isinstance(payload, dict) else None
    config = models.get(model) if isinstance(models, dict) else None
    features = config.get("features") if isinstance(config, dict) else None
    if not isinstance(features, list) or not features:
        raise ValueError(f"Manifest has no non-empty models.{model}.features")
    if not all(isinstance(feature, str) and feature for feature in features):
        raise ValueError(f"Manifest models.{model}.features contains invalid names")
    if len(features) != len(set(features)):
        raise ValueError(f"Manifest models.{model}.features contains duplicates")
    market = [feature for feature in features if is_current_market_feature(feature)]
    if market:
        raise ValueError(
            f"Baseline model {model!r} is not market-blind: " + ", ".join(market)
        )
    return list(features)


def graph_experiment_feature_sets(
    baseline: Sequence[str],
) -> dict[str, list[str]]:
    """Return the planned nested A/B/C/D feature groups."""
    a = list(dict.fromkeys(baseline))
    b = [*a, *EXPERIENCED_ABSOLUTE_FEATURES]
    relative = [
        f"{source}_{suffix}"
        for source in RACE_RELATIVE_SOURCES
        for suffix in ("rank_in_race", "minus_race_mean")
    ]
    c = [*b, *relative]
    d = [*c, *DEBUTANT_RELATIONSHIP_FEATURES, *AVAILABILITY_FEATURES]
    result = {
        "graph_a": list(dict.fromkeys(a)),
        "graph_b": list(dict.fromkeys(b)),
        "graph_c": list(dict.fromkeys(c)),
        "graph_d": list(dict.fromkeys(d)),
    }
    known_graph = set(GRAPH_FEATURE_NAMES)
    unavailable = sorted(
        feature
        for features in result.values()
        for feature in features
        if feature.startswith("graph_") and feature not in known_graph
    )
    if unavailable:
        raise AssertionError("Unknown graph experiment features: " + ", ".join(unavailable))
    return result


def load_joined_rows(
    database: Path,
    graph_table: str,
    required_features: Sequence[str],
) -> pd.DataFrame:
    """Load exact source-row graph joins for finished active winner races."""
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    graph_table = validate_identifier(graph_table, "graph table")
    metadata = (
        "race_id",
        "start_time_iso",
        "competition_id",
        "competition_name",
        "race_number",
        "race_name",
        "runner_number",
        "runner_name",
        "career_starts",
        "status",
        "runner_mask",
        "is_winner",
    )
    graph_features = [
        feature for feature in required_features if feature.startswith("graph_")
    ]
    race_features = [
        feature
        for feature in required_features
        if not feature.startswith("graph_") and feature not in metadata
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
        missing_race = sorted(set([*metadata, *race_features]) - race_schema)
        missing_graph = sorted(set(graph_features) - graph_schema)
        if missing_race:
            raise ValueError("race_runners is missing: " + ", ".join(missing_race))
        if missing_graph:
            raise ValueError(f"{graph_table} is missing: " + ", ".join(missing_graph))
        selected = [
            "r.rowid AS source_rowid",
            *(f"r.{quote_identifier(name)}" for name in metadata),
            *(f"r.{quote_identifier(name)}" for name in race_features),
            "g.snapshot_date",
            *(f"g.{quote_identifier(name)}" for name in graph_features),
        ]
        frame = pd.read_sql_query(
            f"SELECT {', '.join(selected)} "
            'FROM "race_runners" AS r '
            f"JOIN {quote_identifier(graph_table)} AS g ON g.source_rowid = r.rowid "
            "WHERE r.status = 'finished' AND r.runner_mask = 1 "
            "AND r.is_winner IN (0, 1) "
            "ORDER BY r.start_time_iso, r.race_id, r.runner_number",
            connection,
        )
    if frame["source_rowid"].duplicated().any():
        raise ValueError("Graph join produced duplicate source rows")
    start = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    snapshot = pd.to_datetime(frame["snapshot_date"], utc=True, errors="coerce")
    if start.isna().any() or snapshot.isna().any():
        raise ValueError("Graph experiment contains invalid start/snapshot timestamps")
    invalid = snapshot.ge(start)
    if invalid.any():
        raise ValueError(
            f"Graph chronology violation: {int(invalid.sum()):,} snapshots are not earlier"
        )
    return frame


def chronological_split(
    races: pd.DataFrame,
    validation_races: int,
    test_races: int,
) -> tuple[list[int], list[int], list[int]]:
    if validation_races < 1 or test_races < 1:
        raise ValueError("validation-races and test-races must be positive")
    if len(races) <= validation_races + test_races:
        raise ValueError(
            f"Need more than {validation_races + test_races:,} eligible races; "
            f"found {len(races):,}"
        )
    ids = races["race_id"].astype(int).tolist()
    train_end = len(ids) - validation_races - test_races
    validation_end = len(ids) - test_races
    return ids[:train_end], ids[train_end:validation_end], ids[validation_end:]


def rank_metrics(report: pd.DataFrame) -> dict[str, float]:
    ranks = pd.to_numeric(report["winner_rank"], errors="coerce")
    if ranks.isna().any() or report.empty:
        raise ValueError("Winner report contains invalid ranks")
    return {
        "races": float(len(ranks)),
        "top1_hit_rate": float(ranks.le(1).mean()),
        "top2_hit_rate": float(ranks.le(2).mean()),
        "top3_hit_rate": float(ranks.le(3).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "mean_winner_rank": float(ranks.mean()),
    }


def winner_start_bands(
    frame: pd.DataFrame,
    targets: np.ndarray,
    scores: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    report = winner_race_report(frame, targets, scores)
    winner_positions = np.flatnonzero(np.asarray(targets, dtype=np.int64) == 1)
    winners = frame.iloc[winner_positions].loc[:, ["race_id", "career_starts"]].copy()
    # The frame contract has exactly one winner per race, validated before this call.
    winners["career_starts"] = pd.to_numeric(
        winners["career_starts"], errors="coerce"
    )
    report = report.merge(winners, on="race_id", how="left", validate="one_to_one")
    starts = report["career_starts"]
    band = pd.Series("missing", index=report.index, dtype="string")
    band.loc[starts.eq(0)] = "0"
    band.loc[starts.between(1, 2, inclusive="both")] = "1-2"
    band.loc[starts.between(3, 5, inclusive="both")] = "3-5"
    band.loc[starts.ge(6)] = "6+"
    report["winner_start_band"] = band
    by_band = {
        str(band): rank_metrics(group)
        for band, group in report.groupby(
            "winner_start_band", sort=False, observed=True
        )
    }
    return report, by_band


def debutant_presence_metrics(
    frame: pd.DataFrame,
    report: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    starts = pd.to_numeric(frame["career_starts"], errors="coerce")
    has_debutant = starts.eq(0).groupby(frame["race_id"], sort=False).any()
    sliced = report.copy()
    sliced["debutant_presence"] = (
        sliced["race_id"]
        .map(has_debutant)
        .map({True: "races_with_debutant", False: "races_without_debutant"})
    )
    return {
        str(label): rank_metrics(group)
        for label, group in sliced.groupby(
            "debutant_presence", sort=False, observed=True
        )
    }


def evaluate_models(
    models_by_label: Mapping[str, list[Any]],
    feature_sets: Mapping[str, list[str]],
    cohort: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, float]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, dict[str, float]]],
]:
    targets = cohort["is_winner"].to_numpy(dtype=np.int64)
    race_ids = cohort["race_id"].to_numpy(dtype=np.int64)
    scored = cohort.loc[:, [
        "race_id", "start_time_iso", "competition_id", "competition_name",
        "race_number", "race_name", "runner_number", "runner_name",
        "career_starts", "is_winner",
    ]].copy()
    overall: dict[str, dict[str, float]] = {}
    bands: dict[str, dict[str, dict[str, float]]] = {}
    debutant_presence: dict[str, dict[str, dict[str, float]]] = {}
    for label, models in models_by_label.items():
        scores = ensemble_rank_scores(
            models, model_feature_matrix(cohort, feature_sets[label]), race_ids
        )
        report, band_metrics = winner_start_bands(cohort, targets, scores)
        overall[label] = rank_metrics(report)
        bands[label] = band_metrics
        debutant_presence[label] = debutant_presence_metrics(cohort, report)
        scored[f"{label}_score"] = scores
        scored[f"{label}_rank"] = pd.Series(scores).groupby(
            cohort["race_id"], sort=False
        ).rank(method="average", ascending=False).to_numpy()
    return scored, overall, bands, debutant_presence


def metrics_table(metrics: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics).T.loc[:, [
        "races", "top1_hit_rate", "top2_hit_rate", "top3_hit_rate", "mrr",
        "mean_winner_rank",
    ]]


def band_metrics_table(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    order = {"0": 0, "1-2": 1, "3-5": 2, "6+": 3, "missing": 4}
    for model, model_bands in metrics.items():
        for band, values in model_bands.items():
            rows.append({"model": model, "winner_start_band": band, **values})
    return pd.DataFrame(rows).sort_values(
        ["model", "winner_start_band"],
        key=lambda values: values.map(order) if values.name == "winner_start_band" else values,
        kind="stable",
        ignore_index=True,
    )


def debutant_presence_table(
    metrics: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> pd.DataFrame:
    rows = [
        {"model": model, "debutant_presence": label, **values}
        for model, model_slices in metrics.items()
        for label, values in model_slices.items()
    ]
    order = {"races_with_debutant": 0, "races_without_debutant": 1}
    return pd.DataFrame(rows).sort_values(
        ["model", "debutant_presence"],
        key=lambda values: (
            values.map(order) if values.name == "debutant_presence" else values
        ),
        kind="stable",
        ignore_index=True,
    )


def paired_bootstrap_table(
    scored: pd.DataFrame,
    baseline_model: str,
    samples: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap paired per-race metric differences against one model."""
    winners = scored.loc[pd.to_numeric(scored["is_winner"], errors="coerce").eq(1)]
    baseline_column = f"{baseline_model}_rank"
    if baseline_column not in winners:
        raise ValueError(f"Bootstrap baseline is unavailable: {baseline_model}")
    candidate_models = [
        column.removesuffix("_rank")
        for column in winners.columns
        if column.endswith("_rank") and column != baseline_column
    ]
    baseline_ranks = pd.to_numeric(winners[baseline_column], errors="coerce").to_numpy()
    if winners.empty or not np.isfinite(baseline_ranks).all():
        raise ValueError("Bootstrap input contains invalid baseline winner ranks")

    transforms = {
        "top1_hit_rate": lambda ranks: (ranks <= 1).astype(float),
        "top2_hit_rate": lambda ranks: (ranks <= 2).astype(float),
        "top3_hit_rate": lambda ranks: (ranks <= 3).astype(float),
        "mrr": lambda ranks: 1.0 / ranks,
        "mean_winner_rank": lambda ranks: ranks,
    }
    rng = np.random.default_rng(seed)
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, Any]] = []
    # Bound the temporary index matrix for larger cohorts and bootstrap counts.
    chunk_size = max(1, min(500, samples))
    for candidate_model in candidate_models:
        candidate_ranks = pd.to_numeric(
            winners[f"{candidate_model}_rank"], errors="coerce"
        ).to_numpy()
        if not np.isfinite(candidate_ranks).all():
            raise ValueError(
                f"Bootstrap input contains invalid {candidate_model} winner ranks"
            )
        for metric, transform in transforms.items():
            differences = transform(candidate_ranks) - transform(baseline_ranks)
            bootstrapped = np.empty(samples, dtype=np.float64)
            for start in range(0, samples, chunk_size):
                stop = min(start + chunk_size, samples)
                indices = rng.integers(
                    0, len(differences), size=(stop - start, len(differences))
                )
                bootstrapped[start:stop] = differences[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrapped, [alpha, 1.0 - alpha])
            display_scale = 100.0 if metric.startswith("top") else 1.0
            rows.append(
                {
                    "baseline_model": baseline_model,
                    "candidate_model": candidate_model,
                    "metric": metric,
                    "races": len(differences),
                    "delta": float(differences.mean()),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "confidence": confidence,
                    "display_unit": (
                        "percentage_points" if display_scale == 100.0 else "raw"
                    ),
                    "display_delta": float(differences.mean() * display_scale),
                    "display_ci_lower": float(lower * display_scale),
                    "display_ci_upper": float(upper * display_scale),
                }
            )
    return pd.DataFrame(rows)


def selection_eval_metrics(objective: str) -> list[str]:
    if objective == "top1":
        return ["map", "ndcg@3", "ndcg@1"]
    if objective == "top3":
        return ["map", "ndcg@1", "ndcg@3"]
    if objective == "map":
        return ["ndcg@1", "ndcg@3", "map"]
    raise ValueError(f"Unknown selection objective: {objective}")


def train_experiment_ensemble(
    args: argparse.Namespace,
    label: str,
    features: Sequence[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
) -> tuple[list[XGBRanker], list[int]]:
    train_targets = train["is_winner"].to_numpy(dtype=np.int64)
    validation_targets = validation["is_winner"].to_numpy(dtype=np.int64)
    train_groups = group_sizes(train)
    validation_groups = group_sizes(validation)
    train_matrix = model_feature_matrix(train, list(features))
    validation_matrix = model_feature_matrix(validation, list(features))
    models: list[XGBRanker] = []
    trees: list[int] = []
    for member in range(args.ensemble_size):
        seed = args.seed + member * 1009
        parameters = model_parameters(args, seed, args.max_estimators)
        parameters["eval_metric"] = selection_eval_metrics(args.selection_objective)
        model = XGBRanker(
            **parameters,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        model.fit(
            train_matrix,
            train_targets,
            group=train_groups,
            eval_set=[
                (train_matrix, train_targets),
                (validation_matrix, validation_targets),
            ],
            eval_group=[train_groups, validation_groups],
            verbose=False,
        )
        selected = int(model.best_iteration) + 1
        path = output_dir / f"{label}_evaluation_seed_{seed}.json"
        model.save_model(path)
        models.append(model)
        trees.append(selected)
        print(
            f"trained={label} member={member + 1}/{args.ensemble_size} "
            f"seed={seed} selection_metric={selection_eval_metrics(args.selection_objective)[-1]} "
            f"best_trees={selected} path={path}",
            flush=True,
        )
    return models, trees


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--graph-table", default="graph_features")
    parser.add_argument("--features-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-model", default="c2")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("graph_a", "graph_b", "graph_c", "graph_d"),
        default=["graph_a", "graph_b", "graph_c", "graph_d"],
    )
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument("--minimum-runners", type=int, default=4)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--max-estimators", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument(
        "--selection-objective", choices=("top1", "top3", "map"), default="top1"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--jobs", type=int, default=max(1, int(CPU_THREADS * 0.8)))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in (
        "validation_races", "test_races", "minimum_runners", "ensemble_size",
        "max_estimators", "early_stopping_rounds", "jobs", "bootstrap_samples",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        parser.error("--bootstrap-confidence must be between 0 and 1")
    args.models = list(dict.fromkeys(args.models))
    return args


def main() -> int:
    args = parse_args()
    baseline = load_baseline_features(args.features_json.resolve(), args.baseline_model)
    all_feature_sets = graph_experiment_feature_sets(baseline)
    feature_sets = {label: all_feature_sets[label] for label in args.models}
    required = list(dict.fromkeys(
        feature for features in feature_sets.values() for feature in features
    ))
    frame = load_joined_rows(args.db.resolve(), args.graph_table, required)
    races = eligible_races(frame, args.minimum_runners)
    train_ids, validation_ids, test_ids = chronological_split(
        races, args.validation_races, args.test_races
    )
    train = rows_for_races(frame, train_ids)
    validation = rows_for_races(frame, validation_ids)
    test = rows_for_races(frame, test_ids)
    for cohort in (train, validation, test):
        validate_ranker_groups(
            cohort,
            cohort["is_winner"].to_numpy(dtype=np.int64),
            group_sizes(cohort),
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "graph_experiment_features.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "baseline_manifest": str(args.features_json.resolve()),
                "baseline_model": args.baseline_model,
                "graph_table": args.graph_table,
                "models": {
                    label: {"features": features}
                    for label, features in feature_sets.items()
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"database={args.db.resolve()} graph_table={args.graph_table} "
        f"eligible_races={len(races):,} train_races={len(train_ids):,} "
        f"validation_races={len(validation_ids):,} chronological_test_races={len(test_ids):,}\n"
        f"baseline_model={args.baseline_model} models={','.join(feature_sets)} "
        f"feature_counts={json.dumps({k: len(v) for k, v in feature_sets.items()})}\n"
        f"train_end={train['start_time_iso'].iloc[-1]} "
        f"validation_end={validation['start_time_iso'].iloc[-1]} "
        f"test_end={test['start_time_iso'].iloc[-1]}",
        flush=True,
    )

    models_by_label: dict[str, list[Any]] = {}
    selected_trees: dict[str, list[int]] = {}
    for label, features in feature_sets.items():
        models, trees = train_experiment_ensemble(
            args,
            label,
            features,
            train,
            validation,
            output_dir,
        )
        models_by_label[label] = models
        selected_trees[label] = trees

    (
        validation_scores,
        validation_metrics,
        validation_bands,
        validation_debutant_presence,
    ) = evaluate_models(models_by_label, feature_sets, validation)
    (
        test_scores,
        test_metrics,
        test_bands,
        test_debutant_presence,
    ) = evaluate_models(models_by_label, feature_sets, test)
    comparison_baseline = "graph_a" if "graph_a" in feature_sets else next(iter(feature_sets))
    validation_bootstrap = paired_bootstrap_table(
        validation_scores,
        comparison_baseline,
        args.bootstrap_samples,
        args.bootstrap_confidence,
        args.seed,
    )
    test_bootstrap = paired_bootstrap_table(
        test_scores,
        comparison_baseline,
        args.bootstrap_samples,
        args.bootstrap_confidence,
        args.seed + 1,
    )
    validation_scores.to_csv(output_dir / "validation_predictions.csv", index=False)
    test_scores.to_csv(output_dir / "chronological_test_predictions.csv", index=False)
    validation_table = metrics_table(validation_metrics)
    test_table = metrics_table(test_metrics)
    validation_band_table = band_metrics_table(validation_bands)
    test_band_table = band_metrics_table(test_bands)
    validation_debutant_table = debutant_presence_table(
        validation_debutant_presence
    )
    test_debutant_table = debutant_presence_table(test_debutant_presence)
    validation_table.to_csv(output_dir / "validation_metrics.csv")
    test_table.to_csv(output_dir / "chronological_test_metrics.csv")
    validation_band_table.to_csv(
        output_dir / "validation_winner_start_bands.csv", index=False
    )
    test_band_table.to_csv(
        output_dir / "chronological_test_winner_start_bands.csv", index=False
    )
    validation_debutant_table.to_csv(
        output_dir / "validation_debutant_presence.csv", index=False
    )
    test_debutant_table.to_csv(
        output_dir / "chronological_test_debutant_presence.csv", index=False
    )
    validation_bootstrap.to_csv(
        output_dir / "validation_paired_bootstrap.csv", index=False
    )
    test_bootstrap.to_csv(
        output_dir / "chronological_test_paired_bootstrap.csv", index=False
    )
    report = {
        "schema_version": 1,
        "database": str(args.db.resolve()),
        "graph_table": args.graph_table,
        "baseline_model": args.baseline_model,
        "test_cohort_status": (
            "chronological_holdout_for_this_run; prior exposure through other "
            "experiments is not ruled out"
        ),
        "feature_sets": feature_sets,
        "selected_trees": selected_trees,
        "selection_objective": args.selection_objective,
        "paired_bootstrap": {
            "comparison_baseline": comparison_baseline,
            "samples": args.bootstrap_samples,
            "confidence": args.bootstrap_confidence,
        },
        "cohorts": {
            "train_races": len(train_ids),
            "validation_races": len(validation_ids),
            "chronological_test_races": len(test_ids),
            "train_end": str(train["start_time_iso"].iloc[-1]),
            "validation_end": str(validation["start_time_iso"].iloc[-1]),
            "test_end": str(test["start_time_iso"].iloc[-1]),
        },
        "validation_metrics": validation_metrics,
        "validation_winner_start_bands": validation_bands,
        "validation_debutant_presence": validation_debutant_presence,
        "validation_paired_bootstrap": validation_bootstrap.to_dict(orient="records"),
        "chronological_test_metrics": test_metrics,
        "chronological_test_winner_start_bands": test_bands,
        "chronological_test_debutant_presence": test_debutant_presence,
        "chronological_test_paired_bootstrap": test_bootstrap.to_dict(
            orient="records"
        ),
    }
    (output_dir / "graph_experiment_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nVALIDATION METRICS")
    print(validation_table.to_string(float_format=lambda value: f"{value:.5f}"))
    print("\nCHRONOLOGICAL TEST METRICS")
    print(test_table.to_string(float_format=lambda value: f"{value:.5f}"))
    print("\nCHRONOLOGICAL TEST BY WINNER PREVIOUS STARTS")
    print(test_band_table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nCHRONOLOGICAL TEST BY DEBUTANT PRESENCE")
    print(test_debutant_table.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    if not test_bootstrap.empty:
        print(
            f"\nCHRONOLOGICAL TEST PAIRED BOOTSTRAP VS {comparison_baseline} "
            f"({args.bootstrap_confidence:.0%} CI)"
        )
        display_columns = [
            "candidate_model", "metric", "display_delta", "display_ci_lower",
            "display_ci_upper", "display_unit",
        ]
        print(
            test_bootstrap.loc[:, display_columns].to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
    print(f"report={output_dir / 'graph_experiment_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
