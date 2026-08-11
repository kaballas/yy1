#!/usr/bin/env python3
"""Evaluate every additive model branch without rerunning model variants."""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from debug_race import (
    _forward_base,
    attention_distribution_metrics,
    attention_overlap_metrics,
    build_tensors,
    feature_family,
    load_race_and_context,
    logit_margin,
    positive_probability,
    ranks,
    run_model_with_pre_icl_trace,
)
from predict_race import load_model, read_feature_columns
from src.config import DEFAULT_DB
from src.constants import TRAINING_ROWS_VIEW
from src.database import quote_identifier, require_training_rows_view


STAGE_COMBINATIONS = {
    "base_only": (),
    "base+label": ("label",),
    "base+race": ("race",),
    "base+prototype": ("prototype",),
    "base+label+race": ("label", "race"),
    "base+label+prototype": ("label", "prototype"),
    "base+race+prototype": ("race", "prototype"),
    "full_model": ("label", "race", "prototype"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate additive-stage, race-head usefulness, and probability "
            "cardinality diagnostics over complete chronological races."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--competition-id", type=int)
    parser.add_argument(
        "--competition-ids",
        help="Comma-separated competition IDs; conflicts with --competition-id.",
    )
    parser.add_argument("--race-limit", type=int)
    parser.add_argument(
        "--race-source",
        choices=(
            "all", "is_validation", "checkpoint_training", "checkpoint_validation"
        ),
        default="all",
        help=(
            "Evaluate the current view, current is_validation rows, or exact race "
            "IDs embedded by a newer checkpoint."
        ),
    )
    parser.add_argument(
        "--race-head-scales", default="0,0.1,0.25,0.5,0.75,1",
        help="Comma-separated inference-only race-head logit scales to compare.",
    )
    parser.add_argument("--feature-columns-file", type=Path)
    parser.add_argument("--feature-columns")
    parser.add_argument("--strict-load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Save aggregate stage/retrieval metrics for later A/B/C comparison.",
    )
    parser.add_argument(
        "--allow-evaluation-leakage",
        action="store_true",
        help="Allow an explicitly watermarked validation cohort to overlap training.",
    )
    parser.add_argument(
        "--feature-ablation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run inference-only cohort market/race-relative base-feature masking.",
    )
    parser.add_argument(
        "--attention-query-similarity-warning-threshold",
        type=float,
        default=0.95,
        help="Warn above this mean pairwise query-attention cosine.",
    )
    return parser.parse_args()


def parse_scales(value: str) -> list[float]:
    scales = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not scales or any(not np.isfinite(scale) or scale < 0 for scale in scales):
        raise ValueError("--race-head-scales must contain finite non-negative values")
    return list(dict.fromkeys(scales))


def eligible_race_ids(
    db_path: Path,
    competition_ids: list[int] | None,
    race_source: str,
) -> list[int]:
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        require_training_rows_view(connection)
        filters = ["top3_mask IN (0, 1)"]
        parameters: list[Any] = []
        if competition_ids:
            placeholders = ",".join("?" for _ in competition_ids)
            filters.append(f"competition_id IN ({placeholders})")
            parameters.extend(competition_ids)
        if race_source == "is_validation":
            filters.append("is_validation = 1")
        rows = connection.execute(
            f"SELECT race_id, MIN(start_time_iso) AS race_time "
            f"FROM {quote_identifier(TRAINING_ROWS_VIEW)} "
            f"WHERE {' AND '.join(filters)} GROUP BY race_id "
            "HAVING COUNT(*) >= 4 AND SUM(top3_mask) = 3 "
            "ORDER BY race_time, race_id",
            parameters,
        ).fetchall()
    finally:
        connection.close()
    return [int(row[0]) for row in rows]


def zeros_like(delta: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference) if delta is None else delta


def race_metrics(
    target: np.ndarray,
    finish: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    order = np.argsort(-probability, kind="stable")
    actual_top3 = set(np.flatnonzero(target == 1).tolist())
    actual_top2 = set(np.flatnonzero(np.isfinite(finish) & (finish <= 2)).tolist())
    winner = np.flatnonzero(finish == 1)
    winner_rank = (
        int(np.flatnonzero(order == winner[0])[0]) + 1 if len(winner) == 1 else np.nan
    )
    clipped = np.clip(probability.astype(np.float64), 1e-7, 1 - 1e-7)
    hits = len(actual_top3 & set(order[:3].tolist()))
    discounts = 1.0 / np.log2(np.arange(2, 5, dtype=np.float64))
    ndcg3 = float(np.dot(target[order[:3]], discounts) / discounts.sum())
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    pairwise_correct = 0.0
    pairwise_count = 0
    for positive_index in positive:
        differences = probability[positive_index] - probability[negative]
        pairwise_correct += float(
            np.sum(differences > 0) + 0.5 * np.sum(differences == 0)
        )
        pairwise_count += len(negative)
    rank_by_index = {int(index): rank for rank, index in enumerate(order, 1)}
    top3_mrr = float(np.mean([1.0 / rank_by_index[int(index)] for index in positive]))
    return {
        "top1_hit": float(len(winner) == 1 and order[0] == winner[0]),
        "top2_containment": float(
            len(actual_top2) == 2 and actual_top2 == set(order[:2].tolist())
        ),
        "top3_hits": float(hits),
        "top3_recall": hits / 3.0,
        "exact_top3": float(actual_top3 == set(order[:3].tolist())),
        "mrr": top3_mrr,
        "ndcg3": ndcg3,
        "pairwise_ranking_accuracy": (
            pairwise_correct / pairwise_count if pairwise_count else float("nan")
        ),
        "winner_rank": float(winner_rank),
        "logloss": float(
            -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean()
        ),
        "probability_sum": float(probability.sum()),
        "cardinality_abs_error": float(abs(probability.sum() - 3.0)),
    }


def aggregate_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        "races": len(records),
        "top1_hit_rate": float(np.mean([row["top1_hit"] for row in records])),
        "top2_containment": float(
            np.mean([row["top2_containment"] for row in records])
        ),
        "top3_recall": float(np.mean([row["top3_recall"] for row in records])),
        "exact_top3_set": float(np.mean([row["exact_top3"] for row in records])),
        "mrr": float(np.mean([row["mrr"] for row in records])),
        "ndcg3": float(np.mean([row["ndcg3"] for row in records])),
        "pairwise_ranking_accuracy": float(np.mean([
            row["pairwise_ranking_accuracy"] for row in records
        ])),
        "race_logloss": float(np.mean([row["logloss"] for row in records])),
        "mean_probability_sum": float(
            np.mean([row["probability_sum"] for row in records])
        ),
        "mean_abs_sum_error": float(np.mean([
            row["cardinality_abs_error"] for row in records
        ])),
        "sum_in_2.5_to_3.5_rate": float(np.mean([
            2.5 <= row["probability_sum"] <= 3.5 for row in records
        ])),
    }


def cardinality_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    sums = np.asarray([row["probability_sum"] for row in records])
    errors = np.abs(sums - 3.0)
    recall = np.asarray([row["top3_recall"] for row in records])
    relationship = (
        float(np.corrcoef(errors, recall)[0, 1])
        if len(records) > 1 and np.std(errors) > 0 and np.std(recall) > 0
        else float("nan")
    )
    return {
        "mean": float(np.mean(sums)),
        "median": float(np.median(sums)),
        "p10": float(np.quantile(sums, 0.10)),
        "p90": float(np.quantile(sums, 0.90)),
        "mean_abs_error_from_3": float(np.mean(errors)),
        "in_2.5_to_3.5_rate": float(np.mean((sums >= 2.5) & (sums <= 3.5))),
        "abs_error_vs_top3_recall_correlation": relationship,
    }


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
        category=UserWarning,
    )
    if args.race_limit is not None and args.race_limit < 1:
        raise ValueError("--race-limit must be positive")
    if not 0 <= args.attention_query_similarity_warning_threshold <= 1:
        raise ValueError(
            "--attention-query-similarity-warning-threshold must be between 0 and 1"
        )
    scales = parse_scales(args.race_head_scales)
    if args.competition_id is not None and args.competition_ids:
        raise ValueError("Use either --competition-id or --competition-ids, not both")
    competition_ids = (
        [args.competition_id]
        if args.competition_id is not None
        else [
            int(value.strip())
            for value in (args.competition_ids or "").split(",")
            if value.strip()
        ]
    )
    if len(competition_ids) != len(set(competition_ids)):
        competition_ids = list(dict.fromkeys(competition_ids))
    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint.resolve(), device, args.strict_load)
    feature_columns = read_feature_columns(args, metadata)
    context_count = int(metadata.get("context_races_per_step", 9))
    current_source = "is_validation" if args.race_source == "is_validation" else "all"
    current_candidates = eligible_race_ids(args.db, competition_ids, current_source)
    manifest_key = {
        "checkpoint_training": "eligible_training_query_race_ids",
        "checkpoint_validation": "validation_query_race_ids",
    }.get(args.race_source)
    if manifest_key is None:
        candidates = current_candidates
    else:
        if competition_ids:
            raise ValueError(
                "Checkpoint race manifests already define the cohort; do not combine "
                "them with --competition-id/--competition-ids"
            )
        embedded = metadata.get(manifest_key)
        if not embedded:
            raise ValueError(
                f"Checkpoint does not embed {manifest_key}; use a current-view race "
                "source and treat provenance as unknown, or train a new checkpoint."
            )
        candidates = [int(race_id) for race_id in embedded]
    if args.race_limit is not None:
        candidates = candidates[-args.race_limit:]

    training_manifest = {
        int(race_id)
        for race_id in metadata.get("eligible_training_query_race_ids", [])
    }
    validation_manifest = {
        int(race_id) for race_id in metadata.get("validation_query_race_ids", [])
    }
    manifest_overlap = training_manifest & validation_manifest
    candidate_overlap = training_manifest & set(candidates)
    print(
        "PROVENANCE "
        f"training_races={len(training_manifest) if training_manifest else 'unknown'} "
        f"validation_races={len(validation_manifest) if validation_manifest else 'unknown'} "
        f"manifest_overlap={len(manifest_overlap) if training_manifest and validation_manifest else 'unknown'} "
        f"evaluated_overlap={len(candidate_overlap) if training_manifest else 'unknown'}"
    )
    supposed_validation = args.race_source in {
        "is_validation", "checkpoint_validation"
    }
    if supposed_validation and training_manifest and candidate_overlap:
        message = (
            "Validation evaluation contains training races: "
            f"overlap_count={len(candidate_overlap)} preview="
            f"{sorted(candidate_overlap)[:10]}"
        )
        if not args.allow_evaluation_leakage:
            raise ValueError(message + "; pass --allow-evaluation-leakage only for a classroom diagnostic")
        print("WARNING LEAKAGE OVERRIDE: " + message)

    experiment_only = bool(metadata.get("experiment_only", False))
    provenance_matched = manifest_key is not None
    print("AGGREGATE MODEL STAGE EVALUATION")
    print(f"checkpoint={args.checkpoint.resolve()}")
    print(
        f"candidate_races={len(candidates)} source={args.race_source} "
        f"competitions={competition_ids or 'all'} context_races={context_count}"
    )
    recorded_training_pool = metadata.get("eligible_training_race_count")
    print(
        "evidence_scope="
        + (
            "CLASSROOM CHECKPOINT — its training-time validation overlapped "
            "training; no cohort from this checkpoint is held out"
            if experiment_only
            else (
                "checkpoint-embedded race manifest"
                if provenance_matched
                else "current database view; training overlap is not known"
            )
        )
    )
    if recorded_training_pool is not None and not metadata.get(
        "eligible_training_query_race_ids"
    ):
        print(
            f"checkpoint_recorded_training_pool_races={recorded_training_pool}; "
            "race IDs were not embedded, so exact overlap cannot be reconstructed "
            "after the source view changes."
        )

    configuration_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    scale_records: dict[float, list[dict[str, float]]] = defaultdict(list)
    detailed_rows: list[dict[str, Any]] = []
    race_head_changes = {"improved": 0, "unchanged": 0, "degraded": 0}
    winner_rank_changes = {"improved": 0, "unchanged": 0, "degraded": 0}
    before_hits: list[float] = []
    after_hits: list[float] = []
    branch_spreads: dict[str, list[float]] = defaultdict(list)
    attention_rows: list[dict[str, float]] = []
    feature_ablation_records: dict[str, list[dict[str, float]]] = defaultdict(list)
    feature_ablation_rank_changes: dict[str, list[float]] = defaultdict(list)
    stage_change_counts = {
        "base -> label": {"improved": 0, "unchanged": 0, "degraded": 0},
        "label -> race": {"improved": 0, "unchanged": 0, "degraded": 0},
        "race -> prototype": {"improved": 0, "unchanged": 0, "degraded": 0},
    }
    skipped: list[tuple[int, str]] = []

    for candidate_index, race_id in enumerate(candidates, start=1):
        try:
            query, context, _ = load_race_and_context(
                args.db, race_id, feature_columns, context_count
            )
            tensors, _, _ = build_tensors(
                model, metadata, context, query, feature_columns, device
            )
            logits, auxiliary, _, label_representations = (
                run_model_with_pre_icl_trace(model, tensors)
            )
        except (ValueError, KeyError) as error:
            skipped.append((race_id, str(error)))
            continue

        query_start = len(context)
        final = logits[0, query_start:, :2]
        raw_race = zeros_like(
            auxiliary.get("race_delta"), logits
        )[0, query_start:, :2]
        scaled_race = zeros_like(
            auxiliary.get("scaled_race_delta"), logits
        )[0, query_start:, :2]
        if auxiliary.get("scaled_race_delta") is None:
            scaled_race = getattr(model, "race_head_scale", 1.0) * raw_race
        label = zeros_like(
            auxiliary.get("label_context_delta"), logits
        )[0, query_start:, :2]
        prototype = zeros_like(
            auxiliary.get("context_prototype_delta"), logits
        )[0, query_start:, :2]
        base = final - scaled_race - label - prototype
        pieces = {"label": label, "race": scaled_race, "prototype": prototype}
        target = pd.to_numeric(query["top3_mask"], errors="raise").to_numpy(int)
        finish = pd.to_numeric(query["finish_place"], errors="coerce").to_numpy(float)

        per_configuration: dict[str, dict[str, float]] = {}
        for name, included in STAGE_COMBINATIONS.items():
            stage_logits = base.clone()
            for branch in included:
                stage_logits = stage_logits + pieces[branch]
            probability = positive_probability(stage_logits)
            metrics = race_metrics(target, finish, probability)
            configuration_records[name].append(metrics)
            per_configuration[name] = metrics
            detailed_rows.append({"race_id": race_id, "configuration": name, **metrics})

        for transition, before_name, after_name in (
            ("base -> label", "base_only", "base+label"),
            ("label -> race", "base+label", "base+label+race"),
            ("race -> prototype", "base+label+race", "full_model"),
        ):
            delta = (
                per_configuration[after_name]["top3_hits"]
                - per_configuration[before_name]["top3_hits"]
            )
            stage_change_counts[transition][
                "improved" if delta > 0 else "degraded" if delta < 0 else "unchanged"
            ] += 1

        if args.feature_ablation:
            full_base_probability = positive_probability(base)
            feature_sets = {
                "full base model": [],
                "market median": [
                    index for index, feature in enumerate(feature_columns)
                    if feature_family(feature) == "market"
                ],
                "race-relative median": [
                    index for index, feature in enumerate(feature_columns)
                    if feature_family(feature) == "race_relative"
                ],
            }
            feature_sets["market + race-relative median"] = sorted(set(
                feature_sets["market median"] + feature_sets["race-relative median"]
            ))
            for ablation_name, columns in feature_sets.items():
                if columns:
                    changed_x = tensors["x"].clone()
                    changed_x[:, query_start:, columns] = 0.0
                    with torch.inference_mode():
                        changed_base = _forward_base(model, tensors, changed_x)[
                            0, query_start:, :2
                        ]
                    probability = positive_probability(changed_base)
                else:
                    probability = full_base_probability
                feature_ablation_records[ablation_name].append(
                    race_metrics(target, finish, probability)
                )
                feature_ablation_rank_changes[ablation_name].append(float(
                    np.sum(ranks(probability) != ranks(full_base_probability))
                ))

        before = per_configuration["base+label"]
        after = per_configuration["base+label+race"]
        before_hits.append(before["top3_hits"])
        after_hits.append(after["top3_hits"])
        hit_delta = after["top3_hits"] - before["top3_hits"]
        race_head_changes[
            "improved" if hit_delta > 0 else "degraded" if hit_delta < 0 else "unchanged"
        ] += 1
        winner_delta = after["winner_rank"] - before["winner_rank"]
        winner_rank_changes[
            "improved"
            if winner_delta < 0
            else "degraded"
            if winner_delta > 0
            else "unchanged"
        ] += 1

        for scale in scales:
            probability = positive_probability(base + label + scale * raw_race + prototype)
            scale_records[scale].append(race_metrics(target, finish, probability))

        for branch, delta in (
            ("label_context", label),
            ("race_head_raw", raw_race),
            ("race_head_scaled", scaled_race),
            ("prototype", prototype),
        ):
            margins = logit_margin(delta)
            branch_spreads[branch].append(float(np.ptp(margins)))

        label_head = getattr(model, "label_context_head", None)
        if label_head is not None and label_representations is not None:
            with torch.inference_mode():
                _, trace = label_head.correction_from_context(
                    label_representations[:, query_start:],
                    label_representations[:, :query_start],
                    tensors["y"][:, :query_start],
                    context_valid_mask=tensors["valid_row_mask"][:, :query_start],
                    query_valid_mask=tensors["valid_row_mask"][:, query_start:],
                    return_attention_diagnostics=True,
                )
            weights_by_query = trace["attention"][0].detach().cpu().numpy()
            weights_by_head = trace["attention_by_head"][0].detach().cpu().numpy()
            historical_labels = tensors["y"][0, :query_start].detach().cpu().numpy()
            historical_races = context["race_id"].to_numpy(dtype=np.int64)
            logits_by_query = trace["attention_logits"][0].detach().cpu().numpy()
            query_projection = trace["projected_query"][0].detach().cpu().numpy()
            key_projection = trace["projected_key"][0].detach().cpu().numpy()
            value_projection = trace["projected_value"][0].detach().cpu().numpy()
            views = [("head_average", weights_by_query)] + [
                (f"head_{index}", values)
                for index, values in enumerate(weights_by_head)
            ]
            for view, view_weights in views:
                overlap = attention_overlap_metrics(view_weights)
                for query_index, weights in enumerate(view_weights):
                    distribution = attention_distribution_metrics(
                        weights, historical_races, historical_labels
                    )
                    finite_logits = logits_by_query[:, query_index, :]
                    finite_logits = finite_logits[np.isfinite(finite_logits)]
                    attention_rows.append({
                        "view": view,
                        "entropy": distribution["entropy"],
                        "normalised_entropy": distribution["normalised_entropy"],
                        "effective_retrieval_count": distribution["effective_runners"],
                        "effective_race_count": distribution["effective_races"],
                        "context_runner_count": float(len(weights)),
                        "context_race_count": float(len(set(historical_races))),
                        "top1_mass": distribution["top1"],
                        "top3_mass": distribution["top3"],
                        "top5_mass": distribution["top5"],
                        "top10_mass": distribution["top10"],
                        "top3_label_mass": distribution["positive_mass"],
                        "top3_attention_lift": distribution["positive_lift"],
                        "mean_pairwise_query_cosine": overlap["mean_cosine"],
                        "mean_top10_jaccard": overlap["mean_top10_jaccard"],
                        "attention_logit_std": float(np.std(finite_logits)),
                        "query_projection_norm": float(np.linalg.norm(query_projection[query_index])),
                        "key_projection_norm": float(np.mean(np.linalg.norm(key_projection, axis=-1))),
                        "value_projection_norm": float(np.mean(np.linalg.norm(value_projection, axis=-1))),
                    })

        if candidate_index % 25 == 0:
            print(
                f"progress={candidate_index}/{len(candidates)} "
                f"evaluated={len(configuration_records['full_model'])} "
                f"skipped={len(skipped)}",
                flush=True,
            )

    evaluated = len(configuration_records["full_model"])
    if evaluated == 0:
        raise ValueError("No races had sufficient chronological context for evaluation")

    stage_rows = []
    for name in STAGE_COMBINATIONS:
        metrics = aggregate_metrics(configuration_records[name])
        stage_rows.append({"configuration": name, **metrics})
    print_table("STAGE ABLATION METRICS", stage_rows)

    print_table(
        "SEQUENTIAL STAGE TOP-3 SET EFFECT",
        [
            {"transition": transition, **counts}
            for transition, counts in stage_change_counts.items()
        ],
    )

    print("\nRACE-HEAD USEFULNESS (base+label -> base+label+race)")
    print(
        f"races_improved={race_head_changes['improved']} "
        f"races_unchanged={race_head_changes['unchanged']} "
        f"races_degraded={race_head_changes['degraded']}"
    )
    print(
        f"mean_top3_hits_before={np.mean(before_hits):.4f} "
        f"mean_top3_hits_after={np.mean(after_hits):.4f}"
    )
    print(
        f"winner_rank_improved={winner_rank_changes['improved']} "
        f"winner_rank_unchanged={winner_rank_changes['unchanged']} "
        f"winner_rank_degraded={winner_rank_changes['degraded']}"
    )

    scale_rows = [
        {"race_head_scale": scale, **aggregate_metrics(scale_records[scale])}
        for scale in scales
    ]
    print_table("INFERENCE-ONLY RACE-HEAD SCALE SWEEP", scale_rows)

    cardinality_rows = []
    for name in STAGE_COMBINATIONS:
        cardinality_rows.append({
            "stage": name,
            **cardinality_metrics(configuration_records[name]),
        })
    print_table("PROBABILITY CARDINALITY", cardinality_rows)

    spread_rows = [
        {
            "branch": branch,
            "mean_within_race_logit_spread": float(np.mean(values)),
            "median_spread": float(np.median(values)),
            "p90_spread": float(np.quantile(values, 0.90)),
        }
        for branch, values in branch_spreads.items()
    ]
    print_table("BRANCH CORRECTION SPREAD", spread_rows)
    attention_summaries: list[dict[str, Any]] = []
    if attention_rows:
        attention_frame = pd.DataFrame(attention_rows)
        for view, frame in attention_frame.groupby("view", sort=False):
            summary = {
                "view": view,
                "mean_effective_runners": float(frame["effective_retrieval_count"].mean()),
                "median_effective_runners": float(frame["effective_retrieval_count"].median()),
                "mean_effective_races": float(frame["effective_race_count"].mean()),
                "mean_top1_mass": float(frame["top1_mass"].mean()),
                "mean_top5_mass": float(frame["top5_mass"].mean()),
                "mean_top10_mass": float(frame["top10_mass"].mean()),
                "mean_positive_label_lift": float(frame["top3_attention_lift"].mean()),
                "mean_pairwise_query_cosine": float(frame["mean_pairwise_query_cosine"].mean()),
                "mean_top10_jaccard": float(frame["mean_top10_jaccard"].mean()),
                "effective_runner_fraction": float(np.mean(
                    frame["effective_retrieval_count"] / frame["context_runner_count"]
                )),
                "effective_race_fraction": float(np.mean(
                    frame["effective_race_count"] / frame["context_race_count"]
                )),
            }
            attention_summaries.append(summary)
        print_table("LABEL-CONTEXT ATTENTION BY HEAD", attention_summaries)
        averaged = next(row for row in attention_summaries if row["view"] == "head_average")
        if averaged["effective_runner_fraction"] >= 0.80:
            print(
                "WARNING: label-context attention is effectively uniform on average; "
                f"effective_fraction={averaged['effective_runner_fraction']:.1%}."
            )
        if averaged["mean_pairwise_query_cosine"] > args.attention_query_similarity_warning_threshold:
            print(
                "WARNING: label-context retrieval is weakly query-specific; "
                f"mean_pairwise_query_cosine={averaged['mean_pairwise_query_cosine']:.4f} "
                f"> {args.attention_query_similarity_warning_threshold:.4f}."
            )
    if args.feature_ablation:
        feature_rows = []
        for name, records in feature_ablation_records.items():
            feature_rows.append({
                "ablation": name,
                **aggregate_metrics(records),
                "mean_runner_rank_changes": float(np.mean(
                    feature_ablation_rank_changes[name]
                )),
            })
        print_table("COHORT BASE-FEATURE ABLATION (NO RETRAINING)", feature_rows)
    print(f"\nevaluated_races={evaluated} skipped_races={len(skipped)}")
    if skipped:
        print("skipped_preview=" + repr(skipped[:10]))
    if experiment_only:
        if not metadata.get("eligible_training_query_race_ids"):
            print(
                "WARNING: this checkpoint did not embed its training race IDs. "
                "Current-view results may mix seen and unseen races and establish "
                "neither memorisation nor generalisation."
            )
        else:
            print(
                "WARNING: the checkpoint is classroom-only; even an exact embedded "
                "validation manifest overlaps optimization and is not generalisation "
                "evidence."
            )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detailed_rows).to_csv(args.output_csv, index=False)
        print(f"saved_per_race_metrics={args.output_csv.resolve()}")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_kind": metadata.get("checkpoint_kind"),
            "best_epoch": metadata.get("best_epoch"),
            "experiment_only": experiment_only,
            "race_source": args.race_source,
            "evaluated_races": evaluated,
            "training_races": len(training_manifest) if training_manifest else None,
            "validation_races": len(validation_manifest) if validation_manifest else None,
            "overlap_count": len(candidate_overlap) if training_manifest else None,
            "stage_metrics": stage_rows,
            "stage_change_counts": stage_change_counts,
            "attention_metrics": attention_summaries,
            "cardinality_metrics": cardinality_rows,
            "race_head_scale_metrics": scale_rows,
        }
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved_summary={args.summary_json.resolve()}")


if __name__ == "__main__":
    main()
