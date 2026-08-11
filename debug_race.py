#!/usr/bin/env python3
"""Explain one race as it moves through a race-aware TabFM checkpoint."""

from __future__ import annotations

import argparse
import sqlite3
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from predict_race import (
    apply_checkpoint_preprocessing,
    checkpoint_cat_mask,
    checkpoint_context_size,
    load_training_context_for_target,
    load_model,
    matrix_from_frame,
    read_feature_columns,
)
from src.config import DEFAULT_DB
from src.database import quote_identifier
from src.prediction import ablate_context_labels
from src.sampling import build_race_group_ids


DISPLAY_METADATA = (
    "race_id", "race_number", "race_name", "competition_id",
    "competition_name", "start_time_iso", "runner_number", "runner_name",
    "open_price", "fluc1", "fluc2", "top3_mask", "is_winner", "finish_place",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show how every runner in one race moves through preprocessing, "
            "ICL, label-aware historical attention, race-context, prototype, "
            "and final scoring."
        )
    )
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--feature-columns-file", type=Path,
        help="Override the feature order embedded in the checkpoint.",
    )
    parser.add_argument(
        "--feature-columns",
        help="Comma-separated override for the feature order embedded in the checkpoint.",
    )
    parser.add_argument("--top-features", type=int, default=5)
    parser.add_argument(
        "--race-head-scale", type=float,
        help=(
            "Temporarily override the checkpoint's race-head logit scale for "
            "this diagnostic run; the checkpoint is not modified."
        ),
    )
    parser.add_argument(
        "--base-attribution", action=argparse.BooleanOptionalAction, default=True,
        help="Run controlled feature-family ablations and integrated gradients.",
    )
    parser.add_argument(
        "--attribution-runner-number", type=int,
        help="Runner to attribute; defaults to the worst-ranked actual top-three miss.",
    )
    parser.add_argument(
        "--attribution-steps", type=int, default=24,
        help="Integration steps for base-score integrated gradients (default: 24).",
    )
    parser.add_argument(
        "--context-ablation", action=argparse.BooleanOptionalAction, default=True,
        help="Compare correct context labels with permuted, zeroed, and flipped labels.",
    )
    parser.add_argument(
        "--debug-attention-details",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print full per-runner race-mass and pairwise attention matrices.",
    )
    parser.add_argument(
        "--attention-query-similarity-warning-threshold",
        type=float,
        default=0.95,
        help="Warn when mean pairwise query-attention cosine exceeds this value.",
    )
    parser.add_argument(
        "--label-context-temperature-grid",
        default="0.25,0.5,1,2",
        help=(
            "Inference-only explicit label-context temperatures. Logits are divided "
            "by temperature, so smaller values sharpen retrieval. Empty disables."
        ),
    )
    parser.add_argument(
        "--strict-load", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-csv", type=Path, help="Optionally save the stage table.")
    return parser.parse_args()


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def load_race_and_context(
    db_path: Path,
    race_id: int,
    feature_columns: Sequence[str],
    metadata: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one target using the exact native context policy from predict_race.py."""
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    context, _query, selected_ids = load_training_context_for_target(
        db_path, str(race_id), feature_columns, metadata
    )

    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        target_columns = _unique([*DISPLAY_METADATA, *feature_columns])
        target = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(c) for c in target_columns)} "
            "FROM race_runners WHERE race_id = ? "
            "ORDER BY runner_number",
            connection,
            params=(race_id,),
        )
        placeholders = ", ".join("?" for _ in selected_ids)
        display = pd.read_sql_query(
            "SELECT race_id, runner_number, runner_name "
            f"FROM race_runners WHERE race_id IN ({placeholders})",
            connection,
            params=selected_ids,
        )
    finally:
        connection.close()

    # Keep the canonical training-view rows and ordering; add display-only fields.
    context = context.merge(
        display, on=["race_id", "runner_number"], how="left", validate="one_to_one"
    )
    summary = (
        context.groupby("race_id", sort=False)
        .agg(
            start_time_iso=("start_time_iso", "min"),
            runners=("race_id", "size"),
            top3=("top3_mask", "sum"),
        )
        .reindex(selected_ids)
        .reset_index()
    )
    summary["context_source"] = "predict_race chronological training context"
    return target.reset_index(drop=True), context.reset_index(drop=True), summary


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values), kind="stable")
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.arange(1, len(order) + 1)
    return result


def positive_probability(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.float(), dim=-1)[:, 1].detach().cpu().numpy()


def logit_margin(logits: torch.Tensor) -> np.ndarray:
    """Return the binary class-1 minus class-0 logit used by softmax ranking."""
    return (logits.float()[:, 1] - logits.float()[:, 0]).detach().cpu().numpy()


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def architecture_description(model: Any) -> str:
    label_head = getattr(model, "label_context_head", None)
    if label_head is None:
        label_context = "OFF"
    elif getattr(label_head, "labels_in_values_only", False):
        label_context = (
            "ON (runner-only keys; label-aware values; "
            f"temperature={getattr(label_head, 'temperature', 1.0):g}; "
            f"top-k={getattr(label_head, 'top_k', 0) or 'disabled'})"
        )
    else:
        label_context = "ON (legacy label-aware keys and values)"
    return (
        f"pre-ICL race encoder={'ON' if model.pre_icl_race_encoder is not None else 'OFF'}; "
        f"post-ICL race head={'ON' if model.race_set_head is not None else 'OFF'} "
        f"(logit scale={getattr(model, 'race_head_scale', 1.0):g}); "
        f"prototype branch={'ON' if model.context_prototype_head is not None else 'OFF'}; "
        f"label-aware cross-attention={label_context}"
    )


def print_context(summary: pd.DataFrame, context: pd.DataFrame) -> None:
    shown = summary.copy()
    positives = context.groupby("race_id")["top3_mask"].sum().astype(int)
    shown["top3"] = shown["race_id"].map(positives)
    shown["start_time_iso"] = shown["start_time_iso"].astype(str).str.slice(0, 19)
    print("\nSTAGE 1 — CHRONOLOGICAL CONTEXT")
    print("These labelled races are the evidence supplied before the target race.")
    print(shown[["race_id", "start_time_iso", "runners", "top3"]].to_string(index=False))


def print_feature_diagnostics(
    query: pd.DataFrame,
    raw: np.ndarray,
    scaled: np.ndarray,
    feature_columns: Sequence[str],
    top_features: int,
) -> None:
    missing = np.isnan(raw).sum()
    infinite = np.isinf(raw).sum()
    print("\nSTAGE 2 — INPUT AND PREPROCESSING")
    print(
        f"Raw query matrix={raw.shape[0]} runners x {raw.shape[1]} features; "
        f"missing values={missing}; infinite values={infinite}."
    )
    print("Largest absolute standardized inputs per runner (useful for spotting outliers):")
    rows = []
    count = max(0, min(top_features, len(feature_columns)))
    for index, values in enumerate(scaled):
        top = np.argsort(-np.abs(values), kind="stable")[:count]
        details = ", ".join(
            f"{feature_columns[column]}={values[column]:+.2f}" for column in top
        )
        rows.append({
            "No.": query.iloc[index]["runner_number"],
            "Runner": query.iloc[index].get("runner_name", "-"),
            "largest standardized features": details,
        })
    print(pd.DataFrame(rows).to_string(index=False))


def build_tensors(
    model: Any,
    metadata: Mapping[str, Any],
    context: pd.DataFrame,
    query: pd.DataFrame,
    feature_columns: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor | None], np.ndarray, np.ndarray]:
    context_raw = matrix_from_frame(context, feature_columns, None)
    query_raw = matrix_from_frame(query, feature_columns, None)
    context_x = apply_checkpoint_preprocessing(context_raw, feature_columns, metadata)
    query_x = apply_checkpoint_preprocessing(query_raw, feature_columns, metadata)
    combined = np.concatenate([context_x, query_x])
    context_y = pd.to_numeric(context["top3_mask"], errors="raise").to_numpy(np.float32)
    y_values = np.concatenate([context_y, np.full(len(query), -100, np.float32)])
    context_race_ids = context["race_id"].to_numpy(np.int64)
    query_race_ids = np.full(len(query), int(query.iloc[0]["race_id"]), np.int64)
    race_groups = build_race_group_ids(
        query_race_ids,
        len(context),
        context_race_ids=context_race_ids if model.encode_races_before_icl else None,
    ).to(device)
    feature_count = len(feature_columns)
    cat_array = checkpoint_cat_mask(metadata, feature_count)
    cat_mask = (
        None if cat_array is None else
        torch.from_numpy(cat_array).unsqueeze(0).to(device=device, dtype=torch.bool)
    )
    tensors: dict[str, torch.Tensor | None] = {
        "x": torch.from_numpy(combined).unsqueeze(0).to(device),
        "y": torch.from_numpy(y_values).unsqueeze(0).to(device),
        "train_size": torch.tensor([len(context)], dtype=torch.long, device=device),
        "d": torch.tensor([feature_count], dtype=torch.long, device=device),
        "cat_mask": cat_mask,
        "race_group_ids": race_groups,
        "valid_row_mask": torch.ones(
            (1, len(context) + len(query)), dtype=torch.bool, device=device
        ),
    }
    return tensors, query_raw, query_x


def run_model(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor | None]]:
    with torch.inference_mode():
        return model(
            tensors["x"],
            tensors["y"] if labels is None else labels,
            tensors["train_size"],
            cat_mask=tensors["cat_mask"],
            d=tensors["d"],
            race_group_ids=tensors["race_group_ids"],
            valid_row_mask=tensors["valid_row_mask"],
            return_auxiliary_deltas=True,
        )


def run_model_with_pre_icl_trace(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
) -> tuple[
    torch.Tensor,
    Mapping[str, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor] | None,
    torch.Tensor | None,
]:
    """Run normally while observing representation and context boundaries."""
    encoder = model.pre_icl_race_encoder
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def capture(_module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        captured["before"] = inputs[0].detach()
        captured["after"] = output.detach()

    def capture_label_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        captured["label_representations"] = inputs[0].detach()

    if encoder is not None:
        handles.append(encoder.register_forward_hook(capture))
    label_head = getattr(model, "label_context_head", None)
    if label_head is not None:
        handles.append(label_head.register_forward_pre_hook(capture_label_input))
    try:
        logits, auxiliary = run_model(model, tensors)
    finally:
        for handle in handles:
            handle.remove()
    pre_icl_trace = (
        None if encoder is None else (captured["before"], captured["after"])
    )
    return (
        logits,
        auxiliary,
        pre_icl_trace,
        captured.get("label_representations"),
    )


def print_pre_icl_trace(
    query: pd.DataFrame,
    trace: tuple[torch.Tensor, torch.Tensor] | None,
    query_start: int,
) -> None:
    print("\nSTAGE 3 — PRE-ICL FIELD ENCODER")
    if trace is None:
        print("This checkpoint has no pre-ICL race encoder; runner representations are unchanged here.")
        return
    before, after = trace
    before = before[0, query_start:].float()
    after = after[0, query_start:].float()
    movement = torch.linalg.vector_norm(after - before, dim=-1).cpu().numpy()
    cosine = torch.nn.functional.cosine_similarity(before, after, dim=-1).cpu().numpy()
    shown = pd.DataFrame({
        "No.": query["runner_number"].to_numpy(),
        "Runner": query.get("runner_name", pd.Series("-", index=query.index)).to_numpy(),
        "representation ΔL2": movement,
        "before/after cosine": cosine,
    })
    shown["representation ΔL2"] = shown["representation ΔL2"].map(
        lambda value: f"{value:.5f}"
    )
    shown["before/after cosine"] = shown["before/after cosine"].map(
        lambda value: f"{value:.5f}"
    )
    print("This stage mixes each runner with the other runners in its own race before ICL.")
    print(shown.to_string(index=False))


def attention_distribution_metrics(
    weights: np.ndarray,
    historical_race_ids: np.ndarray,
    historical_labels: np.ndarray,
) -> dict[str, Any]:
    """Summarise one historical-runner attention distribution."""
    weights = np.asarray(weights, dtype=np.float64)
    positive = weights[weights > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    available = int(np.count_nonzero(weights >= 0))
    sorted_weights = np.sort(weights)[::-1]
    race_order = list(dict.fromkeys(map(int, historical_race_ids)))
    race_mass = np.asarray([
        weights[historical_race_ids == race_id].sum() for race_id in race_order
    ])
    positive_race_mass = race_mass[race_mass > 0]
    race_entropy = float(-np.sum(positive_race_mass * np.log(positive_race_mass)))
    positive_base_rate = float(np.mean(historical_labels == 1))
    positive_mass = float(weights[historical_labels == 1].sum())
    race_rank = np.argsort(-race_mass, kind="stable")
    return {
        "entropy": entropy,
        "normalised_entropy": entropy / np.log(len(weights)) if len(weights) > 1 else 0.0,
        "effective_runners": float(np.exp(entropy)),
        "race_entropy": race_entropy,
        "effective_races": float(np.exp(race_entropy)),
        "top1": float(sorted_weights[:1].sum()),
        "top3": float(sorted_weights[:3].sum()),
        "top5": float(sorted_weights[:5].sum()),
        "top10": float(sorted_weights[:10].sum()),
        "positive_mass": positive_mass,
        "negative_mass": float(weights[historical_labels == 0].sum()),
        "positive_lift": positive_mass / positive_base_rate,
        "race_order": race_order,
        "race_mass": race_mass,
        "top_race_mass": float(race_mass[race_rank[:1]].sum()),
        "top3_race_mass": float(race_mass[race_rank[:3]].sum()),
        "top_races": [race_order[index] for index in race_rank],
        "available": available,
    }


def attention_overlap_metrics(attention: np.ndarray, top_k: int = 10) -> dict[str, Any]:
    """Compare historical retrieval vectors across query runners."""
    attention = np.asarray(attention, dtype=np.float64)
    query_count = len(attention)
    cosine_matrix = np.eye(query_count, dtype=np.float64)
    js_matrix = np.zeros((query_count, query_count), dtype=np.float64)
    jaccard_matrix = np.ones((query_count, query_count), dtype=np.float64)
    cosines: list[float] = []
    divergences: list[float] = []
    jaccards: list[float] = []
    for left in range(query_count):
        for right in range(left + 1, query_count):
            a, b = attention[left], attention[right]
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosine = float(np.dot(a, b) / denominator) if denominator else 0.0
            midpoint = 0.5 * (a + b)
            with np.errstate(divide="ignore", invalid="ignore"):
                left_kl = np.where(a > 0, a * np.log(a / midpoint), 0.0)
                right_kl = np.where(b > 0, b * np.log(b / midpoint), 0.0)
            divergence = float(0.5 * (left_kl.sum() + right_kl.sum()))
            count = min(top_k, len(a))
            left_top = set(np.argsort(-a, kind="stable")[:count].tolist())
            right_top = set(np.argsort(-b, kind="stable")[:count].tolist())
            union = left_top | right_top
            jaccard = len(left_top & right_top) / len(union) if union else 1.0
            cosine_matrix[left, right] = cosine_matrix[right, left] = cosine
            js_matrix[left, right] = js_matrix[right, left] = divergence
            jaccard_matrix[left, right] = jaccard_matrix[right, left] = jaccard
            cosines.append(cosine)
            divergences.append(divergence)
            jaccards.append(jaccard)
    return {
        "mean_cosine": float(np.mean(cosines)) if cosines else 1.0,
        "min_cosine": float(np.min(cosines)) if cosines else 1.0,
        "max_cosine": float(np.max(cosines)) if cosines else 1.0,
        "mean_js": float(np.mean(divergences)) if divergences else 0.0,
        "mean_top10_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
        "cosine_matrix": cosine_matrix,
        "js_matrix": js_matrix,
        "jaccard_matrix": jaccard_matrix,
    }


def print_label_context_attention(
    model: Any,
    context: pd.DataFrame,
    query: pd.DataFrame,
    representations: torch.Tensor | None,
    tensors: Mapping[str, torch.Tensor | None],
    query_start: int,
    top_examples: int = 3,
    *,
    verbose: bool = False,
    similarity_warning_threshold: float = 0.95,
) -> dict[str, Any] | None:
    print("\nSTAGE 4 — LABEL-AWARE HISTORICAL ATTENTION")
    head = getattr(model, "label_context_head", None)
    if head is None:
        print(
            "OFF: this checkpoint does not contain the label-aware cross-attention "
            "branch. LabelCtx Δp will therefore be exactly zero."
        )
        print(
            "Historical-label ablations can still move predictions through the base "
            "ICL and prototype paths; those movements are not evidence that this new "
            "branch exists in the checkpoint."
        )
        return None
    if representations is None:
        print("The branch is enabled, but its representation trace was unavailable.")
        return None
    if getattr(head, "labels_in_values_only", False):
        print(
            "Retrieval mode: runner-only keys choose similar historical runners; "
            "labels are injected only into the retrieved values."
        )
    else:
        print(
            "Retrieval mode: LEGACY labels are injected into both attention keys "
            "and values. Retrain with --label-context-labels-in-values-only to "
            "enable similarity-first retrieval."
        )
    with torch.inference_mode():
        _, attention_trace = head.correction_from_context(
            representations[:, query_start:],
            representations[:, :query_start],
            tensors["y"][:, :query_start],
            context_valid_mask=tensors["valid_row_mask"][:, :query_start],
            query_valid_mask=tensors["valid_row_mask"][:, query_start:],
            return_attention_diagnostics=True,
        )
    attention = attention_trace["attention"][0].detach().cpu().numpy()
    attention_by_head = attention_trace["attention_by_head"][0].detach().cpu().numpy()
    attention_logits = attention_trace["attention_logits"][0].detach().cpu().numpy()
    projected_query = attention_trace["projected_query"][0].detach().cpu().numpy()
    historical_labels = pd.to_numeric(
        context["top3_mask"], errors="raise"
    ).to_numpy(dtype=np.int64)
    positive_base_rate = float(np.mean(historical_labels == 1))
    historical_race_ids = context["race_id"].to_numpy(dtype=np.int64)
    rows = []
    configured_top_k = int(getattr(head, "top_k", 0) or 0)
    count = min(max(1, top_examples, configured_top_k), query_start)
    for query_index, weights in enumerate(attention):
        selected = np.argsort(-weights, kind="stable")[:count]
        metrics = attention_distribution_metrics(
            weights, historical_race_ids, historical_labels
        )
        finite_logits = attention_logits[:, query_index, :]
        finite_logits = finite_logits[np.isfinite(finite_logits)]
        examples = []
        for context_index in selected:
            historical = context.iloc[int(context_index)]
            examples.append(
                f"race {int(historical['race_id'])}/No.{int(historical['runner_number'])} "
                f"{historical.get('runner_name', '-')} "
                f"label={int(historical['top3_mask'])} weight={weights[context_index]:.3f}"
            )
        rows.append({
            "No.": query.iloc[query_index]["runner_number"],
            "Runner": query.iloc[query_index].get("runner_name", "-"),
            "entropy": metrics["entropy"],
            "normalised entropy": metrics["normalised_entropy"],
            "effective runners": metrics["effective_runners"],
            "effective races": metrics["effective_races"],
            "available runners": len(weights),
            "available races": len(metrics["race_order"]),
            "top-1 mass": metrics["top1"],
            "top-3 mass": metrics["top3"],
            "top-5 mass": metrics["top5"],
            "top-10 mass": metrics["top10"],
            "attention to top-3": metrics["positive_mass"],
            "attention to others": metrics["negative_mass"],
            "top-3 attention lift": metrics["positive_lift"],
            "top race mass": metrics["top_race_mass"],
            "top-3 race mass": metrics["top3_race_mass"],
            "strongest historical races": ", ".join(map(str, metrics["top_races"][:3])),
            "attention-logit std": float(np.std(finite_logits)),
            "query projection norm": float(np.linalg.norm(projected_query[query_index])),
            "strongest historical attention": "; ".join(examples),
        })
    shown = pd.DataFrame(rows)
    for column in (
        "entropy", "normalised entropy", "effective runners", "effective races", "top-1 mass",
        "top-3 mass", "top-5 mass", "top-10 mass", "attention to top-3",
        "attention to others", "top race mass", "top-3 race mass",
        "attention-logit std", "query projection norm",
    ):
        shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    shown["top-3 attention lift"] = shown["top-3 attention lift"].map(
        lambda value: f"{value:.2f}x"
    )
    projection_norm = float(
        torch.linalg.vector_norm(head.output_projection.weight.detach()).cpu()
    )
    print(
        "Each row shows how attention is divided between historical top-three and "
        "other runners, followed by the strongest individual matches."
    )
    print(
        f"Historical top-three base rate={positive_base_rate:.3f}; a lift above "
        "1.00x means the query focuses on top-three examples more than uniform "
        "attention would."
    )
    print(shown.to_string(index=False))
    runner_labels = [
        f"No.{int(number)}" for number in query["runner_number"].to_numpy()
    ]
    head_rows: list[dict[str, Any]] = []
    print("\nPER-HEAD LABEL-CONTEXT RETRIEVAL")
    for head_index, head_attention in enumerate(attention_by_head):
        for query_index, weights in enumerate(head_attention):
            metrics = attention_distribution_metrics(
                weights, historical_race_ids, historical_labels
            )
            selected = np.argsort(-weights, kind="stable")[:count]
            top_runner_text = ", ".join(
                f"{int(context.iloc[index]['race_id'])}/No."
                f"{int(context.iloc[index]['runner_number'])}"
                for index in selected
            )
            head_rows.append({
                "No.": query.iloc[query_index]["runner_number"],
                "head": head_index,
                "entropy": metrics["entropy"],
                "normalised entropy": metrics["normalised_entropy"],
                "effective runners": metrics["effective_runners"],
                "effective races": metrics["effective_races"],
                "available runners": len(weights),
                "available races": len(metrics["race_order"]),
                "top-1": metrics["top1"],
                "top-3": metrics["top3"],
                "top-5": metrics["top5"],
                "top-10": metrics["top10"],
                "positive mass": metrics["positive_mass"],
                "top runners": top_runner_text,
                "top races": ",".join(map(str, metrics["top_races"][:3])),
            })
    head_frame = pd.DataFrame(head_rows)
    numeric_head_columns = [
        "entropy", "normalised entropy", "effective runners", "effective races",
        "top-1", "top-3", "top-5", "top-10", "positive mass",
    ]
    shown_heads = head_frame.copy()
    for column in numeric_head_columns:
        shown_heads[column] = shown_heads[column].map(lambda value: f"{value:.3f}")
    print(shown_heads.to_string(index=False))

    overlap_rows = []
    all_overlap = [("average", attention)] + [
        (f"head_{index}", values) for index, values in enumerate(attention_by_head)
    ]
    overlap_by_view: dict[str, dict[str, Any]] = {}
    for label, values in all_overlap:
        overlap = attention_overlap_metrics(values)
        overlap_by_view[label] = overlap
        overlap_rows.append({
            "view": label,
            "mean cosine": overlap["mean_cosine"],
            "min cosine": overlap["min_cosine"],
            "max cosine": overlap["max_cosine"],
            "mean JS divergence": overlap["mean_js"],
            "mean top-10 Jaccard": overlap["mean_top10_jaccard"],
        })
    print("\nQUERY-TO-QUERY RETRIEVAL OVERLAP")
    print(pd.DataFrame(overlap_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if overlap_by_view["average"]["mean_cosine"] > similarity_warning_threshold:
        print(
            "WARNING: label-context retrieval is weakly query-specific: "
            f"mean pairwise attention cosine "
            f"{overlap_by_view['average']['mean_cosine']:.4f} > "
            f"{similarity_warning_threshold:.4f}."
        )
    print("\nHEAD-AVERAGED ATTENTION MASS BY HISTORICAL RACE")
    race_ids = list(dict.fromkeys(map(int, historical_race_ids)))
    race_mass_rows = []
    for query_index, weights in enumerate(attention):
        metrics = attention_distribution_metrics(
            weights, historical_race_ids, historical_labels
        )
        row = {"No.": query.iloc[query_index]["runner_number"]}
        row.update({
            str(race_id): mass
            for race_id, mass in zip(race_ids, metrics["race_mass"])
        })
        row["top race mass"] = metrics["top_race_mass"]
        row["top-3 race mass"] = metrics["top3_race_mass"]
        row["effective races"] = metrics["effective_races"]
        race_mass_rows.append(row)
    print(pd.DataFrame(race_mass_rows).to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))
    if verbose:
        print("\nHEAD-AVERAGED PAIRWISE ATTENTION COSINE")
        print(pd.DataFrame(
            overlap_by_view["average"]["cosine_matrix"],
            index=runner_labels, columns=runner_labels,
        ).to_string(float_format=lambda value: f"{value:.3f}"))
        print("\nHEAD-AVERAGED PAIRWISE JENSEN-SHANNON DIVERGENCE")
        print(pd.DataFrame(
            overlap_by_view["average"]["js_matrix"],
            index=runner_labels, columns=runner_labels,
        ).to_string(float_format=lambda value: f"{value:.3f}"))
    projected_key = attention_trace["projected_key"][0].detach().cpu().float()
    query_heads = attention_trace["query_heads"][0].detach().cpu().float()
    key_heads = attention_trace["key_heads"][0].detach().cpu().float()
    finite_attention_logits = attention_trace["attention_logits"][0].detach().cpu()
    finite_attention_logits = finite_attention_logits[torch.isfinite(finite_attention_logits)]
    finite_base_logits = attention_trace["base_attention_logits"][0].detach().cpu()
    finite_base_logits = finite_base_logits[torch.isfinite(finite_base_logits)]
    dot_products = finite_base_logits / float(attention_trace["temperature_scale"])
    query_unit = torch.nn.functional.normalize(query_heads, dim=-1)
    key_unit = torch.nn.functional.normalize(key_heads, dim=-1)
    cosine = torch.matmul(query_unit, key_unit.transpose(-2, -1)).reshape(-1)

    def pairwise_cosine_summary(values: torch.Tensor) -> tuple[float, float]:
        unit = torch.nn.functional.normalize(values, dim=-1)
        matrix = unit @ unit.transpose(0, 1)
        mask = ~torch.eye(len(values), dtype=torch.bool)
        off_diagonal = matrix[mask]
        return float(off_diagonal.mean()), float(off_diagonal.std(unbiased=False))

    raw_key_mean, raw_key_std = pairwise_cosine_summary(
        attention_trace["context_before_norm"][0, :query_start].detach().cpu().float()
    )
    normalized_key_mean, normalized_key_std = pairwise_cosine_summary(
        attention_trace["attention_keys"][0, :query_start].detach().cpu().float()
    )
    effective = np.asarray([float(value) for value in pd.DataFrame(rows)["effective runners"]])
    uniform_fraction = effective / query_start
    print(
        "Attention mechanics: "
        f"temperature_scale={float(attention_trace['temperature_scale']):.4f} "
        f"explicit_temperature={float(attention_trace['temperature']):.4f} "
        f"top_k={int(attention_trace['top_k']) or 'disabled'} "
        f"key_projection_norm_mean={float(projected_key.norm(dim=-1).mean()):.4f} "
        f"dot_product_mean/std={float(dot_products.mean()):+.4f}/"
        f"{float(dot_products.std(unbiased=False)):.4f} "
        f"cosine_mean/std={float(cosine.mean()):+.4f}/"
        f"{float(cosine.std(unbiased=False)):.4f} "
        f"attention_logit_std={float(finite_attention_logits.std(unbiased=False)):.4f}."
    )
    print(
        "Key similarity before/after LayerNorm: "
        f"pairwise_cosine={raw_key_mean:+.4f}±{raw_key_std:.4f} -> "
        f"{normalized_key_mean:+.4f}±{normalized_key_std:.4f}."
    )
    if float(np.mean(uniform_fraction)) >= 0.80:
        print(
            "WARNING effectively uniform attention: mean effective retrieval count="
            f"{float(np.mean(effective)):.1f}/{query_start} context runners "
            f"({float(np.mean(uniform_fraction)):.1%})."
        )
    print(
        f"Label-context output projection weight norm={projection_norm:.6f}. "
        "Attention describes retrieval focus; LabelCtx Δp below measures the "
        "branch's actual score effect."
    )
    if projection_norm < 1e-8:
        print(
            "WARNING: the branch exists but its zero-initialized output projection "
            "has not learned yet, so it cannot change predictions."
        )
    return {
        "trace": attention_trace,
        "representations": representations,
        "attention": attention,
        "attention_by_head": attention_by_head,
        "overlap": overlap_by_view,
        "rows": rows,
        "head_rows": head_rows,
    }


def prototype_similarities(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    query_start: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    head = model.context_prototype_head
    if head is None:
        return None
    with torch.inference_mode():
        prototypes, valid = head.build_prototypes(
            tensors["x"], tensors["y"], tensors["train_size"],
            valid_row_mask=tensors["valid_row_mask"],
        )
        projected = head.project(tensors["x"])[0, query_start:]
        negative = (projected * prototypes[0, 0]).sum(dim=-1)
        positive = (projected * prototypes[0, 1]).sum(dim=-1)
    if not bool(valid[0]):
        return None
    return negative.cpu().numpy(), positive.cpu().numpy()


def build_stage_table(
    query: pd.DataFrame,
    final_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor | None],
    query_start: int,
    prototype_similarity: tuple[np.ndarray, np.ndarray] | None,
) -> pd.DataFrame:
    final = final_logits[0, query_start:, :2]
    raw_race_delta = auxiliary["race_delta"]
    scaled_race_delta = auxiliary.get("scaled_race_delta")
    prototype_delta = auxiliary["context_prototype_delta"]
    label_context_delta = auxiliary.get("label_context_delta")
    raw_race = (
        torch.zeros_like(final)
        if raw_race_delta is None else raw_race_delta[0, query_start:, :2]
    )
    race = (
        raw_race
        if scaled_race_delta is None
        else scaled_race_delta[0, query_start:, :2]
    )
    prototype = (
        torch.zeros_like(final)
        if prototype_delta is None else prototype_delta[0, query_start:, :2]
    )
    label_context = (
        torch.zeros_like(final)
        if label_context_delta is None else label_context_delta[0, query_start:, :2]
    )
    base = final - race - prototype - label_context
    after_label_context = base + label_context
    after_race = after_label_context + race
    base_p = positive_probability(base)
    label_context_p = positive_probability(after_label_context)
    race_p = positive_probability(after_race)
    final_p = positive_probability(final)
    market = pd.to_numeric(query.get("fluc2"), errors="coerce").to_numpy()
    market_score = np.where(np.isfinite(market) & (market > 0), 1.0 / market, -np.inf)
    result = pd.DataFrame({
        "No.": query["runner_number"].to_numpy(),
        "Runner": query.get("runner_name", pd.Series("-", index=query.index)).to_numpy(),
        "Base logit": logit_margin(base),
        "Base ICL p": base_p,
        "Base rank": ranks(base_p),
        "LabelCtx logit Δ": logit_margin(label_context),
        "LabelCtx Δp": label_context_p - base_p,
        "LabelCtx rank": ranks(label_context_p),
        "Race raw logit Δ": logit_margin(raw_race),
        "Race scaled logit Δ": logit_margin(race),
        "Race Δp": race_p - label_context_p,
        "Race rank": ranks(race_p),
        "Proto logit Δ": logit_margin(prototype),
        "Proto Δp": final_p - race_p,
        "Final p": final_p,
        "Final rank": ranks(final_p),
        "Market": market,
        "Market rank": ranks(market_score),
        "Actual": query.get("finish_place", pd.Series(np.nan, index=query.index)).to_numpy(),
    })
    if prototype_similarity is not None:
        negative, positive = prototype_similarity
        prototype_index = result.columns.get_loc("Proto logit Δ")
        result.insert(prototype_index, "Proto sim−", negative)
        result.insert(prototype_index + 1, "Proto sim+", positive)
        result.insert(prototype_index + 2, "Proto gap", positive - negative)
    return result.sort_values(["Final rank", "No."], kind="stable").reset_index(drop=True)


def extract_base_logits(
    final_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor | None],
) -> torch.Tensor:
    """Remove every additive post-base branch from final logits."""
    base = final_logits
    for name in (
        "scaled_race_delta", "label_context_delta", "context_prototype_delta"
    ):
        delta = auxiliary.get(name)
        if delta is not None:
            base = base - delta
    if auxiliary.get("scaled_race_delta") is None:
        race_delta = auxiliary.get("race_delta")
        if race_delta is not None:
            base = base - race_delta
    return base


def feature_family(feature: str) -> str:
    """Assign one transparent, non-overlapping diagnostic feature family."""
    market_names = {"open_price", "fluc1", "fluc2"}
    if (
        feature in market_names
        or feature.startswith("market_")
        or "market_edge" in feature
        or feature in {"open_price_rank", "fluc1_price_rank", "fluc2_price_rank"}
        or feature.startswith("race_overlay")
    ):
        return "market"
    if feature.startswith("horse_jockey_") or feature == "winningPartnership":
        return "connections"
    if feature.endswith("_rank") or feature.startswith("race_consensus") or feature.startswith("race_signal"):
        return "race_relative"
    if feature in {"distance_m", "active_field_size", "field_size", "draw_number"}:
        return "race_context"
    return "runner_profile"


def _forward_base(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    x: torch.Tensor,
) -> torch.Tensor:
    output, auxiliary = model(
        x,
        tensors["y"],
        tensors["train_size"],
        cat_mask=tensors["cat_mask"],
        d=tensors["d"],
        race_group_ids=tensors["race_group_ids"],
        valid_row_mask=tensors["valid_row_mask"],
        return_auxiliary_deltas=True,
    )
    return extract_base_logits(output, auxiliary)


def print_base_attribution(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    query: pd.DataFrame,
    table: pd.DataFrame,
    feature_columns: Sequence[str],
    query_start: int,
    requested_runner_number: int | None,
    integration_steps: int,
) -> None:
    print("\nBASE-SCORE ATTRIBUTION")
    if integration_steps < 2:
        raise ValueError("--attribution-steps must be at least 2")
    actual_top3 = table.loc[pd.to_numeric(table["Actual"], errors="coerce") <= 3]
    if requested_runner_number is None:
        misses = actual_top3.loc[actual_top3["Base rank"] > 3]
        candidates = misses if not misses.empty else actual_top3
        selected = candidates.sort_values("Base rank", ascending=False).iloc[0]
        runner_number = int(selected["No."])
    else:
        runner_number = requested_runner_number
    matching = np.flatnonzero(query["runner_number"].to_numpy() == runner_number)
    if len(matching) != 1:
        print(f"Attribution runner No. {runner_number} is not uniquely present; skipped.")
        return
    local_index = int(matching[0])
    sequence_index = query_start + local_index
    selected_row = table.loc[table["No."] == runner_number].iloc[0]
    print(
        f"runner=No. {runner_number} {selected_row['Runner']} base_logit="
        f"{selected_row['Base logit']:+.5f} base_probability="
        f"{selected_row['Base ICL p']:.5f} base_rank={selected_row['Base rank']}"
    )
    print(
        "Feature-family ablations set that family to its fitted median (z=0) for "
        "every query runner; they are controlled masking tests, not learned modules."
    )
    full_base_probability = table.set_index("No.").loc[runner_number, "Base ICL p"]
    full_base_logit = table.set_index("No.").loc[runner_number, "Base logit"]
    full_base_rank = int(table.set_index("No.").loc[runner_number, "Base rank"])
    family_columns: dict[str, list[int]] = {}
    for column_index, name in enumerate(feature_columns):
        family_columns.setdefault(feature_family(name), []).append(column_index)
    rows = [{
        "ablation": "full base prediction",
        "features": len(feature_columns),
        "probability": full_base_probability,
        "probability change": 0.0,
        "logit": full_base_logit,
        "logit change": 0.0,
        "rank": full_base_rank,
        "rank change": 0,
    }]
    with torch.inference_mode():
        for family in ("market", "race_relative", "race_context", "connections", "runner_profile"):
            columns = family_columns.get(family, [])
            masked_x = tensors["x"].clone()
            masked_x[:, query_start:, columns] = 0
            base = _forward_base(model, tensors, masked_x)[0, query_start:, :2]
            probability = positive_probability(base)
            margins = logit_margin(base)
            rank = int(ranks(probability)[local_index])
            rows.append({
                "ablation": f"without {family}",
                "features": len(columns),
                "probability": probability[local_index],
                "probability change": probability[local_index] - full_base_probability,
                "logit": margins[local_index],
                "logit change": margins[local_index] - full_base_logit,
                "rank": rank,
                "rank change": rank - full_base_rank,
            })
        if model.pre_icl_race_encoder is not None:
            handle = model.pre_icl_race_encoder.register_forward_hook(
                lambda _module, inputs, _output: inputs[0]
            )
            try:
                base = _forward_base(model, tensors, tensors["x"])[0, query_start:, :2]
            finally:
                handle.remove()
            probability = positive_probability(base)
            margins = logit_margin(base)
            rank = int(ranks(probability)[local_index])
            rows.append({
                "ablation": "without pre-ICL race encoder",
                "features": 0,
                "probability": probability[local_index],
                "probability change": probability[local_index] - full_base_probability,
                "logit": margins[local_index],
                "logit change": margins[local_index] - full_base_logit,
                "rank": rank,
                "rank change": rank - full_base_rank,
            })
    shown = pd.DataFrame(rows)
    for column in ("probability", "probability change", "logit", "logit change"):
        shown[column] = shown[column].map(lambda value: f"{value:+.5f}")
    print(shown.to_string(index=False))

    print(
        f"Integrated gradients: {integration_steps} points from the fitted-median "
        "baseline for this runner's own standardized features; all other rows fixed."
    )
    original_x = tensors["x"].detach()
    baseline_x = original_x.clone()
    baseline_x[0, sequence_index, :] = 0
    difference = original_x[0, sequence_index, :] - baseline_x[0, sequence_index, :]
    accumulated_gradient = torch.zeros_like(difference)
    alphas = torch.linspace(
        0.0, 1.0, integration_steps, device=original_x.device,
        dtype=original_x.dtype,
    )
    endpoint_logits = []
    for alpha_index, alpha in enumerate(alphas):
        interpolated = baseline_x.clone()
        interpolated[0, sequence_index, :] = (
            baseline_x[0, sequence_index, :] + alpha * difference
        )
        interpolated.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        base = _forward_base(model, tensors, interpolated)
        margin = base[0, sequence_index, 1] - base[0, sequence_index, 0]
        gradient = torch.autograd.grad(margin, interpolated)[0][0, sequence_index]
        weight = 0.5 if alpha_index in (0, integration_steps - 1) else 1.0
        accumulated_gradient += weight * gradient.detach()
        if alpha_index in (0, integration_steps - 1):
            endpoint_logits.append(float(margin.detach()))
    average_gradient = accumulated_gradient / (integration_steps - 1)
    attribution = (difference * average_gradient).detach().cpu().numpy()
    completeness_target = endpoint_logits[-1] - endpoint_logits[0]
    completeness_error = float(attribution.sum() - completeness_target)
    attribution_frame = pd.DataFrame({
        "feature": list(feature_columns),
        "standardized value": difference.detach().cpu().numpy(),
        "integrated-gradient contribution": attribution,
    })
    positive = attribution_frame.nlargest(10, "integrated-gradient contribution")
    negative = attribution_frame.nsmallest(10, "integrated-gradient contribution")
    print("Strongest positive contributors to the base logit margin:")
    print(positive.to_string(index=False, float_format=lambda value: f"{value:+.5f}"))
    print("Strongest negative contributors to the base logit margin:")
    print(negative.to_string(index=False, float_format=lambda value: f"{value:+.5f}"))
    requested_features = {
        "recent_wins", "goodGroundPro", "distance_wins",
        "recent_weighted_win_rate", "recentPodium",
    }
    requested = attribution_frame.loc[attribution_frame["feature"].isin(requested_features)]
    print("Requested Napoleonic feature contributions:")
    print(requested.to_string(index=False, float_format=lambda value: f"{value:+.5f}"))
    print(
        f"Integrated-gradients completeness: attributed_sum={attribution.sum():+.5f} "
        f"observed_base_logit_change={completeness_target:+.5f} "
        f"error={completeness_error:+.5f}."
    )


def print_stage_table(table: pd.DataFrame) -> None:
    print("\nSTAGES 5–8 — RUNNER SCORE DEVELOPMENT")
    print(
        "Base ICL p → label-aware historical cross-attention → "
        "post-race correction → prototype correction → final probability."
    )
    print("Logits and logit deltas are binary class margins: top3 logit minus non-top3 logit.")
    shown = table.copy()
    probability_columns = [
        "Base ICL p", "LabelCtx Δp", "Race Δp", "Proto Δp", "Final p"
    ]
    logit_columns = [
        "Base logit", "LabelCtx logit Δ", "Race raw logit Δ",
        "Race scaled logit Δ", "Proto logit Δ",
    ]
    similarity_columns = ["Proto sim−", "Proto sim+", "Proto gap"]
    for column in probability_columns:
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:.4f}")
    for column in similarity_columns:
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:+.3f}")
    for column in logit_columns:
        if column in shown:
            shown[column] = shown[column].map(lambda value: f"{value:+.4f}")
    shown["Market"] = shown["Market"].map(lambda value: fmt(value, 2))
    shown["Actual"] = shown["Actual"].map(lambda value: fmt(value, 0))
    print(shown.to_string(index=False))
    print("All Δp columns are sequential probability changes, not raw logits.")


def print_ablation(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    query: pd.DataFrame,
    correct_probability: np.ndarray,
    query_start: int,
) -> pd.DataFrame:
    context_rows = query_start
    original = tensors["y"]
    original_context = original[0, :context_rows].detach().cpu().numpy()
    rows = []
    for mode in ("correct", "permuted", "zeroed", "flipped"):
        labels = original.clone()
        changed = ablate_context_labels(original_context, mode, seed=int(query.iloc[0]["race_id"]))
        labels[0, :context_rows] = torch.from_numpy(changed).to(labels.device, labels.dtype)
        logits, _ = run_model(model, tensors, labels=labels)
        probability = positive_probability(logits[0, query_start:, :2])
        changed_top3 = len(set(np.argsort(-probability)[:3]) ^ set(np.argsort(-correct_probability)[:3])) // 2
        rows.append({
            "labels": mode,
            "mean |Δp|": np.mean(np.abs(probability - correct_probability)),
            "max |Δp|": np.max(np.abs(probability - correct_probability)),
            "rank changes": int(np.sum(ranks(probability) != ranks(correct_probability))),
            "top-3 swaps": changed_top3,
        })
    result = pd.DataFrame(rows)
    shown = result.copy()
    for column in ("mean |Δp|", "max |Δp|"):
        shown[column] = shown[column].map(lambda value: f"{value:.6f}")
    print("\nSTAGE 9 — DOES HISTORICAL LABEL CONTEXT MATTER?")
    print("Counterfactual runs change only the labels attached to context runners.")
    print(shown.to_string(index=False))
    return result


def _counterfactual_race_metrics(
    probability: np.ndarray,
    baseline_probability: np.ndarray,
    query: pd.DataFrame,
) -> dict[str, Any]:
    baseline_rank = ranks(baseline_probability)
    changed_rank = ranks(probability)
    baseline_top3 = set(np.argsort(-baseline_probability, kind="stable")[:3])
    changed_top3 = set(np.argsort(-probability, kind="stable")[:3])
    target = pd.to_numeric(query["top3_mask"], errors="coerce").to_numpy()
    complete = int(np.nansum(target == 1)) == 3
    actual = set(np.flatnonzero(target == 1).tolist()) if complete else set()
    clipped = np.clip(probability, 1e-7, 1 - 1e-7)
    return {
        "mean |Δp|": float(np.mean(np.abs(probability - baseline_probability))),
        "max |Δp|": float(np.max(np.abs(probability - baseline_probability))),
        "rank changes": int(np.sum(changed_rank != baseline_rank)),
        "top-3 swaps": len(baseline_top3 ^ changed_top3) // 2,
        "top3 recall": (
            len(actual & changed_top3) / 3.0 if complete else float("nan")
        ),
        "exact top3": float(actual == changed_top3) if complete else float("nan"),
        "logloss": (
            float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))
            if complete else float("nan")
        ),
        "sum(p)": float(probability.sum()),
    }


def print_label_context_retrieval_counterfactuals(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    query: pd.DataFrame,
    context: pd.DataFrame,
    query_start: int,
    final_logits: torch.Tensor,
    auxiliary: Mapping[str, torch.Tensor | None],
    attention_payload: Mapping[str, Any] | None,
    temperature_grid: Sequence[float],
) -> None:
    """Change only the explicit label-context retrieval in a captured forward."""
    head = getattr(model, "label_context_head", None)
    old_delta_full = auxiliary.get("label_context_delta")
    if head is None or old_delta_full is None or attention_payload is None:
        return
    representations = attention_payload["representations"]
    old_delta = old_delta_full[:, query_start:, :2]
    base_without_label = final_logits[:, query_start:, :2] - old_delta
    context_labels = tensors["y"][:, :query_start].long()
    context_valid = tensors["valid_row_mask"][:, :query_start]
    query_valid = tensors["valid_row_mask"][:, query_start:]

    def evaluate(
        labels: torch.Tensor,
        *,
        top_k: int | None = None,
        temperature: float | None = None,
        force_uniform_attention: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
        with torch.inference_mode():
            delta, trace = head.correction_from_context(
                representations[:, query_start:],
                representations[:, :query_start],
                labels,
                context_valid_mask=context_valid,
                query_valid_mask=query_valid,
                return_attention_diagnostics=True,
                top_k=top_k,
                temperature=temperature,
                force_uniform_attention=force_uniform_attention,
            )
            changed_logits = base_without_label + delta
        return (
            positive_probability(changed_logits[0]),
            delta.detach().cpu().numpy(),
            trace,
        )

    correct_probability = positive_probability(final_logits[0, query_start:, :2])
    original_context = context_labels[0].detach().cpu().numpy()
    conditions: list[tuple[str, torch.Tensor, dict[str, Any]]] = [
        ("correct labels", context_labels, {}),
        (
            "permuted labels",
            torch.from_numpy(ablate_context_labels(
                original_context, "permuted", seed=int(query.iloc[0]["race_id"])
            )).to(context_labels.device, context_labels.dtype)[None, :],
            {},
        ),
        ("zeroed labels", torch.zeros_like(context_labels), {}),
        ("flipped labels", 1 - context_labels, {}),
        ("uniform attention", context_labels, {"force_uniform_attention": True}),
        ("top-5 retrieval", context_labels, {"top_k": 5}),
        ("top-10 retrieval", context_labels, {"top_k": 10}),
        ("top-20 retrieval", context_labels, {"top_k": 20}),
    ]
    rows = []
    historical_race_ids = context["race_id"].to_numpy(dtype=np.int64)
    historical_labels = context_labels[0].detach().cpu().numpy()
    for condition, labels, overrides in conditions:
        probability, _, trace = evaluate(labels, **overrides)
        attention = trace["attention"][0].detach().cpu().numpy()
        distribution = [
            attention_distribution_metrics(row, historical_race_ids, historical_labels)
            for row in attention
        ]
        rows.append({
            "condition": condition,
            **_counterfactual_race_metrics(probability, correct_probability, query),
            "effective runners": float(np.mean([row["effective_runners"] for row in distribution])),
            "effective races": float(np.mean([row["effective_races"] for row in distribution])),
        })
    print("\nLABEL-CONTEXT-ONLY RETRIEVAL COUNTERFACTUALS")
    print(
        "Only the explicit label-context branch is recomputed; base ICL, race head, "
        "and prototype outputs remain fixed. Top-k masks are inference-only."
    )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.5f}"))

    temperature_rows = []
    for temperature in temperature_grid:
        probability, _, trace = evaluate(context_labels, temperature=temperature)
        attention = trace["attention"][0].detach().cpu().numpy()
        distribution = [
            attention_distribution_metrics(row, historical_race_ids, historical_labels)
            for row in attention
        ]
        overlap = attention_overlap_metrics(attention)
        temperature_rows.append({
            "temperature": temperature,
            "effective runners": float(np.mean([row["effective_runners"] for row in distribution])),
            "effective races": float(np.mean([row["effective_races"] for row in distribution])),
            "query cosine": overlap["mean_cosine"],
            **_counterfactual_race_metrics(probability, correct_probability, query),
        })
    if temperature_rows:
        print("\nINFERENCE-ONLY LABEL-CONTEXT TEMPERATURE SWEEP")
        print("Smaller temperature divides by a smaller value and therefore sharpens attention.")
        print(pd.DataFrame(temperature_rows).to_string(index=False, float_format=lambda value: f"{value:.5f}"))


def permutation_equivariance_check(
    model: Any,
    tensors: Mapping[str, torch.Tensor | None],
    correct_probability: np.ndarray,
    query_start: int,
) -> tuple[float, bool]:
    """Reverse query-row order and verify runner probabilities follow the rows."""
    total_rows = tensors["x"].shape[1]
    query_rows = total_rows - query_start
    context_order = torch.arange(query_start, device=tensors["x"].device)
    reverse_query = torch.arange(
        total_rows - 1, query_start - 1, -1, device=tensors["x"].device
    )
    order = torch.cat([context_order, reverse_query])
    permuted = dict(tensors)
    for key in ("x", "y", "race_group_ids", "valid_row_mask"):
        permuted[key] = tensors[key][:, order]
    logits, _ = run_model(model, permuted)
    reversed_probability = positive_probability(logits[0, query_start:, :2])[::-1]
    maximum_delta = float(np.max(np.abs(reversed_probability - correct_probability)))
    same_ranking = bool(np.array_equal(
        ranks(reversed_probability), ranks(correct_probability)
    ))
    if query_rows < 2:
        same_ranking = True
    return maximum_delta, same_ranking


def stage_quality(table: pd.DataFrame) -> pd.DataFrame | None:
    actual_mask = pd.to_numeric(table["Actual"], errors="coerce") <= 3
    if int(actual_mask.sum()) != 3:
        return None
    rows = []
    for stage, rank_column in (
        ("base ICL", "Base rank"),
        ("after label context", "LabelCtx rank"),
        ("after race head", "Race rank"),
        ("final", "Final rank"),
        ("market", "Market rank"),
    ):
        predicted_mask = table[rank_column] <= 3
        hits = int((actual_mask & predicted_mask).sum())
        rows.append({
            "stage": stage,
            "top-3 hits": f"{hits}/3",
            "exact set": "yes" if hits == 3 else "no",
            "selected numbers": ", ".join(
                map(str, sorted(table.loc[predicted_mask, "No."].astype(int).tolist()))
            ),
        })
    return pd.DataFrame(rows)


def print_summary(
    table: pd.DataFrame,
    ablation: pd.DataFrame | None,
    permutation_result: tuple[float, bool],
) -> None:
    predicted = table.nsmallest(3, "Final rank")
    actual = table.loc[pd.to_numeric(table["Actual"], errors="coerce") <= 3]
    predicted_numbers = set(predicted["No."].tolist())
    actual_numbers = set(actual["No."].tolist())
    print("\nDIAGNOSTIC SUMMARY")
    quality = stage_quality(table)
    if quality is not None:
        print("Stage-by-stage top-three result:")
        print(quality.to_string(index=False))
    if len(actual_numbers) == 3:
        hits = len(predicted_numbers & actual_numbers)
        print(f"Final top-3 recall for this race: {hits}/3 = {hits / 3:.3f}.")
        print(f"Predicted top 3: {sorted(predicted_numbers)}; actual top 3: {sorted(actual_numbers)}.")
    else:
        print("The race has no complete three-runner outcome yet; ranking only was shown.")
    largest_race = table.iloc[np.abs(table["Race Δp"]).argmax()]
    largest_raw_race = table.iloc[np.abs(table["Race raw logit Δ"]).argmax()]
    largest_proto = table.iloc[np.abs(table["Proto Δp"]).argmax()]
    print(
        f"Largest race-context movement: No. {largest_race['No.']} "
        f"({largest_race['Race Δp']:+.4f} probability)."
    )
    print(
        f"Largest raw/scaled race-head logit movement: No. "
        f"{largest_raw_race['No.']} ({largest_raw_race['Race raw logit Δ']:+.4f} "
        f"raw, {largest_raw_race['Race scaled logit Δ']:+.4f} scaled)."
    )
    print(
        f"Largest prototype movement: No. {largest_proto['No.']} "
        f"({largest_proto['Proto Δp']:+.4f} probability)."
    )
    base_slots = float(table["Base ICL p"].sum())
    label_slots = float((table["Base ICL p"] + table["LabelCtx Δp"]).sum())
    race_slots = float(
        (table["Base ICL p"] + table["LabelCtx Δp"] + table["Race Δp"]).sum()
    )
    final_slots = float(table["Final p"].sum())
    print(
        "Expected top-three slots (sum of marginal probabilities): "
        f"base={base_slots:.3f}, after_label_context={label_slots:.3f}, "
        f"after_race={race_slots:.3f}, final={final_slots:.3f}; "
        "the structural target is 3.000."
    )
    if abs(final_slots - 3.0) > 0.5:
        print(
            "WARNING cardinality mismatch: probabilities rank runners, but their sum is "
            f"{final_slots:.3f}, which is materially different from three."
        )

    if len(actual_numbers) == 3:
        transition_stages = (
            ("Base ICL", "Base rank"),
            ("label-context", "LabelCtx rank"),
            ("race-head", "Race rank"),
            ("prototype/final", "Final rank"),
        )
        stage_top3 = {
            name: set(table.loc[table[rank_column] <= 3, "No."].tolist())
            for name, rank_column in transition_stages
        }
        for number in sorted(actual_numbers):
            row = table.loc[table["No."] == number].iloc[0]
            if number not in stage_top3["Base ICL"]:
                print(
                    f"STAGE FAILURE: No. {number} {row['Runner']} was already "
                    "outside the predicted top 3 at the Base ICL stage."
                )
                continue
            previous_name = "Base ICL"
            previous_set = stage_top3[previous_name]
            for stage_name, _ in transition_stages[1:]:
                current_set = stage_top3[stage_name]
                if number in previous_set and number not in current_set:
                    entrants = sorted(current_set - previous_set)
                    entrant_text = ", ".join(
                        f"No. {entrant} "
                        f"{table.loc[table['No.'] == entrant, 'Runner'].iloc[0]}"
                        for entrant in entrants
                    ) or "another runner"
                    print(
                        f"STAGE FAILURE: No. {number} {row['Runner']} was inside "
                        "the predicted top 3 at Base ICL and was displaced "
                        f"by {entrant_text} at the {stage_name} stage."
                    )
                    break
                previous_name = stage_name
                previous_set = current_set

        missed = actual_numbers - predicted_numbers
        false_positives = predicted_numbers - actual_numbers
        for number in sorted(missed):
            row = table.loc[table["No."] == number].iloc[0]
            print(
                f"MISSED actual top-3 No. {number} {row['Runner']}: "
                f"base rank {row['Base rank']} → label-context rank "
                f"{row['LabelCtx rank']} → race rank {row['Race rank']} → "
                f"final rank {row['Final rank']} "
                f"(label context {row['LabelCtx Δp']:+.4f}, "
                f"race {row['Race Δp']:+.4f}, prototype {row['Proto Δp']:+.4f})."
            )
        for number in sorted(false_positives):
            row = table.loc[table["No."] == number].iloc[0]
            first_entry = next(
                stage_name
                for stage_name, _ in transition_stages
                if number in stage_top3[stage_name]
            )
            print(
                f"FALSE POSITIVE No. {number} {row['Runner']}: actual place "
                f"{fmt(row['Actual'], 0)}, final rank {row['Final rank']}; first "
                f"entered the predicted top 3 at {first_entry}."
            )

    race_strength = float(np.mean(np.abs(table["Race Δp"])))
    label_context_strength = float(np.mean(np.abs(table["LabelCtx Δp"])))
    prototype_strength = float(np.mean(np.abs(table["Proto Δp"])))
    print(
        f"Mean branch influence: label context={label_context_strength:.4f}, "
        f"race head={race_strength:.4f}, "
        f"prototype={prototype_strength:.4f} probability per runner."
    )
    label_context_spread = float(np.ptp(table["LabelCtx Δp"].to_numpy()))
    label_context_logit_spread = float(
        np.ptp(table["LabelCtx logit Δ"].to_numpy())
    )
    label_context_rank_changes = int(
        np.sum(table["Base rank"].to_numpy() != table["LabelCtx rank"].to_numpy())
    )
    print(
        "Label-context ranking effect: "
        f"probability spread={label_context_spread:.6f}, "
        f"logit-margin spread={label_context_logit_spread:.6f}, "
        f"runner rank changes={label_context_rank_changes}."
    )
    other_context_strength = max(prototype_strength, label_context_strength)
    if other_context_strength > 0 and race_strength > 3 * other_context_strength:
        print(
            "Interpretation: the race head dominates both explicit historical "
            "context branches on this race."
        )
    if not table.empty and label_context_strength < 1e-5:
        print(
            "Interpretation: the label-aware cross-attention branch is absent or "
            "effectively inactive on this race."
        )
    elif label_context_rank_changes == 0 and label_context_logit_spread < 0.01:
        print(
            "Interpretation: the label-aware branch is active, but currently acts "
            "mostly as a shared confidence shift rather than distinguishing runners "
            "in this race."
        )
    elif label_context_rank_changes == 0:
        print(
            "Interpretation: the label-aware branch is runner-specific, but its "
            "corrections preserve or reinforce the base ordering rather than changing "
            "the selected top three in this race."
        )
    if ablation is not None:
        flipped = ablation.loc[ablation["labels"] == "flipped"].iloc[0]
        if int(flipped["top-3 swaps"]) == 0:
            print(
                "Interpretation: historical labels alter probabilities but do not decide "
                "the final top-three set for this race."
            )
        else:
            print(
                "Interpretation: historical labels materially affect top-three selection "
                f"({int(flipped['top-3 swaps'])} flipped-label swap(s))."
            )
    permutation_delta, same_ranking = permutation_result
    print(
        "Runner-order permutation check: "
        f"max probability delta={permutation_delta:.8f}, "
        f"ranking_preserved={'yes' if same_ranking else 'NO'}."
    )
    if permutation_delta > 1e-5 or not same_ranking:
        print("WARNING prediction depends on the database row order within the target race.")


def main() -> None:
    args = parse_args()
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
        category=UserWarning,
    )
    if args.top_features < 0:
        raise ValueError("--top-features must be zero or positive")
    if not 0 <= args.attention_query_similarity_warning_threshold <= 1:
        raise ValueError(
            "--attention-query-similarity-warning-threshold must be between 0 and 1"
        )
    temperature_grid = [
        float(value.strip())
        for value in args.label_context_temperature_grid.split(",")
        if value.strip()
    ]
    if any(not np.isfinite(value) or value <= 0 for value in temperature_grid):
        raise ValueError(
            "--label-context-temperature-grid values must be finite and positive"
        )
    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint.resolve(), device, args.strict_load)
    if args.race_head_scale is not None:
        if not np.isfinite(args.race_head_scale) or args.race_head_scale < 0:
            raise ValueError("--race-head-scale must be finite and non-negative")
        if getattr(model, "race_set_head", None) is None:
            raise ValueError("--race-head-scale requires a post-ICL race head")
        model.race_head_scale = float(args.race_head_scale)
    feature_columns = read_feature_columns(args, metadata)
    context_count = checkpoint_context_size(metadata)
    query, context, context_summary = load_race_and_context(
        args.db,
        args.race_id,
        feature_columns,
        metadata,
    )
    if query["competition_id"].nunique(dropna=False) != 1:
        raise ValueError("Target race has inconsistent competition_id values")

    tensors, query_raw, query_scaled = build_tensors(
        model, metadata, context, query, feature_columns, device
    )
    print("RACE MODEL DEBUGGER")
    print(
        f"checkpoint={args.checkpoint.resolve()}\n"
        f"race={args.race_id} {query.iloc[0].get('race_name', '')} "
        f"competition={query.iloc[0]['competition_id']} "
        f"start={query.iloc[0]['start_time_iso']}\n"
        f"architecture: {architecture_description(model)}\n"
        f"context: {context_count} races from "
        f"{context_summary.iloc[0]['context_source']}"
    )
    print_context(context_summary, context)
    print_feature_diagnostics(
        query, query_raw, query_scaled, feature_columns, args.top_features
    )
    query_start = len(context)
    (
        logits,
        auxiliary,
        pre_icl_trace,
        label_context_representations,
    ) = run_model_with_pre_icl_trace(model, tensors)
    print_pre_icl_trace(query, pre_icl_trace, query_start)
    attention_payload = print_label_context_attention(
        model,
        context,
        query,
        label_context_representations,
        tensors,
        query_start,
        verbose=args.debug_attention_details,
        similarity_warning_threshold=(
            args.attention_query_similarity_warning_threshold
        ),
    )
    prototype_similarity = prototype_similarities(model, tensors, query_start)
    table = build_stage_table(
        query, logits, auxiliary, query_start, prototype_similarity
    )
    print_stage_table(table)
    if args.base_attribution:
        print_base_attribution(
            model,
            tensors,
            query,
            table,
            feature_columns,
            query_start,
            args.attribution_runner_number,
            args.attribution_steps,
        )
    ablation = None
    correct_probability = positive_probability(logits[0, query_start:, :2])
    if args.context_ablation:
        ablation = print_ablation(
            model, tensors, query,
            correct_probability,
            query_start,
        )
        print_label_context_retrieval_counterfactuals(
            model,
            tensors,
            query,
            context,
            query_start,
            logits,
            auxiliary,
            attention_payload,
            temperature_grid,
        )
    permutation_result = permutation_equivariance_check(
        model, tensors, correct_probability, query_start
    )
    print_summary(table, ablation, permutation_result)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output_csv, index=False)
        print(f"\nSaved stage table to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
