"""Winner-ranking metrics and router diagnostics for RaceMixtureOfExperts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.model.race_moe import RaceMixtureOfExperts, router_balance_loss
from src.race_moe_data import batches, pad_batch


def _safe_correlations(values: np.ndarray) -> list[list[float | None]]:
    experts = values.shape[1]
    result: list[list[float | None]] = []
    for left in range(experts):
        row = []
        for right in range(experts):
            a, b = values[:, left], values[:, right]
            if left == right:
                value = 1.0
            elif np.std(a) < 1e-12 or np.std(b) < 1e-12:
                value = np.nan
            else:
                value = float(np.corrcoef(a, b)[0, 1])
            row.append(None if not np.isfinite(value) else value)
        result.append(row)
    return result


def _correlation_matrix(values: np.ndarray, method: str) -> list[list[float | None]]:
    matrix = pd.DataFrame(values).corr(method=method).to_numpy(dtype=float)
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in matrix
    ]


def _mean_race_ranking_correlations(
    race_expert_logits: list[np.ndarray],
) -> list[list[float | None]]:
    experts = race_expert_logits[0].shape[1]
    values = [[[] for _ in range(experts)] for _ in range(experts)]
    for logits in race_expert_logits:
        ranks = pd.DataFrame(logits).rank(method="average").to_numpy()
        correlation = pd.DataFrame(ranks).corr(method="spearman").to_numpy()
        for left in range(experts):
            for right in range(experts):
                if np.isfinite(correlation[left, right]):
                    values[left][right].append(float(correlation[left, right]))
    return [
        [float(np.mean(cell)) if cell else None for cell in row]
        for row in values
    ]


def _expert_prediction_diagnostics(
    race_expert_logits: list[np.ndarray], race_dense_weights: list[np.ndarray],
    winner_indices: list[int],
) -> dict[str, Any]:
    experts = race_expert_logits[0].shape[1]
    picks = np.asarray([
        logits.argmax(axis=0) for logits in race_expert_logits
    ], dtype=np.int64)
    winners = np.asarray(winner_indices, dtype=np.int64)
    agreement = np.zeros((experts, experts), dtype=float)
    for left in range(experts):
        for right in range(experts):
            agreement[left, right] = float(np.mean(picks[:, left] == picks[:, right]))
    unique_rates, unique_winner_rates, unique_winner_given_unique = [], [], []
    expert_hits = []
    for expert in range(experts):
        unique = np.asarray([
            int(np.sum(row == row[expert])) == 1 for row in picks
        ], dtype=bool)
        correct = picks[:, expert] == winners
        unique_rates.append(float(unique.mean()))
        unique_winner_rates.append(float((unique & correct).mean()))
        unique_winner_given_unique.append(
            float(correct[unique].mean()) if unique.any() else None
        )
        expert_hits.append(float(correct.mean()))
    uniform = all(
        np.allclose(weights, 1.0 / experts, atol=1e-7)
        for weights in race_dense_weights
    ) and experts > 1
    if uniform:
        routed_hit_rate = None
        routed_experts = None
    else:
        routed_experts = np.asarray([
            weights.mean(axis=0).argmax() for weights in race_dense_weights
        ], dtype=np.int64)
        routed_picks = picks[np.arange(len(picks)), routed_experts]
        routed_hit_rate = float(np.mean(routed_picks == winners))
    return {
        "top1_selection_agreement": agreement.tolist(),
        "top1_selection_disagreement": (1.0 - agreement).tolist(),
        "unique_top1_race_rate_per_expert": unique_rates,
        "unique_top1_winner_race_rate_per_expert": unique_winner_rates,
        "winner_hit_rate_given_unique_top1_per_expert": unique_winner_given_unique,
        "expert_specific_winner_hit_rate": expert_hits,
        "router_selected_expert_winner_hit_rate": routed_hit_rate,
        "router_selected_expert_frequency": (
            None if routed_experts is None else
            (np.bincount(routed_experts, minlength=experts) / len(routed_experts)).tolist()
        ),
        "router_selection_available": not uniform,
    }


def _characteristic_labels(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    distance = pd.to_numeric(frame.get("distance_m"), errors="coerce")
    field = pd.to_numeric(
        frame.get("active_field_size", frame.get("field_size")), errors="coerce"
    )
    starts = pd.to_numeric(frame.get("career_starts"), errors="coerce")
    labels: dict[str, np.ndarray] = {
        "race_distance": np.select(
            [distance < 1200, distance < 1800, distance >= 1800],
            ["short_<1200m", "middle_1200-1799m", "long_1800m+"], default="unknown",
        ),
        "field_size": np.select(
            [field <= 7, field <= 11, field >= 12],
            ["small_<=7", "medium_8-11", "large_12+"], default="unknown",
        ),
        "track_condition": frame.get(
            "track_status", pd.Series("unknown", index=frame.index)
        ).fillna("unknown").astype(str).to_numpy(),
        "race_class": frame.get(
            "class_name", pd.Series("unknown", index=frame.index)
        ).fillna("unknown").astype(str).to_numpy(),
        "runner_experience": np.select(
            [starts <= 1, starts <= 5, starts > 5],
            ["inexperienced_0-1", "developing_2-5", "experienced_6+"],
            default="unknown",
        ),
    }
    debutants = (starts.fillna(-1) == 0).groupby(frame["race_id"]).transform("sum")
    labels["number_of_debutants"] = np.select(
        [debutants == 0, debutants == 1, debutants >= 2],
        ["none", "one", "two_plus"], default="unknown",
    )
    return labels


def routing_diagnostics(
    weights: np.ndarray, selected: np.ndarray, expert_logits: np.ndarray,
    row_frame: pd.DataFrame, dense_weights: np.ndarray | None = None,
    race_expert_logits: list[np.ndarray] | None = None,
    race_dense_weights: list[np.ndarray] | None = None,
    winner_indices: list[int] | None = None,
) -> dict[str, Any]:
    if dense_weights is None:
        dense_weights = weights
    top = weights.argmax(axis=1)
    experts = weights.shape[1]
    fixed_uniform = (
        experts > 1
        and np.allclose(weights, 1.0 / experts, atol=1e-7)
        and np.allclose(dense_weights, 1.0 / experts, atol=1e-7)
    )
    top_frequency = (
        np.full(experts, 1.0 / experts)
        if fixed_uniform else np.bincount(top, minlength=experts) / len(top)
    )
    entropy = -(
        dense_weights * np.log(np.clip(dense_weights, 1e-12, 1.0))
    ).sum(axis=1)
    mean_weight = weights.mean(axis=0)
    mean_dense_load = dense_weights.mean(axis=0)
    balance = (
        float(experts * np.square(mean_dense_load).sum() - 1.0)
        if experts > 1 else 0.0
    )
    breakdown: dict[str, Any] = {}
    labels = _characteristic_labels(row_frame.reset_index(drop=True))
    for characteristic, values in labels.items():
        groups = []
        for value in pd.unique(values):
            mask = values == value
            if int(mask.sum()) < 10:
                continue
            groups.append({
                "value": str(value), "runners": int(mask.sum()),
                "mean_gate_weight_per_expert": weights[mask].mean(axis=0).tolist(),
                "top1_routed_expert_frequency": (
                    np.bincount(top[mask], minlength=experts) / int(mask.sum())
                ).tolist(),
            })
        breakdown[characteristic] = groups
    descriptions = []
    for expert in range(experts):
        candidates = []
        for characteristic, groups in breakdown.items():
            for group in groups:
                uplift = group["mean_gate_weight_per_expert"][expert] - mean_weight[expert]
                candidates.append((uplift, characteristic, group["value"]))
        uplift, characteristic, value = max(candidates, default=(0.0, "", ""))
        descriptions.append(
            f"stronger usage in {characteristic}={value} (gate uplift {uplift:+.1%})"
            if uplift >= 0.03 else "no obvious specialisation"
        )
    correlations = _safe_correlations(expert_logits)
    spearman = _correlation_matrix(expert_logits, "spearman")
    centred = (
        np.concatenate([
            logits - logits.mean(axis=0, keepdims=True)
            for logits in race_expert_logits
        ]) if race_expert_logits else expert_logits
    )
    centred_pearson = _safe_correlations(centred)
    centred_spearman = _correlation_matrix(centred, "spearman")
    finite_pairs = [
        abs(value) for i, row in enumerate(correlations)
        for j, value in enumerate(row) if j > i and value is not None
    ]
    result = {
        "expert_usage_rate": selected.mean(axis=0).tolist(),
        "mean_gate_weight_per_expert": mean_weight.tolist(),
        "top1_routed_expert_frequency": top_frequency.tolist(),
        "gate_entropy": float(entropy.mean()),
        "dominant_expert_rate": float(top_frequency.max()),
        "average_number_of_active_experts": float(selected.sum(axis=1).mean()),
        "router_balance_loss": balance,
        "router_selection_available": not fixed_uniform,
        "pairwise_expert_logit_correlations": correlations,
        "pairwise_expert_logit_pearson": correlations,
        "pairwise_expert_logit_spearman": spearman,
        "pairwise_race_centred_expert_logit_pearson": centred_pearson,
        "pairwise_race_centred_expert_logit_spearman": centred_spearman,
        "mean_race_level_ranking_correlation": (
            _mean_race_ranking_correlations(race_expert_logits)
            if race_expert_logits else None
        ),
        "maximum_absolute_pairwise_expert_correlation": (
            max(finite_pairs) if finite_pairs else None
        ),
        "specialisation_descriptions": descriptions,
        "breakdown": breakdown,
    }
    if race_expert_logits and race_dense_weights and winner_indices:
        result.update(_expert_prediction_diagnostics(
            race_expert_logits, race_dense_weights, winner_indices
        ))
    return result


def evaluate_model(
    model: RaceMixtureOfExperts, x: np.ndarray, y: np.ndarray,
    race_ids: np.ndarray, indices: dict[int, np.ndarray], row_frame: pd.DataFrame,
    races_per_batch: int, device: torch.device,
) -> tuple[dict[str, float | int], dict[str, Any], pd.DataFrame]:
    model.eval()
    records = []
    all_weights, all_dense_weights, all_selected, all_expert_logits = [], [], [], []
    race_expert_logits, race_dense_weights, winner_indices = [], [], []
    row_cursor = 0
    ordered_rows: list[int] = []
    loss_total = 0.0
    with torch.inference_mode():
        for groups in batches(indices, races_per_batch):
            bx, by, valid = pad_batch(x, y, groups, device)
            output = model(bx, valid, return_diagnostics=True)
            logits = output["logits"]
            for batch_index, rows in enumerate(groups):
                count = len(rows)
                race_logits = logits[batch_index, :count]
                probability = F.softmax(race_logits, dim=0).cpu().numpy()
                race_y = y[rows]
                winner = int(np.flatnonzero(race_y == 1)[0])
                order = np.argsort(-probability, kind="stable")
                winner_rank = int(np.flatnonzero(order == winner)[0]) + 1
                winner_probability = float(probability[winner])
                records.append({
                    "race_id": int(race_ids[rows[0]]), "field_size": count,
                    "winner_rank": winner_rank,
                    "winner_probability": winner_probability,
                    "predicted_runner_number": int(
                        row_frame.iloc[int(rows[order[0]])]["runner_number"]
                    ),
                    "winner_runner_number": int(
                        row_frame.iloc[int(rows[winner])]["runner_number"]
                    ),
                    "race_logloss": -float(np.log(max(winner_probability, 1e-12))),
                })
                loss_total += records[-1]["race_logloss"]
                ordered_rows.extend(map(int, rows))
                all_weights.append(output["router_weights"][batch_index, :count].cpu().numpy())
                all_dense_weights.append(
                    output["dense_router_weights"][batch_index, :count].cpu().numpy()
                )
                all_selected.append(output["selected_experts"][batch_index, :count].cpu().numpy())
                all_expert_logits.append(output["expert_logits"][batch_index, :count].cpu().numpy())
                race_expert_logits.append(
                    output["expert_logits"][batch_index, :count].cpu().numpy()
                )
                race_dense_weights.append(
                    output["dense_router_weights"][batch_index, :count].cpu().numpy()
                )
                winner_indices.append(winner)
                row_cursor += count
    race_results = pd.DataFrame(records)
    metrics: dict[str, float | int] = {
        "races": len(race_results),
        "top1_hit_rate": float(race_results["winner_rank"].eq(1).mean()),
        "top2_containment": float(race_results["winner_rank"].le(2).mean()),
        "top3_containment": float(race_results["winner_rank"].le(3).mean()),
        "mrr": float((1.0 / race_results["winner_rank"]).mean()),
        "race_logloss": float(loss_total / len(race_results)),
        "average_winner_probability": float(race_results["winner_probability"].mean()),
    }
    diagnostics = routing_diagnostics(
        np.concatenate(all_weights), np.concatenate(all_selected),
        np.concatenate(all_expert_logits), row_frame.iloc[ordered_rows].reset_index(drop=True),
        np.concatenate(all_dense_weights),
        race_expert_logits, race_dense_weights, winner_indices,
    )
    return metrics, diagnostics, race_results


def collapse_warnings(
    diagnostics: dict[str, Any], dominant_threshold: float,
    correlation_threshold: float,
) -> list[str]:
    warnings = []
    if (
        len(diagnostics["top1_routed_expert_frequency"]) > 1
        and diagnostics["dominant_expert_rate"] >= dominant_threshold
    ):
        expert = int(np.argmax(diagnostics["top1_routed_expert_frequency"]))
        warnings.append(
            "MoE expert collapse detected. Expert " + str(expert)
            + " is dominant for "
            + f"{diagnostics['dominant_expert_rate']:.1%} of runners."
        )
    correlation = diagnostics["maximum_absolute_pairwise_expert_correlation"]
    if correlation is not None and correlation >= correlation_threshold:
        warnings.append(
            "MoE expert outputs lack meaningful diversity: maximum absolute "
            f"pairwise logit correlation is {correlation:.4f}."
        )
    return warnings
