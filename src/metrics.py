"""TabFM training metrics helpers."""

from __future__ import annotations

import numpy as np


def roc_auc(target: np.ndarray, probability: np.ndarray) -> float:
    """Compute binary ROC AUC with stable tie handling."""
    positive = target == 1
    negative = ~positive

    order = np.argsort(probability, kind="stable")
    sorted_probability = probability[order]
    ranks = np.empty(len(probability), dtype=np.float64)
    start = 0
    while start < len(probability):
        end = start + 1
        while end < len(probability) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def race_top3_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    race_ids: np.ndarray,
) -> dict[str, float | int]:
    """Compute discrete whole-race top-three ranking metrics."""
    complete_races = 0
    total_actual_top3_found = 0
    exact_top3_set_hits = 0
    contained_in_top4 = 0
    contained_in_top5 = 0
    contained_in_top6 = 0

    for race_id in np.unique(race_ids):
        indices = np.flatnonzero(race_ids == race_id)

        actual = indices[target[indices] == 1]
        if len(actual) != 3:
            continue

        ranked = indices[np.argsort(-probability[indices], kind="stable")]
        predicted_top3 = ranked[:3]
        actual_set = set(actual.tolist())

        total_actual_top3_found += len(actual_set & set(predicted_top3.tolist()))
        predicted_top3_set = set(predicted_top3.tolist())
        exact_top3_set_hits += int(actual_set == predicted_top3_set)
        contained_in_top4 += int(actual_set.issubset(set(ranked[:4].tolist())))
        contained_in_top5 += int(actual_set.issubset(set(ranked[:5].tolist())))
        contained_in_top6 += int(actual_set.issubset(set(ranked[:6].tolist())))
        complete_races += 1

    if complete_races == 0:
        return {
            "top3_recall": float("nan"),
            "exact_top3_set_rate": float("nan"),
            "contained_top4_rate": float("nan"),
            "contained_top5_rate": float("nan"),
            "contained_top6_rate": float("nan"),
            "complete_races": 0,
        }

    return {
        "top3_recall": total_actual_top3_found / (complete_races * 3),
        "exact_top3_set_rate": exact_top3_set_hits / complete_races,
        "contained_top4_rate": contained_in_top4 / complete_races,
        "contained_top5_rate": contained_in_top5 / complete_races,
        "contained_top6_rate": contained_in_top6 / complete_races,
        "complete_races": complete_races,
    }


def select_fixed_probe_race_ids(
    target: np.ndarray, race_ids: np.ndarray, max_races: int
) -> np.ndarray:
    """Select the first complete races in the supplied deterministic row order."""
    if len(target) != len(race_ids):
        raise ValueError("Probe targets and race IDs are misaligned")
    if max_races < 1:
        raise ValueError("max_races must be positive")

    selected: list[int] = []
    seen: set[int] = set()
    for race_id_value in race_ids:
        race_id = int(race_id_value)
        if race_id in seen:
            continue
        seen.add(race_id)
        indices = np.flatnonzero(race_ids == race_id)
        if len(indices) >= 4 and int(target[indices].sum()) == 3:
            selected.append(race_id)
            if len(selected) == max_races:
                break

    if not selected:
        raise ValueError("No complete races are available for the fixed probe")
    return np.asarray(selected, dtype=np.int64)


def pre_update_training_batch_metrics(
    target: np.ndarray, logits, race_ids: np.ndarray
) -> dict[str, float | int]:
    """Compute batch ranking metrics from logits before the optimizer update."""
    probabilities = logits.detach().softmax(-1)[:, 1].cpu().numpy()
    return race_top3_metrics(target, probabilities, race_ids)


def checkpoint_selection(
    race_metrics: dict[str, float | int], auc: float, logloss: float
) -> tuple[float, float, float]:
    """Return the ranking metric used for a chronological checkpoint cohort."""
    del auc  # retained in the signature for compatibility; not a selection metric
    return (
        float(race_metrics["top3_recall"]),
        float(race_metrics["contained_top5_rate"]),
        -logloss,
    )


def probability_metrics(
    target: np.ndarray, probability: np.ndarray, race_ids: np.ndarray
) -> dict[str, float | int]:
    """Compute ranking, AUC, and log-loss validation metrics."""
    race_metrics = race_top3_metrics(target, probability, race_ids)
    if race_metrics["complete_races"] == 0:
        raise ValueError("Validation cohort contains no complete top-three races")
    clipped = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return {
        **race_metrics,
        "roc_auc": roc_auc(target, probability),
        "logloss": float(
            -(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean()
        ),
    }


def prediction_change_metrics(
    baseline_probability: np.ndarray,
    ablated_probability: np.ndarray,
    race_ids: np.ndarray,
) -> dict[str, float | int]:
    """Measure probability and within-race ranking changes from an ablation."""
    if not (
        len(baseline_probability) == len(ablated_probability) == len(race_ids)
    ):
        raise ValueError("Prediction comparison arrays are misaligned")

    absolute_delta = np.abs(ablated_probability - baseline_probability)
    ranking_changed_races = 0
    top3_changed_races = 0
    compared_races = 0
    total_rank_displacement = 0.0
    compared_rows = 0
    for race_id in np.unique(race_ids):
        indices = np.flatnonzero(race_ids == race_id)
        if not len(indices):
            continue
        baseline_order = np.argsort(
            -baseline_probability[indices], kind="stable"
        )
        ablated_order = np.argsort(
            -ablated_probability[indices], kind="stable"
        )
        ranking_changed_races += int(not np.array_equal(baseline_order, ablated_order))
        top3_changed_races += int(
            set(baseline_order[:3].tolist()) != set(ablated_order[:3].tolist())
        )
        baseline_rank = np.empty(len(indices), dtype=np.int64)
        ablated_rank = np.empty(len(indices), dtype=np.int64)
        baseline_rank[baseline_order] = np.arange(len(indices))
        ablated_rank[ablated_order] = np.arange(len(indices))
        total_rank_displacement += float(
            np.abs(baseline_rank - ablated_rank).sum()
        )
        compared_rows += len(indices)
        compared_races += 1

    return {
        "max_probability_delta": float(absolute_delta.max(initial=0.0)),
        "mean_probability_delta": float(absolute_delta.mean()) if len(absolute_delta) else 0.0,
        "ranking_changed_races": ranking_changed_races,
        "top3_changed_races": top3_changed_races,
        "compared_races": compared_races,
        "mean_absolute_rank_displacement": (
            total_rank_displacement / compared_rows if compared_rows else 0.0
        ),
    }


def context_permutation_is_ineffective(
    mean_probability_delta: float,
    auc_delta: float,
    logloss_delta: float,
    *,
    minimum_probability_delta: float = 1e-4,
    minimum_auc_change: float = 0.01,
    minimum_logloss_change: float = 0.001,
) -> bool:
    """Return whether permutation has no material probability or metric effect."""
    return (
        abs(mean_probability_delta) < minimum_probability_delta
        and abs(auc_delta) < minimum_auc_change
        and abs(logloss_delta) < minimum_logloss_change
    )


def fixed_probe_has_material_regression(
    best_top3_recall: float,
    best_auc: float,
    current_top3_recall: float,
    current_auc: float,
    *,
    minimum_top3_recall_drop: float = 0.10,
    minimum_auc_drop: float = 0.10,
) -> bool:
    """Return whether both fixed-probe ranking signals regressed materially."""
    return (
        best_top3_recall - current_top3_recall >= minimum_top3_recall_drop
        and best_auc - current_auc >= minimum_auc_drop
    )


def validation_metrics_by_cohort(
    target: np.ndarray,
    probability: np.ndarray,
    race_ids: np.ndarray,
    cohort_labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Compute combined and cohort-specific validation metrics."""
    if not (len(target) == len(probability) == len(race_ids) == len(cohort_labels)):
        raise ValueError("Validation targets, predictions, races, and cohorts are misaligned")
    metrics = {"combined": probability_metrics(target, probability, race_ids)}
    for cohort in sorted(set(map(str, cohort_labels))):
        mask = cohort_labels == cohort
        metrics[cohort] = probability_metrics(
            target[mask], probability[mask], race_ids[mask]
        )
    return metrics


def cohort_checkpoint_selection(
    metrics_by_cohort: dict[str, dict[str, float | int]],
) -> tuple[float, ...]:
    """Return the chronological ranking used for checkpoint selection.

    The broad combined cohort may be dominated by legacy-labelled races, so it
    is deliberately never used for model selection.  The market-miss cohort is
    retained as a diagnostic/guardrail only.
    """
    representative = metrics_by_cohort.get("chronological_representative")
    if representative is None:
        raise ValueError(
            "Checkpoint selection requires the chronological_representative cohort; "
            "combined/legacy validation cannot be used as a fallback"
        )
    if int(representative.get("complete_races", 0)) < 1:
        raise ValueError(
            "Checkpoint selection requires at least one complete chronological race"
        )
    return (
        float(representative["top3_recall"]),
        float(representative["contained_top5_rate"]),
        -float(representative["logloss"]),
    )


def checkpoint_selection_improves(
    candidate_metrics: dict[str, dict[str, float | int]],
    best_metrics: dict[str, dict[str, float | int]],
    recall_tolerance_runners: int = 2,
) -> bool:
    """Return whether chronological ranking selection prefers the candidate.

    Selection is intentionally lexicographic: chronological top-three recall,
    chronological contained-top-5, then chronological log-loss.  The legacy
    combined cohort and market-miss cohort cannot affect this decision.
    """
    del recall_tolerance_runners  # retained for compatibility with older callers
    candidate = cohort_checkpoint_selection(candidate_metrics)
    best = cohort_checkpoint_selection(best_metrics)
    return candidate > best


def stress_guardrail_passes(
    metrics_by_cohort: dict[str, dict[str, float | int]],
    best_observed_stress_recall: float | None,
    max_drop: float,
) -> bool:
    """Check the market-miss recall guardrail for checkpoint selection."""
    stress = metrics_by_cohort.get("market_miss_stress")
    if stress is None or best_observed_stress_recall is None:
        return True
    return float(stress["top3_recall"]) >= best_observed_stress_recall - max_drop


def format_metric_line(
    label: str, metrics: dict[str, float | int]
) -> str:
    """Format a validation metric line for console logging."""
    return (
        f"{label} races={metrics['complete_races']} "
        f"top3_recall={metrics['top3_recall']:.4f} "
        f"exact_top3_set={metrics['exact_top3_set_rate']:.4f} "
        f"contained_top4={metrics['contained_top4_rate']:.4f} "
        f"contained_top5={metrics['contained_top5_rate']:.4f} "
        f"auc={metrics['roc_auc']:.4f} logloss={metrics['logloss']:.5f}"
    )
