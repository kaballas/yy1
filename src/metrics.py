"""TabFM training metrics helpers."""

from __future__ import annotations

import numpy as np
from src.constants import (
    CHECKPOINT_RECALL_TOLERANCE_RUNNERS,
    MIN_CHECKPOINT_SELECTION_RACES,
)


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
        if len(indices) >= 3 and int(target[indices].sum()) == 3:
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
) -> tuple[float, float, float, float, float, float]:
    """Prefer broad race-level capture; use smoother runner metrics as tie-breakers."""
    return (
        float(race_metrics["top3_recall"]),
        auc,
        -logloss,
        float(race_metrics["exact_top3_set_rate"]),
        float(race_metrics["contained_top4_rate"]),
        float(race_metrics["contained_top5_rate"]),
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
    """Select on representative races first, with stress metrics as tie-breakers."""
    representative = metrics_by_cohort.get("chronological_representative")
    primary = (
        representative
        if representative is not None
        and int(representative.get("complete_races", MIN_CHECKPOINT_SELECTION_RACES))
        >= MIN_CHECKPOINT_SELECTION_RACES
        else metrics_by_cohort["combined"]
    )
    stress = metrics_by_cohort.get("market_miss_stress", metrics_by_cohort["combined"])
    return (
        float(primary["top3_recall"]),
        float(primary["contained_top5_rate"]),
        -float(primary["logloss"]),
        float(primary["exact_top3_set_rate"]),
        float(primary["contained_top4_rate"]),
        float(primary["roc_auc"]),
        float(stress["top3_recall"]),
        float(stress["contained_top5_rate"]),
        -float(stress["logloss"]),
    )


def checkpoint_selection_improves(
    candidate_metrics: dict[str, dict[str, float | int]],
    best_metrics: dict[str, dict[str, float | int]],
    recall_tolerance_runners: int = CHECKPOINT_RECALL_TOLERANCE_RUNNERS,
) -> bool:
    """Select only checkpoints that improve discrete race-ranking outcomes.

    A recall difference larger than ``recall_tolerance_runners`` remains the
    primary decision. Within that tolerance, require improvement in contained
    top five, exact top three, or contained top four. AUC and log loss remain
    diagnostics and cannot reset early stopping without a race-ranking gain.
    """
    candidate = cohort_checkpoint_selection(candidate_metrics)
    best = cohort_checkpoint_selection(best_metrics)
    representative = candidate_metrics.get("chronological_representative")
    primary = (
        representative
        if representative is not None
        and int(representative.get("complete_races", MIN_CHECKPOINT_SELECTION_RACES))
        >= MIN_CHECKPOINT_SELECTION_RACES
        else candidate_metrics["combined"]
    )
    complete_races = int(primary["complete_races"])
    if complete_races < 1:
        raise ValueError("Checkpoint selection requires at least one complete race")
    recall_tolerance = recall_tolerance_runners / (complete_races * 3)
    recall_difference = candidate[0] - best[0]
    # Recall is discrete (three targets per complete race), so any positive
    # change is a real additional top-three capture and must not be hidden by
    # the tolerance. The tolerance only prevents a one-or-two-runner decline
    # from vetoing a stronger secondary race-ranking result.
    if recall_difference > 0:
        return True
    if recall_difference < -recall_tolerance:
        return False
    candidate_ranking = (
        candidate[1],  # contained_top5_rate
        candidate[3],  # exact_top3_set_rate
        candidate[4],  # contained_top4_rate
    )
    best_ranking = (
        best[1],
        best[3],
        best[4],
    )
    return candidate_ranking > best_ranking


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
