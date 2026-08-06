import numpy as np
import pytest
import torch

from src.metrics import (
    checkpoint_selection_improves,
    cohort_checkpoint_selection,
    pre_update_training_batch_metrics,
    select_fixed_probe_race_ids,
)


def test_fixed_probe_selection_is_deterministic_and_whole_race():
    race_ids = np.asarray([20, 20, 20, 20, 10, 10, 10, 10, 30, 30, 30])
    target = np.asarray([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0])

    selected = select_fixed_probe_race_ids(target, race_ids, max_races=2)

    np.testing.assert_array_equal(selected, np.asarray([20, 10]))


def _first_pre_update_metrics(learning_rate: float):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    features = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],
            [0.0, 0.0, 0.5],
            [0.5, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    target = torch.tensor([1, 1, 1, 0, 1, 1, 1, 0])
    race_ids = np.asarray([10, 10, 10, 10, 20, 20, 20, 20])

    logits = model(features)
    metrics_before_update = pre_update_training_batch_metrics(
        target.numpy(), logits, race_ids
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    torch.nn.functional.cross_entropy(logits, target).backward()
    optimizer.step()
    return metrics_before_update


def test_learning_rate_does_not_change_first_pre_update_batch_metrics():
    high_rate_metrics = _first_pre_update_metrics(0.003)
    low_rate_metrics = _first_pre_update_metrics(0.00003)

    assert high_rate_metrics == low_rate_metrics


def _cohort_metrics(*, recall, contained5, logloss, auc=0.99):
    return {
        "complete_races": 30,
        "top3_recall": recall,
        "contained_top5_rate": contained5,
        "logloss": logloss,
        "roc_auc": auc,
        "exact_top3_set_rate": 0.5,
        "contained_top4_rate": 0.8,
    }


def test_checkpoint_selection_uses_chronological_lexicographic_metrics():
    best = {
        "combined": _cohort_metrics(recall=0.99, contained5=0.99, logloss=0.01),
        "chronological_representative": _cohort_metrics(
            recall=0.50, contained5=0.70, logloss=0.60
        ),
    }
    candidate = {
        "combined": _cohort_metrics(recall=0.10, contained5=0.10, logloss=1.0),
        "chronological_representative": _cohort_metrics(
            recall=0.50, contained5=0.70, logloss=0.55
        ),
    }

    assert cohort_checkpoint_selection(candidate) == (0.50, 0.70, -0.55)
    assert checkpoint_selection_improves(candidate, best)


def test_checkpoint_selection_rejects_chronological_recall_drop_despite_logloss_gain():
    best = {
        "combined": _cohort_metrics(recall=0.01, contained5=0.01, logloss=1.0),
        "chronological_representative": _cohort_metrics(
            recall=0.60, contained5=0.70, logloss=0.60
        ),
    }
    candidate = {
        "combined": _cohort_metrics(recall=0.99, contained5=0.99, logloss=0.01),
        "chronological_representative": _cohort_metrics(
            recall=0.50, contained5=0.90, logloss=0.10
        ),
    }

    assert not checkpoint_selection_improves(candidate, best)


def test_checkpoint_selection_requires_chronological_cohort():
    with pytest.raises(ValueError, match="chronological_representative"):
        cohort_checkpoint_selection({"combined": _cohort_metrics(
            recall=0.5, contained5=0.5, logloss=0.5
        )})
