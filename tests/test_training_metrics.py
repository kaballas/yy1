import numpy as np
import torch

from src.metrics import pre_update_training_batch_metrics, select_fixed_probe_race_ids


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
