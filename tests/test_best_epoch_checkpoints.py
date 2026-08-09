from datetime import datetime, timezone

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.training import (
    save_best_epoch_checkpoint,
    timestamped_best_checkpoint_path,
)


def test_timestamped_best_checkpoint_path_contains_epoch_and_timestamp(tmp_path):
    saved_at = datetime(2026, 8, 9, 12, 34, 56, 123456, tzinfo=timezone.utc)

    path, iso_timestamp = timestamped_best_checkpoint_path(
        tmp_path / "model.pt", 7, saved_at
    )

    assert path.name == "model.best-epoch-007.20260809T123456123456+0000.pt"
    assert iso_timestamp == "2026-08-09T12:34:56.123456+00:00"


def test_save_best_epoch_checkpoint_writes_reviewable_bundle(tmp_path):
    path = save_best_epoch_checkpoint(
        output=tmp_path / "model.pt",
        epoch=2,
        state_dict={"weight": torch.tensor([1.0])},
        model_kwargs={"dim": 4},
        feature_columns=["speed"],
        median=np.asarray([1.0]),
        scale=np.asarray([2.0]),
        context_races_per_step=8,
        zeroed_features=[],
        metrics={"top3_recall": 0.5},
        metrics_by_cohort={"combined": {"top3_recall": 0.5}},
    )

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert bundle["best_epoch"] == 2
    assert bundle["checkpoint_kind"] == "best_epoch_snapshot"
    assert bundle["label"] == "top3_mask"
    assert bundle["context_races_per_step"] == 8
    assert bundle["best_metrics"]["top3_recall"] == 0.5
    assert not path.with_suffix(path.suffix + ".tmp").exists()
