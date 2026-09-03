import argparse

import numpy as np
import pandas as pd
import pytest

import backtest_winner_ranker_checkpoints as backtest
from backtest_winner_ranker_checkpoints import (
    build_model_from_checkpoint,
    cohort_key,
    filter_race_ids_by_number,
    iso_date,
    leaderboard,
    prepare_checkpoint_data,
    preprocessing_fingerprint,
    race_ids_in_date_range,
    race_ids_on_date,
    validate_date_filters,
)


def test_cohort_key_changes_when_race_order_changes():
    assert cohort_key([1, 2, 3]) == cohort_key([1, 2, 3])
    assert cohort_key([1, 2, 3]) != cohort_key([3, 2, 1])


def test_iso_date_accepts_calendar_date_and_rejects_invalid_value():
    assert iso_date("2026-08-29") == "2026-08-29"
    with pytest.raises(argparse.ArgumentTypeError, match="YYYY-MM-DD"):
        iso_date("29/08/2026")


def test_race_ids_on_date_uses_every_matching_race_in_database_order():
    frame = pd.DataFrame({
        "race_id": [3, 1, 2, 4],
        "start_time_iso": [
            "2026-08-29T10:00:00+00:00",
            "2026-08-28T10:00:00+00:00",
            "2026-08-29T09:00:00+00:00",
            None,
        ],
    })

    assert race_ids_on_date(frame, "2026-08-29") == [3, 2]


def test_race_ids_in_date_range_is_inclusive_and_preserves_database_order():
    frame = pd.DataFrame({
        "race_id": [4, 1, 2, 3, 2],
        "start_time_iso": [
            "2026-09-03T10:00:00+00:00",
            "2026-08-31T10:00:00+00:00",
            "2026-09-01T09:00:00+00:00",
            "2026-09-02T12:00:00+00:00",
            "2026-09-01T09:00:00+00:00",
        ],
    })

    assert race_ids_in_date_range(
        frame, "2026-09-01", "2026-09-02",
    ) == [2, 3]
    assert race_ids_in_date_range(frame, "2026-09-02", None) == [4, 3]
    assert race_ids_in_date_range(frame, None, "2026-09-01") == [1, 2]


def test_validate_date_filters_rejects_ambiguous_and_reversed_ranges():
    validate_date_filters(None, "2026-09-01", "2026-09-02")
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_date_filters("2026-09-01", "2026-09-01", None)
    with pytest.raises(ValueError, match="cannot be after"):
        validate_date_filters(None, "2026-09-02", "2026-09-01")


def test_filter_race_ids_by_number_preserves_selected_cohort_order():
    frame = pd.DataFrame({
        "race_id": [3, 3, 1, 1, 2, 2],
        "race_number": [5, 5, 4, 4, 5, 5],
    })

    assert filter_race_ids_by_number(frame, [1, 2, 3], 5) == [2, 3]
    assert filter_race_ids_by_number(frame, [1, 2, 3], None) == [1, 2, 3]


def test_filter_race_ids_by_number_validates_value_and_column():
    with pytest.raises(ValueError, match="must be positive"):
        filter_race_ids_by_number(pd.DataFrame(), [1], 0)
    with pytest.raises(ValueError, match="has no race_number"):
        filter_race_ids_by_number(pd.DataFrame({"race_id": [1]}), [1], 1)


def test_preprocessing_fingerprint_supports_numpy_arrays_and_scalars():
    first = {
        "version": np.int64(3),
        "median": np.asarray([1.0, 2.0], dtype=np.float32),
        "scale": np.asarray([3.0, 4.0], dtype=np.float32),
    }
    equivalent = {
        "scale": first["scale"].copy(),
        "median": first["median"].copy(),
        "version": 3,
    }
    changed = {**equivalent, "scale": np.asarray([3.0, 5.0], dtype=np.float32)}

    assert preprocessing_fingerprint(first) == preprocessing_fingerprint(equivalent)
    assert preprocessing_fingerprint(first) != preprocessing_fingerprint(changed)


def test_prepare_checkpoint_data_reuses_matching_preprocessing(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "race_id": [2, 2, 1, 1],
        "runner_number": [2, 1, 2, 1],
        "is_winner": [0, 1, 1, 0],
        "speed": [20.0, 10.0, 40.0, 30.0],
    })
    checkpoint = {
        "raw_feature_columns": ["speed"],
        "zeroed_features": [],
        "preprocessing": {"version": "test"},
    }
    calls = 0

    def fake_transform(raw, *_args):
        nonlocal calls
        calls += 1
        return raw + 1

    monkeypatch.setattr(backtest, "transform_raceformer", fake_transform)
    cache = {}

    first = prepare_checkpoint_data(tmp_path / "races.db", checkpoint, frame, [1, 2], cache)
    second = prepare_checkpoint_data(tmp_path / "races.db", checkpoint, frame, [1, 2], cache)

    assert first is second
    assert calls == 1
    assert first["race_id_array"].tolist() == [1, 1, 2, 2]
    np.testing.assert_array_equal(first["values"].ravel(), [31, 41, 11, 21])

    changed = {**checkpoint, "preprocessing": {"version": "other"}}
    prepare_checkpoint_data(tmp_path / "races.db", changed, frame, [1, 2], cache)
    assert calls == 2


def test_evaluate_checkpoint_shares_frame_and_prepared_data(monkeypatch, tmp_path):
    frame = pd.DataFrame({
        "race_id": [1, 1],
        "runner_number": [1, 2],
        "is_winner": [1, 0],
        "speed": [10.0, 20.0],
    })
    checkpoint = {
        "checkpoint_type": "race_winner_moe",
        "raw_feature_columns": ["speed"],
        "zeroed_features": [],
        "preprocessing": {"version": "test"},
        "partition": {"test_race_ids": [1]},
    }
    load_calls = 0
    transform_calls = 0

    def fake_load(_database, features):
        nonlocal load_calls
        load_calls += 1
        assert features == ("speed",)
        return frame

    def fake_transform(raw, *_args):
        nonlocal transform_calls
        transform_calls += 1
        return raw

    metrics = {
        "top1_hit_rate": 1.0,
        "top2_containment": 1.0,
        "top3_containment": 1.0,
        "mrr": 1.0,
        "race_logloss": 0.0,
        "average_winner_probability": 1.0,
    }
    monkeypatch.setattr(backtest, "load_finished_winner_rows", fake_load)
    monkeypatch.setattr(backtest, "transform_raceformer", fake_transform)
    monkeypatch.setattr(
        backtest, "evaluate_model", lambda *_args: (metrics, {}, pd.DataFrame()),
    )
    frame_cache = {}
    prepared_cache = {}

    for model_name in ("one.pt", "two.pt"):
        backtest.evaluate_checkpoint(
            tmp_path / model_name,
            checkpoint,
            object(),
            "test",
            tmp_path / "races.db",
            64,
            backtest.torch.device("cpu"),
            frame_cache=frame_cache,
            prepared_cache=prepared_cache,
        )

    assert load_calls == 1
    assert transform_calls == 1


def test_build_model_uses_already_loaded_checkpoint(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.loaded = None
            self.device = None

        def load_state_dict(self, state, strict):
            self.loaded = (state, strict)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    model = FakeModel()
    monkeypatch.setattr(backtest, "build_race_winner_model", lambda _config: model)
    checkpoint = {
        "checkpoint_type": "race_winner_moe",
        "model_config": {"experts": 2},
        "model_state_dict": {"weight": "already loaded"},
    }

    result = build_model_from_checkpoint(checkpoint, backtest.torch.device("cpu"))

    assert result is model
    assert model.loaded == (checkpoint["model_state_dict"], True)
    assert model.device == backtest.torch.device("cpu")


def test_leaderboard_ranks_models_by_logloss_then_mrr_and_top1():
    records = [
        {
            "path": "a.pt", "model": "a", "checkpoint_type": "race_winner_moe",
            "cohort_id": "shared", "races": 2,
            "metrics": {
                "top1_hit_rate": 0.60, "top2_containment": 0.8,
                "top3_containment": 0.9, "mrr": 0.70,
                "race_logloss": 1.20, "average_winner_probability": 0.3,
            },
        },
        {
            "path": "b.pt", "model": "b", "checkpoint_type": "race_winner_moe",
            "cohort_id": "shared", "races": 2,
            "metrics": {
                "top1_hit_rate": 0.50, "top2_containment": 0.7,
                "top3_containment": 0.8, "mrr": 0.60,
                "race_logloss": 1.10, "average_winner_probability": 0.4,
            },
        },
    ]

    result = leaderboard(records)

    assert result["model"].tolist() == ["b", "a"]
    assert result["cohort_rank"].tolist() == [1, 2]
