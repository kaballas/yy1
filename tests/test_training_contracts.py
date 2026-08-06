"""Focused contracts for scopes, chronology, scheduling, and top-three output."""

from datetime import datetime, timezone
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from src.cli import parse_args
from src.model import TabFM
from src.prediction import predict, predict_with_chronological_context
from src.sampling import (
    build_query_race_schedule,
    sample_independent_race_batch,
    sample_race_batch,
)
from src.training import (
    configure_trainable_parameters,
    resolve_training_scope,
    trainable_parameter_names,
    validate_fine_tune_learning_rate,
)
from src.training import early_stopping_is_enabled


class _FakeModel(nn.Module):
    def __init__(self, with_race_head=True):
        super().__init__()
        self.cell_embedder = nn.Linear(2, 2)
        self.col_embedder = nn.Linear(2, 2)
        self.col_embedder_2 = nn.Linear(2, 2)
        self.row_interactor = nn.Linear(2, 2)
        self.row_interactor_2 = nn.Linear(2, 2)
        self.cls_tokens = nn.Parameter(torch.zeros(2, 2))
        self.icl_predictor = nn.Module()
        self.icl_predictor.decoder = nn.Linear(2, 2)
        self.icl_predictor.tf_icl = nn.Linear(2, 2)
        self.icl_predictor.ln = nn.LayerNorm(2)
        self.icl_predictor.y_encoder = nn.Linear(2, 2)
        self.pre_icl_race_encoder = nn.Linear(2, 2)
        self.race_set_head = nn.Module() if with_race_head else None
        if self.race_set_head is not None:
            self.race_set_head.input_projection = nn.Linear(2, 2)
            self.race_set_head.encoder = nn.Linear(2, 2)
            self.race_set_head.output_norm = nn.LayerNorm(2)
            self.race_set_head.output_projection = nn.Linear(2, 2)


def test_fine_tune_scopes_reset_and_select_exact_modules():
    model = _FakeModel()
    _, decoder_count, _ = configure_trainable_parameters(model, "decoder_and_race_head")
    decoder_names = trainable_parameter_names(model)
    assert decoder_count > 0
    assert decoder_names
    assert all(
        name.startswith("icl_predictor.decoder.") or name.startswith("race_set_head.")
        for name in decoder_names
    )
    assert any(name.startswith("icl_predictor.decoder.") for name in decoder_names)
    assert any(name.startswith("race_set_head.") for name in decoder_names)

    configure_trainable_parameters(model, "attention_head_only")
    assert all(name.startswith("race_set_head.") for name in trainable_parameter_names(model))
    configure_trainable_parameters(model, "icl_and_race_head")
    assert all(
        name.startswith("icl_predictor.") or name.startswith("race_set_head.")
        for name in trainable_parameter_names(model)
    )
    configure_trainable_parameters(model, "full_model")
    assert len(trainable_parameter_names(model)) == sum(1 for _ in model.named_parameters())
    configure_trainable_parameters(model, "attention_head_only")
    assert all(name.startswith("race_set_head.") for name in trainable_parameter_names(model))


def test_partial_scope_requires_race_head():
    with pytest.raises(ValueError, match="race_context_mode=self_attention"):
        configure_trainable_parameters(_FakeModel(with_race_head=False), "decoder_and_race_head")


def test_cli_accepts_decoder_and_race_head(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--fine-tune-scope", "decoder_and_race_head"])
    assert parse_args().fine_tune_scope == "decoder_and_race_head"


def test_cli_can_explicitly_enable_small_cohort_early_stopping(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_model.py", "--allow-small-cohort-early-stopping"])
    assert parse_args().allow_small_cohort_early_stopping


def test_resumed_self_attention_defaults_to_narrow_fine_tune_scope():
    assert resolve_training_scope(None, False, True, "self_attention") == "icl_and_race_head"
    assert resolve_training_scope(None, False, False, "self_attention") == "full_model"
    assert resolve_training_scope("full_model", False, True, "self_attention") == "full_model"


def test_high_fine_tune_learning_rate_requires_explicit_unlock():
    validate_fine_tune_learning_rate(3e-5, True, False)
    validate_fine_tune_learning_rate(3e-4, False, False)
    validate_fine_tune_learning_rate(3e-4, True, True)
    with pytest.raises(ValueError, match="safe ceiling"):
        validate_fine_tune_learning_rate(3e-4, True, False)


def test_small_selection_cohort_disables_early_stopping():
    assert not early_stopping_is_enabled(19)
    assert early_stopping_is_enabled(20)


def test_normal_training_has_no_database_mutation_flag_or_update():
    cli_source = open("src/cli.py", encoding="utf-8").read()
    training_source = open("src/training.py", encoding="utf-8").read()
    assert "mark-trained-races-validation" not in cli_source
    assert "UPDATE race_runners SET is_validation" not in training_source


def _race_data():
    race_ids = [1, 2, 3, 4, 5, 6]
    race_indices = {
        race_id: np.arange(index * 3, index * 3 + 3)
        for index, race_id in enumerate(race_ids)
    }
    x = np.arange(18, dtype=np.float32).reshape(18, 1)
    y = np.tile(np.asarray([1, 1, 1], dtype=np.int64), 6)
    times = {
        race_id: datetime(2026, 1, race_id, tzinfo=timezone.utc)
        for race_id in race_ids
    }
    return x, y, race_indices, times


def test_chronological_sampling_is_complete_and_disjoint():
    x, y, race_indices, times = _race_data()
    batch = sample_race_batch(
        x, y, race_indices, 2, 2, np.random.default_rng(4),
        forced_query_race_ids=np.asarray([5, 6]), race_time_by_id=times,
    )
    _, _, context_rows, context_ids, query_ids, query_rows, groups = batch
    assert max(times[int(race_id)] for race_id in context_ids) < min(
        times[int(race_id)] for race_id in query_ids
    )
    assert set(map(int, context_ids)).isdisjoint(map(int, query_ids))
    assert len(query_rows) == 6
    assert groups.shape == (1, context_rows + len(query_rows))

    with pytest.raises(ValueError, match="Insufficient earlier context"):
        sample_race_batch(
            x, y, race_indices, 4, 1, np.random.default_rng(4),
            forced_query_race_ids=np.asarray([2]), race_time_by_id=times,
        )


def test_independent_training_batch_matches_validation_context_per_query():
    x, y, race_indices, times = _race_data()
    (
        batch_x,
        batch_y,
        train_sizes,
        context_ids,
        query_ids,
        groups,
        valid,
    ) = sample_independent_race_batch(
        x,
        y,
        race_indices,
        2,
        np.asarray([5, 6]),
        times,
    )

    np.testing.assert_array_equal(context_ids, np.asarray([3, 4, 4, 5]))
    np.testing.assert_array_equal(query_ids, np.asarray([5, 6]))
    np.testing.assert_array_equal(train_sizes.numpy(), np.asarray([6, 6]))
    assert batch_x.shape == (2, 9, 1)
    assert batch_y.shape == groups.shape == valid.shape == (2, 9)
    assert valid.all()
    assert set(groups[0, 6:].tolist()) == {5}
    assert set(groups[1, 6:].tolist()) == {6}


def test_independent_training_batch_pads_only_after_complete_query():
    x, y, race_indices, times = _race_data()
    race_indices = dict(race_indices)
    race_indices[6] = race_indices[6][:2]
    (
        _,
        batch_y,
        train_sizes,
        _,
        _,
        groups,
        valid,
    ) = sample_independent_race_batch(
        x,
        y,
        race_indices,
        2,
        np.asarray([5, 6]),
        times,
    )

    assert train_sizes.tolist() == [6, 6]
    assert valid[0].all()
    assert not valid[1, -1]
    assert batch_y[1, -1] == -100
    assert groups[1, -1] == -1


def test_independent_padded_batch_matches_separate_model_sequences():
    x, y, race_indices, times = _race_data()
    race_indices = dict(race_indices)
    race_indices[6] = race_indices[6][:2]
    batch_x, batch_y, train_sizes, _, _, groups, valid = (
        sample_independent_race_batch(
            x,
            y,
            race_indices,
            2,
            np.asarray([5, 6]),
            times,
        )
    )
    torch.manual_seed(9)
    model = TabFM(
        embed_dim=8,
        max_classes=2,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=2,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=2,
        icl_num_blocks=1,
        icl_nhead=2,
        ff_factor=2,
        feature_group_size=1,
        num_freq=4,
        decoder_hidden=6,
        race_context_mode="self_attention",
        race_context_dim=8,
        race_context_heads=2,
        race_context_ff_dim=12,
    ).eval()

    with torch.no_grad():
        batched = model(
            batch_x,
            batch_y,
            train_sizes,
            race_group_ids=groups,
            valid_row_mask=valid,
        )
        for batch_index in range(batch_x.shape[0]):
            row_count = int(valid[batch_index].sum())
            separate = model(
                batch_x[batch_index:batch_index + 1, :row_count],
                batch_y[batch_index:batch_index + 1, :row_count],
                train_sizes[batch_index:batch_index + 1],
                race_group_ids=groups[batch_index:batch_index + 1, :row_count],
                valid_row_mask=valid[batch_index:batch_index + 1, :row_count],
            )
            torch.testing.assert_close(
                batched[batch_index, :row_count], separate[0], rtol=0, atol=1e-6
            )


def test_query_schedule_has_no_within_step_duplicates_across_cycles():
    schedule = build_query_race_schedule([1, 2, 3], 11, 2, np.random.default_rng(1))
    assert schedule.shape == (11, 2)
    assert all(len(set(map(int, row))) == 2 for row in schedule)
    with pytest.raises(ValueError, match="cannot exceed"):
        build_query_race_schedule([1, 2], 1, 3, np.random.default_rng(1))


class _PredictionModel(nn.Module):
    encode_races_before_icl = False

    def forward(self, x, y, train_size, race_group_ids=None, valid_row_mask=None):
        score = x[..., 0]
        return torch.stack((torch.zeros_like(score), score), dim=-1)


def test_prediction_returns_binary_top3_scores_without_race_sum_normalization():
    context_x = np.zeros((2, 1), dtype=np.float32)
    context_y = np.asarray([0, 1], dtype=np.int64)
    query_x = np.asarray([[2.0], [1.0], [0.5], [0.0]], dtype=np.float32)
    query_ids = np.asarray([10, 10, 20, 20], dtype=np.int64)
    output = predict(
        _PredictionModel(), context_x, context_y, query_x, query_ids,
        2, torch.device("cpu"),
    )
    assert np.isfinite(output).all()
    assert ((output >= 0) & (output <= 1)).all()
    assert output[0] > output[1] > output[2] > output[3]
    assert not np.isclose(output[:2].sum(), 1.0)
    assert not np.isclose(output[2:].sum(), 1.0)


class _ContextRecordingPredictionModel(_PredictionModel):
    def __init__(self):
        super().__init__()
        self.context_values: list[np.ndarray] = []

    def forward(self, x, y, train_size, race_group_ids=None, valid_row_mask=None):
        self.context_values.append(x[0, :int(train_size[0]), 0].cpu().numpy())
        return super().forward(x, y, train_size, race_group_ids, valid_row_mask)


def test_causal_validation_uses_only_most_recent_earlier_training_races():
    context_x = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    context_y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    context_indices = {
        1: np.asarray([0]),
        2: np.asarray([1]),
        3: np.asarray([2]),
        4: np.asarray([3]),
    }
    times = {
        1: datetime(2026, 1, 1, tzinfo=timezone.utc),
        2: datetime(2026, 1, 2, tzinfo=timezone.utc),
        3: datetime(2026, 1, 3, tzinfo=timezone.utc),
        4: datetime(2026, 1, 4, tzinfo=timezone.utc),
        10: datetime(2026, 1, 5, tzinfo=timezone.utc),
    }
    query_x = np.asarray([[2.0], [1.0]], dtype=np.float32)
    model = _ContextRecordingPredictionModel()

    output = predict_with_chronological_context(
        model,
        context_x,
        context_y,
        context_indices,
        times,
        query_x,
        {10: np.asarray([0, 1])},
        2,
        torch.device("cpu"),
    )

    np.testing.assert_array_equal(model.context_values[0], np.asarray([3.0, 4.0]))
    assert output[0] > output[1]
