"""Contracts for RaceFormer final refitting on every eligible race."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from predict_raceformer import _backtest_frame
from finetune_raceformer import (
    _validate_args as validate_finetune_args,
    parse_args as parse_finetune_args,
)
from train_raceformer import _validate_args, parse_args


def _args(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["train_raceformer.py", *extra])
    return parse_args()


def _finetune_args(monkeypatch, *extra):
    monkeypatch.setattr(
        sys, "argv",
        [
            "finetune_raceformer.py", "--checkpoint", "source.pt",
            "--output", "result.pt", *extra,
        ],
    )
    return parse_finetune_args()


def test_train_all_races_is_a_valid_fixed_epoch_mode(monkeypatch):
    args = _args(monkeypatch, "--train-all-races", "--epochs", "7")

    _validate_args(args)

    assert args.train_all_races
    assert args.epochs == 7


@pytest.mark.parametrize(
    "conflict",
    [
        ("--competition-split",),
        ("--training-competition-id", "1"),
        ("--validation-competition-id", "2"),
        ("--max-training-races", "10"),
        ("--max-validation-races", "10"),
    ],
)
def test_train_all_races_rejects_partial_data_options(monkeypatch, conflict):
    args = _args(monkeypatch, "--train-all-races", *conflict)

    with pytest.raises(ValueError, match="train-all-races"):
        _validate_args(args)


def test_full_data_checkpoint_has_no_automatic_held_out_backtest():
    args = SimpleNamespace(competition_id=None)
    checkpoint = {"partition": {"mode": "full_data_fit"}}

    with pytest.raises(ValueError, match="no held-out backtest cohort"):
        _backtest_frame(args, checkpoint, ["fluc2"])


def test_backtest_can_explicitly_select_sealed_test_ids(monkeypatch):
    args = SimpleNamespace(
        competition_id=None, backtest_cohort="test", db="unused",
        backtest_max_races=0,
    )
    checkpoint = {
        "partition": {"validation_race_ids": [1], "test_race_ids": [2, 3]}
    }
    captured = {}

    def fake_load(db, race_ids, features, maximum, competition_id):
        captured["race_ids"] = race_ids
        return "frame"

    monkeypatch.setattr("predict_raceformer.load_checkpoint_backtest", fake_load)

    frame, source = _backtest_frame(args, checkpoint, ["fluc2"])

    assert frame == "frame"
    assert captured["race_ids"] == [2, 3]
    assert source == "checkpoint_test_races"


def test_three_way_training_arguments_are_valid(monkeypatch):
    args = _args(
        monkeypatch, "--chronological-validation-races", "1000",
        "--chronological-test-races", "1000",
    )

    _validate_args(args)

    assert args.chronological_test_races == 1000


def test_sealed_test_requires_a_validation_cohort(monkeypatch):
    args = _args(
        monkeypatch, "--chronological-validation-races", "0",
        "--chronological-test-races", "1000",
    )

    with pytest.raises(ValueError, match="requires"):
        _validate_args(args)


def test_finetune_sealed_test_requires_inherited_partition(monkeypatch):
    args = _finetune_args(monkeypatch, "--evaluate-sealed-test")

    with pytest.raises(ValueError, match="requires"):
        validate_finetune_args(args)


def test_finetune_can_inherit_partition_and_evaluate_test(monkeypatch):
    args = _finetune_args(
        monkeypatch, "--inherit-checkpoint-partition", "--evaluate-sealed-test"
    )

    validate_finetune_args(args)

    assert args.inherit_checkpoint_partition
