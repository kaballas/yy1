"""Contracts for RaceFormer final refitting on every eligible race."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from predict_raceformer import _backtest_frame
from train_raceformer import _validate_args, parse_args


def _args(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["train_raceformer.py", *extra])
    return parse_args()


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
