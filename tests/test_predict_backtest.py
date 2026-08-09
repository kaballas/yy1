"""Focused tests for native backtest target selection."""

import sqlite3

import pytest

pytest.importorskip("torch")

from predict_race import (
    finished_race_ids,
    load_competition_context_race_ids,
    load_training_context_for_target,
    prepare_backtest_native_data,
)


def _database(tmp_path):
    path = tmp_path / "races.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE race_runners (
            race_id INTEGER,
            start_time_iso TEXT,
            runner_number INTEGER,
            race_number INTEGER,
            competition_id INTEGER,
            status TEXT,
            feature REAL,
            top3_mask INTEGER,
            is_winner INTEGER
        );
        CREATE VIEW tabfm_trainable_validation_runners AS
        SELECT * FROM race_runners WHERE top3_mask IN (0, 1);
        """
    )
    rows = []
    for race_id, day, competition_id in (
        (1, 1, 590),
        (2, 2, 591),
        (3, 3, 590),
        (4, 4, 590),
    ):
        for runner in range(1, 5):
            rows.append(
                (
                    race_id,
                    f"2026-01-{day:02d}T00:00:00+00:00",
                    runner,
                    1,
                    competition_id,
                    "finished",
                    float(runner),
                    int(runner <= 3),
                    int(runner == 1),
                )
            )
    connection.executemany(
        "INSERT INTO race_runners VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    connection.commit()
    connection.close()
    return path


def test_finished_race_cap_uses_latest_matching_races(tmp_path):
    path = _database(tmp_path)
    assert finished_race_ids(path, 2) == [3, 4]
    assert finished_race_ids(path, 2, competition_id=590) == [3, 4]
    assert finished_race_ids(path, 0, competition_id=591) == [2]


def test_prepared_targets_filter_competition_and_keep_chronological_order(tmp_path):
    path = _database(tmp_path)
    _, targets, _ = prepare_backtest_native_data(
        path, ["feature"], {}, maximum=2, competition_id=590
    )
    assert list(targets) == [3, 4]


def test_prepared_targets_can_select_one_backtest_race(tmp_path):
    path = _database(tmp_path)
    _, targets, _ = prepare_backtest_native_data(
        path, ["feature"], {}, maximum=0, target_race_id="3"
    )
    assert list(targets) == [3]


def test_single_backtest_race_respects_competition_filter(tmp_path):
    path = _database(tmp_path)
    _, targets, _ = prepare_backtest_native_data(
        path,
        ["feature"],
        {},
        maximum=0,
        competition_id=591,
        target_race_id="3",
    )
    assert not targets


def test_negative_backtest_cap_is_rejected(tmp_path):
    path = _database(tmp_path)
    with pytest.raises(ValueError, match="zero or positive"):
        prepare_backtest_native_data(path, ["feature"], {}, maximum=-1)


def test_competition_context_uses_all_complete_strictly_earlier_races(tmp_path):
    path = _database(tmp_path)

    race_ids, competition_id, target_race_number = (
        load_competition_context_race_ids(path, "3")
    )

    assert race_ids == [1]
    assert competition_id == 590
    assert target_race_number == 1


def test_single_race_context_can_be_augmented_with_competition_history(tmp_path):
    path = _database(tmp_path)

    context, query, race_ids = load_training_context_for_target(
        path,
        "4",
        ["feature"],
        {"context_races_per_step": 1},
        include_competition_history=True,
    )

    assert race_ids == [1, 3]
    assert context["race_id"].drop_duplicates().tolist() == [1, 3]
    assert query["race_id"].drop_duplicates().tolist() == [4]
