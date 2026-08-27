import sqlite3

import numpy as np

from update_text_vectors import (
    EpochProgress,
    SQLiteSentences,
    populate_vectors,
    train_model,
    value_token,
)


def make_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE race_runners (runner_name TEXT, country TEXT, "
            "runner_name_vec BLOB, country_vec BLOB)"
        )
        connection.executemany(
            "INSERT INTO race_runners (runner_name, country) VALUES (?, ?)",
            [("Fast Horse", "AU"), ("  ", "NZ"), (None, "AU")],
        )


def test_value_token_scopes_complete_value_by_column():
    assert value_token("runner_name", " Fast Horse ") == "runner_name\x1fFast Horse"
    assert value_token("trainer", "Smith") != value_token("jockey", "Smith")
    assert value_token("name", " ") is None


def test_sqlite_sentences_are_restartable(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)
    corpus = SQLiteSentences(database, "race_runners", ["runner_name", "country"])

    first = list(corpus)

    assert first == list(corpus)
    assert first[0] == ["runner_name\x1fFast Horse", "country\x1fAU"]
    assert first[1] == ["country\x1fNZ"]


def test_populate_vectors_writes_float32_blobs_and_nulls(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)
    vectors = {
        "runner_name\x1fFast Horse": np.array([1, 2], dtype=np.float32),
        "country\x1fAU": np.array([3, 4], dtype=np.float32),
        "country\x1fNZ": np.array([5, 6], dtype=np.float32),
    }

    rows, written, missing = populate_vectors(
        database,
        "race_runners",
        [("runner_name", "runner_name_vec"), ("country", "country_vec")],
        vectors,
        batch_size=2,
    )

    assert (rows, written, missing) == (3, 4, 0)
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT runner_name_vec, country_vec FROM race_runners ORDER BY rowid"
        ).fetchall()
    np.testing.assert_array_equal(np.frombuffer(stored[0][0], dtype="<f4"), [1, 2])
    assert stored[1][0] is None
    assert stored[2][0] is None


def test_train_model_saves_reloadable_model_and_metadata(tmp_path):
    database = tmp_path / "races.sqlite"
    model_path = tmp_path / "text.model"
    make_database(database)

    model = train_model(
        database, "race_runners", ["runner_name", "country"], model_path,
        dimensions=4, window=10, epochs=2, min_count=1, workers=1, seed=42,
    )

    assert model.vector_size == 4
    assert "country\x1fAU" in model.wv
    assert model_path.is_file()
    assert model_path.with_suffix(".model.json").is_file()


def test_epoch_progress_prints_loss_and_eta(monkeypatch, capsys):
    times = iter([10.0, 12.0, 13.0])
    monkeypatch.setattr("update_text_vectors.time.monotonic", lambda: next(times))

    class Model:
        @staticmethod
        def get_latest_training_loss():
            return 7.5

    progress = EpochProgress(2)
    progress.on_train_begin(Model())
    progress.on_epoch_end(Model())
    progress.on_train_end(Model())

    output = capsys.readouterr().out
    assert "epoch=1/2" in output
    assert "loss=7.500000" in output
    assert "eta=2.0s" in output
