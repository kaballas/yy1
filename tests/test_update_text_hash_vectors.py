import sqlite3

import numpy as np

from update_text_hash_vectors import hash_vector, populate_hash_vectors


def make_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE race_runners (name TEXT, country TEXT, "
            "name_vec BLOB, country_vec BLOB)"
        )
        connection.executemany(
            "INSERT INTO race_runners (name, country) VALUES (?, ?)",
            [("Horse A", "AU"), ("Horse A", "NZ"), (None, "AU")],
        )


def test_hash_vector_is_deterministic_normalized_and_seeded():
    first = hash_vector("name\x1fHorse A", 32)
    second = hash_vector("name\x1fHorse A", 32)

    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.dtype("<f4")
    assert first.shape == (32,)
    np.testing.assert_allclose(np.linalg.norm(first), 1.0, rtol=1e-6)
    assert not np.array_equal(first, hash_vector("name\x1fHorse A", 32, "other"))


def test_populate_hash_vectors_writes_expected_blobs(tmp_path):
    database = tmp_path / "races.sqlite"
    make_database(database)

    rows, written = populate_hash_vectors(
        database,
        "race_runners",
        [("name", "name_vec"), ("country", "country_vec")],
        dimensions=4,
        batch_size=2,
    )

    assert (rows, written) == (3, 5)
    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT name_vec, country_vec FROM race_runners ORDER BY rowid"
        ).fetchall()
    assert stored[0][0] == stored[1][0]
    assert stored[2][0] is None
    assert len(stored[0][0]) == 4 * 4
