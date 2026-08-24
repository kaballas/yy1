import math
import sqlite3

import numpy as np
import pandas as pd

from build_racing_graph_features import (
    GRAPH_FEATURE_NAMES,
    add_node_identities,
    adjacency_from_edges,
    assign_snapshots_strictly_before,
    biased_walk,
    calculate_graph_features,
    graph_edges,
    historical_rows,
    normalize_identity,
    output_columns,
    snapshot_boundaries,
    write_graph_feature_table,
)


def _raw_rows():
    return pd.DataFrame(
        {
            "source_rowid": [1, 2, 3],
            "race_id": [10, 10, 10],
            "runner_number": [1, 2, 3],
            "runner_name": [" Horse  One ", "Horse Two", "Debutant"],
            "runner_country": ["AU", "AU", "NZ"],
            "jockey": ["J One", "J Two", "J One"],
            "trainer": ["T One", "T Two", "T One"],
            "sire": ["S One", "S Two", "S One"],
            "dam": ["D One", "D Two", "D Three"],
            "competition_name": ["Darwin", "Darwin", "Darwin"],
            "competition_id": [6, 6, 6],
            "start_time_iso": ["2026-03-10T01:00:00Z"] * 3,
            "status": ["finished"] * 3,
            "runner_mask": [1, 1, 1],
            "source_betting_status": ["RESULTED"] * 3,
            "active_field_size": [3, 3, 3],
            "career_starts": [5, 2, 0],
        }
    )


def test_identity_normalization_types_nodes_and_applies_verified_venue_alias():
    assert normalize_identity("  JAMES\u00a0  McDONALD ") == "james mcdonald"

    result = add_node_identities(_raw_rows())

    assert result.loc[0, "graph_horse_node"] == "horse:au:horse one"
    assert result.loc[0, "graph_jockey_node"] == "jockey:j one"
    assert result.loc[0, "graph_venue_node"] == "venue:fannie bay"


def test_snapshot_assignment_is_strict_at_exact_boundary():
    times = pd.Series(
        pd.to_datetime(
            ["2026-01-01", "2026-01-15", "2026-02-01", "2026-02-02"], utc=True
        )
    )
    snapshots = list(pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True))

    assignments, unassigned = assign_snapshots_strictly_before(times, snapshots)

    assert unassigned.tolist() == [0]
    assert assignments[0].timestamp == snapshots[0]
    assert assignments[0].target_indices.tolist() == [1, 2]
    assert assignments[1].timestamp == snapshots[1]
    assert assignments[1].target_indices.tolist() == [3]


def test_automatic_schedule_precedes_target_exactly_on_month_boundary():
    times = pd.Series(pd.to_datetime(["2026-02-01T00:00:00Z"], utc=True))

    snapshots = snapshot_boundaries(times, "monthly")
    assignments, unassigned = assign_snapshots_strictly_before(times, snapshots)

    assert snapshots == [
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2026-02-01", tz="UTC"),
    ]
    assert unassigned.size == 0
    assert assignments[0].timestamp == snapshots[0]
    assert assignments[0].target_indices.tolist() == [0]


def test_graph_uses_only_finished_active_rows_strictly_before_snapshot():
    frame = add_node_identities(_raw_rows())
    frame["_start_time"] = pd.to_datetime(
        ["2026-01-01", "2026-02-01", "2026-01-15"], utc=True
    )
    frame.loc[2, "runner_mask"] = 0
    snapshot = pd.Timestamp("2026-02-01", tz="UTC")

    history = historical_rows(frame, snapshot)
    edges = graph_edges(history, snapshot, half_life_days=None)

    assert history["source_rowid"].tolist() == [1]
    assert (
        "horse:au:horse one",
        "trainer:t one",
    ) in edges
    assert not any("horse two" in node for edge in edges for node in edge)
    assert not any("debutant" in node for edge in edges for node in edge)
    assert ("sire:s one", "trainer:t one") not in edges


def test_pedigree_edges_are_static_while_interactions_accumulate_and_decay():
    raw = pd.concat([_raw_rows().iloc[[0]], _raw_rows().iloc[[0]]], ignore_index=True)
    raw["source_rowid"] = [1, 2]
    frame = add_node_identities(raw)
    frame["_start_time"] = pd.to_datetime(
        ["2026-01-01", "2026-01-16"], utc=True
    )
    snapshot = pd.Timestamp("2026-02-01", tz="UTC")

    edges = graph_edges(frame, snapshot, half_life_days=30.0)

    horse = "horse:au:horse one"
    trainer = "trainer:t one"
    sire = "sire:s one"
    dam = "dam:d one"
    expected_repeated = 2 ** (-31 / 30) + 2 ** (-16 / 30)
    assert math.isclose(edges[tuple(sorted((horse, trainer)))], expected_repeated)
    assert edges[tuple(sorted((horse, sire)))] == 1.0
    assert edges[tuple(sorted((horse, dam)))] == 1.0


def test_biased_walk_is_reproducible_and_follows_edges():
    adjacency = adjacency_from_edges({("a", "b"): 1.0, ("b", "c"): 1.0})
    first = biased_walk(adjacency, "a", 8, 1.0, 1.0, np.random.default_rng(7))
    second = biased_walk(adjacency, "a", 8, 1.0, 1.0, np.random.default_rng(7))

    assert first == second
    assert len(first) == 8
    assert all(right in adjacency[left] for left, right in zip(first, first[1:]))


def test_similarity_availability_and_race_relative_features_support_debutant():
    target = add_node_identities(_raw_rows())
    embeddings = {
        "horse:au:horse one": np.array([1.0, 0.0]),
        "horse:au:horse two": np.array([0.0, 1.0]),
        "jockey:j one": np.array([1.0, 0.0]),
        "jockey:j two": np.array([1.0, 0.0]),
        "trainer:t one": np.array([1.0, 0.0]),
        "trainer:t two": np.array([0.0, 1.0]),
        "sire:s one": np.array([1.0, 0.0]),
        "sire:s two": np.array([0.0, 1.0]),
        "dam:d one": np.array([1.0, 0.0]),
        "dam:d two": np.array([0.0, 1.0]),
        "venue:fannie bay": np.array([1.0, 0.0]),
    }

    result = calculate_graph_features(target, embeddings)

    assert result.loc[0, "graph_horse_trainer_similarity"] == 1.0
    assert result.loc[1, "graph_horse_trainer_similarity"] == 1.0
    assert math.isnan(result.loc[2, "graph_horse_trainer_similarity"])
    assert result.loc[2, "graph_horse_embedding_available"] == 0.0
    assert result.loc[2, "graph_sire_embedding_available"] == 1.0
    assert result.loc[2, "graph_jockey_trainer_similarity"] == 1.0
    assert result.loc[0, "graph_horse_trainer_similarity_rank_in_race"] == 1.0
    assert result.loc[1, "graph_horse_trainer_similarity_rank_in_race"] == 1.0
    assert math.isnan(
        result.loc[2, "graph_horse_trainer_similarity_rank_in_race"]
    )


def test_graph_feature_table_is_separate_and_preserves_nulls(tmp_path):
    database = tmp_path / "races.sqlite"
    target = add_node_identities(_raw_rows().iloc[:1])
    embeddings = {
        "horse:au:horse one": np.array([1.0, 0.0]),
        "trainer:t one": np.array([1.0, 0.0]),
    }
    features = calculate_graph_features(target, embeddings)
    metadata = target.loc[
        :, ["source_rowid", "race_id", "runner_number", "runner_name", "runner_country"]
    ].copy()
    metadata["snapshot_date"] = "2026-03-01T00:00:00+00:00"
    output = pd.concat([metadata, features], axis=1).loc[:, output_columns()]

    write_graph_feature_table(database, "graph_features", output, replace=False)

    with sqlite3.connect(database) as connection:
        stored = connection.execute(
            "SELECT source_rowid, graph_horse_trainer_similarity, "
            "graph_horse_jockey_similarity FROM graph_features"
        ).fetchone()
        columns = connection.execute("PRAGMA table_info(graph_features)").fetchall()
    assert stored == (1, 1.0, None)
    assert len(columns) == len(output_columns())
    assert set(GRAPH_FEATURE_NAMES) <= {row[1] for row in columns}
