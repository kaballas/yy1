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
    first_order_samplers,
    graph_edges,
    historical_rows,
    normalize_identity,
    output_columns,
    repeated_edge_weight_statistics,
    snapshot_boundaries,
    write_graph_feature_table,
)


def _raw_rows():
    frame = pd.DataFrame(
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
            "country": ["Australia", "Australia", "Australia"],
            "competition_id": [6, 6, 6],
            "start_time_iso": ["2026-03-10T01:00:00Z"] * 3,
            "status": ["finished"] * 3,
            "runner_mask": [1, 1, 1],
            "source_betting_status": ["RESULTED"] * 3,
            "active_field_size": [3, 3, 3],
            "career_starts": [5, 2, 0],
            "distance_m": [1200, 1400, 1200],
            "track_status": ["Good (4)", "Soft (5)", "Good"],
        }
    )
    for slot in range(1, 7):
        frame[f"recent_{slot}_date"] = None
        frame[f"recent_{slot}_class"] = None
        frame[f"recent_{slot}_jockey"] = None
        frame[f"recent_{slot}_track_name"] = None
        frame[f"recent_{slot}_distance_m"] = None
        frame[f"recent_{slot}_place"] = None
        frame[f"recent_{slot}_track_status"] = None
    return frame


def test_identity_normalization_types_nodes_and_applies_verified_venue_alias():
    assert normalize_identity("  JAMES\u00a0  McDONALD ") == "james mcdonald"

    result = add_node_identities(_raw_rows())

    assert result.loc[0, "graph_horse_node"] == "horse:au:horse one"
    assert result.loc[0, "graph_jockey_node"] == "jockey:j one"
    assert result.loc[0, "graph_venue_node"] == "venue:fannie bay"
    assert result.loc[0, "graph_distance_node"] == "distance:1200m"
    assert result.loc[0, "graph_track_status_node"] == "track_status:good"


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


def test_repeated_edge_weight_modes_do_not_transform_pedigree():
    raw = pd.concat([_raw_rows().iloc[[0]], _raw_rows().iloc[[0]]], ignore_index=True)
    raw["source_rowid"] = [1, 2]
    frame = add_node_identities(raw)
    frame["_start_time"] = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    snapshot = pd.Timestamp("2026-02-01", tz="UTC")
    horse_trainer = tuple(sorted(("horse:au:horse one", "trainer:t one")))
    horse_sire = tuple(sorted(("horse:au:horse one", "sire:s one")))

    raw_edges = graph_edges(frame, snapshot, None, "raw")
    log_edges = graph_edges(frame, snapshot, None, "log")
    binary_edges = graph_edges(frame, snapshot, None, "binary")

    assert raw_edges[horse_trainer] == 2.0
    assert math.isclose(log_edges[horse_trainer], math.log1p(2.0))
    assert binary_edges[horse_trainer] == 1.0
    assert raw_edges[horse_sire] == log_edges[horse_sire] == 1.0
    assert binary_edges[horse_sire] == 1.0

    statistics = repeated_edge_weight_statistics(raw_edges)
    assert statistics["count"] == 4
    assert statistics["min"] == statistics["max"] == 2.0


def test_recent_run_edges_are_deduplicated_and_future_form_is_rejected():
    raw = pd.concat([_raw_rows().iloc[[0]], _raw_rows().iloc[[0]]], ignore_index=True)
    raw["source_rowid"] = [1, 2]
    for row in raw.index:
        raw.loc[row, "recent_1_date"] = "2025-12-20"
        raw.loc[row, "recent_1_class"] = "BM64"
        raw.loc[row, "recent_1_jockey"] = "J Previous"
        raw.loc[row, "recent_1_track_name"] = "Belmont"
        raw.loc[row, "recent_1_distance_m"] = 1200
        raw.loc[row, "recent_1_place"] = 2
        raw.loc[row, "recent_1_track_status"] = "Soft"
        raw.loc[row, "recent_2_date"] = "2026-03-01"
        raw.loc[row, "recent_2_track_name"] = "Future Track"
        raw.loc[row, "recent_2_distance_m"] = 1400
    frame = add_node_identities(raw)
    frame["_start_time"] = pd.to_datetime(["2026-01-10", "2026-01-15"], utc=True)

    edges = graph_edges(
        frame, pd.Timestamp("2026-02-01", tz="UTC"), None, "raw"
    )

    run_nodes = {
        node for edge in edges for node in edge if node.startswith("run:")
    }
    assert len(run_nodes) == 3
    run_node = next(node for node in run_nodes if "|2025-12-20|" in node)
    run_edges = [edge for edge in edges if run_node in edge]
    assert len(run_edges) == 7
    assert all(edges[edge] == 1.0 for edge in run_edges)
    assert not any("future track" in node for edge in edges for node in edge)


def test_finished_current_run_is_available_to_a_later_target():
    history = add_node_identities(_raw_rows().iloc[[0]])
    history["_start_time"] = pd.to_datetime(["2026-01-10"], utc=True)
    target_raw = _raw_rows().iloc[[0]].copy()
    target_raw.loc[:, "recent_1_date"] = "10/01/2026"
    target_raw.loc[:, "recent_1_track_name"] = "Darwin"
    target_raw.loc[:, "recent_1_distance_m"] = 1200
    target = add_node_identities(target_raw)

    edges = graph_edges(
        history, pd.Timestamp("2026-02-01", tz="UTC"), None, "raw"
    )
    target_latest_run = target.iloc[0]["graph_recent_1_run_node"]

    assert target_latest_run in {node for edge in edges for node in edge}
    assert any(
        target_latest_run in edge and "horse:au:horse one" in edge
        for edge in edges
    )


def test_current_run_uses_local_race_date_for_new_zealand():
    raw = _raw_rows().iloc[[0]].copy()
    raw.loc[:, "country"] = "New Zealand"
    raw.loc[:, "competition_name"] = "Riccarton"
    history = add_node_identities(raw)
    history["_start_time"] = pd.to_datetime(["2026-01-09T23:30:00Z"], utc=True)
    target_raw = raw.copy()
    target_raw.loc[:, "recent_1_date"] = "10/01/2026"
    target_raw.loc[:, "recent_1_track_name"] = "Riccarton"
    target_raw.loc[:, "recent_1_distance_m"] = 1200
    target = add_node_identities(target_raw)

    edges = graph_edges(
        history, pd.Timestamp("2026-02-01", tz="UTC"), None, "raw"
    )

    assert target.iloc[0]["graph_recent_1_run_node"] in {
        node for edge in edges for node in edge
    }


def test_biased_walk_is_reproducible_and_follows_edges():
    adjacency = adjacency_from_edges({("a", "b"): 1.0, ("b", "c"): 1.0})
    first = biased_walk(adjacency, "a", 8, 1.0, 1.0, np.random.default_rng(7))
    second = biased_walk(adjacency, "a", 8, 1.0, 1.0, np.random.default_rng(7))

    assert first == second
    assert len(first) == 8
    assert all(right in adjacency[left] for left, right in zip(first, first[1:]))


def test_p1_q1_cached_sampler_preserves_weighted_transitions():
    adjacency = adjacency_from_edges({("a", "b"): 1.0, ("a", "c"): 9.0})
    samplers = first_order_samplers(adjacency)
    rng = np.random.default_rng(42)

    destinations = [
        biased_walk(adjacency, "a", 2, 1.0, 1.0, rng, samplers)[1]
        for _ in range(2000)
    ]

    assert destinations.count("c") / len(destinations) > 0.85


def test_seeded_walk_is_independent_of_neighbor_insertion_order():
    forward = adjacency_from_edges({
        ("center", f"neighbor-{index}"): 1.0 for index in range(20)
    })
    reverse = adjacency_from_edges(dict(reversed([
        (("center", f"neighbor-{index}"), 1.0) for index in range(20)
    ])))

    cached_forward = biased_walk(
        forward, "center", 8, 1.0, 1.0, np.random.default_rng(42),
        first_order_samplers(forward),
    )
    cached_reverse = biased_walk(
        reverse, "center", 8, 1.0, 1.0, np.random.default_rng(42),
        first_order_samplers(reverse),
    )
    biased_forward = biased_walk(
        forward, "center", 8, 0.5, 2.0, np.random.default_rng(42)
    )
    biased_reverse = biased_walk(
        reverse, "center", 8, 0.5, 2.0, np.random.default_rng(42)
    )

    assert cached_forward == cached_reverse
    assert biased_forward == biased_reverse


def test_similarity_availability_and_race_relative_features_support_debutant():
    raw = _raw_rows()
    raw.loc[0, "recent_1_date"] = "2026-02-01"
    raw.loc[0, "recent_1_class"] = "BM64"
    raw.loc[0, "recent_1_jockey"] = "J Previous"
    raw.loc[0, "recent_1_track_name"] = "Belmont"
    raw.loc[0, "recent_1_distance_m"] = 1200
    raw.loc[0, "recent_1_place"] = 2
    raw.loc[0, "recent_1_track_status"] = "Soft"
    target = add_node_identities(raw)
    recent_run = target.loc[0, "graph_recent_1_run_node"]
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
        "distance:1200m": np.array([1.0, 0.0]),
        "distance:1400m": np.array([0.0, 1.0]),
        "track_status:good": np.array([1.0, 0.0]),
        "track_status:soft": np.array([0.0, 1.0]),
        "jockey:j previous": np.array([1.0, 0.0]),
        "venue:belmont park": np.array([1.0, 0.0]),
        "class:bm64": np.array([1.0, 0.0]),
        "place:2": np.array([1.0, 0.0]),
        recent_run: np.array([1.0, 0.0]),
    }

    result = calculate_graph_features(target, embeddings)

    assert result.loc[0, "graph_horse_trainer_similarity"] == 1.0
    assert result.loc[1, "graph_horse_trainer_similarity"] == 1.0
    assert math.isnan(result.loc[2, "graph_horse_trainer_similarity"])
    assert result.loc[2, "graph_horse_embedding_available"] == 0.0
    assert result.loc[2, "graph_sire_embedding_available"] == 1.0
    assert result.loc[2, "graph_jockey_trainer_similarity"] == 1.0
    assert result.loc[0, "graph_horse_distance_similarity"] == 1.0
    assert result.loc[0, "graph_horse_recent_run_similarity"] == 1.0
    assert result.loc[0, "graph_distance_recent_distance_similarity"] == 1.0
    assert result.loc[0, "graph_recent_run_embedding_available"] == 1.0
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
