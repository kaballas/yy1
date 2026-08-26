import pandas as pd
import pytest

from train_predict_graph_month import (
    RACE_DETAIL_COLUMNS,
    exact_xgboost_input_frame,
    expand_validation_models,
    print_target_race_details,
    select_validation_model,
    utc_timestamp,
    write_race_input_audit,
)


def test_naive_month_boundaries_use_requested_local_timezone():
    parsed = utc_timestamp("2026-07-01", "train-start", "Australia/Sydney")

    assert parsed == pd.Timestamp("2026-06-30T14:00:00Z")


def test_explicit_utc_boundary_is_not_reinterpreted_as_local_time():
    parsed = utc_timestamp("2026-07-01T00:00:00Z", "train-start", "Australia/Sydney")

    assert parsed == pd.Timestamp("2026-07-01T00:00:00Z")


def test_invalid_timezone_is_rejected_for_naive_boundary():
    with pytest.raises(ValueError, match="Invalid --timezone"):
        utc_timestamp("2026-07-01", "train-start", "Not/A_Timezone")


def test_validation_model_selection_uses_top1_then_mrr_then_simplicity():
    metrics = {
        "graph_a": {"top1_hit_rate": 0.25, "mrr": 0.48},
        "graph_b": {"top1_hit_rate": 0.27, "mrr": 0.49},
        "graph_c": {"top1_hit_rate": 0.27, "mrr": 0.51},
        "graph_d": {"top1_hit_rate": 0.26, "mrr": 0.52},
        "graph_e": {"top1_hit_rate": 0.26, "mrr": 0.50},
    }
    assert select_validation_model(metrics) == "graph_c"

    metrics["graph_b"]["mrr"] = 0.51
    assert select_validation_model(metrics) == "graph_b"


def test_partial_model_request_expands_to_required_abcd_selection_set():
    assert expand_validation_models(["graph_c"]) == [
        "graph_a", "graph_b", "graph_c", "graph_d", "graph_e",
    ]
    assert expand_validation_models(["graph_only"]) == ["graph_only"]


def test_race_details_print_and_complete_input_audit(tmp_path, capsys):
    values = {
        "source_rowid": 1,
        "race_id": 10,
        "start_time_iso": "2026-08-25T06:30:00Z",
        "snapshot_date": "2026-08-01T00:00:00Z",
        "competition_id": 6,
        "competition_name": "Venue",
        "race_number": 3,
        "race_name": "Test Race",
        "status": "no_result",
        "source_betting_status": "PRICED",
        "active_field_size": 1,
        "runner_mask": 0,
        "runner_number": 4,
        "runner_name": "Horse",
        "graph_horse_distance_similarity": 0.75,
    }
    values.update({column: None for column in RACE_DETAIL_COLUMNS})
    values.update({
        "runner_country": "AU", "jockey": "Jockey", "trainer": "Trainer",
        "sire": "Sire", "dam": "Dam", "distance_m": 1200,
        "track_status": "Good (4)", "recent_1_place": 2,
    })
    target = pd.DataFrame([values])

    print_target_race_details(target)
    path = write_race_input_audit(
        target,
        {"graph_a": ["distance_m"], "graph_b": [
            "distance_m", "graph_horse_distance_similarity",
        ]},
        tmp_path,
    )

    printed = capsys.readouterr().out
    stored = pd.read_csv(path)
    assert "race_id=10" in printed
    assert "recent_1:" in printed
    assert "place=2" in printed
    assert stored.loc[0, "graph_horse_distance_similarity"] == 0.75
    assert set(RACE_DETAIL_COLUMNS) <= set(stored.columns)


def test_exact_xgboost_audit_matches_numeric_matrix_order_and_missing_values():
    frame = pd.DataFrame(
        {
            "race_id": [1, 1],
            "runner_number": [1, 2],
            "runner_name": ["A", "B"],
            "is_winner": [1, 0],
            "feature_b": ["2.5", "bad"],
            "feature_a": [float("inf"), 4],
        }
    )

    audit = exact_xgboost_input_frame(frame, ["feature_a", "feature_b"])

    assert audit.columns.tolist() == [
        "race_id", "runner_number", "runner_name", "is_winner",
        "feature_a", "feature_b",
    ]
    assert pd.isna(audit.loc[0, "feature_a"])
    assert pd.isna(audit.loc[1, "feature_b"])
    assert audit.loc[0, "feature_b"] == 2.5
