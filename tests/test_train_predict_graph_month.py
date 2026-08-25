import pandas as pd
import pytest

from train_predict_graph_month import select_validation_model, utc_timestamp


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
    }
    assert select_validation_model(metrics) == "graph_c"

    metrics["graph_b"]["mrr"] = 0.51
    assert select_validation_model(metrics) == "graph_b"
