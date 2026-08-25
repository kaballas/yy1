import pandas as pd
import pytest

from train_predict_graph_month import utc_timestamp


def test_naive_month_boundaries_use_requested_local_timezone():
    parsed = utc_timestamp("2026-07-01", "train-start", "Australia/Sydney")

    assert parsed == pd.Timestamp("2026-06-30T14:00:00Z")


def test_explicit_utc_boundary_is_not_reinterpreted_as_local_time():
    parsed = utc_timestamp("2026-07-01T00:00:00Z", "train-start", "Australia/Sydney")

    assert parsed == pd.Timestamp("2026-07-01T00:00:00Z")


def test_invalid_timezone_is_rejected_for_naive_boundary():
    with pytest.raises(ValueError, match="Invalid --timezone"):
        utc_timestamp("2026-07-01", "train-start", "Not/A_Timezone")
