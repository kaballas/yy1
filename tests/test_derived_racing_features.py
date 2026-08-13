import sqlite3

import numpy as np
import pandas as pd

from audit_market_residual_features import fit_pairwise_alpha, is_market_derived
from src.derived_racing_features import derive_racing_features


def _frame():
    row = {"distance_m": 1200.0}
    for run in range(1, 7):
        row.update({
            f"recent_{run}_place": float(run),
            f"recent_{run}_margin": float(run - 1),
            f"recent_{run}_total_runners": 10.0,
            f"recent_{run}_barrier": float(run),
            f"recent_{run}_starting_price": 5.0 + run,
            f"recent_{run}_distance_m": 1200.0,
        })
    return pd.DataFrame([row])


def test_barrier_percentile_is_recency_weighted_and_field_normalized():
    result = derive_racing_features(_frame())
    weights = 1.0 / np.arange(1, 7)
    expected = np.average(np.arange(6) / 9.0, weights=weights)

    assert result.loc[0, "form_barrier_percentile_weighted_6"] == np.float32(
        expected
    )


def test_derived_features_use_only_input_history_and_handle_missing():
    frame = _frame()
    frame.loc[0, "recent_1_barrier"] = np.nan
    result = derive_racing_features(frame)
    expected = np.average(
        np.arange(1, 6) / 9.0, weights=(1.0 / np.arange(2, 7))
    )

    assert result.loc[0, "form_barrier_percentile_weighted_6"] == np.float32(
        expected
    )


def test_pairwise_audit_fits_adjustment_in_evidence_direction():
    market = np.zeros(4)
    feature = np.asarray([1.0, 0.5, -0.5, -1.0])

    alpha = fit_pairwise_alpha(
        market, feature, np.asarray([0, 1]), np.asarray([2, 3]), 0.05, 1.0
    )

    assert 0 < alpha <= 1.0


def test_non_market_audit_excludes_current_and_historical_prices():
    assert is_market_derived("fluc2")
    assert is_market_derived("recent_3_starting_price")
    assert is_market_derived("recent_market_edge_weighted")
    assert not is_market_derived("recent_best_margin")
    assert not is_market_derived("form_barrier_percentile_weighted_6")
