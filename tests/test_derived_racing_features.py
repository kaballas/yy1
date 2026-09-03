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


def test_recent_market_and_weight_trajectory_features():
    frame = _frame()
    frame["weight_kg"] = 54.0
    frame["recent_1_starting_price"] = 5.0
    frame["recent_2_starting_price"] = 10.0
    frame["recent_1_weight_kg"] = 57.0
    frame["recent_2_weight_kg"] = 59.0

    result = derive_racing_features(frame)

    assert result.loc[0, "recent_1_starting_price_log"] == np.float32(np.log(5.0))
    assert result.loc[0, "recent_1_implied_probability"] == np.float32(0.2)
    assert result.loc[0, "historical_market_expectation_change"] == np.float32(
        np.log(0.5)
    )
    assert result.loc[0, "recent_2_weight_change_from_current"] == -5.0
    assert result.loc[0, "recent_2_weight_change_pct_from_current"] == np.float32(
        -5.0 / 59.0
    )
    assert result.loc[0, "recent_1_vs_recent_2_weight_change"] == -2.0
    assert result.loc[0, "recent_1_vs_recent_2_weight_change_pct"] == np.float32(
        -2.0 / 59.0
    )


def test_recent_market_and_weight_features_reject_invalid_inputs():
    frame = _frame()
    frame["weight_kg"] = 54.0
    frame["recent_1_starting_price"] = 0.0
    frame["recent_2_starting_price"] = -2.0
    frame["recent_1_weight_kg"] = 57.0
    frame["recent_2_weight_kg"] = 0.0

    result = derive_racing_features(frame)

    names = [
        "recent_1_starting_price_log",
        "recent_1_implied_probability",
        "historical_market_expectation_change",
        "recent_2_weight_change_from_current",
        "recent_2_weight_change_pct_from_current",
        "recent_1_vs_recent_2_weight_change",
        "recent_1_vs_recent_2_weight_change_pct",
    ]
    assert result.loc[0, names].isna().all()


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
