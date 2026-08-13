"""Leakage-safe numeric features built from the six stored pre-race starts.

All matrices are ordered newest to oldest (``recent_1`` to ``recent_6``).
Missing observations are ignored and weights are renormalised per runner.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


DERIVED_FEATURE_NAMES = (
    # Existing persisted feature retained for backwards compatibility.
    "form_barrier_percentile_weighted_6",
    # Finish percentile (one is best).
    "recent_finish_percentile_last1", "recent_finish_percentile_avg_3",
    "recent_finish_percentile_avg_6", "recent_finish_percentile_weighted_3",
    "recent_finish_percentile_weighted_6", "recent_finish_percentile_best_3",
    "recent_finish_percentile_best_6", "recent_finish_percentile_std_6",
    "recent_finish_percentile_slope_3", "recent_finish_percentile_slope_6",
    "recent_finish_percentile_last_minus_avg3",
    "recent_finish_percentile_avg3_minus_avg6",
    # Raw margins (lower is better) and bounded margin quality (higher is better).
    "recent_margin_avg_3", "recent_margin_avg_6", "recent_margin_weighted_3",
    "recent_margin_weighted_6", "recent_margin_best_3", "recent_margin_best_6",
    "recent_margin_std_6", "recent_margin_slope_3", "recent_margin_slope_6",
    "recent_margin_last_minus_avg3", "recent_margin_quality_last1",
    "recent_margin_quality_avg_3", "recent_margin_quality_avg_6",
    "recent_margin_quality_weighted_3", "recent_margin_quality_weighted_6",
    "recent_margin_quality_best_3", "recent_margin_quality_slope_3",
    # Historical market.
    "historical_market_overperformance_last1",
    "historical_market_overperformance_avg_3",
    "historical_market_overperformance_avg_6",
    "historical_market_overperformance_weighted_3",
    "historical_market_overperformance_weighted_6",
    "historical_market_overperformance_best_3",
    "historical_market_overperformance_slope_3",
    "historical_implied_probability_weighted_3",
    "historical_implied_probability_weighted_6",
    # Distance suitability and transitions.
    "distance_change_last_run", "abs_distance_change_last_run",
    "distance_change_pct_last_run", "distance_minus_recent_avg",
    "distance_minus_recent_weighted_avg", "abs_distance_minus_recent_weighted_avg",
    "similar_distance_runs", "similar_distance_finish_percentile_avg",
    "similar_distance_finish_percentile_weighted",
    "similar_distance_margin_quality_avg", "similar_distance_margin_quality_weighted",
    "similar_distance_top3_rate", "distance_step_up_m", "distance_step_down_m",
    "distance_step_up_pct", "distance_step_down_pct",
    "historical_step_up_finish_percentile", "historical_step_up_observations",
    "historical_step_down_finish_percentile", "historical_step_down_observations",
    # Barrier and weight context.
    "recent_barrier_percentile_last1", "recent_barrier_percentile_avg_3",
    "recent_barrier_percentile_weighted_6", "recent_barrier_percentile_std_6",
    "current_barrier_percentile", "current_minus_recent_barrier_percentile",
    "current_weight_minus_last", "current_weight_minus_recent_avg",
    "current_weight_minus_recent_weighted_avg", "current_weight_change_pct",
    "best_finish_percentile_at_equal_or_higher_weight",
    "best_margin_quality_at_equal_or_higher_weight",
    # Ceiling, consistency, and best-run recency.
    "form_ceiling_finish_percentile_6", "form_average_finish_percentile_6",
    "form_consistency_finish_percentile_6", "form_ceiling_minus_average",
    "margin_quality_ceiling_6", "margin_quality_average_6",
    "margin_quality_consistency_6", "best_run_recency_index",
    "second_best_run_recency_index", "best_run_is_last_start",
    "best_run_within_last_2", "best_run_within_last_3",
    # Available career comparisons.
    "recent3_vs_career_finish_percentile", "recent6_vs_career_finish_percentile",
    "recent3_vs_career_place_rate", "recent6_vs_career_place_rate",
    "recent_form_vs_career_form",
    # Track and going suitability from the six stored historical starts.
    "same_track_runs", "same_track_finish_percentile", "same_track_margin_quality",
    "same_track_top3_rate", "same_track_distance_runs",
    "same_track_distance_finish_percentile", "same_track_distance_margin_quality",
    "same_track_distance_top3_rate", "condition_runs",
    "condition_win_rate_smoothed", "condition_top3_rate_smoothed",
    "condition_finish_percentile", "condition_margin_quality",
)


def _matrix(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return np.column_stack([
        pd.to_numeric(frame[f"recent_{run}_{stem}"], errors="coerce").to_numpy(float)
        for run in range(1, 7)
    ])


def _optional_matrix(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return np.column_stack([
        pd.to_numeric(frame.get(f"recent_{run}_{stem}", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
        for run in range(1, 7)
    ])


def _weighted_mean(values: np.ndarray, count: int) -> np.ndarray:
    """Weighted mean using [N, ..., 1], with recent_1 receiving weight N."""
    selected = values[:, :count]
    weights = np.arange(count, 0, -1, dtype=float)
    valid = np.isfinite(selected)
    denominator = (valid * weights).sum(axis=1)
    return np.divide(np.where(valid, selected, 0).dot(weights), denominator,
                     out=np.full(len(values), np.nan), where=denominator > 0)


def _mean(values: np.ndarray, count: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values[:, :count], axis=1)


def _std(values: np.ndarray, count: int) -> np.ndarray:
    selected = values[:, :count]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        answer = np.nanstd(selected, axis=1, ddof=1)
    answer[np.isfinite(selected).sum(axis=1) < 2] = np.nan
    return answer


def _best(values: np.ndarray, count: int, higher: bool = True) -> np.ndarray:
    selected = values[:, :count]
    valid = np.isfinite(selected)
    fill = -np.inf if higher else np.inf
    answer = (np.max if higher else np.min)(np.where(valid, selected, fill), axis=1)
    answer[~valid.any(axis=1)] = np.nan
    return answer


def _slope(values: np.ndarray, count: int, higher_is_better: bool = True) -> np.ndarray:
    """OLS slope in chronological order, so positive always means improving."""
    selected = values[:, :count][:, ::-1]  # oldest -> newest
    answer = np.full(len(values), np.nan)
    for row, observations in enumerate(selected):
        valid = np.isfinite(observations)
        if valid.sum() >= 2:
            answer[row] = np.polyfit(np.arange(count)[valid], observations[valid], 1)[0]
    return answer if higher_is_better else -answer


def _finish_percentile(place: np.ndarray, field: np.ndarray) -> np.ndarray:
    valid = np.isfinite(place) & np.isfinite(field) & (place >= 1) & (field > 0)
    result = np.full(place.shape, np.nan)
    result[valid] = 1 - (place[valid] - 1) / np.maximum(field[valid] - 1, 1)
    return np.clip(result, 0, 1)


def derive_racing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic features without reading target/future outcomes."""
    required = {"distance_m"}
    required.update(f"recent_{run}_{stem}" for run in range(1, 7) for stem in (
        "place", "margin", "total_runners", "barrier", "starting_price",
        "distance_m"))
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError("Cannot derive racing features; missing: " + ", ".join(missing))

    place, margin, field = (_matrix(frame, x) for x in ("place", "margin", "total_runners"))
    barrier, price, distance = (_matrix(frame, x) for x in
                                ("barrier", "starting_price", "distance_m"))
    weight = _optional_matrix(frame, "weight_kg")
    finish = _finish_percentile(place, field)
    margin = np.where(np.isfinite(margin), margin, np.nan)
    margin_quality = np.exp(-np.maximum(margin, 0) / 5)
    implied = np.divide(1, price, out=np.full(price.shape, np.nan),
                        where=np.isfinite(price) & (price > 0))
    # No historical field-level price table exists: this transparent proxy compares
    # realised finish percentile with the horse's own historical implied probability.
    market_over = finish - implied
    valid_barrier = (np.isfinite(barrier) & np.isfinite(field) & (barrier >= 1)
                     & (field > 0) & (barrier <= field))
    barrier_pct = np.full(barrier.shape, np.nan)
    barrier_pct[valid_barrier] = ((barrier[valid_barrier] - 1) /
                                  np.maximum(field[valid_barrier] - 1, 1))

    current_distance = pd.to_numeric(frame["distance_m"], errors="coerce").to_numpy(float)
    valid_current_distance = np.isfinite(current_distance) & (current_distance > 0)
    distance = np.where(np.isfinite(distance) & (distance > 0), distance, np.nan)
    delta = current_distance - distance[:, 0]
    delta[~valid_current_distance | ~np.isfinite(distance[:, 0])] = np.nan
    recent_distance_avg = _mean(distance, 6)
    recent_distance_weighted = _weighted_mean(distance, 6)
    similar = (np.abs(distance - current_distance[:, None]) /
               current_distance[:, None] <= .10) & valid_current_distance[:, None]
    similar_finish, similar_margin = np.where(similar, finish, np.nan), np.where(similar, margin_quality, np.nan)

    # Prior transition performance uses only transitions among stored historical starts.
    prior_delta = distance[:, :-1] - distance[:, 1:]
    transition_finish = finish[:, :-1]
    step_up_finish = np.where(prior_delta > 0, transition_finish, np.nan)
    step_down_finish = np.where(prior_delta < 0, transition_finish, np.nan)

    current_weight = pd.to_numeric(frame.get("weight_kg", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    valid_weight = np.isfinite(weight) & (weight > 0)
    weight = np.where(valid_weight, weight, np.nan)
    equal_higher = valid_weight & np.isfinite(current_weight[:, None]) & (weight >= current_weight[:, None])

    result = pd.DataFrame(index=frame.index)
    legacy_weights = 1 / np.arange(1, 7, dtype=float)
    legacy_valid = np.isfinite(barrier_pct)
    result["form_barrier_percentile_weighted_6"] = np.divide(
        np.where(legacy_valid, barrier_pct, 0).dot(legacy_weights),
        (legacy_valid * legacy_weights).sum(axis=1), out=np.full(len(frame), np.nan),
        where=(legacy_valid * legacy_weights).sum(axis=1) > 0)
    result["recent_finish_percentile_last1"] = finish[:, 0]
    for n in (3, 6):
        result[f"recent_finish_percentile_avg_{n}"] = _mean(finish, n)
        result[f"recent_finish_percentile_weighted_{n}"] = _weighted_mean(finish, n)
        result[f"recent_finish_percentile_best_{n}"] = _best(finish, n)
    result["recent_finish_percentile_std_6"] = _std(finish, 6)
    result["recent_finish_percentile_slope_3"] = _slope(finish, 3)
    result["recent_finish_percentile_slope_6"] = _slope(finish, 6)
    result["recent_finish_percentile_last_minus_avg3"] = finish[:, 0] - _mean(finish, 3)
    result["recent_finish_percentile_avg3_minus_avg6"] = _mean(finish, 3) - _mean(finish, 6)
    for n in (3, 6):
        result[f"recent_margin_avg_{n}"] = _mean(margin, n)
        result[f"recent_margin_weighted_{n}"] = _weighted_mean(margin, n)
        result[f"recent_margin_best_{n}"] = _best(margin, n, False)
    result["recent_margin_std_6"] = _std(margin, 6)
    result["recent_margin_slope_3"] = _slope(margin, 3, False)
    result["recent_margin_slope_6"] = _slope(margin, 6, False)
    result["recent_margin_last_minus_avg3"] = margin[:, 0] - _mean(margin, 3)
    result["recent_margin_quality_last1"] = margin_quality[:, 0]
    for n in (3, 6):
        result[f"recent_margin_quality_avg_{n}"] = _mean(margin_quality, n)
        result[f"recent_margin_quality_weighted_{n}"] = _weighted_mean(margin_quality, n)
    result["recent_margin_quality_best_3"] = _best(margin_quality, 3)
    result["recent_margin_quality_slope_3"] = _slope(margin_quality, 3)
    result["historical_market_overperformance_last1"] = market_over[:, 0]
    for n in (3, 6):
        result[f"historical_market_overperformance_avg_{n}"] = _mean(market_over, n)
        result[f"historical_market_overperformance_weighted_{n}"] = _weighted_mean(market_over, n)
        result[f"historical_implied_probability_weighted_{n}"] = _weighted_mean(implied, n)
    result["historical_market_overperformance_best_3"] = _best(market_over, 3)
    result["historical_market_overperformance_slope_3"] = _slope(market_over, 3)
    result["distance_change_last_run"] = delta
    result["abs_distance_change_last_run"] = np.abs(delta)
    result["distance_change_pct_last_run"] = np.divide(delta, distance[:, 0], out=np.full(len(frame), np.nan), where=np.isfinite(distance[:, 0]) & (distance[:, 0] > 0))
    result["distance_minus_recent_avg"] = current_distance - recent_distance_avg
    result["distance_minus_recent_weighted_avg"] = current_distance - recent_distance_weighted
    result["abs_distance_minus_recent_weighted_avg"] = np.abs(current_distance - recent_distance_weighted)
    result["similar_distance_runs"] = np.where(valid_current_distance, similar.sum(axis=1), np.nan)
    result["similar_distance_finish_percentile_avg"] = _mean(similar_finish, 6)
    result["similar_distance_finish_percentile_weighted"] = _weighted_mean(similar_finish, 6)
    result["similar_distance_margin_quality_avg"] = _mean(similar_margin, 6)
    result["similar_distance_margin_quality_weighted"] = _weighted_mean(similar_margin, 6)
    result["similar_distance_top3_rate"] = _mean(np.where(similar, np.where(np.isfinite(place), (place <= 3).astype(float), np.nan), np.nan), 6)
    result["distance_step_up_m"] = np.maximum(delta, 0)
    result["distance_step_down_m"] = np.maximum(-delta, 0)
    result["distance_step_up_pct"] = np.maximum(result["distance_change_pct_last_run"], 0)
    result["distance_step_down_pct"] = np.maximum(-result["distance_change_pct_last_run"], 0)
    result["historical_step_up_finish_percentile"] = _mean(step_up_finish, 5)
    result["historical_step_up_observations"] = np.isfinite(step_up_finish).sum(axis=1)
    result["historical_step_down_finish_percentile"] = _mean(step_down_finish, 5)
    result["historical_step_down_observations"] = np.isfinite(step_down_finish).sum(axis=1)
    result["recent_barrier_percentile_last1"] = barrier_pct[:, 0]
    result["recent_barrier_percentile_avg_3"] = _mean(barrier_pct, 3)
    result["recent_barrier_percentile_weighted_6"] = _weighted_mean(barrier_pct, 6)
    result["recent_barrier_percentile_std_6"] = _std(barrier_pct, 6)
    current_field = pd.to_numeric(frame.get("active_field_size", pd.Series(np.nan, index=frame.index)), errors="coerce").fillna(pd.to_numeric(frame.get("field_size", pd.Series(np.nan, index=frame.index)), errors="coerce")).to_numpy(float)
    draw = pd.to_numeric(frame.get("draw_number", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    current_barrier_pct = np.divide(draw - 1, np.maximum(current_field - 1, 1), out=np.full(len(frame), np.nan), where=np.isfinite(draw) & (draw >= 1) & np.isfinite(current_field) & (current_field > 0) & (draw <= current_field))
    result["current_barrier_percentile"] = current_barrier_pct
    result["current_minus_recent_barrier_percentile"] = current_barrier_pct - _weighted_mean(barrier_pct, 6)
    result["current_weight_minus_last"] = current_weight - weight[:, 0]
    result["current_weight_minus_recent_avg"] = current_weight - _mean(weight, 6)
    result["current_weight_minus_recent_weighted_avg"] = current_weight - _weighted_mean(weight, 6)
    result["current_weight_change_pct"] = np.divide(current_weight - weight[:, 0], weight[:, 0], out=np.full(len(frame), np.nan), where=np.isfinite(weight[:, 0]) & (weight[:, 0] > 0))
    result["best_finish_percentile_at_equal_or_higher_weight"] = _best(np.where(equal_higher, finish, np.nan), 6)
    result["best_margin_quality_at_equal_or_higher_weight"] = _best(np.where(equal_higher, margin_quality, np.nan), 6)
    finish_std, margin_std = _std(finish, 6), _std(margin_quality, 6)
    result["form_ceiling_finish_percentile_6"] = _best(finish, 6)
    result["form_average_finish_percentile_6"] = _mean(finish, 6)
    result["form_consistency_finish_percentile_6"] = 1 / (1 + finish_std)
    result["form_ceiling_minus_average"] = _best(finish, 6) - _mean(finish, 6)
    result["margin_quality_ceiling_6"] = _best(margin_quality, 6)
    result["margin_quality_average_6"] = _mean(margin_quality, 6)
    result["margin_quality_consistency_6"] = 1 / (1 + margin_std)
    run_quality = np.nanmean(np.stack([finish, margin_quality]), axis=0)
    for row in range(len(frame)):
        valid = np.flatnonzero(np.isfinite(run_quality[row]))
        if len(valid):
            order = valid[np.argsort(-run_quality[row, valid], kind="stable")]
            result.loc[frame.index[row], "best_run_recency_index"] = order[0] + 1
            result.loc[frame.index[row], "second_best_run_recency_index"] = order[1] + 1 if len(order) > 1 else np.nan
    best_index = result["best_run_recency_index"]
    result["best_run_is_last_start"] = np.where(best_index.notna(), (best_index == 1).astype(float), np.nan)
    result["best_run_within_last_2"] = np.where(best_index.notna(), (best_index <= 2).astype(float), np.nan)
    result["best_run_within_last_3"] = np.where(best_index.notna(), (best_index <= 3).astype(float), np.nan)
    career_starts = pd.to_numeric(frame.get("career_starts", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
    career_top3 = sum(pd.to_numeric(frame.get(x, pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float) for x in ("career_wins", "career_seconds", "career_thirds"))
    career_place_rate = np.divide(career_top3, career_starts, out=np.full(len(frame), np.nan), where=career_starts > 0)
    supplied_place_rate = pd.to_numeric(frame.get("place_percentage", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float) / 100
    career_place_rate = np.where(np.isfinite(career_place_rate), career_place_rate, supplied_place_rate)
    for n in (3, 6):
        recent_rate = _mean(np.where(np.isfinite(place), (place <= 3).astype(float), np.nan), n)
        # The schema has career top-three rate but no career finish-percentile
        # aggregate. Keep the non-equivalent finish comparison null rather than
        # fabricating it from a different statistic.
        result[f"recent{n}_vs_career_finish_percentile"] = np.nan
        result[f"recent{n}_vs_career_place_rate"] = recent_rate - career_place_rate
    result["recent_form_vs_career_form"] = result["recent3_vs_career_place_rate"]
    def normalized_text(series: pd.Series) -> np.ndarray:
        return series.astype("string").str.strip().str.casefold().replace("", pd.NA).fillna("\x00missing").to_numpy(dtype=str)
    current_track = normalized_text(frame.get("competition_name", pd.Series(pd.NA, index=frame.index)))
    historical_track = np.column_stack([normalized_text(frame.get(f"recent_{run}_track_name", pd.Series(pd.NA, index=frame.index))) for run in range(1, 7)])
    # Going labels may include a rating, e.g. "Good (4)"; the leading word is
    # the stable categorical condition shared with historical form.
    condition_text = lambda series: series.astype("string").str.extract(r"^\s*([A-Za-z]+)", expand=False).str.casefold().fillna("\x00missing").to_numpy(dtype=str)
    current_condition = condition_text(frame.get("track_status", pd.Series(pd.NA, index=frame.index)))
    historical_condition = np.column_stack([condition_text(frame.get(f"recent_{run}_track_status", pd.Series(pd.NA, index=frame.index))) for run in range(1, 7)])
    same_track = (current_track[:, None] != "\x00missing") & (historical_track == current_track[:, None])
    same_track_distance = same_track & similar
    current_condition_valid = current_condition != "\x00missing"
    same_condition = current_condition_valid[:, None] & (historical_condition == current_condition[:, None])
    # A result is usable only when its field size also supports a valid finish
    # percentile; this keeps successes and observation counts identical.
    top3_history = np.where(np.isfinite(finish), (place <= 3).astype(float), np.nan)
    win_history = np.where(np.isfinite(finish), (place == 1).astype(float), np.nan)
    result["same_track_runs"] = same_track.sum(axis=1)
    result["same_track_finish_percentile"] = _mean(np.where(same_track, finish, np.nan), 6)
    result["same_track_margin_quality"] = _mean(np.where(same_track, margin_quality, np.nan), 6)
    result["same_track_top3_rate"] = _mean(np.where(same_track, top3_history, np.nan), 6)
    result["same_track_distance_runs"] = same_track_distance.sum(axis=1)
    result["same_track_distance_finish_percentile"] = _mean(np.where(same_track_distance, finish, np.nan), 6)
    result["same_track_distance_margin_quality"] = _mean(np.where(same_track_distance, margin_quality, np.nan), 6)
    result["same_track_distance_top3_rate"] = _mean(np.where(same_track_distance, top3_history, np.nan), 6)
    condition_runs = np.isfinite(np.where(same_condition, finish, np.nan)).sum(axis=1).astype(float)
    condition_wins = np.nansum(np.where(same_condition, win_history, np.nan), axis=1)
    condition_top3 = np.nansum(np.where(same_condition, top3_history, np.nan), axis=1)
    population_win = np.divide(1, current_field, out=np.full(len(frame), np.nan), where=current_field > 0)
    population_top3 = np.divide(np.minimum(3, current_field), current_field, out=np.full(len(frame), np.nan), where=current_field > 0)
    result["condition_runs"] = condition_runs
    result["condition_win_rate_smoothed"] = np.divide(condition_wins + 5 * population_win, condition_runs + 5, out=np.full(len(frame), np.nan), where=condition_runs > 0)
    result["condition_top3_rate_smoothed"] = np.divide(condition_top3 + 5 * population_top3, condition_runs + 5, out=np.full(len(frame), np.nan), where=condition_runs > 0)
    result["condition_finish_percentile"] = _mean(np.where(same_condition, finish, np.nan), 6)
    result["condition_margin_quality"] = _mean(np.where(same_condition, margin_quality, np.nan), 6)
    return result.loc[:, DERIVED_FEATURE_NAMES].astype(np.float32)
