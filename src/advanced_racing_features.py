"""Leakage-safe sectional, class, and entity-history racing features."""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd


HISTORY_FEATURE_NAMES = (
    "sectional_last600_seconds_weighted_3",
    "sectional_last600_seconds_weighted_6",
    "sectional_last600_best_6",
    "sectional_last600_std_6",
    "sectional_last600_trend",
    "sectional_closing_speed_ratio_weighted_3",
    "sectional_closing_speed_ratio_weighted_6",
    "current_class_level",
    "form_class_level_weighted_3",
    "form_class_level_weighted_6",
    "class_change_vs_recent_3",
    "class_drop_from_recent_best",
    "recent_last600_avg_3", "recent_last600_avg_6",
    "recent_last600_weighted_3", "recent_last600_weighted_6",
    "recent_last600_best_3", "recent_last600_best_6", "recent_last600_std_6",
    "recent_last600_slope_3", "recent_last600_slope_6",
    "historical_run_quality_last1", "historical_run_quality_avg_3",
    "historical_run_quality_weighted_3", "historical_run_quality_weighted_6",
    "historical_run_quality_best_3", "historical_run_quality_slope_3",
    "hidden_sectional_run_last1", "hidden_sectional_run_avg_3",
    "hidden_sectional_run_best_3", "sectional_ceiling_6",
    "sectional_average_6", "sectional_consistency_6",
    "class_change_last_run", "class_step_up", "class_step_down",
    "recent_class_avg_3", "recent_class_weighted_6", "recent_best_class",
    "performance_at_current_or_stronger_class",
    "best_finish_percentile_at_stronger_class",
    "best_margin_quality_at_stronger_class",
    "class_adjusted_finish_percentile_last1",
    "class_adjusted_finish_percentile_weighted_3",
    "class_adjusted_finish_percentile_weighted_6",
    "class_adjusted_margin_quality_weighted_3",
    "class_adjusted_margin_quality_weighted_6",
)

ENTITY_FEATURE_NAMES = tuple(f"{prefix}_{suffix}"
    for prefix in ("jockey", "trainer", "jockey_trainer")
    for suffix in ("history_starts", "history_top3_excess", "recent_top3_excess")) + (
    "jockey_trainer_history_runs", "jockey_trainer_history_wins",
    "jockey_trainer_history_top3", "jockey_trainer_history_win_rate",
    "jockey_trainer_history_top3_rate", "jockey_trainer_history_smoothed_win_rate",
    "jockey_trainer_history_smoothed_top3_rate", "jockey_trainer_history_win_excess",
    "jockey_trainer_synergy",
)

RACE_RELATIVE_SOURCES = (
    "recent_finish_percentile_weighted_3",
    "recent_finish_percentile_weighted_6",
    "recent_margin_quality_weighted_3",
    "historical_market_overperformance_weighted_3",
    "hidden_sectional_run_best_3",
    "similar_distance_finish_percentile_weighted",
    "class_adjusted_finish_percentile_weighted_3",
    "jockey_trainer_history_smoothed_top3_rate",
    "current_form_strength",
)
RACE_RELATIVE_SUFFIXES = (
    "rank_in_race", "pct_in_race", "minus_race_mean", "minus_race_median",
    "zscore_in_race", "gap_to_best",
)
CONTEXT_FEATURE_NAMES = ("current_form_strength",) + tuple(
    f"{source}_{suffix}" for source in RACE_RELATIVE_SOURCES
    for suffix in RACE_RELATIVE_SUFFIXES
)
ADVANCED_FEATURE_NAMES = HISTORY_FEATURE_NAMES + ENTITY_FEATURE_NAMES + CONTEXT_FEATURE_NAMES

_GRADE_LEVELS = {"ONE": 125.0, "TWO": 120.0, "THREE": 115.0, "LR": 110.0}


def _weighted_mean(values: np.ndarray, count: int) -> np.ndarray:
    selected = values[:, :count]
    # Same documented convention as derived_racing_features: [N, ..., 1].
    weights = np.arange(count, 0, -1, dtype=np.float64)
    valid = np.isfinite(selected)
    denominator = (valid * weights).sum(axis=1)
    numerator = np.where(valid, selected, 0.0).dot(weights)
    return np.divide(
        numerator, denominator,
        out=np.full(len(values), np.nan), where=denominator > 0,
    )


def _linear_slope(values: np.ndarray, count: int, higher_is_better: bool) -> np.ndarray:
    selected = values[:, :count][:, ::-1]
    result = np.full(len(values), np.nan)
    for row, observations in enumerate(selected):
        valid = np.isfinite(observations)
        if valid.sum() >= 2:
            result[row] = np.polyfit(np.arange(count)[valid], observations[valid], 1)[0]
    return result if higher_is_better else -result


def _duration_seconds(values: pd.Series) -> np.ndarray:
    text = values.astype("string").str.strip()
    colon = text.str.extract(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")
    colon_minutes = pd.to_numeric(colon[0], errors="coerce").fillna(0.0)
    colon_seconds = pd.to_numeric(colon[1], errors="coerce")
    result = colon_minutes.to_numpy(dtype=np.float64) * 60.0 + colon_seconds.to_numpy(
        dtype=np.float64
    )
    dot = text.str.extract(r"^(\d+)\.(\d{2})\.(\d{2})$")
    dot_minutes = pd.to_numeric(dot[0], errors="coerce")
    dot_seconds = pd.to_numeric(dot[1], errors="coerce")
    dot_hundredths = pd.to_numeric(dot[2], errors="coerce")
    dot_valid = dot_minutes.notna() & dot_seconds.notna() & dot_hundredths.notna()
    result[dot_valid.to_numpy()] = (
        dot_minutes[dot_valid].to_numpy(dtype=np.float64) * 60.0
        + dot_seconds[dot_valid].to_numpy(dtype=np.float64)
        + dot_hundredths[dot_valid].to_numpy(dtype=np.float64) / 100.0
    )
    valid = colon_seconds.notna() | dot_valid
    result[~valid.to_numpy()] = np.nan
    return result


def _duration_matrix(frame: pd.DataFrame, stem: str) -> np.ndarray:
    return np.column_stack([
        _duration_seconds(frame[f"recent_{run}_{stem}"])
        for run in range(1, 7)
    ])


def parse_class_level(value: object, grade: object = None) -> float:
    """Map common Australasian class labels onto an approximate rating scale."""
    grade_text = "" if pd.isna(grade) else str(grade).strip().upper()
    if grade_text in _GRADE_LEVELS:
        return _GRADE_LEVELS[grade_text]
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if not text or "BARRIER TRIAL" in text:
        return np.nan
    if re.search(r"\b(?:GROUP\s*1|G1)\b", text):
        return 125.0
    if re.search(r"\b(?:GROUP\s*2|G2)\b", text):
        return 120.0
    if re.search(r"\b(?:GROUP\s*3|G3)\b", text):
        return 115.0
    if re.search(r"\b(?:LISTED|LR)\b", text):
        return 110.0
    benchmark = re.search(r"\b(?:BM|BENCHMARK|RST)\s*([0-9]{2,3})", text)
    if benchmark:
        return float(benchmark.group(1))
    handicap_rating = re.search(r"\bHCP\s*\(([0-9]{2,3})\)", text)
    if handicap_rating:
        return float(handicap_rating.group(1))
    class_number = re.search(r"\b(?:CLS|CLASS|C)\s*([1-6])\b", text)
    if class_number:
        return 52.0 + 4.0 * float(class_number.group(1))
    if re.search(r"\b(?:MDN|MAIDEN)\b", text):
        return 50.0
    if re.search(r"\bOPEN\b", text):
        return 100.0
    return np.nan


def derive_sectional_class_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive features solely from the current race card and stored prior runs."""
    required = {"race_name", "grade", "distance_m"}
    required.update(
        f"recent_{run}_{stem}"
        for run in range(1, 7)
        for stem in ("last600", "time", "distance_m", "class")
    )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Cannot derive advanced history features; missing: " + ", ".join(missing))

    last600 = _duration_matrix(frame, "last600")
    # A 600 m sectional outside this range is a malformed/full-race duration,
    # not a plausible thoroughbred closing split.
    last600[(last600 < 25.0) | (last600 > 60.0)] = np.nan
    race_time = _duration_matrix(frame, "time")
    recent_distance = np.column_stack([
        pd.to_numeric(frame[f"recent_{run}_distance_m"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for run in range(1, 7)
    ])
    average_speed = np.divide(
        recent_distance, race_time,
        out=np.full(last600.shape, np.nan),
        where=np.isfinite(recent_distance) & np.isfinite(race_time) & (race_time > 0),
    )
    closing_speed = np.divide(
        600.0, last600,
        out=np.full(last600.shape, np.nan),
        where=np.isfinite(last600) & (last600 > 0),
    )
    closing_ratio = np.divide(
        closing_speed, average_speed,
        out=np.full(last600.shape, np.nan),
        where=np.isfinite(closing_speed) & np.isfinite(average_speed) & (average_speed > 0),
    )
    closing_ratio[(closing_ratio < 0.5) | (closing_ratio > 2.0)] = np.nan

    valid_last600 = np.isfinite(last600)
    with np.errstate(invalid="ignore"):
        best_last600 = np.min(np.where(valid_last600, last600, np.inf), axis=1)
    best_last600[~valid_last600.any(axis=1)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        last600_std = np.nanstd(last600, axis=1, ddof=1)
    last600_std[valid_last600.sum(axis=1) < 2] = np.nan
    older = _weighted_mean(np.column_stack([last600[:, 1:], np.full(len(frame), np.nan)]), 5)
    trend = older - last600[:, 0]
    trend[~np.isfinite(last600[:, 0]) | ~np.isfinite(older)] = np.nan

    current_class = np.asarray([
        parse_class_level(name, grade)
        for name, grade in zip(frame["race_name"], frame["grade"])
    ], dtype=np.float64)
    recent_class = np.column_stack([
        np.asarray([
            parse_class_level(value)
            for value in frame[f"recent_{run}_class"]
        ], dtype=np.float64)
        for run in range(1, 7)
    ])
    class_3 = _weighted_mean(recent_class, 3)
    class_6 = _weighted_mean(recent_class, 6)
    with np.errstate(invalid="ignore"):
        best_class = np.max(
            np.where(np.isfinite(recent_class), recent_class, -np.inf), axis=1
        )
    best_class[~np.isfinite(recent_class).any(axis=1)] = np.nan

    optional = lambda stem: np.column_stack([pd.to_numeric(frame.get(f"recent_{run}_{stem}", pd.Series(np.nan, index=frame.index)), errors="coerce") for run in range(1, 7)])
    place, field, margin, barrier = (optional(stem) for stem in ("place", "total_runners", "margin", "barrier"))
    valid_finish = np.isfinite(place) & np.isfinite(field) & (place >= 1) & (field > 0)
    finish = np.full(place.shape, np.nan)
    finish[valid_finish] = 1 - (place[valid_finish] - 1) / np.maximum(field[valid_finish] - 1, 1)
    finish = np.clip(finish, 0, 1)
    margin = np.where(np.isfinite(margin), np.where(place == 1, 0., margin), np.nan)
    margin_quality = np.exp(-np.maximum(margin, 0) / 5)
    # Closing speed ratio is higher-is-better. Map its validated [0.5, 2.0]
    # range onto [0, 1] before equal-weight composites.
    sectional_quality = np.clip((closing_ratio - .5) / 1.5, 0, 1)
    class_ratio = np.divide(recent_class, current_class[:, None], out=np.full(recent_class.shape, np.nan),
                            where=np.isfinite(recent_class) & np.isfinite(current_class[:, None]) & (current_class[:, None] > 0))
    class_factor = np.clip(class_ratio, .5, 1.5)
    adjusted_finish = finish * class_factor
    adjusted_margin = margin_quality * class_factor
    # Equal-weight historical run quality: class-adjusted finish,
    # beaten-margin quality, and sectional quality. Barrier is excluded because
    # draw quality is context dependent.
    quality_components = np.stack([
        np.clip(adjusted_finish, 0, 1),
        margin_quality,
        sectional_quality,
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        run_quality = np.nanmean(quality_components, axis=0)
    hidden_sectional = sectional_quality * (1 - finish)

    result = pd.DataFrame(index=frame.index)
    result["sectional_last600_seconds_weighted_3"] = _weighted_mean(last600, 3)
    result["sectional_last600_seconds_weighted_6"] = _weighted_mean(last600, 6)
    result["sectional_last600_best_6"] = best_last600
    result["sectional_last600_std_6"] = last600_std
    result["sectional_last600_trend"] = trend
    result["sectional_closing_speed_ratio_weighted_3"] = _weighted_mean(
        closing_ratio, 3
    )
    result["sectional_closing_speed_ratio_weighted_6"] = _weighted_mean(
        closing_ratio, 6
    )
    result["current_class_level"] = current_class
    result["form_class_level_weighted_3"] = class_3
    result["form_class_level_weighted_6"] = class_6
    result["class_change_vs_recent_3"] = current_class - class_3
    result["class_drop_from_recent_best"] = best_class - current_class
    result["recent_last600_avg_3"] = np.nanmean(last600[:, :3], axis=1)
    result["recent_last600_avg_6"] = np.nanmean(last600, axis=1)
    result["recent_last600_weighted_3"] = _weighted_mean(last600, 3)
    result["recent_last600_weighted_6"] = _weighted_mean(last600, 6)
    result["recent_last600_best_3"] = np.nanmin(last600[:, :3], axis=1)
    result["recent_last600_best_6"] = best_last600
    result["recent_last600_std_6"] = last600_std
    result["recent_last600_slope_3"] = _linear_slope(last600, 3, False)
    result["recent_last600_slope_6"] = _linear_slope(last600, 6, False)
    result["historical_run_quality_last1"] = run_quality[:, 0]
    result["historical_run_quality_avg_3"] = np.nanmean(run_quality[:, :3], axis=1)
    result["historical_run_quality_weighted_3"] = _weighted_mean(run_quality, 3)
    result["historical_run_quality_weighted_6"] = _weighted_mean(run_quality, 6)
    result["historical_run_quality_best_3"] = np.nanmax(run_quality[:, :3], axis=1)
    result["historical_run_quality_slope_3"] = _linear_slope(run_quality, 3, True)
    result["hidden_sectional_run_last1"] = hidden_sectional[:, 0]
    result["hidden_sectional_run_avg_3"] = np.nanmean(hidden_sectional[:, :3], axis=1)
    result["hidden_sectional_run_best_3"] = np.nanmax(hidden_sectional[:, :3], axis=1)
    result["sectional_ceiling_6"] = np.nanmax(sectional_quality, axis=1)
    result["sectional_average_6"] = np.nanmean(sectional_quality, axis=1)
    sectional_std = np.nanstd(sectional_quality, axis=1, ddof=1)
    sectional_std[np.isfinite(sectional_quality).sum(axis=1) < 2] = np.nan
    result["sectional_consistency_6"] = 1 / (1 + sectional_std)
    class_delta = current_class - recent_class[:, 0]
    result["class_change_last_run"] = class_delta
    result["class_step_up"] = np.maximum(class_delta, 0)
    result["class_step_down"] = np.maximum(-class_delta, 0)
    result["recent_class_avg_3"] = np.nanmean(recent_class[:, :3], axis=1)
    result["recent_class_weighted_6"] = class_6
    result["recent_best_class"] = best_class
    at_stronger = np.isfinite(recent_class) & (recent_class >= current_class[:, None])
    result["performance_at_current_or_stronger_class"] = np.nanmean(np.where(at_stronger, finish, np.nan), axis=1)
    result["best_finish_percentile_at_stronger_class"] = np.nanmax(np.where(at_stronger, finish, np.nan), axis=1)
    result["best_margin_quality_at_stronger_class"] = np.nanmax(np.where(at_stronger, margin_quality, np.nan), axis=1)
    result["class_adjusted_finish_percentile_last1"] = adjusted_finish[:, 0]
    result["class_adjusted_finish_percentile_weighted_3"] = _weighted_mean(adjusted_finish, 3)
    result["class_adjusted_finish_percentile_weighted_6"] = _weighted_mean(adjusted_finish, 6)
    result["class_adjusted_margin_quality_weighted_3"] = _weighted_mean(adjusted_margin, 3)
    result["class_adjusted_margin_quality_weighted_6"] = _weighted_mean(adjusted_margin, 6)
    return result.astype(np.float32)


def _normalize_entity(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.mask(normalized.eq(""))


def _causal_entity_features(
    frame: pd.DataFrame,
    entity: pd.Series,
    prefix: str,
    recent_span: int,
) -> pd.DataFrame:
    """Calculate entity statistics using only strictly earlier start times."""
    result = pd.DataFrame(index=frame.index)
    names = [
        f"{prefix}_history_starts",
        f"{prefix}_history_top3_excess",
        f"{prefix}_recent_top3_excess",
    ]
    for name in names:
        result[name] = np.nan

    start_time = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    field_size = pd.to_numeric(
        frame["active_field_size"], errors="coerce"
    ).fillna(pd.to_numeric(frame["field_size"], errors="coerce"))
    top3 = pd.to_numeric(frame["top3_mask"], errors="coerce")
    eligible = (
        entity.notna() & start_time.notna() & frame["status"].eq("finished")
        & pd.to_numeric(frame["runner_mask"], errors="coerce").eq(1)
        & top3.isin([0, 1]) & field_size.gt(0)
    )
    expected = np.minimum(3.0, field_size) / field_size
    history = pd.DataFrame({
        "entity": entity[eligible],
        "time": start_time[eligible],
        "starts": 1.0,
        "residual": top3[eligible].to_numpy(dtype=np.float64)
        - expected[eligible].to_numpy(dtype=np.float64),
    })
    if history.empty:
        return result
    events = history.groupby(["entity", "time"], as_index=False, sort=True).agg(
        starts=("starts", "sum"), residual=("residual", "sum")
    )
    events = events.sort_values(["entity", "time"], kind="stable")
    events["post_starts"] = events.groupby("entity", sort=False)["starts"].cumsum()
    events["post_residual"] = events.groupby("entity", sort=False)["residual"].cumsum()
    events["event_excess"] = events["residual"] / events["starts"]
    events["post_recent"] = events.groupby("entity", sort=False)[
        "event_excess"
    ].transform(lambda values: values.ewm(span=recent_span, adjust=False).mean())

    targets = pd.DataFrame({
        "entity": entity,
        "time": start_time,
        "row": np.arange(len(frame), dtype=np.int64),
    }).dropna(subset=["entity", "time"])
    event_groups = {
        entity_name: group
        for entity_name, group in events.groupby("entity", sort=False)
    }
    for entity_name, target_group in targets.groupby("entity", sort=False):
        event_group = event_groups.get(entity_name)
        if event_group is None:
            continue
        event_ns = event_group["time"].astype("int64").to_numpy()
        target_ns = target_group["time"].astype("int64").to_numpy()
        prior_index = np.searchsorted(event_ns, target_ns, side="left") - 1
        usable = prior_index >= 0
        if not usable.any():
            continue
        rows = target_group["row"].to_numpy(dtype=np.int64)[usable]
        indices = prior_index[usable]
        prior_starts = event_group["post_starts"].to_numpy(dtype=np.float64)[indices]
        prior_residual = event_group["post_residual"].to_numpy(dtype=np.float64)[indices]
        recent = event_group["post_recent"].to_numpy(dtype=np.float64)[indices]
        reliability = prior_starts / (prior_starts + 10.0)
        result.iloc[rows, result.columns.get_loc(names[0])] = prior_starts
        result.iloc[rows, result.columns.get_loc(names[1])] = (
            prior_residual / (prior_starts + 20.0)
        )
        result.iloc[rows, result.columns.get_loc(names[2])] = recent * reliability
    return result.astype(np.float32)


def _causal_success_totals(
    frame: pd.DataFrame, entity: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return prior runs/wins/top3; same-time events are atomically excluded.

    ``finish_place`` is read only for completed historical events. The left-sided
    timestamp lookup guarantees that neither the target result, another runner in
    the target race, nor a future result can enter a target's aggregates.
    """
    times = pd.to_datetime(frame["start_time_iso"], utc=True, errors="coerce")
    if "finish_place" in frame:
        place = pd.to_numeric(frame["finish_place"], errors="coerce")
    else:
        # Compatibility fallback for callers that only carry the historical
        # top-three label. It can support top3 totals but intentionally cannot
        # infer wins.
        top3 = pd.to_numeric(frame["top3_mask"], errors="coerce")
        place = pd.Series(np.where(top3.eq(1), 3., 4.), index=frame.index)
    eligible = (entity.notna() & times.notna() & frame["status"].eq("finished")
                & pd.to_numeric(frame["runner_mask"], errors="coerce").eq(1)
                & place.ge(1))
    history = pd.DataFrame({"entity": entity[eligible], "time": times[eligible],
                            "runs": 1., "wins": place[eligible].eq(1).astype(float),
                            "top3": place[eligible].le(3).astype(float)})
    outputs = tuple(np.full(len(frame), np.nan) for _ in range(3))
    if history.empty:
        return outputs
    events = history.groupby(["entity", "time"], as_index=False, sort=True)[["runs", "wins", "top3"]].sum()
    events = events.sort_values(["entity", "time"], kind="stable")
    for column in ("runs", "wins", "top3"):
        events[column] = events.groupby("entity", sort=False)[column].cumsum()
    groups = {key: value for key, value in events.groupby("entity", sort=False)}
    targets = pd.DataFrame({"entity": entity, "time": times,
                            "row": np.arange(len(frame))}).dropna(subset=["entity", "time"])
    for key, target_group in targets.groupby("entity", sort=False):
        event_group = groups.get(key)
        if event_group is None:
            continue
        positions = np.searchsorted(event_group["time"].astype("int64"),
                                    target_group["time"].astype("int64"), side="left") - 1
        usable = positions >= 0
        rows = target_group["row"].to_numpy()[usable]
        for output, column in zip(outputs, ("runs", "wins", "top3")):
            output[rows] = event_group[column].to_numpy()[positions[usable]]
    return outputs


def derive_entity_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return causal jockey, trainer, and partnership history features."""
    required = {
        "start_time_iso", "status", "runner_mask", "top3_mask",
        "active_field_size", "field_size", "jockey", "trainer",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Cannot derive entity history features; missing: " + ", ".join(missing))
    jockey = _normalize_entity(frame["jockey"])
    trainer = _normalize_entity(frame["trainer"])
    partnership = (jockey + "\x1f" + trainer).where(jockey.notna() & trainer.notna())
    result = pd.concat([
        _causal_entity_features(frame, jockey, "jockey", 50),
        _causal_entity_features(frame, trainer, "trainer", 100),
        _causal_entity_features(frame, partnership, "jockey_trainer", 30),
    ], axis=1)
    jockey_totals = _causal_success_totals(frame, jockey)
    trainer_totals = _causal_success_totals(frame, trainer)
    pair_runs, pair_wins, pair_top3 = _causal_success_totals(frame, partnership)
    field = pd.to_numeric(frame["active_field_size"], errors="coerce").fillna(
        pd.to_numeric(frame["field_size"], errors="coerce")).to_numpy(float)
    population_win = np.divide(1, field, out=np.full(len(frame), np.nan), where=field > 0)
    population_top3 = np.divide(np.minimum(3, field), field,
                                out=np.full(len(frame), np.nan), where=field > 0)
    prior_strength = 10.0
    pair_win_rate = np.divide(pair_wins, pair_runs, out=np.full(len(frame), np.nan), where=pair_runs > 0)
    pair_top3_rate = np.divide(pair_top3, pair_runs, out=np.full(len(frame), np.nan), where=pair_runs > 0)
    smooth_win = np.divide(pair_wins + prior_strength * population_win,
                           pair_runs + prior_strength, out=np.full(len(frame), np.nan), where=pair_runs > 0)
    smooth_top3 = np.divide(pair_top3 + prior_strength * population_top3,
                            pair_runs + prior_strength, out=np.full(len(frame), np.nan), where=pair_runs > 0)
    jockey_top3_rate = np.divide(jockey_totals[2], jockey_totals[0], out=np.full(len(frame), np.nan), where=jockey_totals[0] > 0)
    trainer_top3_rate = np.divide(trainer_totals[2], trainer_totals[0], out=np.full(len(frame), np.nan), where=trainer_totals[0] > 0)
    # Expected partnership performance is the equal-weight mean of the two
    # independently observed histories; synergy is positive when the pair beats it.
    expected_pair_top3 = np.nanmean(np.column_stack([jockey_top3_rate, trainer_top3_rate]), axis=1)
    result["jockey_trainer_history_runs"] = pair_runs
    result["jockey_trainer_history_wins"] = pair_wins
    result["jockey_trainer_history_top3"] = pair_top3
    result["jockey_trainer_history_win_rate"] = pair_win_rate
    result["jockey_trainer_history_top3_rate"] = pair_top3_rate
    result["jockey_trainer_history_smoothed_win_rate"] = smooth_win
    result["jockey_trainer_history_smoothed_top3_rate"] = smooth_top3
    result["jockey_trainer_history_win_excess"] = smooth_win - population_win
    result["jockey_trainer_synergy"] = smooth_top3 - expected_pair_top3
    return result.loc[:, ENTITY_FEATURE_NAMES].astype(np.float32)


def race_relative_runner_mask(frame: pd.DataFrame) -> pd.Series:
    """Return runners eligible to participate in within-race transforms.

    For resulted races, ``runner_mask == 1`` remains mandatory and therefore
    scratched/inactive runners can never contaminate training features.

    The live feed intentionally leaves ``runner_mask`` at zero until results
    arrive.  For an unfinished race we may use its stored rows only when all
    three pre-race checks agree: betting is PRICED/OFF, ``active_field_size`` is
    positive and consistent, and the number of stored rows exactly matches that
    active size.  This fails closed for partial downloads or retained scratches.
    It does not mutate the database's result/training mask.
    """
    active = pd.to_numeric(frame["runner_mask"], errors="coerce").eq(1)
    required = {
        "race_id", "status", "source_betting_status", "active_field_size",
    }
    if not required <= set(frame.columns):
        return active
    status = frame["status"].astype("string").str.strip().str.casefold()
    betting = (
        frame["source_betting_status"].astype("string").str.strip().str.upper()
    )
    active_size = pd.to_numeric(frame["active_field_size"], errors="coerce")
    stored_rows = frame.groupby("race_id", sort=False, dropna=False)[
        "race_id"
    ].transform("size")
    live_complete_field = (
        status.ne("finished")
        & betting.isin(["PRICED", "OFF"])
        & active_size.gt(0)
        & active_size.eq(stored_rows)
        & active_size.groupby(frame["race_id"], sort=False, dropna=False)
        .transform("min")
        .eq(active_size.groupby(frame["race_id"], sort=False, dropna=False)
            .transform("max"))
    )
    return active | live_complete_field


def derive_context_features(frame: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """Build the equal-weight form composite and within-race transforms.

    Every source is higher-is-better. Ranks use deterministic ``method='min'``
    for ties; percentile is one for the best and zero for the worst; gap is zero
    for the best and increasingly positive for weaker runners.
    """
    required = {"race_id", "runner_mask", "recent_finish_percentile_weighted_3",
                "recent_margin_quality_weighted_3",
                "sectional_closing_speed_ratio_weighted_3",
                "class_adjusted_finish_percentile_weighted_3"}
    missing = sorted(required - (set(frame) | set(derived)))
    if missing:
        raise ValueError("Cannot derive context features; missing: " + ", ".join(missing))
    components = np.column_stack([
        derived["recent_finish_percentile_weighted_3"],
        derived["recent_margin_quality_weighted_3"],
        np.clip((derived["sectional_closing_speed_ratio_weighted_3"] - .5) / 1.5, 0, 1),
        np.clip(derived["class_adjusted_finish_percentile_weighted_3"], 0, 1),
    ]).astype(float)
    valid = np.isfinite(components)
    current_form = np.divide(np.where(valid, components, 0).sum(axis=1), valid.sum(axis=1),
                             out=np.full(len(frame), np.nan), where=valid.sum(axis=1) > 0)
    working = derived.copy()
    working["current_form_strength"] = current_form
    result = pd.DataFrame({"current_form_strength": current_form}, index=frame.index)
    race_id = frame["race_id"]
    active = race_relative_runner_mask(frame)
    for source in RACE_RELATIVE_SOURCES:
        # Resulted races require runner_mask=1. Complete live fields use the
        # fail-closed pre-race fallback documented in race_relative_runner_mask.
        values = pd.to_numeric(working[source], errors="coerce").where(active)
        grouped = values.groupby(race_id, sort=False, dropna=False)
        count = grouped.transform("count")
        rank = grouped.rank(method="min", ascending=False)
        result[f"{source}_rank_in_race"] = rank
        result[f"{source}_pct_in_race"] = np.where(
            values.notna() & (count > 1), (count - rank) / (count - 1),
            np.where(values.notna() & (count == 1), 1., np.nan))
        result[f"{source}_minus_race_mean"] = values - grouped.transform("mean")
        result[f"{source}_minus_race_median"] = values - grouped.transform("median")
        std = grouped.transform("std")
        result[f"{source}_zscore_in_race"] = np.where(std > 0, (values - grouped.transform("mean")) / std, np.nan)
        result[f"{source}_gap_to_best"] = grouped.transform("max") - values
    return result.loc[:, CONTEXT_FEATURE_NAMES].astype(np.float32)
