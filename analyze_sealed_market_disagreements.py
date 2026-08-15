#!/usr/bin/env python3
"""Describe where frozen winner models help or hurt the market favourite."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHALLENGERS = ("frozen_blend", "market_corrector", "gpt_pick", "gpt_fluc2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path,
        default=Path("outputs/winner_ranker_chronological"),
    )
    parser.add_argument("--db", type=Path, default=Path("db/race_runners.sqlite"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/winner_ranker_chronological/disagreement_analysis"),
    )
    return parser.parse_args()


def add_frozen_blend_score(
    frame: pd.DataFrame, weights: dict[str, float]
) -> pd.DataFrame:
    frame = frame.copy()
    model_labels = [label for label in weights if label != "market"]
    missing = sorted(
        f"{label}_score" for label in model_labels
        if f"{label}_score" not in frame
    )
    if missing:
        raise ValueError("Predictions are missing frozen blend scores: " + ", ".join(missing))
    frame["frozen_blend_score"] = sum(
        float(weights[label]) * frame[f"{label}_score"] for label in model_labels
    )
    frame["frozen_blend_rank"] = (
        frame.groupby("race_id", sort=False)["frozen_blend_score"]
        .rank(method="first", ascending=False).astype(int)
    )
    return frame


def load_race_context(database: Path, race_ids: set[int]) -> pd.DataFrame:
    identifiers = ",".join(str(int(race_id)) for race_id in sorted(race_ids))
    if not identifiers:
        raise ValueError("No race IDs supplied for context loading")
    query = f"""
        SELECT race_id,
               MIN(distance_m) AS distance_min,
               MAX(distance_m) AS distance_max,
               MIN(current_class_level) AS class_min,
               MAX(current_class_level) AS class_max
        FROM race_runners
        WHERE race_id IN ({identifiers})
        GROUP BY race_id
    """
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        context = pd.read_sql_query(query, connection)
    if set(context["race_id"].astype(int)) != race_ids:
        raise ValueError("Database context does not cover every prediction race")
    distance_bad = context["distance_min"] != context["distance_max"]
    class_bad = (
        context["class_min"].notna() & context["class_max"].notna()
        & (context["class_min"] != context["class_max"])
    )
    if distance_bad.any() or class_bad.any():
        raise ValueError("Race-level distance or class is inconsistent within a race")
    return context.rename(columns={
        "distance_min": "distance_m", "class_min": "class_level",
    }).loc[:, ["race_id", "distance_m", "class_level"]]


def top_score_margin(frame: pd.DataFrame, name: str) -> pd.Series:
    score_column = f"{name}_score"
    ordered = frame.sort_values(
        ["race_id", score_column], ascending=[True, False], kind="stable"
    )
    top_two = ordered.groupby("race_id", sort=False).head(2)
    counts = top_two.groupby("race_id", sort=False).size()
    if not (counts == 2).all():
        raise ValueError(f"{name} confidence requires at least two runners per race")
    return top_two.groupby("race_id", sort=False)[score_column].agg(
        lambda values: float(values.iloc[0] - values.iloc[1])
    ).rename(f"{name}_confidence")


def confidence_edges(validation: pd.DataFrame) -> list[float]:
    values = top_score_margin(validation, "market_corrector")
    quartiles = values.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
    return [-np.inf, *np.unique(quartiles).tolist(), np.inf]


def race_level_frame(
    predictions: pd.DataFrame,
    context: pd.DataFrame,
    corrector_confidence_edges: list[float],
) -> pd.DataFrame:
    required = {
        "race_id", "runner_number", "fluc2", "is_winner", "market_rank",
        *(f"{name}_rank" for name in CHALLENGERS),
    }
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError("Predictions are missing: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    confidence = {
        name: top_score_margin(predictions, name) for name in CHALLENGERS
    }
    for race_id, race in predictions.groupby("race_id", sort=False):
        market = race.loc[race["market_rank"] == 1].iloc[0]
        record: dict[str, Any] = {
            "race_id": int(race_id),
            "field_size": len(race),
            "favourite_price": float(market["fluc2"]),
            "market_runner_number": int(market["runner_number"]),
            "market_win": int(market["is_winner"]),
        }
        for name in CHALLENGERS:
            pick = race.loc[race[f"{name}_rank"] == 1].iloc[0]
            model_win = int(pick["is_winner"])
            changed = int(pick["runner_number"] != market["runner_number"])
            record[f"{name}_runner_number"] = int(pick["runner_number"])
            record[f"{name}_win"] = model_win
            record[f"{name}_changed"] = changed
            record[f"{name}_corrected"] = int(changed and model_win and not record["market_win"])
            record[f"{name}_damaged"] = int(changed and not model_win and record["market_win"])
            record[f"{name}_confidence"] = float(confidence[name].loc[race_id])
        corrector_pick = race.loc[race["market_corrector_rank"] == 1].iloc[0]
        record["corrector_pick_market_rank"] = int(corrector_pick["market_rank"])
        record["corrector_market_rank_gap"] = int(corrector_pick["market_rank"] - 1)
        record["gpt_pick_market_agreement"] = (
            "agree" if not record["gpt_pick_changed"] else "disagree"
        )
        record["gpt_fluc2_market_agreement"] = (
            "agree" if not record["gpt_fluc2_changed"] else "disagree"
        )
        rows.append(record)
    races = pd.DataFrame(rows).merge(context, on="race_id", validate="one_to_one")
    races["favourite_price_segment"] = pd.cut(
        races["favourite_price"],
        bins=[-np.inf, 1.5, 2.5, 4.0, 6.0, np.inf], right=False,
        labels=["<1.50", "1.50-2.49", "2.50-3.99", "4.00-5.99", "6.00+"],
    )
    races["field_size_segment"] = pd.cut(
        races["field_size"], bins=[0, 7, 10, 14, np.inf],
        labels=["4-7", "8-10", "11-14", "15+"],
    )
    races["distance_segment"] = pd.cut(
        races["distance_m"], bins=[0, 1200, 1400, 1600, 2000, np.inf],
        labels=["<=1200m", "1201-1400m", "1401-1600m", "1601-2000m", "2001m+"],
        include_lowest=True,
    )
    races["class_segment"] = pd.cut(
        races["class_level"], bins=[-np.inf, 50, 58, 65, 75, np.inf],
        labels=["<=50", "51-58", "59-65", "66-75", "76+"],
        include_lowest=True,
    ).astype("object").fillna("unknown")
    confidence_labels = [
        f"validation_Q{index + 1}" for index in range(len(corrector_confidence_edges) - 1)
    ]
    races["corrector_confidence_segment"] = pd.cut(
        races["market_corrector_confidence"], bins=corrector_confidence_edges,
        labels=confidence_labels, include_lowest=True,
    )
    races["corrector_market_rank_gap_segment"] = pd.cut(
        races["corrector_market_rank_gap"], bins=[-1, 0, 1, 2, np.inf],
        labels=["0 (agrees)", "1", "2", "3+"],
    )
    return races


def segment_metrics(races: pd.DataFrame, cohort: str) -> pd.DataFrame:
    dimensions = {
        "favourite_price": "favourite_price_segment",
        "field_size": "field_size_segment",
        "distance": "distance_segment",
        "race_class": "class_segment",
        "corrector_confidence": "corrector_confidence_segment",
        "corrector_market_rank_gap": "corrector_market_rank_gap_segment",
        "gpt_pick_market_agreement": "gpt_pick_market_agreement",
        "gpt_fluc2_market_agreement": "gpt_fluc2_market_agreement",
    }
    output: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        for segment, group in races.groupby(column, observed=True, sort=False):
            for challenger in CHALLENGERS:
                corrected = int(group[f"{challenger}_corrected"].sum())
                damaged = int(group[f"{challenger}_damaged"].sum())
                output.append({
                    "cohort": cohort,
                    "dimension": dimension,
                    "segment": str(segment),
                    "challenger": challenger,
                    "races": len(group),
                    "market_wins": int(group["market_win"].sum()),
                    "model_wins": int(group[f"{challenger}_win"].sum()),
                    "pick_changes": int(group[f"{challenger}_changed"].sum()),
                    "market_losses_corrected": corrected,
                    "market_winners_damaged": damaged,
                    "net_winners_gained": corrected - damaged,
                })
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = json.loads(
        (artifact_dir / "chronological_validation.json").read_text(encoding="utf-8")
    )
    validation = add_frozen_blend_score(
        pd.read_csv(artifact_dir / "validation_predictions.csv"),
        result["selected_weights"],
    )
    sealed = add_frozen_blend_score(
        pd.read_csv(artifact_dir / "sealed_test_predictions.csv"),
        result["selected_weights"],
    )
    all_ids = set(pd.concat([validation["race_id"], sealed["race_id"]]).astype(int))
    context = load_race_context(args.db, all_ids)
    edges = confidence_edges(validation)
    validation_races = race_level_frame(
        validation, context.loc[context["race_id"].isin(validation["race_id"])], edges
    )
    sealed_races = race_level_frame(
        sealed, context.loc[context["race_id"].isin(sealed["race_id"])], edges
    )
    validation_metrics = segment_metrics(validation_races, "validation")
    sealed_metrics = segment_metrics(sealed_races, "sealed_test")
    metrics = pd.concat([validation_metrics, sealed_metrics], ignore_index=True)
    stability = validation_metrics.merge(
        sealed_metrics,
        on=["dimension", "segment", "challenger"],
        suffixes=("_validation", "_sealed_test"),
    )
    metrics.to_csv(output_dir / "segment_metrics.csv", index=False)
    stability.to_csv(output_dir / "segment_stability.csv", index=False)
    validation_races.to_csv(output_dir / "validation_races.csv", index=False)
    sealed_races.to_csv(output_dir / "sealed_test_races.csv", index=False)
    metadata = {
        "schema_version": 1,
        "analysis_role": "exploratory_only_no_gate_tuning",
        "challengers": list(CHALLENGERS),
        "corrector_confidence_edges_selected_on_validation": edges,
        "source_result": str((artifact_dir / "chronological_validation.json").resolve()),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    display = sealed_metrics.loc[
        sealed_metrics["challenger"] == "frozen_blend"
    ]
    print("SEALED TEST SEGMENTS — FROZEN BLEND VS MARKET")
    for dimension, group in display.groupby("dimension", sort=False):
        print(f"\n{dimension.upper()}")
        print(group[[
            "segment", "races", "market_wins", "model_wins", "pick_changes",
            "market_losses_corrected", "market_winners_damaged", "net_winners_gained",
        ]].to_string(index=False))
    print(f"\nsaved_analysis={output_dir}")


if __name__ == "__main__":
    main()
