#!/usr/bin/env python3
"""Predict or backtest a current-race-only RaceFormerTop3 checkpoint."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_DB
from src.constants import VALIDATION_ROWS_VIEW
from src.database import quote_identifier, require_rows_view
from src.dataset import load_feature_manifest
from src.metrics import probability_metrics
from src.model.raceformer import RaceFormerTop3
from src.prediction import market_rank_scores
from src.raceformer_preprocessing import transform_raceformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict current races with RaceFormerTop3 (no historical context)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument(
        "--checkpoint-dir", type=Path,
        help="Backtest every compatible checkpoint in this directory.",
    )
    parser.add_argument(
        "--checkpoint-pattern", default="*.pt",
        help="Glob used with --checkpoint-dir (default: *.pt).",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--features-json", type=Path,
        help=(
            "Optional feature manifest. Ordered features must match each checkpoint; "
            "zeroed_features override prediction without modifying checkpoints."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--race-id", type=int)
    mode.add_argument("--backtest", action="store_true")
    parser.add_argument(
        "--backtest-view", default=VALIDATION_ROWS_VIEW,
        help=f"Completed labelled view used for backtesting (default: {VALIDATION_ROWS_VIEW}).",
    )
    parser.add_argument("--backtest-max-races", type=int, default=0)
    parser.add_argument(
        "--backtest-cohort", choices=("validation", "test"),
        default="validation",
        help=(
            "Use checkpoint validation races or an explicitly sealed test cohort "
            "(default: validation)."
        ),
    )
    parser.add_argument(
        "--competition-id", type=int,
        help=(
            "Backtest all complete finished races from this competition, including "
            "races that may have been used for training."
        ),
    )
    parser.add_argument(
        "--layoff-cohort-report", action="store_true",
        help="Print performance by raw days-since-last-run cohort (single checkpoint).",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def apply_feature_manifest(
    checkpoint: dict[str, Any], path: Path | None
) -> tuple[dict[str, Any], list[str]]:
    features = list(checkpoint.get("raw_feature_columns", checkpoint["feature_columns"]))
    if path is None:
        return checkpoint, features
    manifest_features, manifest_zeroed = load_feature_manifest(path)
    if manifest_features != features:
        checkpoint_only = [name for name in features if name not in manifest_features]
        manifest_only = [name for name in manifest_features if name not in features]
        order_only = not checkpoint_only and not manifest_only
        detail = (
            "same columns but different order" if order_only else
            f"checkpoint_only={checkpoint_only} manifest_only={manifest_only}"
        )
        raise ValueError(
            "--features-json is incompatible with checkpoint feature contract: "
            + detail
        )
    overridden = dict(checkpoint)
    overridden["zeroed_features"] = manifest_zeroed
    return overridden, features


def _backtest_frame(
    args: argparse.Namespace, checkpoint: dict[str, Any], features: list[str]
) -> tuple[pd.DataFrame, str]:
    if args.competition_id is not None:
        frame = load_competition_backtest(
            args.db, args.competition_id, features, args.backtest_max_races
        )
        return frame, "all_finished_competition_races_in_sample_allowed"
    if checkpoint.get("partition", {}).get("mode") == "full_data_fit":
        raise ValueError(
            "Full-data-fit checkpoint has no held-out backtest cohort; evaluate it "
            "only on races collected after the refit cutoff"
        )
    cohort = getattr(args, "backtest_cohort", "validation")
    saved_ids = checkpoint.get("partition", {}).get(f"{cohort}_race_ids")
    if saved_ids is not None:
        frame = load_checkpoint_backtest(
            args.db, list(map(int, saved_ids)), features,
            args.backtest_max_races, args.competition_id,
        )
        return frame, f"checkpoint_{cohort}_races"
    if cohort == "test":
        raise ValueError("Checkpoint has no sealed test cohort")
    frame = load_backtest(
        args.db, args.backtest_view, features, args.backtest_max_races,
        args.competition_id,
    )
    return frame, args.backtest_view


def _backtest_metrics(scored: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    target = pd.to_numeric(scored["top3_mask"], errors="raise").to_numpy(
        dtype=np.int64
    )
    race_ids = scored["race_id"].to_numpy(dtype=np.int64)
    metrics = probability_metrics(
        target, scored["probability"].to_numpy(dtype=np.float64), race_ids
    )
    market_probability = market_rank_scores(
        pd.to_numeric(scored["fluc2"], errors="coerce").to_numpy()
    )
    return metrics, probability_metrics(target, market_probability, race_ids)


def layoff_cohort_report(scored: pd.DataFrame) -> pd.DataFrame:
    """Measure selection bias and ranking quality by raw layoff regime."""
    days = pd.to_numeric(scored["recent_days_since_last_run"], errors="coerce")
    cohort = pd.Series("missing", index=scored.index, dtype=object)
    cohort.loc[days < 30] = "0-29"
    cohort.loc[(days >= 30) & (days < 60)] = "30-59"
    cohort.loc[(days >= 60) & (days < 90)] = "60-89"
    cohort.loc[(days >= 90) & (days < 180)] = "90-179"
    cohort.loc[(days >= 180) & (days < 365)] = "180-364"
    cohort.loc[days >= 365] = "365+"
    work = scored.copy()
    work["layoff_cohort"] = cohort
    work["actual_top3"] = pd.to_numeric(work["top3_mask"], errors="raise").astype(int)
    work["predicted_top3"] = (work["model_rank"] <= 3).astype(int)
    discounts = 1.0 / np.log2(np.arange(2, 5, dtype=np.float64))
    rows = []
    for label in ("0-29", "30-59", "60-89", "90-179", "180-364", "365+", "missing"):
        selected = work.loc[work["layoff_cohort"] == label]
        if selected.empty:
            continue
        positives = int(selected["actual_top3"].sum())
        hits = int((selected["actual_top3"] & selected["predicted_top3"]).sum())
        pairwise_correct = 0.0
        pairwise_count = 0
        ndcg_values = []
        for race_id, race in work.groupby("race_id", sort=False):
            cohort_race = race.loc[race["layoff_cohort"] == label]
            cohort_positives = cohort_race.loc[cohort_race["actual_top3"] == 1]
            negatives = race.loc[race["actual_top3"] == 0, "probability"].to_numpy()
            for probability in cohort_positives["probability"]:
                differences = float(probability) - negatives
                pairwise_correct += float(
                    np.sum(differences > 0) + 0.5 * np.sum(differences == 0)
                )
                pairwise_count += len(negatives)
            if len(cohort_positives):
                ranked = race.sort_values("model_rank").head(3)
                gains = (
                    (ranked["actual_top3"] == 1)
                    & (ranked["layoff_cohort"] == label)
                ).to_numpy(dtype=np.float64)
                ideal_count = min(3, len(cohort_positives))
                ndcg_values.append(
                    float(np.dot(gains, discounts) / discounts[:ideal_count].sum())
                )
        probability = np.clip(
            selected["probability"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7
        )
        target = selected["actual_top3"].to_numpy(dtype=np.float64)
        rows.append({
            "layoff": label,
            "races": selected["race_id"].nunique(),
            "runners": len(selected),
            "actual_top3_rate": target.mean(),
            "predicted_top3_rate": selected["predicted_top3"].mean(),
            "top3_recall": hits / positives if positives else float("nan"),
            "cohort_ndcg3": float(np.mean(ndcg_values)) if ndcg_values else float("nan"),
            "pairwise_accuracy": (
                pairwise_correct / pairwise_count if pairwise_count else float("nan")
            ),
            "logloss": float(
                -(target * np.log(probability) + (1 - target) * np.log(1 - probability)).mean()
            ),
        })
    return pd.DataFrame(rows)


def backtest_directory(args: argparse.Namespace, device: torch.device) -> None:
    assert args.checkpoint_dir is not None
    if not args.backtest:
        raise ValueError("--checkpoint-dir requires --backtest")
    if args.output is not None:
        raise ValueError("--output is only supported with one --checkpoint")
    if args.layoff_cohort_report:
        raise ValueError("--layoff-cohort-report requires one --checkpoint")
    directory = args.checkpoint_dir.resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    paths = sorted(path for path in directory.glob(args.checkpoint_pattern) if path.is_file())
    if not paths:
        raise ValueError(
            f"No checkpoints match {args.checkpoint_pattern!r} in {directory}"
        )
    rows = []
    skipped = []
    loaded = []
    unbounded_args = argparse.Namespace(**vars(args))
    unbounded_args.backtest_max_races = 0
    for path in paths:
        try:
            model, checkpoint = load_checkpoint(path, device)
            checkpoint, features = apply_feature_manifest(
                checkpoint, args.features_json
            )
            frame, source = _backtest_frame(unbounded_args, checkpoint, features)
            loaded.append((path, model, checkpoint, frame, source))
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            skipped.append((path.name, str(error)))
            print(f"skipped={path.name} reason={error}", flush=True)
    if not loaded:
        raise RuntimeError("No compatible RaceFormer checkpoints could be loaded")

    common_ids = set(map(int, loaded[0][3]["race_id"].unique()))
    for _, _, _, frame, _ in loaded[1:]:
        common_ids &= set(map(int, frame["race_id"].unique()))
    if not common_ids:
        raise ValueError(
            "Compatible checkpoints have no common backtest races after filtering"
        )
    ordered_common = [
        int(race_id) for race_id in loaded[0][3]["race_id"].drop_duplicates()
        if int(race_id) in common_ids
    ]
    if args.backtest_max_races:
        ordered_common = ordered_common[-args.backtest_max_races:]
    common_ids = set(ordered_common)
    print(
        f"common_backtest_cohort races={len(common_ids)} checkpoints={len(loaded)}",
        flush=True,
    )

    for path, model, checkpoint, frame, source in loaded:
        try:
            common_frame = frame.loc[frame["race_id"].isin(common_ids)].copy()
            scored = score_frame(model, checkpoint, common_frame, device)
            metrics, _ = _backtest_metrics(scored)
            config = checkpoint["model_config"]
            rows.append({
                "checkpoint": path.name,
                "variant": config["variant"],
                "model_dim": config["model_dim"],
                "layers": config["layers"],
                "best_epoch": checkpoint.get("best_epoch", "-"),
                "cohort": source,
                **metrics,
            })
            print(
                f"backtested={path.name} races={metrics['complete_races']} "
                f"top3={metrics['top3_recall']:.4f} ndcg3={metrics['ndcg3']:.4f}",
                flush=True,
            )
        except (ValueError, KeyError, RuntimeError, OSError) as error:
            skipped.append((path.name, str(error)))
            print(f"skipped={path.name} reason={error}", flush=True)
    if not rows:
        raise RuntimeError("No compatible RaceFormer checkpoints could be backtested")
    summary = pd.DataFrame(rows).sort_values(
        ["top3_recall", "ndcg3", "pairwise_ranking_accuracy"],
        ascending=False, kind="stable",
    )
    print("\nRACEFORMER DIRECTORY BACKTEST")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if skipped:
        print(f"skipped_checkpoints={len(skipped)}")


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[RaceFormerTop3, dict[str, Any]]:
    checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "raceformer_top3":
        raise ValueError(f"{path} is not a RaceFormerTop3 checkpoint")
    model = RaceFormerTop3(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _columns(features: list[str]) -> list[str]:
    return list(dict.fromkeys([
        "race_id", "start_time_iso", "competition_id", "race_number",
        "race_name", "runner_number", "runner_name", "fluc2", "top3_mask",
        "finish_place", *features,
    ]))


def load_race(db: Path, race_id: int, features: list[str]) -> pd.DataFrame:
    columns = _columns(features)
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
            "FROM race_runners WHERE race_id = ? ORDER BY runner_number",
            connection,
            params=(race_id,),
        )
    finally:
        connection.close()
    if frame.empty:
        raise ValueError(f"Race {race_id} was not found")
    return frame


def load_backtest(
    db: Path, view: str, features: list[str], maximum: int,
    competition_id: int | None = None,
) -> pd.DataFrame:
    if maximum < 0:
        raise ValueError("--backtest-max-races must be zero or positive")
    columns = _columns(features)
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        require_rows_view(connection, view)
        competition_filter = " AND competition_id = ?" if competition_id is not None else ""
        frame = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
            f"FROM {quote_identifier(view)} WHERE status = 'finished' "
            f"{competition_filter} "
            "ORDER BY start_time_iso, race_id, runner_number",
            connection,
            params=(() if competition_id is None else (competition_id,)),
        )
    finally:
        connection.close()
    complete_ids = []
    for race_id, race in frame.groupby("race_id", sort=False):
        labels = pd.to_numeric(race["top3_mask"], errors="coerce")
        if len(race) >= 4 and labels.notna().all() and int(labels.sum()) == 3:
            complete_ids.append(int(race_id))
    if maximum:
        complete_ids = complete_ids[-maximum:]
    frame = frame.loc[frame["race_id"].isin(complete_ids)].copy()
    if frame.empty:
        suffix = "" if competition_id is None else f" for competition {competition_id}"
        raise ValueError(f"No complete backtest races found in {view}{suffix}")
    return frame


def load_competition_backtest(
    db: Path, competition_id: int, features: list[str], maximum: int
) -> pd.DataFrame:
    """Load all complete labelled races for a competition, including training races."""
    if maximum < 0:
        raise ValueError("--backtest-max-races must be zero or positive")
    columns = _columns(features)
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
            "FROM race_runners WHERE competition_id = ? AND status = 'finished' "
            "AND top3_mask IN (0, 1) "
            "ORDER BY start_time_iso, race_id, runner_number",
            connection,
            params=(competition_id,),
        )
    finally:
        connection.close()
    complete_ids = []
    for race_id, race in frame.groupby("race_id", sort=False):
        labels = pd.to_numeric(race["top3_mask"], errors="coerce")
        if len(race) >= 4 and labels.notna().all() and int(labels.sum()) == 3:
            complete_ids.append(int(race_id))
    if maximum:
        complete_ids = complete_ids[-maximum:]
    frame = frame.loc[frame["race_id"].isin(complete_ids)].copy()
    if frame.empty:
        raise ValueError(
            f"No complete finished races found for competition {competition_id}"
        )
    return frame


def load_checkpoint_backtest(
    db: Path, race_ids: list[int], features: list[str], maximum: int,
    competition_id: int | None = None,
) -> pd.DataFrame:
    """Load the exact chronological validation cohort saved in a checkpoint."""
    if maximum < 0:
        raise ValueError("--backtest-max-races must be zero or positive")
    selected = race_ids
    if not selected:
        raise ValueError("Checkpoint validation cohort is empty")
    columns = _columns(features)
    placeholders = ", ".join("?" for _ in selected)
    connection = sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True)
    try:
        competition_filter = " AND competition_id = ?" if competition_id is not None else ""
        params = [*selected]
        if competition_id is not None:
            params.append(competition_id)
        frame = pd.read_sql_query(
            f"SELECT {', '.join(quote_identifier(column) for column in columns)} "
            f"FROM race_runners WHERE race_id IN ({placeholders}) "
            f"AND top3_mask IN (0, 1){competition_filter} "
            "ORDER BY start_time_iso, race_id, runner_number",
            connection,
            params=params,
        )
    finally:
        connection.close()
    if frame.empty:
        suffix = "" if competition_id is None else f" for competition {competition_id}"
        raise ValueError(f"Checkpoint validation cohort contains no races{suffix}")
    found = set(map(int, frame["race_id"].unique()))
    missing = sorted(set(selected) - found)
    if missing and competition_id is None:
        raise ValueError(
            f"Database is missing {len(missing)} checkpoint validation races"
        )
    if maximum:
        keep = frame["race_id"].drop_duplicates().tail(maximum)
        frame = frame.loc[frame["race_id"].isin(keep)].copy()
    return frame


def score_frame(
    model: RaceFormerTop3, checkpoint: dict[str, Any], frame: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    features = list(checkpoint.get("raw_feature_columns", checkpoint["feature_columns"]))
    zeroed = list(checkpoint.get("zeroed_features", []))
    results = []
    with torch.inference_mode():
        for _, race in frame.groupby("race_id", sort=False):
            raw = race.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(
                dtype=np.float32
            )
            race_ids = race["race_id"].to_numpy(dtype=np.int64)
            values = transform_raceformer(
                raw, race_ids, features, zeroed, checkpoint.get("preprocessing"),
                legacy_median=np.asarray(checkpoint["median"], dtype=np.float32),
                legacy_scale=np.asarray(checkpoint["scale"], dtype=np.float32),
            )
            x = torch.from_numpy(values).unsqueeze(0).to(device)
            valid = torch.ones((1, len(race)), dtype=torch.bool, device=device)
            logits, anchor_logits, residual_logits = model.forward_parts(x, valid)
            probability = torch.sigmoid(logits[0]).cpu().numpy()
            scored = race.copy()
            scored["probability"] = probability
            if model.variant == "market_residual":
                scored["anchor_logit"] = anchor_logits[0].cpu().numpy()
                scored["residual_logit"] = residual_logits[0].cpu().numpy()
            scored["model_rank"] = (
                scored["probability"].rank(method="first", ascending=False).astype(int)
            )
            market = pd.to_numeric(scored["fluc2"], errors="coerce").to_numpy()
            market_score = market_rank_scores(market)
            market_order = np.argsort(-market_score, kind="stable")
            market_rank = np.empty(len(market_order), dtype=np.int64)
            market_rank[market_order] = np.arange(1, len(market_order) + 1)
            scored["market_rank"] = market_rank
            results.append(scored)
    return pd.concat(results, ignore_index=True)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.backtest and args.competition_id is not None:
        print(
            "WARNING competition_backtest_includes_training_races=yes "
            f"competition_id={args.competition_id}",
            flush=True,
        )
    if args.checkpoint_dir is not None:
        backtest_directory(args, device)
        return
    assert args.checkpoint is not None
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint, features = apply_feature_manifest(checkpoint, args.features_json)
    if args.features_json is not None:
        print(
            f"feature_manifest={args.features_json.resolve()} "
            f"features={len(features)} "
            f"zeroed={len(checkpoint.get('zeroed_features', []))} "
            "checkpoint_modified=no prediction_override=yes",
            flush=True,
        )
    if args.race_id is not None:
        frame = load_race(args.db, args.race_id, features)
    else:
        frame, source = _backtest_frame(args, checkpoint, features)
        print(
            f"backtest_source={source} races={frame['race_id'].nunique()}", flush=True,
        )
    scored = score_frame(model, checkpoint, frame, device)
    if args.backtest:
        metrics, market_metrics = _backtest_metrics(scored)
        print("RACEFORMER BACKTEST")
        print(pd.DataFrame([metrics, market_metrics], index=["model", "market"]).to_string())
        if args.layoff_cohort_report:
            print("\nLAYOFF COHORT REPORT")
            print(
                layoff_cohort_report(scored).to_string(
                    index=False, float_format=lambda value: f"{value:.4f}"
                )
            )
    else:
        race = scored.iloc[0]
        print(
            "RACEFORMER TOP3\n"
            f"checkpoint={args.checkpoint.resolve()} variant={model.variant}\n"
            f"race={int(race['race_id'])} {race['race_name']} competition="
            f"{int(race['competition_id'])} start={race['start_time_iso']}\n"
            "historical_context=OFF icl=OFF"
        )
        shown = scored.sort_values(["model_rank", "runner_number"])[[
            "runner_number", "runner_name", "probability", "model_rank",
            "fluc2", "market_rank", "finish_place",
        ]]
        print(shown.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        print(f"sum_probability={scored['probability'].sum():.4f}")
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(args.output.resolve(), index=False)
        print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()
