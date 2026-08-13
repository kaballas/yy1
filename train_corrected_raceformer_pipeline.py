#!/usr/bin/env python3
"""Train comparable corrected-feature RaceFormer models from one data snapshot.

The first model is market-residual but is allowed materially larger corrections.
The second model is an unanchored race-token model with current-race market inputs
zeroed. Both manifests have exactly the same ordered feature columns, so the
second training run safely reuses the CSVs exported by the first run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from src.config import DEFAULT_DB


# Distinct candidates from the 300-race chronological winner audit. The omitted
# recent_class_weighted_6 is exactly equal to form_class_level_weighted_6, whose
# canonical existing name is activated below.
CORRECTED_CANDIDATE_FEATURES = (
    "current_weight_minus_recent_weighted_avg",
    "recent_6_barrier",
    "dry_rating_rank",
    "historical_market_overperformance_weighted_3_zscore_in_race",
    "historical_market_overperformance_slope_3",
    "speed_rating_rank",
    "distance_minus_recent_avg",
    "recent_2_weight_kg",
    "jockey_trainer_synergy",
    "recent_barrier_percentile_weighted_6",
    "heavy_starts",
    "recent_avg_place",
    "similar_distance_margin_quality_avg",
    "goodGroundPro",
    "current_form_strength",
    "first_up_win_rate_rank",
    "form_class_level_weighted_6",
    "recent_best_speed_rank",
    "historical_market_overperformance_weighted_3_rank_in_race",
    "similar_distance_runs",
)


def is_current_market_feature(name: str) -> bool:
    """Match current-race prices and transforms, not historical market form."""
    lowered = name.casefold()
    return (
        name in {"open_price", "fluc1", "fluc2"}
        or lowered.startswith((
            "open_price", "fluc1", "fluc2", "market_open_", "market_fluc1_",
            "market_total_", "market_price_", "market_implied_prob_",
            "market_steam", "race_overlay", "race_consensus", "race_signal_",
        ))
    )


def _read_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    zeroed = payload.get("zeroed_features")
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError(f"{path} has an invalid features list")
    if not isinstance(zeroed, list) or not all(isinstance(x, str) for x in zeroed):
        raise ValueError(f"{path} has an invalid zeroed_features list")
    return payload


def _database_columns(database: Path) -> set[str]:
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        return {str(row[1]) for row in connection.execute(
            'PRAGMA table_info("race_runners")'
        )}


def build_manifests(
    base_path: Path,
    residual_path: Path,
    unanchored_path: Path,
    database_columns: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Write compatible residual/unanchored manifests and return active lists."""
    payload = _read_manifest(base_path)
    features = list(dict.fromkeys(payload["features"]))  # type: ignore[arg-type]
    missing_database = sorted(
        feature for feature in CORRECTED_CANDIDATE_FEATURES
        if feature not in database_columns
    )
    if missing_database:
        raise ValueError(
            "Run update_derived_racing_features.py first; database is missing: "
            + ", ".join(missing_database)
        )
    for feature in CORRECTED_CANDIDATE_FEATURES:
        if feature not in features:
            features.append(feature)

    base_zeroed = set(payload["zeroed_features"])  # type: ignore[arg-type]
    candidate_set = set(CORRECTED_CANDIDATE_FEATURES)
    residual_zeroed = [
        feature for feature in features
        if feature in base_zeroed and feature not in candidate_set
    ]
    unanchored_zeroed = list(dict.fromkeys([
        *residual_zeroed,
        *(feature for feature in features if is_current_market_feature(feature)),
    ]))
    residual_payload = {
        **payload,
        "label": "top3_mask",
        "features": features,
        "zeroed_features": residual_zeroed,
        "profile": "corrected_market_residual",
    }
    unanchored_payload = {
        **payload,
        "label": "top3_mask",
        "features": features,
        "zeroed_features": unanchored_zeroed,
        "profile": "corrected_unanchored",
    }
    for path, content in (
        (residual_path, residual_payload),
        (unanchored_path, unanchored_payload),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    residual_active = [x for x in features if x not in set(residual_zeroed)]
    unanchored_active = [x for x in features if x not in set(unanchored_zeroed)]
    return features, residual_active, unanchored_active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--base-features-json", type=Path, default=Path("tabfm_features.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/corrected_raceformer"))
    parser.add_argument("--training-csv", type=Path, default=Path("outputs/corrected_raceformer_training.csv"))
    parser.add_argument("--validation-csv", type=Path, default=Path("outputs/corrected_raceformer_validation.csv"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--chronological-validation-races", type=int, default=1000)
    parser.add_argument("--chronological-test-races", type=int, default=1000)
    parser.add_argument("--races-per-batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-feature-update", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build/validate manifests and print commands without training.",
    )
    return parser.parse_args()


def _run(command: list[str], dry_run: bool) -> None:
    print("command=" + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def _common_training_args(args: argparse.Namespace) -> list[str]:
    return [
        "--db", str(args.db.resolve()),
        "--training-csv", str(args.training_csv.resolve()),
        "--validation-csv", str(args.validation_csv.resolve()),
        "--standardized-clip", "5",
        "--hidden-dim", "128", "--model-dim", "64",
        "--attention-heads", "4", "--race-layers", "1",
        "--feedforward-dim", "128", "--dropout", "0.15",
        "--races-per-batch", str(args.races_per_batch),
        "--learning-rate", str(args.learning_rate), "--weight-decay", "0.001",
        "--ranking-loss-weight", "2", "--cardinality-loss-weight", "0.5",
        "--listwise-loss-weight", "0.5", "--bce-loss-weight", "0.5",
        "--max-grad-norm", "0.5", "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--checkpoint-metric", "composite",
        "--chronological-validation-races", str(args.chronological_validation_races),
        "--chronological-test-races", str(args.chronological_test_races),
        "--seed", str(args.seed), "--device", args.device,
    ]


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.early_stopping_patience < 1:
        raise ValueError("epochs and patience must be positive")
    database = args.db.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_feature_update:
        _run([
            sys.executable, str(Path(__file__).with_name("update_derived_racing_features.py")),
            "--db", str(database),
        ], args.dry_run)

    residual_manifest = output_dir / "corrected_market_residual_features.json"
    unanchored_manifest = output_dir / "corrected_unanchored_features.json"
    features, residual_active, unanchored_active = build_manifests(
        args.base_features_json.resolve(), residual_manifest, unanchored_manifest,
        _database_columns(database),
    )
    print(
        f"manifest_features={len(features)} residual_active={len(residual_active)} "
        f"unanchored_active={len(unanchored_active)}\n"
        f"residual_manifest={residual_manifest}\n"
        f"unanchored_manifest={unanchored_manifest}",
        flush=True,
    )

    training_script = str(Path(__file__).with_name("train_raceformer.py"))
    common = _common_training_args(args)
    residual_checkpoint = output_dir / "raceformer_corrected_market_residual.pt"
    unanchored_checkpoint = output_dir / "raceformer_corrected_unanchored.pt"
    residual_command = [
        sys.executable, training_script, *common,
        "--features-json", str(residual_manifest),
        "--output", str(residual_checkpoint),
        "--variant", "market_residual",
        "--market-residual-scale", "1.0",
        "--market-residual-weight", "0.005",
        "--market-error-loss-weight", "1.0",
        "--market-correct-stability-weight", "0.05",
        "--market-uninvolved-stability-weight", "0.10",
    ]
    unanchored_command = [
        sys.executable, training_script, *common,
        "--features-json", str(unanchored_manifest),
        "--output", str(unanchored_checkpoint),
        "--variant", "race_token", "--market-residual-weight", "0",
        "--no-export",
    ]
    _run(residual_command, args.dry_run)
    _run(unanchored_command, args.dry_run)

    if not args.skip_backtest:
        prediction_script = str(Path(__file__).with_name("predict_raceformer.py"))
        for label, checkpoint in (
            ("market_residual", residual_checkpoint),
            ("unanchored", unanchored_checkpoint),
        ):
            _run([
                sys.executable, prediction_script,
                "--checkpoint", str(checkpoint), "--db", str(database),
                "--backtest", "--backtest-cohort", "test", "--device", args.device,
                "--output", str(output_dir / f"{label}_sealed_test_predictions.csv"),
            ], args.dry_run)

    print(
        "pipeline_complete=" + ("dry_run" if args.dry_run else "yes") + "\n"
        f"rank_command={sys.executable} "
        f"{Path(__file__).with_name('rank_raceformer_models.py')} --race-id RACE_ID "
        f"--market-checkpoint {residual_checkpoint} "
        f"--unanchored-checkpoint {unanchored_checkpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()
