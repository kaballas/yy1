#!/usr/bin/env python3
"""Build one market-base-plus-one-feature model per numeric database feature."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from feature_hinter import candidate_features, database_schema
from src.config import DEFAULT_DB

BASE_FEATURES = [
"distance_m",
  "draw_number",
  "field_size",
"jockey_trainer_history_smoothed_win_rate"

  
]

# Optional permanent exclusions. Add race_runners feature names here when they
# should be neither a base feature nor an independently tested tN feature.
EXCLUDED_FEATURES: list[str] = [
    "fluc1",
    "fluc2",
    "fluc2_price_rank",
    "fluc1_price_rank",
    "open_price",
    "race_consensus_rank",
    "race_consensus_score",
    "market_fluc1_to_fluc2_move",
    "open_price_rank",
    "market_implied_prob_change_open_to_fluc2",
    "market_total_abs_movement",
    "market_implied_prob_change_fluc1_to_fluc2",
    "market_open_to_fluc1_move",
]


def parse_feature_list(value: str) -> list[str]:
    features = [feature.strip() for feature in value.split(",")]
    if not features or any(not feature for feature in features):
        raise argparse.ArgumentTypeError(
            "features must be a comma-separated list of column names"
        )
    return list(dict.fromkeys(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=Path("test.json"))
    parser.add_argument(
        "--exclude-features",
        type=parse_feature_list,
        metavar="FEATURE[,FEATURE...]",
        help=(
            "Additional race_runners features to omit from all tN tests. "
            "These are combined with EXCLUDED_FEATURES in this script."
        ),
    )
    return parser.parse_args()


def build_manifest(
    database: Path,
    base_features: list[str] | None = None,
    excluded_features: list[str] | None = None,
) -> dict[str, object]:
    base_features = list(BASE_FEATURES if base_features is None else base_features)
    excluded_features = list(
        EXCLUDED_FEATURES if excluded_features is None else excluded_features
    )
    if not base_features:
        raise ValueError("At least one base feature is required")
    duplicates = sorted(
        {name for name in base_features if base_features.count(name) > 1}
    )
    if duplicates:
        raise ValueError("Duplicate base features: " + ", ".join(duplicates))
    excluded_features = list(dict.fromkeys(excluded_features))
    base_exclusion_overlap = sorted(set(base_features) & set(excluded_features))
    if base_exclusion_overlap:
        raise ValueError(
            "Features cannot be both base and excluded: "
            + ", ".join(base_exclusion_overlap)
        )
    database = database.resolve()
    if not database.is_file():
        raise ValueError(f"Database does not exist: {database}")

    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        schema = database_schema(connection)

    column_names = {name for name, _ in schema}
    missing_base = [name for name in base_features if name not in column_names]
    if missing_base:
        raise ValueError("Missing market base features: " + ", ".join(missing_base))
    unknown_exclusions = [
        name for name in excluded_features if name not in column_names
    ]
    if unknown_exclusions:
        raise ValueError(
            "Excluded features are absent from race_runners: "
            + ", ".join(unknown_exclusions)
        )

    omitted = set(base_features) | set(excluded_features)

    additions = [
        feature for feature in candidate_features(schema) if feature not in omitted
    ]
    if not additions:
        raise ValueError("No additional usable numeric race_runners features found")

    models = {
        f"t{index}": {
            "features": [*base_features, feature],
        }
        for index, feature in enumerate(additions, start=1)
    }
    return {
        "schema_version": 1,
        "description": (
            "Market-mover ablation models: every shared base feature plus one "
            "independently tested race_runners feature."
        ),
        "base_features": base_features,
        "excluded_features": excluded_features,
        "models": models,
    }


def main() -> None:
    args = parse_args()
    database = args.db.resolve()
    output = args.output.resolve()
    excluded_features = list(EXCLUDED_FEATURES)
    if args.exclude_features:
        excluded_features.extend(args.exclude_features)
    manifest = build_manifest(
        database,
        excluded_features=list(dict.fromkeys(excluded_features)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    models = manifest["models"]
    assert isinstance(models, dict)
    first_label = next(iter(models))
    last_label = next(reversed(models))
    print(f"database={database}")
    print(f"output={output}")
    print(f"base_features={json.dumps(manifest['base_features'])}")
    print(f"excluded_features={json.dumps(manifest['excluded_features'])}")
    print(f"models={len(models):,} labels={first_label}..{last_label}")
    print(f"first={json.dumps({first_label: models[first_label]})}")
    print(f"last={json.dumps({last_label: models[last_label]})}")


if __name__ == "__main__":
    main()
