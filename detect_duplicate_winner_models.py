#!/usr/bin/env python3
"""Detect duplicate feature groups in a winner-ranker feature manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_FEATURES = Path(__file__).with_name("winner_ranker_features.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    return parser.parse_args()


def load_feature_groups(path: Path) -> dict[str, tuple[str, ...]]:
    """Load model feature groups and validate the subset needed for comparison."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Feature manifest does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{resolved} must contain a non-empty models object")

    groups: dict[str, tuple[str, ...]] = {}
    for label, model in models.items():
        features = model.get("features") if isinstance(model, dict) else None
        if (
            not isinstance(label, str)
            or not isinstance(features, list)
            or not features
            or not all(isinstance(feature, str) and feature for feature in features)
        ):
            raise ValueError(
                f"{resolved} models.{label}.features must be a non-empty string list"
            )
        if len(features) != len(set(features)):
            raise ValueError(f"{resolved} models.{label}.features contains duplicates")
        groups[label] = tuple(features)
    return groups


def duplicate_groups(
    groups: dict[str, tuple[str, ...]],
) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    """Return exact duplicates and order-only duplicates, respectively."""
    by_order: dict[tuple[str, ...], list[str]] = defaultdict(list)
    by_feature_set: dict[frozenset[str], list[str]] = defaultdict(list)
    for label, features in groups.items():
        by_order[features].append(label)
        by_feature_set[frozenset(features)].append(label)

    exact = sorted(
        tuple(labels) for labels in by_order.values() if len(labels) > 1
    )
    reordered = sorted(
        tuple(labels)
        for feature_set, labels in by_feature_set.items()
        if len(labels) > 1
        and len({groups[label] for label in labels}) > 1
        and len(feature_set) == len(groups[labels[0]])
    )
    return exact, reordered


def format_group(labels: tuple[str, ...], groups: dict[str, tuple[str, ...]]) -> str:
    return f"  {', '.join(labels)} ({len(groups[labels[0]])} features)"


def main() -> None:
    args = parse_args()
    groups = load_feature_groups(args.features_json)
    exact, reordered = duplicate_groups(groups)
    if not exact and not reordered:
        print(f"No duplicate model groups found across {len(groups)} models.")
        return

    print("Duplicate winner-ranker model groups found:")
    if exact:
        print("Exact ordered feature lists:")
        print("\n".join(format_group(labels, groups) for labels in exact))
    if reordered:
        print("Equivalent feature sets with different ordering:")
        print("\n".join(format_group(labels, groups) for labels in reordered))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
