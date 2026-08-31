#!/usr/bin/env python3
"""Print the unique feature names contained in a MoE feature-map JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_FEATURE_MAP = Path("configs/race_moe_feature_map.json")


def unique_features(payload: Any) -> list[str]:
    """Return shared, router, and expert features once, in first-seen order."""
    if not isinstance(payload, dict):
        raise ValueError("feature-map JSON must contain an object")

    groups: list[tuple[str, Any]] = [
        ("shared_features", payload.get("shared_features", [])),
        ("router_features", payload.get("router_features", [])),
    ]
    experts = payload.get("experts", {})
    if not isinstance(experts, dict):
        raise ValueError("'experts' must be an object mapping expert IDs to feature lists")
    groups.extend((f"experts.{expert_id}", features) for expert_id, features in experts.items())

    result: list[str] = []
    seen: set[str] = set()
    for group_name, features in groups:
        if not isinstance(features, list):
            raise ValueError(f"'{group_name}' must be a list")
        for feature in features:
            if not isinstance(feature, str) or not feature:
                raise ValueError(f"'{group_name}' contains an invalid feature name: {feature!r}")
            if feature not in seen:
                seen.add(feature)
                result.append(feature)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "feature_map",
        nargs="?",
        type=Path,
        default=DEFAULT_FEATURE_MAP,
        help=f"feature-map JSON path (default: {DEFAULT_FEATURE_MAP})",
    )
    parser.add_argument("--sort", action="store_true", help="sort features alphabetically")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.feature_map.read_text())
    features = unique_features(payload)
    if args.sort:
        features.sort()
    print(json.dumps(features, indent=2))


if __name__ == "__main__":
    main()
