#!/usr/bin/env python3
"""Generate a strict feature-to-expert mapping for the feature-mapped MoE model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import load_feature_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shared-features", type=str, nargs="*", default=[])
    parser.add_argument(
        "--include-market-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep market and pricing-derived features in the map.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.experts < 1:
        raise ValueError("--experts must be >= 1")
    features, _ = load_feature_manifest(args.features_json)
    from src.race_moe_data import market_blind_features
    features, _ = market_blind_features(features, include_market=args.include_market_features)
    shared = set(args.shared_features)
    feature_names = [name for name in features if name not in shared]
    assignment = {str(i): [] for i in range(args.experts)}
    for index, name in enumerate(feature_names):
        expert_id = index % args.experts
        assignment[str(expert_id)].append(name)
    for expert_id in range(args.experts):
        assignment[str(expert_id)].extend(sorted(shared))
    payload = {"shared_features": sorted(shared), "experts": assignment}
    if args.output is None:
        print(json.dumps(payload, indent=2))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
