#!/usr/bin/env python3
"""Create a feature-map MoE config with one expert per configured feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.dataset import load_feature_manifest
from src.race_moe_data import market_blind_features


DEFAULT_INPUT = Path("configs/race_moe_feature_map.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Existing feature-map JSON to transform.",
    )
    parser.add_argument(
        "--features-json",
        type=Path,
        help=(
            "Restrict experts to the effective features in this training "
            "manifest."
        ),
    )
    parser.add_argument(
        "--include-market-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the trainer's market-feature filtering mode.",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help=(
            "Destination JSON. By default, writes beside the input with "
            "'.one_feature_per_expert' added to its name."
        ),
    )
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the input atomically instead of creating a separate file.",
    )
    parser.add_argument(
        "--keep-shared",
        action="store_true",
        help=(
            "Keep shared_features shared across all experts and create one "
            "expert per non-shared feature. By default, shared features also "
            "become individual experts."
        ),
    )
    parser.add_argument(
        "--exclude-feature",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Feature to omit from the generated map. Repeat this option for "
            "multiple unavailable features."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def _feature_list(value: Any, location: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    if not all(isinstance(feature, str) and feature for feature in value):
        raise ValueError(f"{location} must contain non-empty feature names")
    return value


def one_feature_per_expert(
    payload: dict[str, Any],
    *,
    keep_shared: bool = False,
    excluded_features: set[str] | None = None,
    allowed_features: set[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic map containing one unique feature per expert."""
    if not isinstance(payload, dict):
        raise ValueError("Feature-map JSON must be an object")

    excluded = excluded_features or set()
    shared = [
        feature
        for feature in _feature_list(
            payload.get("shared_features", []), "shared_features"
        )
        if feature not in excluded
        and (allowed_features is None or feature in allowed_features)
    ]
    raw_experts = payload.get("experts")
    if not isinstance(raw_experts, dict) or not raw_experts:
        raise ValueError("Feature-map JSON must contain a non-empty experts object")

    configured: list[str] = []
    seen: set[str] = set()

    def add(features: list[str]) -> None:
        for feature in features:
            if (
                feature not in excluded
                and (allowed_features is None or feature in allowed_features)
                and feature not in seen
            ):
                seen.add(feature)
                configured.append(feature)

    add(shared)
    for expert_id, features in raw_experts.items():
        add(_feature_list(features, f"experts.{expert_id}"))

    shared_set = set(shared)
    expert_features = (
        [feature for feature in configured if feature not in shared_set]
        if keep_shared
        else configured
    )
    if not expert_features:
        raise ValueError("No features remain to create individual experts")

    return {
        "shared_features": shared if keep_shared else [],
        "experts": {
            str(index): [feature]
            for index, feature in enumerate(expert_features)
        },
    }


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(
        f"{input_path.stem}.one_feature_per_expert{input_path.suffix}"
    )


def write_json(path: Path, payload: dict[str, Any], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(
            f"Refusing to replace existing output {path}; pass --force"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    allowed_features = None
    if args.features_json is not None:
        manifest_features, _ = load_feature_manifest(args.features_json.resolve())
        effective_features, _ = market_blind_features(
            manifest_features,
            include_market=args.include_market_features,
        )
        allowed_features = set(effective_features)
    output_payload = one_feature_per_expert(
        payload,
        keep_shared=args.keep_shared,
        excluded_features=set(args.exclude_feature),
        allowed_features=allowed_features,
    )
    output_path = (
        input_path
        if args.in_place
        else (args.output.resolve() if args.output else default_output_path(input_path))
    )
    write_json(
        output_path,
        output_payload,
        replace=args.in_place or args.force,
    )

    expert_count = len(output_payload["experts"])
    print(f"wrote={output_path}")
    print(f"experts={expert_count}")
    print(f"shared_features={len(output_payload['shared_features'])}")
    print(f"excluded_features={len(set(args.exclude_feature))}")
    if allowed_features is not None:
        print(f"allowed_manifest_features={len(allowed_features)}")
    print(f"training_argument=--moe-num-experts {expert_count}")


if __name__ == "__main__":
    main()
