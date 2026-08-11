#!/usr/bin/env python3
"""Merge compatible native TabFM top-three checkpoint bundles."""

from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch

from src.model import TabFM


warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "outputs"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "merged_model.pt"
REQUIRED_KEYS = {
    "model_state_dict",
    "model_kwargs",
    "feature_columns",
    "median",
    "scale",
    "label",
}
CONTRACT_METADATA_KEYS = (
    "context_races_per_step",
    "validation_context_races_per_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        type=Path,
        nargs="*",
        help=(
            "Two or more TabFM .pt bundles. When omitted, compatible bundles "
            "are selected from --input-dir."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory scanned for native *.pt bundles (default: outputs).",
    )
    parser.add_argument(
        "--output-file",
        "--output",
        dest="output_file",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Merged native TabFM bundle (default: outputs/merged_model.pt).",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Optional non-negative source weights, in displayed source order.",
    )
    parser.add_argument(
        "--metadata-from",
        choices=("first", "last"),
        default="first",
        help="Source bundle whose non-weight metadata is retained.",
    )
    parser.add_argument(
        "--skip-incompatible",
        action="store_true",
        help="Skip incompatible bundles instead of stopping at the first one.",
    )
    return parser.parse_args()


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("checkpoint is not a native bundle dictionary")
    missing = sorted(REQUIRED_KEYS - set(bundle))
    if missing:
        raise ValueError(f"native TabFM bundle fields are missing: {missing}")
    if bundle["label"] != "top3_mask":
        raise ValueError(f"label must be 'top3_mask', found {bundle['label']!r}")
    if not isinstance(bundle["model_state_dict"], (dict, OrderedDict)):
        raise ValueError("model_state_dict is not a mapping")
    return bundle


def source_paths(args: argparse.Namespace) -> list[Path]:
    output = args.output_file.resolve()
    paths = args.models or sorted(args.input_dir.glob("*.pt"))
    paths = [path.resolve() for path in paths if path.resolve() != output]
    if len(paths) < 2:
        raise ValueError("At least two source TabFM checkpoints are required")
    if len(set(paths)) != len(paths):
        raise ValueError("Source checkpoint paths must be unique")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source checkpoints do not exist: {missing}")
    return paths


def arrays_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def require_same_contract(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> None:
    if dict(reference["model_kwargs"]) != dict(candidate["model_kwargs"]):
        raise ValueError("model_kwargs differ")
    if list(reference["feature_columns"]) != list(candidate["feature_columns"]):
        raise ValueError("feature columns or their order differ")
    for key in ("median", "scale"):
        if not arrays_equal(reference[key], candidate[key]):
            raise ValueError(f"preprocessing field {key!r} differs")
    for key in CONTRACT_METADATA_KEYS:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"inference contract field {key!r} differs")


def validate_model_bundle(bundle: dict[str, Any]) -> tuple[set[str], set[str]]:
    model = TabFM(**dict(bundle["model_kwargs"]))
    model.load_state_dict(bundle["model_state_dict"], strict=True)
    parameter_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    state_names = set(bundle["model_state_dict"])
    unknown = state_names - parameter_names - buffer_names
    if unknown:
        raise ValueError(f"state contains unclassified tensors: {sorted(unknown)[:10]}")
    return parameter_names, buffer_names


def normalise_weights(weights: list[float] | None, count: int) -> list[float]:
    if weights is None:
        return [1.0 / count] * count
    if len(weights) != count:
        raise ValueError(f"Expected {count} weights, received {len(weights)}")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("Weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one weight must be positive")
    return [weight / total for weight in weights]


def require_compatible_state(
    reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]
) -> None:
    if reference.keys() != candidate.keys():
        missing = sorted(reference.keys() - candidate.keys())
        unexpected = sorted(candidate.keys() - reference.keys())
        raise ValueError(
            f"state keys differ: missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    for name, reference_tensor in reference.items():
        candidate_tensor = candidate[name]
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(
                f"tensor {name!r} shape differs: "
                f"{tuple(reference_tensor.shape)} != {tuple(candidate_tensor.shape)}"
            )
        if reference_tensor.dtype != candidate_tensor.dtype:
            raise ValueError(
                f"tensor {name!r} dtype differs: "
                f"{reference_tensor.dtype} != {candidate_tensor.dtype}"
            )


def average_parameters(
    bundles: list[dict[str, Any]],
    weights: list[float],
    parameter_names: set[str],
    buffer_names: set[str],
    buffer_source_index: int,
) -> tuple[OrderedDict[str, torch.Tensor], list[str]]:
    states = [bundle["model_state_dict"] for bundle in bundles]
    reference = states[0]
    for state in states[1:]:
        require_compatible_state(reference, state)

    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    differing_buffers: list[str] = []
    with torch.no_grad():
        for name, reference_tensor in reference.items():
            if name in buffer_names:
                if any(
                    not torch.equal(reference_tensor, state[name])
                    for state in states[1:]
                ):
                    differing_buffers.append(name)
                merged[name] = (
                    states[buffer_source_index][name].detach().cpu().clone()
                )
                continue
            if name not in parameter_names:
                raise ValueError(f"state tensor {name!r} is neither parameter nor buffer")
            if not (reference_tensor.is_floating_point() or reference_tensor.is_complex()):
                for state in states[1:]:
                    if not torch.equal(reference_tensor, state[name]):
                        raise ValueError(f"non-floating parameter {name!r} differs")
                merged[name] = reference_tensor.detach().cpu().clone()
                continue
            accumulation_dtype = (
                torch.complex128 if reference_tensor.dtype == torch.complex128
                else torch.complex64 if reference_tensor.is_complex()
                else torch.float64 if reference_tensor.dtype == torch.float64
                else torch.float32
            )
            accumulator = torch.zeros_like(
                reference_tensor, dtype=accumulation_dtype, device="cpu"
            )
            for state, weight in zip(states, weights):
                accumulator.add_(
                    state[name].detach().cpu().to(accumulation_dtype), alpha=weight
                )
            merged[name] = accumulator.to(reference_tensor.dtype)
    return merged, differing_buffers


def write_metadata(path: Path, bundle: dict[str, Any]) -> None:
    metadata = {
        "model": "src.model.TabFM",
        "output": str(path.resolve()),
        "features": list(bundle["feature_columns"]),
        "label": bundle["label"],
        "selection_metric": bundle["selection_metric"],
        "merge_metadata": bundle["merge_metadata"],
        "context_races_per_step": bundle.get("context_races_per_step"),
        "zeroed_features": bundle.get("zeroed_features", []),
    }
    metadata_path = path.with_suffix(path.suffix + ".json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)


def main() -> int:
    args = parse_args()
    paths = source_paths(args)
    selected_paths: list[Path] = []
    bundles: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    parameter_names: set[str] | None = None
    buffer_names: set[str] | None = None

    for path in paths:
        try:
            bundle = load_bundle(path)
            candidate_parameters, candidate_buffers = validate_model_bundle(bundle)
            if reference is not None:
                require_same_contract(reference, bundle)
                require_compatible_state(
                    reference["model_state_dict"], bundle["model_state_dict"]
                )
                if candidate_parameters != parameter_names or candidate_buffers != buffer_names:
                    raise ValueError("model parameter/buffer classification differs")
            else:
                reference = bundle
                parameter_names = candidate_parameters
                buffer_names = candidate_buffers
            bundles.append(bundle)
            selected_paths.append(path)
            print(f"INCLUDED {path.name}")
        except Exception as error:
            if not args.skip_incompatible:
                raise RuntimeError(f"Cannot merge {path}: {error}") from error
            print(f"SKIPPED {path.name}: {error}")

    if len(bundles) < 2:
        raise RuntimeError("Fewer than two compatible native TabFM bundles remain")
    if args.weights is not None and len(args.weights) != len(paths):
        raise ValueError(
            "When --weights is supplied, its count must match all requested sources"
        )
    selected_weights = (
        [weight for path, weight in zip(paths, args.weights) if path in selected_paths]
        if args.weights is not None
        else None
    )
    weights = normalise_weights(selected_weights, len(bundles))
    assert parameter_names is not None and buffer_names is not None
    metadata_source_index = 0 if args.metadata_from == "first" else len(bundles) - 1
    merged_state, differing_buffers = average_parameters(
        bundles,
        weights,
        parameter_names,
        buffer_names,
        metadata_source_index,
    )

    metadata_source = bundles[metadata_source_index]
    output_bundle = copy.deepcopy(metadata_source)
    output_bundle["model_state_dict"] = merged_state
    output_bundle["best_epoch"] = None
    output_bundle["best_metrics"] = None
    output_bundle["best_metrics_by_validation_cohort"] = None
    output_bundle["selection_metric"] = "experimental_weighted_parameter_average"
    output_bundle["training_mode"] = "merged_native_tabfm_checkpoints"
    output_bundle["zeroed_features"] = []
    output_bundle["merge_metadata"] = {
        "method": "weighted_parameter_average",
        "model": "src.model.TabFM",
        "created_at": datetime.now().astimezone().isoformat(),
        "source_files": [str(path) for path in selected_paths],
        "normalised_weights": weights,
        "model_count": len(bundles),
        "metadata_source": args.metadata_from,
        "buffer_source_file": str(selected_paths[metadata_source_index]),
        "differing_buffers_copied_from_source": differing_buffers,
        "source_zeroed_features": [
            list(bundle.get("zeroed_features", [])) for bundle in bundles
        ],
        "merged_zeroed_features": [],
    }

    # Prove that predict_race.py can reconstruct the merged architecture strictly.
    validate_model_bundle(output_bundle)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    torch.save(output_bundle, temporary_output)
    temporary_output.replace(args.output_file)
    write_metadata(args.output_file, output_bundle)

    print("\nTABFM MERGE COMPLETE")
    for path, weight in zip(selected_paths, weights):
        print(f"  weight={weight:.6f} checkpoint={path}")
    if differing_buffers:
        print(
            "  copied_differing_buffers="
            + ",".join(differing_buffers)
            + f" source={selected_paths[metadata_source_index]}"
        )
    print(f"  output={args.output_file.resolve()}")
    print(
        "WARNING: parameter averaging is experimental; backtest the merged "
        "checkpoint before using it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
