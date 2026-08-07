#!/usr/bin/env python3
"""Create an experimental weighted average of two compatible TabFM bundles."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import warnings

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False.*",
    category=UserWarning,
)

try:
    from src.model import TabFM
except ImportError as exc:
    raise SystemExit(
        "Cannot import src.model.TabFM. Run this script from the repository root."
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/merged.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models", type=Path, nargs="*",
        help="Two or more checkpoints. Omit to merge every *.pt file in outputs/.",
    )
    parser.add_argument("--weight-a", type=float, default=0.5)
    parser.add_argument(
        "--contract", choices=("a", "b"), default="a",
        help="Use the first (a) or last (b) checkpoint's non-weight metadata.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_equal_contract(a: dict, b: dict) -> None:
    if a["model_kwargs"] != b["model_kwargs"]:
        raise ValueError("Model architectures differ")
    if list(a["feature_columns"]) != list(b["feature_columns"]):
        raise ValueError("Feature columns or their order differ")
    for key in ("median", "scale"):
        if not np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
            raise ValueError(f"Preprocessing field {key!r} differs")


def merge_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    model_kwargs: dict,
    weight_a: float,
) -> dict[str, torch.Tensor]:
    if state_a.keys() != state_b.keys():
        missing_a = sorted(state_b.keys() - state_a.keys())
        missing_b = sorted(state_a.keys() - state_b.keys())
        raise ValueError(
            f"State keys differ: missing_from_a={missing_a} missing_from_b={missing_b}"
        )

    buffer_names = dict(TabFM(**model_kwargs).named_buffers()).keys()
    weight_b = 1.0 - weight_a
    merged = {}
    for key, tensor_a in state_a.items():
        tensor_b = state_b[key]
        if tensor_a.shape != tensor_b.shape or tensor_a.dtype != tensor_b.dtype:
            raise ValueError(
                f"Tensor {key!r} differs: "
                f"a={tuple(tensor_a.shape)}/{tensor_a.dtype} "
                f"b={tuple(tensor_b.shape)}/{tensor_b.dtype}"
            )
        if key in buffer_names:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(
                    f"Required model buffer {key!r} differs; weight merging is unsafe"
                )
            merged[key] = tensor_a.detach().cpu().clone()
        elif torch.is_floating_point(tensor_a) or torch.is_complex(tensor_a):
            merged[key] = (
                tensor_a.detach().cpu() * weight_a
                + tensor_b.detach().cpu() * weight_b
            ).to(tensor_a.dtype)
        else:
            if not torch.equal(tensor_a, tensor_b):
                raise ValueError(f"Non-floating tensor {key!r} differs")
            merged[key] = tensor_a.detach().cpu().clone()
    return merged


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.weight_a <= 1.0:
        raise SystemExit("--weight-a must be between 0 and 1")
    output_path = args.output.resolve()
    source_paths = args.models or sorted(DEFAULT_OUTPUT.parent.glob("*.pt"))
    source_paths = [path.resolve() for path in source_paths if path.resolve() != output_path]
    if len(source_paths) < 2:
        raise SystemExit(
            "At least two source checkpoints are required. With no positional models, "
            "place two or more *.pt files in outputs/."
        )
    if len(set(source_paths)) != len(source_paths):
        raise SystemExit("Source checkpoint paths must be unique")
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Checkpoint files not found: {missing}")

    loaded = [
        (path, torch.load(path, map_location="cpu", weights_only=False))
        for path in source_paths
    ]
    compatible_groups: list[list[tuple[Path, dict]]] = []
    for path, bundle in loaded:
        for group in compatible_groups:
            try:
                require_equal_contract(group[0][1], bundle)
            except (KeyError, ValueError):
                continue
            group.append((path, bundle))
            break
        else:
            compatible_groups.append([(path, bundle)])

    # A weight average is only meaningful inside one shared feature/preprocessing
    # contract.  Prefer the largest compatible set rather than aborting because
    # outputs/ contains older experiments with another feature layout.
    compatible_groups.sort(key=len, reverse=True)
    selected_group = compatible_groups[0]
    if len(selected_group) < 2:
        raise SystemExit(
            "No compatible group contains two checkpoints; cannot safely merge "
            "models with different feature/preprocessing contracts."
        )
    skipped_paths = [
        path for group in compatible_groups[1:] for path, _ in group
    ]
    if skipped_paths:
        print(
            "WARNING excluding incompatible checkpoints from merge: "
            + ", ".join(str(path) for path in skipped_paths),
            flush=True,
        )
    source_paths = [path for path, _ in selected_group]
    bundles = [bundle for _, bundle in selected_group]

    merged_state = bundles[0]["model_state_dict"]
    # Preserve --weight-a's historical behaviour for exactly two explicit models.
    if len(bundles) == 2:
        merged_state = merge_state_dicts(
            merged_state, bundles[1]["model_state_dict"], bundles[0]["model_kwargs"], args.weight_a
        )
        weights = [args.weight_a, 1.0 - args.weight_a]
    else:
        # Running mean: at iteration i, the old aggregate has weight i/(i+1).
        for index, bundle in enumerate(bundles[1:], start=1):
            merged_state = merge_state_dicts(
                merged_state, bundle["model_state_dict"], bundles[0]["model_kwargs"], index / (index + 1)
            )
        weights = [1.0 / len(bundles)] * len(bundles)

    contract = bundles[0] if args.contract == "a" else bundles[-1]
    output_bundle = copy.deepcopy(contract)
    output_bundle["model_state_dict"] = merged_state
    output_bundle["best_epoch"] = None
    output_bundle["best_metrics"] = None
    output_bundle["selection_metric"] = "experimental_weight_merge"
    output_bundle["weight_merge"] = {
        "models": [str(path) for path in source_paths],
        "weights": weights,
        "excluded_incompatible_models": [str(path) for path in skipped_paths],
        "inference_contract": args.contract,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(output_bundle, temporary)
    temporary.replace(args.output)

    metadata = {
        "model": "src.model.TabFM",
        "output": str(args.output.resolve()),
        "features": list(output_bundle["feature_columns"]),
        "train_cutoff_iso": output_bundle.get("train_cutoff_iso"),
        "context_race_ids": output_bundle.get("context_race_ids", []),
        "zeroed_features": output_bundle.get("zeroed_features", []),
        "selection_metric": output_bundle["selection_metric"],
        "weight_merge": output_bundle["weight_merge"],
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"merged_model={args.output.resolve()} models={len(source_paths)} "
        f"weights={','.join(f'{weight:.3f}' for weight in weights)} "
        f"contract={args.contract} "
        f"zeroed_features={output_bundle.get('zeroed_features', [])}",
        flush=True,
    )
    print(
        "WARNING weight averaging is experimental and may perform poorly if "
        "the source models did not descend from aligned initialization.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
