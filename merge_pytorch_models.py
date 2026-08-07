#!/usr/bin/env python3

"""
Merge compatible PyTorch checkpoints using weighted parameter averaging.

The script:

1. Searches the output folder for .pt, .pth and .ckpt files.
2. Prints model information for every checkpoint.
3. Validates that all included models have matching parameter names and shapes.
4. Averages floating-point model parameters.
5. Copies integer and Boolean buffers from the first model.
6. Saves a merged checkpoint.

Examples:

    python merge_pytorch_models.py

    python merge_pytorch_models.py \
        --input-dir output \
        --output-file output/merged_model.pt

    python merge_pytorch_models.py \
        --input-dir output \
        --output-file output/merged_model.pt \
        --weights 0.5 0.3 0.2

    python merge_pytorch_models.py \
        --input-dir output \
        --output-file output/merged_model.pt \
        --recursive \
        --skip-incompatible

Important:

All merged checkpoints must use the same model architecture, parameter names
and tensor shapes.

Only load PyTorch checkpoints from trusted sources because checkpoint files
can contain pickled Python objects.
"""

from __future__ import annotations

import argparse
import copy
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


STATE_DICT_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "network_state_dict",
    "net_state_dict",
    "weights",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average compatible PyTorch model checkpoints."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output"),
        help="Folder containing PyTorch checkpoint files.",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("output/merged_model.pt"),
        help="Path where the merged checkpoint will be saved.",
    )

    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pt", ".pth", ".ckpt"],
        help="Checkpoint file extensions to include.",
    )

    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional model averaging weights. The number of weights must "
            "match the number of compatible checkpoints."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search the input folder and all subfolders.",
    )

    parser.add_argument(
        "--skip-incompatible",
        action="store_true",
        help="Skip checkpoints with incompatible parameter keys or shapes.",
    )

    parser.add_argument(
        "--strip-module-prefix",
        action="store_true",
        help="Remove the 'module.' prefix used by DataParallel checkpoints.",
    )

    return parser.parse_args()


def torch_load_checkpoint(path: Path) -> Any:
    """
    Load a PyTorch checkpoint onto the CPU.
    """
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        return torch.load(
            path,
            map_location="cpu",
        )


def is_raw_state_dict(value: Any) -> bool:
    """
    Return True when value looks like a raw PyTorch state_dict.
    """
    if not isinstance(value, (dict, OrderedDict)):
        return False

    if not value:
        return False

    return all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def extract_state_dict(
    checkpoint: Any,
) -> tuple[OrderedDict[str, torch.Tensor], str | None]:
    """
    Extract the model state_dict from a checkpoint.

    Returns:
        state_dict
        wrapper key, or None when the checkpoint is a raw state_dict
    """
    if is_raw_state_dict(checkpoint):
        return OrderedDict(checkpoint), None

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unsupported checkpoint type: "
            f"{type(checkpoint).__name__}"
        )

    for key in STATE_DICT_KEYS:
        candidate = checkpoint.get(key)

        if is_raw_state_dict(candidate):
            return OrderedDict(candidate), key

    available_keys = list(checkpoint.keys())

    raise ValueError(
        "Could not locate a model state_dict. "
        f"Checked keys: {STATE_DICT_KEYS}. "
        f"Available checkpoint keys: {available_keys[:30]}"
    )


def strip_module_prefix(
    state_dict: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """
    Convert DataParallel parameter names such as:

        module.layer.weight

    into:

        layer.weight
    """
    stripped_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    for key, tensor in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key

        if new_key in stripped_state_dict:
            raise ValueError(
                "Duplicate parameter key after removing the module prefix: "
                f"{new_key}"
            )

        stripped_state_dict[new_key] = tensor

    return stripped_state_dict


def find_checkpoint_files(
    input_dir: Path,
    output_file: Path,
    extensions: list[str],
    recursive: bool,
) -> list[Path]:
    """
    Find all checkpoint files in the requested folder.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    normalised_extensions = {
        extension.lower()
        if extension.startswith(".")
        else f".{extension.lower()}"
        for extension in extensions
    }

    iterator = (
        input_dir.rglob("*")
        if recursive
        else input_dir.glob("*")
    )

    output_resolved = output_file.resolve()

    checkpoint_files = [
        path
        for path in iterator
        if (
            path.is_file()
            and path.suffix.lower() in normalised_extensions
            and path.resolve() != output_resolved
        )
    ]

    return sorted(
        checkpoint_files,
        key=lambda path: str(path).lower(),
    )


def first_available(
    checkpoint: Any,
    keys: tuple[str, ...],
    default: Any = None,
) -> Any:
    """
    Search common checkpoint metadata dictionaries and return the first value
    matching one of the requested keys.
    """
    if not isinstance(checkpoint, dict):
        return default

    search_locations: list[Any] = [
        checkpoint,
        checkpoint.get("metadata"),
        checkpoint.get("meta"),
        checkpoint.get("config"),
        checkpoint.get("model_config"),
        checkpoint.get("training_config"),
        checkpoint.get("trainer_config"),
        checkpoint.get("hyperparameters"),
        checkpoint.get("hparams"),
        checkpoint.get("args"),
        checkpoint.get("metrics"),
        checkpoint.get("validation_metrics"),
        checkpoint.get("results"),
    ]

    for location in search_locations:
        if not isinstance(location, dict):
            continue

        for key in keys:
            value = location.get(key)

            if value is not None:
                return value

    return default


def format_info_value(value: Any) -> str:
    """
    Format checkpoint metadata for readable console output.
    """
    if value is None:
        return "unknown"

    if torch.is_tensor(value):
        if value.numel() == 1:
            return str(value.item())

        return (
            f"tensor(shape={tuple(value.shape)}, "
            f"dtype={value.dtype})"
        )

    if isinstance(value, float):
        return f"{value:.6f}"

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (list, tuple)):
        if len(value) <= 10:
            return str(value)

        return (
            f"{type(value).__name__}"
            f"(length={len(value)})"
        )

    if isinstance(value, dict):
        keys = list(value.keys())

        if len(keys) <= 10:
            return f"dict(keys={keys})"

        return (
            f"dict(keys={keys[:10]}, "
            f"total_keys={len(keys)})"
        )

    text = str(value)

    if len(text) > 150:
        return text[:147] + "..."

    return text


def count_state_dict_values(
    state_dict: OrderedDict[str, torch.Tensor],
) -> tuple[int, int, int]:
    """
    Return:

        total number of tensor values
        number of floating-point tensor values
        number of tensors
    """
    total_values = 0
    floating_values = 0
    tensor_count = 0

    for tensor in state_dict.values():
        if not torch.is_tensor(tensor):
            continue

        tensor_count += 1
        total_values += tensor.numel()

        if tensor.is_floating_point() or tensor.is_complex():
            floating_values += tensor.numel()

    return total_values, floating_values, tensor_count


def get_state_dict_dtype_summary(
    state_dict: OrderedDict[str, torch.Tensor],
) -> str:
    """
    Return a summary of tensor data types found in the state_dict.
    """
    dtype_counts: dict[str, int] = {}

    for tensor in state_dict.values():
        dtype_name = str(tensor.dtype).replace("torch.", "")

        dtype_counts[dtype_name] = (
            dtype_counts.get(dtype_name, 0) + 1
        )

    return ", ".join(
        f"{dtype_name}={count}"
        for dtype_name, count in sorted(dtype_counts.items())
    )


def get_checkpoint_size_mb(checkpoint_file: Path) -> float:
    """
    Return the checkpoint file size in megabytes.
    """
    return checkpoint_file.stat().st_size / (1024 * 1024)


def print_model_info(
    checkpoint_number: int,
    checkpoint_file: Path,
    checkpoint: Any,
    state_dict: OrderedDict[str, torch.Tensor],
    wrapper_key: str | None,
) -> None:
    """
    Print available model and training information for one checkpoint.
    """
    total_values, floating_values, tensor_count = (
        count_state_dict_values(state_dict)
    )

    model_name = first_available(
        checkpoint,
        (
            "model_name",
            "model_class",
            "architecture",
            "arch",
            "network_name",
            "model_type",
            "class_name",
        ),
    )

    saved_mode = first_available(
        checkpoint,
        (
            "model_mode",
            "mode",
            "training_mode",
            "run_mode",
        ),
    )

    training_objective = first_available(
        checkpoint,
        (
            "training_objective",
            "objective",
            "loss_name",
            "loss_function",
            "criterion",
        ),
    )

    epoch = first_available(
        checkpoint,
        (
            "epoch",
            "current_epoch",
            "last_epoch",
            "epochs_completed",
        ),
    )

    best_epoch = first_available(
        checkpoint,
        (
            "best_epoch",
            "best_validation_epoch",
            "best_val_epoch",
        ),
    )

    validation_loss = first_available(
        checkpoint,
        (
            "best_val_loss",
            "validation_loss",
            "val_loss",
            "best_validation_loss",
        ),
    )

    top1_metric = first_available(
        checkpoint,
        (
            "win_hit_rate_top1",
            "val_win_hit_rate_top1",
            "validation_win_hit_rate_top1",
            "top1_accuracy",
            "top1",
        ),
    )

    top2_metric = first_available(
        checkpoint,
        (
            "win_hit_rate_top2",
            "val_win_hit_rate_top2",
            "top2_accuracy",
            "top2",
        ),
    )

    top3_metric = first_available(
        checkpoint,
        (
            "win_hit_rate_top3",
            "val_win_hit_rate_top3",
            "top3_accuracy",
            "top3",
        ),
    )

    validation_metric = first_available(
        checkpoint,
        (
            "best_metric",
            "validation_metric",
            "val_metric",
            "best_score",
            "score",
        ),
    )

    mean_reciprocal_rank = first_available(
        checkpoint,
        (
            "mean_reciprocal_rank",
            "mrr",
            "validation_mrr",
            "val_mrr",
        ),
    )

    race_log_loss = first_available(
        checkpoint,
        (
            "race_level_log_loss",
            "race_log_loss",
            "validation_race_log_loss",
        ),
    )

    hidden_dims = first_available(
        checkpoint,
        (
            "hidden_dims",
            "hidden_sizes",
            "layer_sizes",
        ),
    )

    dropout = first_available(
        checkpoint,
        (
            "dropout",
            "dropout_rate",
        ),
    )

    embedding_dropout = first_available(
        checkpoint,
        (
            "embedding_dropout",
            "embedding_dropout_rate",
        ),
    )

    embedding_dim_cap = first_available(
        checkpoint,
        (
            "embedding_dim_cap",
            "max_embedding_dim",
        ),
    )

    activation = first_available(
        checkpoint,
        (
            "activation",
            "activation_name",
        ),
    )

    learning_rate = first_available(
        checkpoint,
        (
            "learning_rate",
            "lr",
        ),
    )

    weight_decay = first_available(
        checkpoint,
        (
            "weight_decay",
            "l2",
        ),
    )

    batch_size = first_available(
        checkpoint,
        (
            "batch_size",
            "train_batch_size",
        ),
    )

    random_seed = first_available(
        checkpoint,
        (
            "seed",
            "random_seed",
        ),
    )

    feature_count = first_available(
        checkpoint,
        (
            "feature_count",
            "num_features",
            "input_dim",
            "input_size",
        ),
    )

    branch_config = first_available(
        checkpoint,
        (
            "branch_config",
            "branches",
            "feature_branches",
        ),
    )

    print()
    print("=" * 90)
    print(f"MODEL {checkpoint_number}")
    print("=" * 90)
    print(f"File:                  {checkpoint_file}")
    print(f"File size:             {get_checkpoint_size_mb(checkpoint_file):,.2f} MB")
    print(f"Model name:            {format_info_value(model_name)}")
    print(f"Saved model mode:      {format_info_value(saved_mode)}")
    print(f"State-dict key:        {wrapper_key or 'raw state_dict'}")
    print(f"Tensor count:          {tensor_count:,}")
    print(f"Total tensor values:   {total_values:,}")
    print(f"Floating values:       {floating_values:,}")
    print(f"Tensor dtypes:         {get_state_dict_dtype_summary(state_dict)}")

    print("-" * 90)
    print("TRAINING INFORMATION")
    print("-" * 90)
    print(f"Epoch:                 {format_info_value(epoch)}")
    print(f"Best epoch:            {format_info_value(best_epoch)}")
    print(f"Objective:             {format_info_value(training_objective)}")
    print(f"Validation loss:       {format_info_value(validation_loss)}")
    print(f"Validation metric:     {format_info_value(validation_metric)}")
    print(f"Top-1 metric:          {format_info_value(top1_metric)}")
    print(f"Top-2 metric:          {format_info_value(top2_metric)}")
    print(f"Top-3 metric:          {format_info_value(top3_metric)}")
    print(f"MRR:                   {format_info_value(mean_reciprocal_rank)}")
    print(f"Race-level log loss:   {format_info_value(race_log_loss)}")
    print(f"Learning rate:         {format_info_value(learning_rate)}")
    print(f"Weight decay:          {format_info_value(weight_decay)}")
    print(f"Batch size:            {format_info_value(batch_size)}")
    print(f"Random seed:           {format_info_value(random_seed)}")

    print("-" * 90)
    print("ARCHITECTURE INFORMATION")
    print("-" * 90)
    print(f"Feature count:         {format_info_value(feature_count)}")
    print(f"Hidden dimensions:     {format_info_value(hidden_dims)}")
    print(f"Activation:            {format_info_value(activation)}")
    print(f"Dropout:               {format_info_value(dropout)}")
    print(f"Embedding dropout:     {format_info_value(embedding_dropout)}")
    print(f"Embedding dim cap:     {format_info_value(embedding_dim_cap)}")
    print(f"Branch configuration:  {format_info_value(branch_config)}")


def compatibility_error(
    reference: OrderedDict[str, torch.Tensor],
    candidate: OrderedDict[str, torch.Tensor],
) -> str | None:
    """
    Return an error message when two state_dicts are incompatible.
    """
    reference_keys = set(reference.keys())
    candidate_keys = set(candidate.keys())

    missing_keys = sorted(
        reference_keys - candidate_keys
    )

    unexpected_keys = sorted(
        candidate_keys - reference_keys
    )

    if missing_keys or unexpected_keys:
        messages: list[str] = []

        if missing_keys:
            messages.append(
                "missing keys: "
                f"{missing_keys[:20]}"
            )

        if unexpected_keys:
            messages.append(
                "unexpected keys: "
                f"{unexpected_keys[:20]}"
            )

        return "; ".join(messages)

    for key in reference:
        reference_tensor = reference[key]
        candidate_tensor = candidate[key]

        if reference_tensor.shape != candidate_tensor.shape:
            return (
                f"shape mismatch for '{key}': "
                f"{tuple(reference_tensor.shape)} versus "
                f"{tuple(candidate_tensor.shape)}"
            )

        if (
            reference_tensor.is_floating_point()
            != candidate_tensor.is_floating_point()
        ):
            return (
                f"tensor type mismatch for '{key}': "
                f"{reference_tensor.dtype} versus "
                f"{candidate_tensor.dtype}"
            )

    return None


def normalise_weights(
    weights: list[float] | None,
    model_count: int,
) -> list[float]:
    """
    Validate and normalise model averaging weights.
    """
    if model_count <= 0:
        raise ValueError(
            "At least one model is required."
        )

    if weights is None:
        equal_weight = 1.0 / model_count

        return [equal_weight] * model_count

    if len(weights) != model_count:
        raise ValueError(
            f"Received {len(weights)} weights for "
            f"{model_count} compatible models."
        )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            "Model weights cannot be negative."
        )

    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError(
            "The total model weight must be greater than zero."
        )

    return [
        weight / total_weight
        for weight in weights
    ]


def accumulation_dtype(
    tensor: torch.Tensor,
) -> torch.dtype:
    """
    Choose a stable data type for parameter accumulation.
    """
    if tensor.is_complex():
        if tensor.dtype == torch.complex128:
            return torch.complex128

        return torch.complex64

    if tensor.dtype == torch.float64:
        return torch.float64

    return torch.float32


def average_state_dicts(
    state_dicts: list[OrderedDict[str, torch.Tensor]],
    weights: list[float],
) -> OrderedDict[str, torch.Tensor]:
    """
    Create a weighted average of compatible state_dicts.

    Floating-point and complex tensors are averaged.

    Integer and Boolean buffers are copied from the model with the highest
    averaging weight. This applies to buffers such as:

        num_batches_tracked
    """
    if not state_dicts:
        raise ValueError(
            "No state_dicts were supplied for merging."
        )

    if len(state_dicts) != len(weights):
        raise ValueError(
            "The number of state_dicts does not match "
            "the number of model weights."
        )

    merged_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    reference_state_dict = state_dicts[0]

    highest_weight_model_index = max(
        range(len(weights)),
        key=lambda index: weights[index],
    )

    with torch.no_grad():
        for parameter_name, reference_tensor in (
            reference_state_dict.items()
        ):
            if (
                reference_tensor.is_floating_point()
                or reference_tensor.is_complex()
            ):
                accumulator = torch.zeros_like(
                    reference_tensor,
                    dtype=accumulation_dtype(reference_tensor),
                    device="cpu",
                )

                for state_dict, model_weight in zip(
                    state_dicts,
                    weights,
                ):
                    source_tensor = (
                        state_dict[parameter_name]
                        .detach()
                        .cpu()
                    )

                    accumulator.add_(
                        source_tensor.to(accumulator.dtype),
                        alpha=model_weight,
                    )

                merged_state_dict[parameter_name] = accumulator.to(
                    reference_tensor.dtype
                )

            else:
                source_tensor = state_dicts[
                    highest_weight_model_index
                ][parameter_name]

                merged_state_dict[parameter_name] = (
                    source_tensor
                    .detach()
                    .cpu()
                    .clone()
                )

    return merged_state_dict


def build_output_checkpoint(
    reference_checkpoint: Any,
    wrapper_key: str | None,
    merged_state_dict: OrderedDict[str, torch.Tensor],
    source_files: list[Path],
    weights: list[float],
) -> Any:
    """
    Build the final checkpoint while retaining metadata from the first model.
    """
    merge_metadata = {
        "method": "weighted_parameter_average",
        "source_files": [
            str(path)
            for path in source_files
        ],
        "normalised_weights": weights,
        "model_count": len(source_files),
    }

    if wrapper_key is None:
        # The source was a raw state_dict. Save a structured checkpoint so
        # the source information and averaging weights are retained.
        return {
            "model_state_dict": merged_state_dict,
            "merge_metadata": merge_metadata,
        }

    output_checkpoint = copy.deepcopy(reference_checkpoint)

    output_checkpoint[wrapper_key] = merged_state_dict

    existing_merge_metadata = output_checkpoint.get(
        "merge_metadata"
    )

    if isinstance(existing_merge_metadata, dict):
        existing_merge_metadata.update(
            merge_metadata
        )
    else:
        output_checkpoint["merge_metadata"] = merge_metadata

    return output_checkpoint


def print_merge_summary(
    compatible_files: list[Path],
    weights: list[float],
) -> None:
    """
    Print the final model inclusion and weighting summary.
    """
    print()
    print("=" * 90)
    print("MERGE SUMMARY")
    print("=" * 90)

    for model_number, (
        checkpoint_file,
        weight,
    ) in enumerate(
        zip(compatible_files, weights),
        start=1,
    ):
        print(
            f"Model {model_number:>3}: "
            f"weight={weight:.6f} | "
            f"{checkpoint_file}"
        )


def main() -> None:
    args = parse_args()

    checkpoint_files = find_checkpoint_files(
        input_dir=args.input_dir,
        output_file=args.output_file,
        extensions=args.extensions,
        recursive=args.recursive,
    )

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files were found in: {args.input_dir}"
        )

    print()
    print("=" * 90)
    print("PYTORCH MODEL MERGER")
    print("=" * 90)
    print(f"Input directory:       {args.input_dir}")
    print(f"Output file:           {args.output_file}")
    print(f"Files discovered:      {len(checkpoint_files)}")
    print(f"Recursive search:      {args.recursive}")
    print(f"Skip incompatible:     {args.skip_incompatible}")
    print(f"Strip module prefix:   {args.strip_module_prefix}")

    compatible_files: list[Path] = []
    compatible_checkpoints: list[Any] = []
    compatible_state_dicts: list[
        OrderedDict[str, torch.Tensor]
    ] = []

    reference_state_dict: (
        OrderedDict[str, torch.Tensor] | None
    ) = None

    reference_wrapper_key: str | None = None
    reference_file: Path | None = None

    for checkpoint_number, checkpoint_file in enumerate(
        checkpoint_files,
        start=1,
    ):
        try:
            checkpoint = torch_load_checkpoint(
                checkpoint_file
            )

            state_dict, wrapper_key = extract_state_dict(
                checkpoint
            )

            if args.strip_module_prefix:
                state_dict = strip_module_prefix(
                    state_dict
                )

            print_model_info(
                checkpoint_number=checkpoint_number,
                checkpoint_file=checkpoint_file,
                checkpoint=checkpoint,
                state_dict=state_dict,
                wrapper_key=wrapper_key,
            )

            if reference_state_dict is None:
                reference_state_dict = state_dict
                reference_wrapper_key = wrapper_key
                reference_file = checkpoint_file

                print("-" * 90)
                print("Compatibility:         REFERENCE MODEL")
                print("Merge status:          INCLUDED")

            else:
                error = compatibility_error(
                    reference=reference_state_dict,
                    candidate=state_dict,
                )

                if error is not None:
                    raise ValueError(error)

                print("-" * 90)
                print(
                    "Compatibility:         "
                    f"COMPATIBLE WITH {reference_file}"
                )
                print("Merge status:          INCLUDED")

            compatible_files.append(
                checkpoint_file
            )

            compatible_checkpoints.append(
                checkpoint
            )

            compatible_state_dicts.append(
                state_dict
            )

        except Exception as error:
            print()
            print("-" * 90)
            print(f"File:                  {checkpoint_file}")
            print("Merge status:          SKIPPED")
            print(f"Reason:                {error}")
            print("-" * 90)

            if not args.skip_incompatible:
                raise RuntimeError(
                    "Checkpoint is incompatible or could not be loaded: "
                    f"{checkpoint_file}\n"
                    f"Reason: {error}"
                ) from error

    if not compatible_state_dicts:
        raise RuntimeError(
            "No compatible checkpoints were available for merging."
        )

    weights = normalise_weights(
        weights=args.weights,
        model_count=len(compatible_state_dicts),
    )

    print_merge_summary(
        compatible_files=compatible_files,
        weights=weights,
    )

    merged_state_dict = average_state_dicts(
        state_dicts=compatible_state_dicts,
        weights=weights,
    )

    output_checkpoint = build_output_checkpoint(
        reference_checkpoint=compatible_checkpoints[0],
        wrapper_key=reference_wrapper_key,
        merged_state_dict=merged_state_dict,
        source_files=compatible_files,
        weights=weights,
    )

    args.output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        output_checkpoint,
        args.output_file,
    )

    output_size_mb = get_checkpoint_size_mb(
        args.output_file
    )

    print()
    print("=" * 90)
    print("MERGE COMPLETED")
    print("=" * 90)
    print(f"Models merged:         {len(compatible_files)}")
    print(f"Output file:           {args.output_file}")
    print(f"Output size:           {output_size_mb:,.2f} MB")
    print(
        "Method:                "
        "weighted parameter averaging"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()