#!/usr/bin/env python3
"""Train a race-winner MoE where each expert is restricted to a feature subset from JSON."""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import DEFAULT_DB, DEFAULT_FEATURES
from src.dataset import load_feature_manifest
from src.metrics import probability_metrics
from src.model.race_moe_feature_map import (
    FeatureMappedRaceWinnerConfig,
    RaceMixtureOfExpertsFeatureMap,
    expand_feature_map_to_model_features,
    expand_feature_indices_to_model_features,
    load_feature_expert_map,
    load_router_feature_indices,
)
from src.race_moe_data import (
    batches, chronological_race_ids, load_finished_winner_rows,
    market_blind_features, numeric_matrix, pad_batch, race_indices,
)
from src.model.raceformer import raceformer_losses
from src.race_moe_evaluation import collapse_warnings, routing_diagnostics
from src.race_moe_snapshot import (
    create_split_snapshot, load_split_snapshot, snapshot_manifest_reference,
)
from src.raceformer_preprocessing import (
    fit_raceformer_preprocessor, model_feature_columns, transform_raceformer,
)


def _dims(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("hidden dims must be comma-separated integers") from error
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("hidden dims must be positive")
    return result


def _optional_dims(value: str) -> tuple[int, ...]:
    if value.lower() in {"none", "off"}:
        return ()
    return _dims(value)


def _top_k(value: str) -> int | None:
    if value.lower() in {"all", "dense"}:
        return None
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be an integer or all") from error
    if result < 1:
        raise argparse.ArgumentTypeError("top-k must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-map-json", type=Path, required=True, help="JSON mapping feature names to expert IDs")
    parser.add_argument("--output", type=Path, default=Path("outputs/race_winner_moe_feature_map.pt"))
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        help="Compatible feature-map checkpoint used to initialize model weights.",
    )
    parser.add_argument(
        "--freeze-base-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze experts and router; train only the judge from --initial-checkpoint.",
    )
    parser.add_argument(
        "--trainable-components",
        nargs="+",
        metavar="COMPONENT",
        help=(
            "Components to update, separated by spaces and/or commas, such "
            "as 'expert_0 router judge' or 'expert_0,router,judge'. Available "
            "components are printed before training; a subset requires "
            "--initial-checkpoint."
        ),
    )
    parser.add_argument(
        "--include-market-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow market-derived price and implied-probability features.",
    )
    parser.add_argument("--validation-races", type=int, default=1000)
    parser.add_argument("--test-races", type=int, default=1000)
    parser.add_argument(
        "--train-competition-id",
        "--competition-id",
        dest="train_competition_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Restrict the population to these competition IDs before chronological splitting.",
    )
    parser.add_argument(
        "--validation-competition-id",
        dest="validation_competition_ids",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Must match --train-competition-id to select checkpoints in-distribution.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--races-per-batch", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=_top_k, default=2)
    parser.add_argument("--moe-gate-temperature", type=float, default=1.0)
    parser.add_argument("--moe-router-balance-weight", type=float, default=0.01)
    parser.add_argument("--moe-expert-hidden-dims", type=_dims, default=(64,))
    parser.add_argument("--moe-router-hidden-dim", type=int, default=64)
    parser.add_argument("--moe-routing-mode", choices=("learned", "fixed_uniform"), default="learned")
    parser.add_argument(
        "--judge-hidden-dims", type=_optional_dims, default=(32,),
        help=(
            "Hidden layers for the final learned re-ranker. The judge only sees "
            "expert logits, router weights, and base winner probabilities; use "
            "'none' to disable it."
        ),
    )
    parser.add_argument("--standardized-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "validation-races": args.validation_races,
        "test-races": args.test_races,
        "epochs": args.epochs,
        "races-per-batch": args.races_per_batch,
        "learning-rate": args.learning_rate,
        "max-grad-norm": args.max_grad_norm,
        "early-stopping-patience": args.early_stopping_patience,
        "moe-num-experts": args.moe_num_experts,
        "moe-gate-temperature": args.moe_gate_temperature,
    }
    bad = [name for name, value in positive.items() if value <= 0]
    if bad:
        raise ValueError("These arguments must be positive: " + ", ".join(bad))
    if args.moe_router_balance_weight < 0:
        raise ValueError("router balance weight must be non-negative")
    if args.moe_top_k is not None and args.moe_top_k > args.moe_num_experts:
        raise ValueError("--moe-top-k cannot exceed --moe-num-experts")
    if args.freeze_base_model and args.initial_checkpoint is None:
        raise ValueError("--freeze-base-model requires --initial-checkpoint")
    if args.freeze_base_model and not args.judge_hidden_dims:
        raise ValueError("--freeze-base-model requires an enabled judge")
    if args.freeze_base_model and args.trainable_components is not None:
        raise ValueError(
            "--freeze-base-model cannot be combined with --trainable-components"
        )
    if (
        args.initial_checkpoint is not None
        and args.initial_checkpoint.resolve() == args.output.resolve()
    ):
        raise ValueError(
            "--output must differ from --initial-checkpoint; overwriting the "
            "initializer destroys checkpoint lineage"
        )


def training_arguments(args: argparse.Namespace) -> dict[str, object]:
    """Return every parsed argument in JSON-safe form for logging/checkpoints."""
    def convert(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return {key: convert(value) for key, value in vars(args).items()}


def training_command(
    args: argparse.Namespace,
    output: Path,
    trainable_components: tuple[str, ...],
) -> str:
    """Build a complete command that reproduces the current training run."""
    trainer = Path(__file__).resolve()
    lines = [shlex.join([sys.executable, str(trainer)])]

    def option(name: str, *values: object) -> None:
        lines.append(shlex.join([name, *(str(value) for value in values)]))

    option("--feature-map-json", args.feature_map_json.resolve())
    option("--db", args.db.resolve())
    option("--features-json", args.features_json.resolve())
    if args.train_competition_ids is not None:
        option("--train-competition-id", *args.train_competition_ids)
    if args.validation_competition_ids is not None:
        option("--validation-competition-id", *args.validation_competition_ids)
    option("--validation-races", args.validation_races)
    option("--test-races", args.test_races)
    lines.append(
        "--include-market-features"
        if args.include_market_features
        else "--no-include-market-features"
    )
    option("--moe-num-experts", args.moe_num_experts)
    option("--moe-top-k", "all" if args.moe_top_k is None else args.moe_top_k)
    option("--moe-router-balance-weight", f"{args.moe_router_balance_weight:g}")
    option("--moe-gate-temperature", f"{args.moe_gate_temperature:g}")
    option(
        "--moe-expert-hidden-dims",
        ",".join(map(str, args.moe_expert_hidden_dims)),
    )
    option("--moe-router-hidden-dim", args.moe_router_hidden_dim)
    option("--moe-routing-mode", args.moe_routing_mode)
    option(
        "--judge-hidden-dims",
        ",".join(map(str, args.judge_hidden_dims))
        if args.judge_hidden_dims else "none",
    )
    option("--races-per-batch", args.races_per_batch)
    option("--learning-rate", f"{args.learning_rate:.10g}")
    option("--weight-decay", f"{args.weight_decay:g}")
    option("--max-grad-norm", f"{args.max_grad_norm:g}")
    option("--dropout", f"{args.dropout:g}")
    option("--standardized-clip", f"{args.standardized_clip:g}")
    option("--epochs", args.epochs)
    option("--early-stopping-patience", args.early_stopping_patience)
    option("--seed", args.seed)
    option("--device", args.device)
    if args.initial_checkpoint is not None:
        option("--initial-checkpoint", args.initial_checkpoint.resolve())
    if args.freeze_base_model:
        lines.append("--freeze-base-model")
    elif args.trainable_components is not None:
        option("--trainable-components", ",".join(trainable_components))
    option("--output", output.resolve())
    return " \\\n  ".join(lines)


def initial_training_command(
    args: argparse.Namespace,
    output: Path,
    trainable_components: tuple[str, ...],
) -> tuple[str, Path, int]:
    """Trace checkpoint reports and reproduce the root, non-initialized run."""
    if args.initial_checkpoint is None:
        resolved_output = output.resolve()
        return (
            training_command(args, resolved_output, trainable_components),
            resolved_output,
            0,
        )

    checkpoint = args.initial_checkpoint.resolve()
    visited: set[Path] = set()
    depth = 0
    root_report: dict[str, object] | None = None
    while checkpoint not in visited:
        visited.add(checkpoint)
        report_path = checkpoint.with_suffix(".report.json")
        if not report_path.is_file():
            predecessor = previous_fine_tune_checkpoint(checkpoint)
            if predecessor is None or predecessor in visited:
                break
            checkpoint = predecessor
            depth += 1
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        root_report = report
        parent_value = report.get("initial_checkpoint")
        if parent_value is None:
            break
        parent = Path(str(parent_value)).resolve()
        if parent == checkpoint or parent in visited:
            predecessor = previous_fine_tune_checkpoint(checkpoint)
            if predecessor is None or predecessor in visited:
                break
            parent = predecessor
        checkpoint = parent
        depth += 1

    if root_report is None or root_report.get("initial_checkpoint") is not None:
        raise ValueError(
            "Cannot print the initial training command because checkpoint "
            f"lineage report is missing or incomplete at {checkpoint.with_suffix('.report.json')}"
        )
    saved_arguments = root_report.get("training_config")
    if not isinstance(saved_arguments, dict):
        raise ValueError("Root checkpoint report has no training_config object")

    root_values = dict(vars(args))
    root_values.update(saved_arguments)
    for name in ("db", "features_json", "feature_map_json", "output"):
        root_values[name] = Path(str(root_values[name]))
    root_values["initial_checkpoint"] = None
    root_values["freeze_base_model"] = bool(root_values.get("freeze_base_model", False))
    for name in ("moe_expert_hidden_dims", "judge_hidden_dims"):
        root_values[name] = tuple(root_values[name])
    root_args = argparse.Namespace(**root_values)
    root_output = checkpoint.resolve()
    root_components_value = root_report.get("trainable_components", ())
    root_components = tuple(map(str, root_components_value))
    return training_command(root_args, root_output, root_components), root_output, depth


def previous_fine_tune_checkpoint(checkpoint: Path) -> Path | None:
    """Infer the predecessor of legacy fine-tune filenames when metadata loops."""
    checkpoint = checkpoint.resolve()
    stem = checkpoint.stem
    numbered = re.fullmatch(r"(.+)_finetuned_(\d+)", stem)
    if numbered:
        base, version_text = numbered.groups()
        version = int(version_text)
        previous_stem = (
            f"{base}_finetuned_{version - 1}"
            if version > 2 else f"{base}_finetuned"
        )
    elif stem.endswith("_finetuned"):
        previous_stem = stem.removesuffix("_finetuned")
    else:
        return None
    predecessor = checkpoint.with_name(
        f"{previous_stem}{checkpoint.suffix or '.pt'}"
    )
    if predecessor.with_suffix(".report.json").is_file():
        return predecessor
    return None


def next_fine_tuned_output(checkpoint: Path) -> Path:
    """Return a readable, non-overwriting continuation checkpoint name."""
    checkpoint = checkpoint.resolve()
    stem = checkpoint.stem
    numbered = re.fullmatch(r"(.+)_finetuned_(\d+)", stem)
    if numbered:
        base, version = numbered.groups()
        next_stem = f"{base}_finetuned_{int(version) + 1}"
    else:
        repeated = re.fullmatch(r"(.+?)((?:_finetuned)+)", stem)
        if repeated:
            base, suffixes = repeated.groups()
            version = suffixes.count("_finetuned") + 1
            next_stem = f"{base}_finetuned_{version}"
        else:
            next_stem = f"{stem}_finetuned"
    suffix = checkpoint.suffix or ".pt"
    return checkpoint.with_name(f"{next_stem}{suffix}")


def fine_tune_command(
    args: argparse.Namespace,
    checkpoint: Path,
    trainable_components: tuple[str, ...],
) -> tuple[str, Path, float, int]:
    """Build a reproducible continuation command for a saved checkpoint."""
    checkpoint = checkpoint.resolve()
    fine_tuned_output = next_fine_tuned_output(checkpoint)
    fine_tune_learning_rate = args.learning_rate / 3.0
    fine_tune_epochs = min(args.epochs, 200)
    trainer = Path(__file__).resolve()
    lines = [shlex.join([sys.executable, str(trainer)])]

    def option(name: str, *values: object) -> None:
        lines.append(shlex.join([name, *(str(value) for value in values)]))

    option("--feature-map-json", args.feature_map_json.resolve())
    option("--db", args.db.resolve())
    option("--features-json", args.features_json.resolve())
    if args.train_competition_ids is not None:
        option("--train-competition-id", *args.train_competition_ids)
    if args.validation_competition_ids is not None:
        option("--validation-competition-id", *args.validation_competition_ids)
    option("--validation-races", args.validation_races)
    option("--test-races", args.test_races)
    lines.append(
        "--include-market-features"
        if args.include_market_features
        else "--no-include-market-features"
    )
    option("--moe-num-experts", args.moe_num_experts)
    option("--moe-top-k", "all" if args.moe_top_k is None else args.moe_top_k)
    option("--moe-router-balance-weight", f"{args.moe_router_balance_weight:g}")
    option("--moe-gate-temperature", f"{args.moe_gate_temperature:g}")
    option(
        "--moe-expert-hidden-dims",
        ",".join(map(str, args.moe_expert_hidden_dims)),
    )
    option("--moe-router-hidden-dim", args.moe_router_hidden_dim)
    option("--moe-routing-mode", args.moe_routing_mode)
    option(
        "--judge-hidden-dims",
        ",".join(map(str, args.judge_hidden_dims))
        if args.judge_hidden_dims else "none",
    )
    option("--races-per-batch", args.races_per_batch)
    option("--learning-rate", f"{fine_tune_learning_rate:.10g}")
    option("--weight-decay", f"{args.weight_decay:g}")
    option("--max-grad-norm", f"{args.max_grad_norm:g}")
    option("--dropout", f"{args.dropout:g}")
    option("--standardized-clip", f"{args.standardized_clip:g}")
    option("--epochs", fine_tune_epochs)
    option("--early-stopping-patience", args.early_stopping_patience)
    option("--seed", args.seed)
    option("--device", args.device)
    option("--initial-checkpoint", checkpoint)
    option("--trainable-components", ",".join(trainable_components))
    option("--output", fine_tuned_output)
    return " \\\n  ".join(lines), fine_tuned_output, fine_tune_learning_rate, fine_tune_epochs


def _selection(metrics: dict[str, float | int]) -> tuple[float]:
    """Select checkpoints and stop solely on exact predicted top-three sets."""
    return (float(metrics["exact_top3_set_rate"]),)


def _competition_population(
    frame: pd.DataFrame,
    competition_ids: list[int] | None,
) -> pd.DataFrame:
    """Restrict the population before creating chronological partitions."""
    if competition_ids is None:
        return frame
    competition = pd.to_numeric(frame["competition_id"], errors="coerce")
    return frame.loc[competition.isin(competition_ids)].copy()


def _exclude_invalid_top3_races(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[int]]:
    """Drop races that cannot supply a valid three-positive top-three target."""
    if "top3_mask" not in frame.columns:
        raise ValueError(
            "race_runners.top3_mask is required for top-3 training and validation"
        )
    top3 = pd.to_numeric(frame["top3_mask"], errors="coerce")
    valid_values = top3.isin((0, 1))
    positive_counts = top3.where(valid_values).groupby(frame["race_id"]).sum()
    row_counts = frame.groupby("race_id").size()
    invalid_ids = set(row_counts.index[
        row_counts.lt(4) | positive_counts.ne(3)
    ].tolist())
    invalid_ids.update(frame.loc[~valid_values, "race_id"].tolist())
    winners = pd.to_numeric(frame["is_winner"], errors="coerce").eq(1)
    invalid_ids.update(frame.loc[winners & top3.ne(1), "race_id"].tolist())
    ordered_invalid_ids = sorted(map(int, invalid_ids))
    filtered = frame.loc[~frame["race_id"].isin(invalid_ids)].copy()
    if filtered.empty:
        raise ValueError(
            "No valid top-3 races remain after excluding races without at least "
            "four runners, exactly three top3_mask=1 rows, and a top-three winner"
        )
    return filtered, ordered_invalid_ids


def available_trainable_components(
    model: RaceMixtureOfExpertsFeatureMap,
) -> tuple[str, ...]:
    """Return the components that have trainable parameters in this model."""
    components = [f"expert_{index}" for index in range(model.num_experts)]
    if model.router is not None:
        components.append("router")
    if model.judge is not None:
        components.append("judge")
    return tuple(components)


def set_trainable_components(
    model: RaceMixtureOfExpertsFeatureMap, components: list[str],
) -> tuple[str, ...]:
    """Freeze all components except the explicitly selected ones."""
    available = available_trainable_components(model)
    selected = tuple(dict.fromkeys(
        component.strip()
        for value in components
        for component in value.split(",")
        if component.strip()
    ))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(
            "Unknown --trainable-components: " + ", ".join(unknown)
            + ". Available: " + ", ".join(available)
        )
    if not selected:
        raise ValueError("--trainable-components must include at least one component")
    for name, parameter in model.named_parameters():
        component = (
            f"expert_{name.split('.', 2)[1]}"
            if name.startswith("experts.")
            else name.split(".", 1)[0]
        )
        parameter.requires_grad_(component in selected)
    return selected


def freeze_base_model_for_judge(model: RaceMixtureOfExpertsFeatureMap) -> None:
    """Freeze every base component while retaining only judge parameters."""
    if model.judge is None:
        raise ValueError("Cannot freeze the base model when the judge is disabled")
    set_trainable_components(model, ["judge"])


def load_initial_checkpoint(
    model: RaceMixtureOfExpertsFeatureMap,
    path: Path,
    *,
    raw_features: list[str],
    model_features: list[str],
    zeroed_features: list[str],
    device: torch.device,
) -> None:
    """Load compatible base weights, allowing a newly added judge to initialize."""
    checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "race_winner_moe_feature_map":
        raise ValueError("--initial-checkpoint is not a feature-map MoE checkpoint")
    required_contract = {
        "raw_feature_columns": raw_features,
        "model_feature_columns": model_features,
        "zeroed_features": zeroed_features,
    }
    for key, expected in required_contract.items():
        if list(checkpoint.get(key, ())) != expected:
            raise ValueError(f"--initial-checkpoint has incompatible {key}")
    saved_config = checkpoint.get("model_config", {})
    current_config = model.model_config
    required_config = {
        "feature_count": current_config.feature_count,
        "num_experts": current_config.num_experts,
        "expert_hidden_dims": list(current_config.expert_hidden_dims),
        "router_hidden_dim": current_config.router_hidden_dim,
        "feature_expert_map": {
            str(index): list(indices)
            for index, indices in enumerate(current_config.feature_map)
        },
        "routing_mode": current_config.routing_mode,
        "gate_temperature": current_config.gate_temperature,
        "router_feature_indices": list(current_config.router_feature_indices),
    }
    for key, expected in required_config.items():
        if saved_config.get(key) != expected:
            raise ValueError(f"--initial-checkpoint has incompatible model_config.{key}")
    saved_judge_dims = saved_config.get("judge_hidden_dims", [])
    if saved_judge_dims and list(current_config.judge_hidden_dims) != saved_judge_dims:
        raise ValueError("--initial-checkpoint has incompatible judge_hidden_dims")
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if unexpected or any(not name.startswith("judge.") for name in missing):
        raise ValueError("--initial-checkpoint model state is incompatible")


def _run_epoch(
    model: RaceMixtureOfExpertsFeatureMap,
    optimizer: torch.optim.Optimizer,
    x: np.ndarray,
    y: np.ndarray,
    indices: dict[int, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    ranking_values, balance_values, total_values = [], [], []
    for groups in batches(indices, args.races_per_batch, rng):
        bx, by, valid = pad_batch(x, y, groups, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(bx, valid, return_diagnostics=True)
        ranking, _ = raceformer_losses(output["logits"], by, valid)
        dense = output["dense_router_weights"]
        weights = dense[valid]
        if weights.numel() > 0 and weights.shape[-1] > 1:
            mean_load = weights.mean(dim=0)
            balance = weights.shape[-1] * mean_load.square().sum() - 1.0
        else:
            balance = dense.new_zeros(())
        total = ranking + args.moe_router_balance_weight * balance
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        ranking_values.append(float(ranking.detach()))
        balance_values.append(float(balance.detach()))
        total_values.append(float(total.detach()))
    return {
        "main_ranking_loss": float(np.mean(ranking_values)),
        "router_balance_loss": float(np.mean(balance_values)),
        "total_loss": float(np.mean(total_values)),
    }


def _evaluate_top3_model(
    model: RaceMixtureOfExpertsFeatureMap,
    x: np.ndarray,
    y: np.ndarray,
    race_ids: np.ndarray,
    indices: dict[int, np.ndarray],
    row_frame: pd.DataFrame,
    races_per_batch: int,
    device: torch.device,
) -> tuple[dict[str, float | int], dict[str, object], pd.DataFrame]:
    """Evaluate predicted top-three sets against top3_mask for every runner."""
    model.eval()
    ordered_rows: list[int] = []
    probabilities: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    dense_weights: list[np.ndarray] = []
    selected: list[np.ndarray] = []
    expert_logits: list[np.ndarray] = []
    race_expert_logits: list[np.ndarray] = []
    race_dense_weights: list[np.ndarray] = []
    winner_indices: list[int] = []
    with torch.inference_mode():
        for groups in batches(indices, races_per_batch):
            bx, by, valid = pad_batch(x, y, groups, device)
            output = model(bx, valid, return_diagnostics=True)
            for batch_index, rows in enumerate(groups):
                count = len(rows)
                ordered_rows.extend(map(int, rows))
                probabilities.append(
                    torch.sigmoid(output["logits"][batch_index, :count])
                    .cpu().numpy()
                )
                race_weights = output["router_weights"][batch_index, :count].cpu().numpy()
                race_dense = output["dense_router_weights"][batch_index, :count].cpu().numpy()
                race_experts = output["expert_logits"][batch_index, :count].cpu().numpy()
                weights.append(race_weights)
                dense_weights.append(race_dense)
                selected.append(output["selected_experts"][batch_index, :count].cpu().numpy())
                expert_logits.append(race_experts)
                race_expert_logits.append(race_experts)
                race_dense_weights.append(race_dense)
                race_winners = pd.to_numeric(
                    row_frame.iloc[rows]["is_winner"], errors="coerce"
                ).to_numpy()
                winner_indices.append(int(np.flatnonzero(race_winners == 1)[0]))

    ordered = np.asarray(ordered_rows, dtype=np.int64)
    probability = np.concatenate(probabilities)
    metrics = probability_metrics(y[ordered].astype(np.int64), probability, race_ids[ordered])
    prediction = row_frame.iloc[ordered][
        ["race_id", "runner_number", "top3_mask"]
    ].copy().reset_index(drop=True)
    prediction["top3_probability"] = probability
    prediction["predicted_top3"] = (
        prediction.groupby("race_id")["top3_probability"]
        .rank(method="first", ascending=False).le(3).astype(int)
    )
    diagnostics = routing_diagnostics(
        np.concatenate(weights), np.concatenate(selected),
        np.concatenate(expert_logits), row_frame.iloc[ordered].reset_index(drop=True),
        np.concatenate(dense_weights), race_expert_logits, race_dense_weights,
        winner_indices,
    )
    return metrics, diagnostics, prediction


def _split_range(frame: pd.DataFrame, race_ids_values: list[int]) -> dict[str, Any]:
    selected = frame.loc[frame["race_id"].isin(race_ids_values)]
    return {
        "races": len(race_ids_values),
        "first_race_id": race_ids_values[0],
        "last_race_id": race_ids_values[-1],
        "start_time": str(selected["start_time_iso"].iloc[0]),
        "end_time": str(selected["start_time_iso"].iloc[-1]),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    train_competitions = (
        None if args.train_competition_ids is None
        else sorted(set(args.train_competition_ids))
    )
    validation_competitions = (
        None if args.validation_competition_ids is None
        else sorted(set(args.validation_competition_ids))
    )
    if train_competitions != validation_competitions:
        raise ValueError(
            "--train-competition-id and --validation-competition-id must match; "
            "use a separate run for cross-competition evaluation"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    configured_features, configured_zeroed = load_feature_manifest(args.features_json)
    features, market_excluded = market_blind_features(configured_features, include_market=args.include_market_features)
    zeroed = [name for name in configured_zeroed if name in features]

    frame = _competition_population(
        load_finished_winner_rows(args.db, features),
        train_competitions,
    )
    original_rows = len(frame)
    frame, invalid_top3_race_ids = _exclude_invalid_top3_races(frame)
    if invalid_top3_race_ids:
        print(
            f"invalid_top3_races_skipped={len(invalid_top3_race_ids):,} "
            f"rows_skipped={original_rows - len(frame):,} "
            f"sample_race_ids={json.dumps(invalid_top3_race_ids[:10])}",
            flush=True,
        )
    train_ids, validation_ids, test_ids = chronological_race_ids(frame, args.validation_races, args.test_races)

    partitions = {
        "training": frame.loc[frame["race_id"].isin(train_ids)].copy(),
        "validation": frame.loc[frame["race_id"].isin(validation_ids)].copy(),
        "test": frame.loc[frame["race_id"].isin(test_ids)].copy(),
    }

    unavailable_features = [
        name for name in features
        if not pd.to_numeric(partitions["training"][name], errors="coerce").notna().any()
    ]
    if unavailable_features:
        features = [name for name in features if name not in unavailable_features]
        zeroed = [name for name in zeroed if name in features]
        print(f"unavailable_training_features_excluded={len(unavailable_features)} {json.dumps(unavailable_features)}", flush=True)

    if not features:
        raise ValueError("No numerical features have training coverage")

    raw = {name: numeric_matrix(part, features) for name, part in partitions.items()}
    preprocessing = fit_raceformer_preprocessor(raw["training"], features, clip=args.standardized_clip, layoff_bucket_mode="none")
    values = {
        name: transform_raceformer(matrix, part["race_id"].to_numpy(dtype=np.int64), features, zeroed, preprocessing)
        for (name, part), matrix in zip(partitions.items(), raw.values())
    }
    expanded_features = model_feature_columns(features, preprocessing)
    arrays = {}
    for name, part in partitions.items():
        race_id_array = part["race_id"].to_numpy(dtype=np.int64)
        arrays[name] = (values[name], part["top3_mask"].to_numpy(dtype=np.float32), race_id_array, race_indices(race_id_array))

    feature_map = load_feature_expert_map(args.feature_map_json, features, args.moe_num_experts)
    feature_map = expand_feature_map_to_model_features(feature_map, features, expanded_features)
    router_features = load_router_feature_indices(args.feature_map_json, features)
    router_feature_indices = expand_feature_indices_to_model_features(
        router_features, features, expanded_features,
    )
    config = FeatureMappedRaceWinnerConfig(
        feature_count=len(expanded_features),
        num_experts=args.moe_num_experts,
        top_k=(None if args.moe_routing_mode == "fixed_uniform" else args.moe_top_k),
        gate_temperature=args.moe_gate_temperature,
        expert_hidden_dims=args.moe_expert_hidden_dims,
        router_hidden_dim=args.moe_router_hidden_dim,
        dropout=args.dropout,
        routing_mode=args.moe_routing_mode,
        feature_map=feature_map,
        router_feature_indices=router_feature_indices,
        judge_hidden_dims=args.judge_hidden_dims,
    )
    model = RaceMixtureOfExpertsFeatureMap(config).to(device)
    if args.initial_checkpoint is not None:
        load_initial_checkpoint(
            model,
            args.initial_checkpoint,
            raw_features=features,
            model_features=expanded_features,
            zeroed_features=zeroed,
            device=device,
        )
    available_components = available_trainable_components(model)
    requested_components = (
        ["judge"] if args.freeze_base_model
        else args.trainable_components
    )
    if requested_components is None:
        selected_components = available_components
    else:
        selected_components = set_trainable_components(model, requested_components)
    if (
        set(selected_components) != set(available_components)
        and args.initial_checkpoint is None
    ):
        raise ValueError(
            "A subset of --trainable-components requires --initial-checkpoint"
        )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    hyperparameters = {
        "arguments": training_arguments(args),
        "effective": {
            "model_type": "moe_feature_map",
            "objective": "top3_mask_ranking",
            "checkpoint_selection_metric": "exact_top3_set_rate",
            "early_stopping_metric": "exact_top3_set_rate",
            "device": str(device),
            "market_blind": not args.include_market_features,
            "raw_feature_count": len(features),
            "model_feature_count": len(expanded_features),
            "zeroed_feature_count": len(zeroed),
            "unavailable_training_features": unavailable_features,
            "invalid_top3_races_skipped": len(invalid_top3_race_ids),
            "invalid_top3_rows_skipped": original_rows - len(frame),
            "training_races": len(train_ids),
            "validation_races": len(validation_ids),
            "validation_rows": len(partitions["validation"]),
            "test_races": len(test_ids),
            "competition_ids": train_competitions,
            "router_feature_count": len(router_feature_indices),
            "effective_top_k": config.top_k,
            "available_trainable_components": list(available_components),
            "trainable_components": list(selected_components),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
            "total_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
    }
    print(
        "TRAINING HYPERPARAMETERS\n"
        + json.dumps(hyperparameters, indent=2, sort_keys=True),
        flush=True,
    )

    print(
        "RACE WINNER FEATURE-MAP MOE EXPERIMENT\n"
        f"model_type=moe_feature_map objective=top3_mask_ranking "
        f"market_blind={not args.include_market_features} raw_features={len(features)} model_features={len(expanded_features)}\n"
        f"train_races={len(train_ids):,} validation_races={len(validation_ids):,} sealed_test_races={len(test_ids):,} device={device}\n"
        "validation_target=top3_mask all_runners_retained=true\n"
        "checkpoint_selection=exact_top3_set_rate early_stopping=exact_top3_set_rate\n"
        f"competition_ids={train_competitions or 'all'} "
        "split_population=shared\n"
        f"router_features={len(router_feature_indices)} "
        f"num_experts={args.moe_num_experts} top_k={args.moe_top_k if args.moe_top_k is not None else 'all'} routing_mode={config.routing_mode} temperature={args.moe_gate_temperature:g} balance_weight={args.moe_router_balance_weight:g} expert_hidden_dims={list(args.moe_expert_hidden_dims)} judge_hidden_dims={list(args.judge_hidden_dims)} "
        f"initial_checkpoint={args.initial_checkpoint or 'none'} trainable_components={','.join(selected_components)}",
        flush=True,
    )
    print(
        f"available_trainable_components={list(available_components)}\n"
        f"trainable_components={list(selected_components)}",
        flush=True,
    )
    print(f"feature_map_path={args.feature_map_json}", flush=True)
    #print(f"feature_expert_distribution={json.dumps({str(i): list(expanded_features[j] for j in indices) for i, indices in enumerate(feature_map)})}", flush=True)

    best_state = None
    best_selection = None
    best_epoch = 0
    stale = 0
    history = []
    rng = np.random.default_rng(args.seed)
    train_x, train_y, _, train_index = arrays["training"]

    for epoch in range(1, args.epochs + 1):
        losses = _run_epoch(model, optimizer, train_x, train_y, train_index, args, device, rng)
        vx, vy, vids, vindices = arrays["validation"]
        validation_metrics, validation_diagnostics, _ = _evaluate_top3_model(model, vx, vy, vids, vindices, partitions["validation"], args.races_per_batch, device)
        selection = _selection(validation_metrics)
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        row = {
            "epoch": epoch,
            **losses,
            "validation_metrics": validation_metrics,
            "validation_router": {
                key: validation_diagnostics[key] for key in ("dominant_expert_rate", "gate_entropy", "average_number_of_active_experts", "router_balance_loss")
            },
            "best": improved,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} main_ranking_loss={losses['main_ranking_loss']:.5f} router_balance_loss={losses['router_balance_loss']:.5f} total_loss={losses['total_loss']:.5f} validation_top3_recall={validation_metrics['top3_recall']:.4f} exact_top3={validation_metrics['exact_top3_set_rate']:.4f} ndcg3={validation_metrics['ndcg3']:.4f} logloss={validation_metrics['logloss']:.5f} dominant={validation_diagnostics['dominant_expert_rate']:.3f} entropy={validation_diagnostics['gate_entropy']:.3f} best={'yes' if improved else 'no'} stale={stale}/{args.early_stopping_patience}",
            flush=True,
        )
        if stale >= args.early_stopping_patience:
            print(f"early_stopping epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    model.load_state_dict(best_state)
    results = {}
    diagnostics = {}
    predictions = {}
    for name in ("training", "validation", "test"):
        x, y, ids, indices = arrays[name]
        results[name], diagnostics[name], predictions[name] = _evaluate_top3_model(model, x, y, ids, indices, partitions[name], args.races_per_batch, device)

    output = args.output.resolve()
    checkpoint = {
        "checkpoint_type": "race_winner_moe_feature_map",
        "checkpoint_version": 1,
        "objective": "top3_mask_ranking",
        "target_column": "top3_mask",
        "model_state_dict": best_state,
        "model_config": {
            "feature_count": len(expanded_features),
            "num_experts": args.moe_num_experts,
            "top_k": args.moe_top_k,
            "expert_hidden_dims": list(args.moe_expert_hidden_dims),
            "router_hidden_dim": args.moe_router_hidden_dim,
            "feature_expert_map": {str(i): list(indices) for i, indices in enumerate(feature_map)},
            "router_feature_indices": list(router_feature_indices),
            "judge_hidden_dims": list(args.judge_hidden_dims),
            "routing_mode": args.moe_routing_mode,
            "gate_temperature": args.moe_gate_temperature,
        },
        "feature_map_path": str(args.feature_map_json) if args.feature_map_json else None,
        "raw_feature_columns": features,
        "model_feature_columns": expanded_features,
        "preprocessing": preprocessing,
        "zeroed_features": zeroed,
        "market_features_excluded": market_excluded,
        "partition": {
            "training_race_ids": train_ids,
            "validation_race_ids": validation_ids,
            "test_race_ids": test_ids,
        },
        "best_epoch": best_epoch,
        "initial_checkpoint": (
            str(args.initial_checkpoint.resolve())
            if args.initial_checkpoint is not None else None
        ),
        "freeze_base_model": args.freeze_base_model,
        "trainable_components": list(selected_components),
        "history": history,
        "metrics": results,
        "router_diagnostics": diagnostics,
        "training_config": training_arguments(args),
        "training_hyperparameters": hyperparameters,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp")
    torch.save(checkpoint, temp)
    temp.replace(output)

    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps({key: value for key, value in checkpoint.items() if key not in {"model_state_dict"}}, indent=2, default=str) + "\n")

    print("\nMODEL RESULTS")
    print("split       top3_recall exact_top3    ndcg3  logloss")
    for name in ("training", "validation", "test"):
        metric = results[name]
        print(
            f"{name:<10} {metric['top3_recall']:>11.2%} "
            f"{metric['exact_top3_set_rate']:>10.2%} "
            f"{metric['ndcg3']:>8.4f} {metric['logloss']:>8.4f}"
        )
    for name in ("training", "validation", "test"):
        diagnostic = diagnostics[name]
        print(f"\n{name.upper()} ROUTER DIAGNOSTICS")
        for expert in range(args.moe_num_experts):
            print(
                f"Expert {expert}: usage={diagnostic['expert_usage_rate'][expert]:.1%} mean_gate={diagnostic['mean_gate_weight_per_expert'][expert]:.1%} top1_frequency={diagnostic['top1_routed_expert_frequency'][expert]:.1%}; {diagnostic['specialisation_descriptions'][expert]}"
            )
        print(
            f"gate_entropy={diagnostic['gate_entropy']:.4f} dominant_expert_rate={diagnostic['dominant_expert_rate']:.2%} average_active_experts={diagnostic['average_number_of_active_experts']:.2f} router_balance_loss={diagnostic['router_balance_loss']:.5f} max_abs_expert_correlation={diagnostic['maximum_absolute_pairwise_expert_correlation']}"
        )
        for warning in collapse_warnings(diagnostic, 0.80, 0.95):
            print("WARNING: " + warning)
    predictor = Path(__file__).resolve().with_name("predict_moe_winner_ranker_feature_map.py")
    prediction_command = shlex.join([
        sys.executable,
        str(predictor),
        "--checkpoint", str(output),
        "--db", str(args.db.resolve()),
        "--race-id", "12345",
    ])
    continuation_command, continuation_output, continuation_lr, continuation_epochs = (
        fine_tune_command(args, output, selected_components)
    )
    reproduction_command = training_command(args, output, selected_components)
    base_training_command, base_checkpoint, lineage_depth = initial_training_command(
        args, output, selected_components,
    )
    current_training_section = (
        "\n\nHOW THIS CHECKPOINT WAS TRAINED:\n"
        f"{reproduction_command}"
        if args.initial_checkpoint is not None else ""
    )
    print(
        f"\nsaved_checkpoint={output}\n"
        f"report={report_path}\n\n"
        "HOW TO PREDICT (replace 12345 with the race ID):\n"
        f"{prediction_command}\n\n"
        "HOW THE BASE MODEL WAS INITIALLY TRAINED "
        "(no initial checkpoint):\n"
        f"base_checkpoint={base_checkpoint} lineage_depth={lineage_depth}\n"
        f"{base_training_command}"
        f"{current_training_section}\n\n"
        "HOW TO FINE-TUNE THIS CHECKPOINT:\n"
        f"fine_tune_learning_rate={continuation_lr:.10g} "
        f"fine_tune_epochs={continuation_epochs} "
        f"fine_tune_output={continuation_output}\n"
        f"{continuation_command}\n"
        "NOTE: Fine-tuning reuses the same chronological validation and test "
        "cohorts. Its test result is not a new sealed estimate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
