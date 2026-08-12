#!/usr/bin/env python3
"""Explain and sanity-check one race scored by a RaceFormerTop3 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from predict_raceformer import load_checkpoint, load_race
from src.config import DEFAULT_DB
from src.dataset import load_feature_manifest
from src.model.raceformer import RaceFormerTop3
from src.prediction import market_rank_scores
from src.raceformer_preprocessing import (
    model_feature_columns,
    raceformer_base_diagnostics,
    transform_raceformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show preprocessing, representations, predictions, attributions, "
            "ablations, and invariance checks for one RaceFormer race."
        )
    )
    parser.add_argument("--race-id", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument(
        "--features-json", type=Path,
        help=(
            "Optional feature manifest. Ordered features must match the checkpoint; "
            "zeroed_features override this diagnostic run only."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top-features", type=int, default=8)
    parser.add_argument(
        "--top-clipped-values", type=int, default=25,
        help="Maximum clipped runner-feature values shown in Stage 1; 0 shows all.",
    )
    parser.add_argument(
        "--attribution-runner-number", type=int,
        help="Runner to explain; defaults to the worst-ranked actual top-three runner.",
    )
    parser.add_argument("--attribution-steps", type=int, default=32)
    parser.add_argument(
        "--feature-ablation", action=argparse.BooleanOptionalAction, default=True,
        help="Measure prediction and ranking changes after neutralizing each feature.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    result = np.empty(len(order), dtype=np.int64)
    result[order] = np.arange(1, len(order) + 1)
    return result


def prepare(
    frame: pd.DataFrame, checkpoint: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    features = list(checkpoint.get("raw_feature_columns", checkpoint["feature_columns"]))
    raw = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    scaled = transform_raceformer(
        raw,
        frame["race_id"].to_numpy(dtype=np.int64),
        features,
        list(checkpoint.get("zeroed_features", [])),
        checkpoint.get("preprocessing"),
        legacy_median=np.asarray(checkpoint["median"], dtype=np.float32),
        legacy_scale=np.asarray(checkpoint["scale"], dtype=np.float32),
    )
    return raw, scaled


def forward(
    model: RaceFormerTop3, values: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(values).unsqueeze(0).to(device)
    valid = torch.ones((1, len(values)), dtype=torch.bool, device=device)
    return model(x, valid)[0], valid


def traced_forward(
    model: RaceFormerTop3, values: np.ndarray, device: torch.device
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    captured: dict[str, np.ndarray] = {}
    handles = []

    def save(name: str):
        def hook(_module: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            captured[f"{name}_input"] = inputs[0].detach().cpu().numpy()[0]
            captured[f"{name}_output"] = output.detach().cpu().numpy()[0]
        return hook

    handles.append(model.feature_encoder.register_forward_hook(save("runner_encoder")))
    if model.race_transformer is not None:
        handles.append(model.race_transformer.register_forward_hook(save("transformer")))
    handles.append(model.prediction_head.register_forward_hook(save("prediction_head")))
    try:
        with torch.inference_mode():
            logits, _ = forward(model, values, device)
    finally:
        for handle in handles:
            handle.remove()
    return logits.cpu().numpy(), captured


def choose_runner(frame: pd.DataFrame, probabilities: np.ndarray, requested: int | None) -> int:
    numbers = pd.to_numeric(frame["runner_number"], errors="raise").to_numpy(dtype=int)
    if requested is not None:
        matches = np.flatnonzero(numbers == requested)
        if not len(matches):
            raise ValueError(f"Runner number {requested} is not in this race")
        return int(matches[0])
    labels = pd.to_numeric(frame["top3_mask"], errors="coerce").to_numpy()
    positives = np.flatnonzero(labels == 1)
    if len(positives):
        return int(positives[np.argmin(probabilities[positives])])
    return int(np.argmax(probabilities))


def integrated_gradients(
    model: RaceFormerTop3, values: np.ndarray, runner: int, steps: int,
    device: torch.device,
) -> np.ndarray:
    if steps < 1:
        raise ValueError("--attribution-steps must be positive")
    source = torch.from_numpy(values).unsqueeze(0).to(device)
    baseline = torch.zeros_like(source)  # zero is the median runner after preprocessing
    valid = torch.ones(source.shape[:2], dtype=torch.bool, device=device)
    gradient_sum = torch.zeros_like(source)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=device):
        point = (baseline + alpha * (source - baseline)).detach().requires_grad_(True)
        score = model(point, valid)[0, runner]
        gradient_sum += torch.autograd.grad(score, point)[0].detach()
    return ((source - baseline) * gradient_sum / steps)[0].cpu().numpy()


def print_checkpoint(model: RaceFormerTop3, checkpoint: dict[str, Any]) -> None:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    preprocessing = checkpoint.get("preprocessing") or {}
    price_transforms = [
        name for name in ("open_price", "fluc1", "fluc2")
        if name in preprocessing.get("log1p_features", [])
    ]
    print("RACEFORMER RACE DEBUGGER")
    print(
        f"variant={model.variant} features={model.feature_count} parameters={parameters:,} "
        f"model_dim={model.model_dim} heads={model.heads} layers={model.layers}"
    )
    print(
        f"best_epoch={checkpoint.get('best_epoch', '-')} "
        f"checkpoint_metric={checkpoint.get('checkpoint_metric', '-')} "
        "historical_context=OFF icl=OFF"
    )
    print(
        f"preprocessing_version={preprocessing.get('version', 1)} "
        f"standardized_clip={preprocessing.get('clip', 'none')} "
        f"log1p_current_prices={price_transforms or 'none'} "
        f"zeroed_features={checkpoint.get('zeroed_features', [])}"
    )


def print_provenance(race_id: int, checkpoint: dict[str, Any]) -> None:
    partition = checkpoint.get("partition", {})
    training_ids = partition.get("training_race_ids")
    validation_ids = partition.get("validation_race_ids")
    membership = "not_recorded"
    if training_ids is not None and race_id in set(map(int, training_ids)):
        membership = "training"
    elif validation_ids is not None and race_id in set(map(int, validation_ids)):
        membership = "validation"
    elif training_ids is not None or validation_ids is not None:
        membership = "outside_checkpoint_partition"
    print(
        f"checkpoint_partition={partition.get('mode', 'not_recorded')} "
        f"race_membership={membership} "
        f"training_competitions={partition.get('training_competition_ids')} "
        f"validation_competitions={partition.get('validation_competition_ids')}"
    )


def print_inputs(
    frame: pd.DataFrame, raw: np.ndarray, scaled: np.ndarray, features: list[str],
    count: int, diagnostics: dict[str, np.ndarray], top_clipped: int,
) -> None:
    print("\nSTAGE 1 — INPUT AND PREPROCESSING")
    print(
        f"matrix={len(frame)} runners x {len(features)} features; "
        f"raw_missing={int(np.isnan(raw).sum())}; "
        f"standardized_nonfinite={int((~np.isfinite(scaled)).sum())}"
    )
    rows = []
    base = scaled[:, :len(features)]
    for index, values in enumerate(base):
        top = np.argsort(-np.abs(values), kind="stable")[:max(0, count)]
        rows.append({
            "No.": frame.iloc[index]["runner_number"],
            "Runner": frame.iloc[index]["runner_name"],
            "largest standardized inputs": ", ".join(
                f"{features[column]}={values[column]:+.2f}" for column in top
            ),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    clipped_rows = []
    clipped_indices = np.argwhere(diagnostics["was_clipped"])
    for runner_index, feature_index in clipped_indices:
        unclipped = float(diagnostics["unclipped_standardized"][runner_index, feature_index])
        clipped = float(diagnostics["clipped_standardized"][runner_index, feature_index])
        clipped_rows.append({
            "No.": frame.iloc[runner_index]["runner_number"],
            "Runner": frame.iloc[runner_index]["runner_name"],
            "feature": features[feature_index],
            "missing": bool(diagnostics["was_missing"][runner_index, feature_index]),
            "raw_value": diagnostics["raw"][runner_index, feature_index],
            "transformed_value": diagnostics["transformed"][runner_index, feature_index],
            "training_median": diagnostics["training_median"][feature_index],
            "training_scale": diagnostics["training_scale"][feature_index],
            "unclipped_z": unclipped,
            "clipped_z": clipped,
            "clip_excess": abs(unclipped) - abs(clipped),
        })
    print(f"\nClipped raw-feature values={len(clipped_rows)}")
    if clipped_rows:
        clipped_frame = pd.DataFrame(clipped_rows).sort_values(
            "clip_excess", ascending=False, kind="stable"
        )
        if top_clipped:
            clipped_frame = clipped_frame.head(top_clipped)
        print(
            clipped_frame.drop(columns="clip_excess").to_string(
                index=False, float_format=lambda value: f"{value:.5f}"
            )
        )
        if top_clipped and len(clipped_rows) > top_clipped:
            print(
                f"showing={top_clipped} hidden={len(clipped_rows) - top_clipped} "
                "(use --top-clipped-values 0 to show all)"
            )
    else:
        clip = diagnostics.get("clip")
        threshold = "none" if clip is None or not np.isfinite(clip) else f"+/-{clip:g}"
        print(f"No base-feature values reached the checkpoint clipping threshold ({threshold}).")


def representation_report(
    frame: pd.DataFrame, trace: dict[str, np.ndarray], model: RaceFormerTop3
) -> None:
    encoded = trace["runner_encoder_output"]
    print("\nSTAGE 2 — REPRESENTATIONS")
    print(f"runner_encoder output={encoded.shape}; mean_norm={np.linalg.norm(encoded, axis=1).mean():.4f}")
    if model.variant == "independent":
        print("field self-attention=OFF; every score depends only on that runner's features")
        return
    transformed = trace["transformer_output"]
    uses_race_token = model.variant in {"race_token", "market_residual"}
    runner_context = transformed[1:] if uses_race_token else transformed
    delta = np.linalg.norm(runner_context - encoded, axis=1)
    shown = pd.DataFrame({
        "No.": frame["runner_number"].to_numpy(),
        "Runner": frame["runner_name"].to_numpy(),
        "context_shift_norm": delta,
    })
    print(f"transformer output={transformed.shape}")
    if uses_race_token:
        print(f"race_token_norm={np.linalg.norm(transformed[0]):.4f}")
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def prediction_report(
    frame: pd.DataFrame,
    logits: np.ndarray,
    anchor_logits: np.ndarray | None = None,
    residual_logits: np.ndarray | None = None,
) -> pd.DataFrame:
    probability = 1.0 / (1.0 + np.exp(-logits))
    result = frame[[
        "runner_number", "runner_name", "fluc2", "finish_place", "top3_mask"
    ]].copy()
    result["logit"] = logits
    result["probability"] = probability
    result["model_rank"] = ranks(probability)
    market = pd.to_numeric(result["fluc2"], errors="coerce").to_numpy()
    result["market_rank"] = ranks(market_rank_scores(market))
    if anchor_logits is not None and residual_logits is not None:
        result["anchor_logit"] = anchor_logits
        result["residual_logit"] = residual_logits
    print("\nSTAGE 3 — FINAL SCORES")
    print(
        result.sort_values(["model_rank", "runner_number"]).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print(f"sum_probability={probability.sum():.4f} (training target is approximately 3.0)")
    if residual_logits is not None:
        print(
            f"mean_abs_residual_logit={np.abs(residual_logits).mean():.6f} "
            f"max_abs_residual_logit={np.abs(residual_logits).max():.6f}"
        )
    return result


def attribution_report(
    model: RaceFormerTop3, frame: pd.DataFrame, values: np.ndarray,
    probabilities: np.ndarray, features: list[str], requested: int | None,
    steps: int, count: int, device: torch.device,
) -> None:
    runner = choose_runner(frame, probabilities, requested)
    attribution = integrated_gradients(model, values, runner, steps, device)
    own = attribution[runner]
    top = np.argsort(-np.abs(own), kind="stable")[:max(0, count)]
    print("\nSTAGE 4 — INTEGRATED-GRADIENT ATTRIBUTION")
    print(
        f"explained_runner={int(frame.iloc[runner]['runner_number'])} "
        f"{frame.iloc[runner]['runner_name']} baseline=all-features-at-training-median steps={steps}"
    )
    shown = pd.DataFrame({
        "feature": [features[i] for i in top],
        "standardized_value": values[runner, top],
        "logit_attribution": own[top],
    })
    print(shown.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    other = np.delete(attribution.sum(axis=1), runner)
    print(
        f"other_runner_total_attribution={other.sum():+.5f} "
        "(non-zero means this runner's score depends on the field)"
    )


def ablation_report(
    model: RaceFormerTop3, values: np.ndarray, baseline_probability: np.ndarray,
    features: list[str], count: int, device: torch.device,
) -> None:
    baseline_top3 = set(np.argsort(-baseline_probability, kind="stable")[:3].tolist())
    rows = []
    with torch.inference_mode():
        for feature_index, feature in enumerate(features):
            ablated = values.copy()
            ablated[:, feature_index] = 0.0
            logits, _ = forward(model, ablated, device)
            probability = torch.sigmoid(logits).cpu().numpy()
            rows.append({
                "feature": feature,
                "mean_abs_probability_delta": np.abs(probability - baseline_probability).mean(),
                "max_abs_probability_delta": np.abs(probability - baseline_probability).max(),
                "top3_changed": baseline_top3 != set(np.argsort(-probability)[:3].tolist()),
            })
    shown = pd.DataFrame(rows).sort_values(
        "mean_abs_probability_delta", ascending=False, kind="stable"
    ).head(max(0, count))
    print("\nSTAGE 5 — WHOLE-FEATURE ABLATION")
    print("Each feature is replaced by its standardized neutral value for every runner.")
    print(shown.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def invariance_report(
    model: RaceFormerTop3, values: np.ndarray, baseline: np.ndarray,
    seed: int, device: torch.device,
) -> None:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(values))
    inverse = np.argsort(permutation)
    with torch.inference_mode():
        permuted_logits, _ = forward(model, values[permutation], device)
        permuted = torch.sigmoid(permuted_logits).cpu().numpy()[inverse]
        padded_x = np.concatenate([values, np.zeros((3, values.shape[1]), np.float32)])
        x = torch.from_numpy(padded_x).unsqueeze(0).to(device)
        valid = torch.zeros((1, len(padded_x)), dtype=torch.bool, device=device)
        valid[:, :len(values)] = True
        padded = torch.sigmoid(model(x, valid)[0, :len(values)]).cpu().numpy()
    print("\nSTAGE 6 — STRUCTURAL SANITY CHECKS")
    print(f"runner_permutation_max_probability_delta={np.max(np.abs(permuted - baseline)):.9f}")
    print(f"padding_max_probability_delta={np.max(np.abs(padded - baseline)):.9f}")


def main() -> None:
    args = parse_args()
    if args.top_features < 0:
        raise ValueError("--top-features must be zero or positive")
    if args.top_clipped_values < 0:
        raise ValueError("--top-clipped-values must be zero or positive")
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    features = list(checkpoint.get("raw_feature_columns", checkpoint["feature_columns"]))
    if args.features_json is not None:
        manifest_features, manifest_zeroed = load_feature_manifest(args.features_json)
        if manifest_features != features:
            checkpoint_only = [name for name in features if name not in manifest_features]
            manifest_only = [name for name in manifest_features if name not in features]
            order_only = not checkpoint_only and not manifest_only
            detail = (
                "same columns but different order" if order_only else
                f"checkpoint_only={checkpoint_only} manifest_only={manifest_only}"
            )
            raise ValueError(
                "--features-json is incompatible with the checkpoint feature contract: "
                + detail
            )
        checkpoint = dict(checkpoint)
        checkpoint["zeroed_features"] = manifest_zeroed
        print(
            f"feature_manifest={args.features_json.resolve()} "
            f"features={len(features)} zeroed={len(manifest_zeroed)} "
            "checkpoint_modified=no diagnostic_override=yes",
            flush=True,
        )
    expanded_features = list(
        checkpoint.get(
            "model_feature_columns",
            model_feature_columns(features, checkpoint.get("preprocessing")),
        )
    )
    frame = load_race(args.db, args.race_id, features)
    raw, values = prepare(frame, checkpoint)
    diagnostics = raceformer_base_diagnostics(
        raw, features, checkpoint.get("preprocessing"),
        legacy_median=np.asarray(checkpoint["median"], dtype=np.float32),
        legacy_scale=np.asarray(checkpoint["scale"], dtype=np.float32),
    )
    logits, trace = traced_forward(model, values, device)
    probability = 1.0 / (1.0 + np.exp(-logits))

    print_checkpoint(model, checkpoint)
    print_provenance(args.race_id, checkpoint)
    print(
        f"race={args.race_id} {frame.iloc[0]['race_name']} "
        f"competition={int(frame.iloc[0]['competition_id'])} "
        f"start={frame.iloc[0]['start_time_iso']}"
    )
    print_inputs(
        frame, raw, values, features, args.top_features,
        diagnostics, args.top_clipped_values,
    )
    representation_report(frame, trace, model)
    anchor_logits = residual_logits = None
    if model.variant == "market_residual":
        with torch.inference_mode():
            x = torch.from_numpy(values).unsqueeze(0).to(device)
            valid = torch.ones((1, len(values)), dtype=torch.bool, device=device)
            _, anchor, residual = model.forward_parts(x, valid)
            anchor_logits = anchor[0].cpu().numpy()
            residual_logits = residual[0].cpu().numpy()
    result = prediction_report(frame, logits, anchor_logits, residual_logits)
    attribution_report(
        model, frame, values, probability, expanded_features,
        args.attribution_runner_number, args.attribution_steps,
        args.top_features, device,
    )
    if args.feature_ablation:
        ablation_report(
            model, values, probability, expanded_features, args.top_features, device
        )
    invariance_report(model, values, probability, args.seed, device)
    if args.output_csv:
        output = args.output_csv.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"saved={output}")


if __name__ == "__main__":
    main()
