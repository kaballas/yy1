#!/usr/bin/env python3
"""Compare baseline and MoE checkpoints on their identical chronological holdouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import DEFAULT_DB
from src.model.race_moe import build_race_winner_model
from src.race_moe_data import load_finished_winner_rows, numeric_matrix, race_indices
from src.race_moe_evaluation import collapse_warnings, evaluate_model
from src.raceformer_preprocessing import transform_raceformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--moe", type=Path, nargs="+", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--races-per-batch", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collapse-threshold", type=float, default=0.80)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load(path: Path, device: torch.device):
    checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "race_winner_moe":
        raise ValueError(f"Unsupported checkpoint type in {path}")
    model = build_race_winner_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def main() -> None:
    args = parse_args(); device = torch.device(args.device)
    paths = [args.baseline, *args.moe]
    loaded = [(path, *load(path, device)) for path in paths]
    reference = loaded[0][2]
    if reference["model_config"]["model_type"] != "baseline":
        raise ValueError("--baseline checkpoint is not a baseline model")
    contract_keys = ("raw_feature_columns", "model_feature_columns", "zeroed_features")
    for path, _, checkpoint in loaded[1:]:
        for key in contract_keys:
            if list(checkpoint[key]) != list(reference[key]):
                raise ValueError(f"{path} does not use identical {key}")
        for split in ("training", "validation", "test"):
            key = f"{split}_race_ids"
            if checkpoint["partition"][key] != reference["partition"][key]:
                raise ValueError(f"{path} does not use the identical {split} races")
        if checkpoint.get("training_objective") != reference.get("training_objective"):
            raise ValueError(f"{path} uses a different objective")
        if checkpoint.get("market_features_enabled") != reference.get("market_features_enabled"):
            raise ValueError(f"{path} differs in market-blind configuration")
        for key in ("median", "scale"):
            if not np.array_equal(
                np.asarray(checkpoint["preprocessing"][key]),
                np.asarray(reference["preprocessing"][key]), equal_nan=True,
            ):
                raise ValueError(f"{path} does not use identical fitted preprocessing")
        for key in (
            "encoder_hidden_dim", "representation_dim", "dropout",
            "expert_hidden_dims", "expert_context_conditioning",
        ):
            if checkpoint["model_config"][key] != reference["model_config"][key]:
                raise ValueError(f"{path} differs from baseline architecture at {key}")
    features = list(reference["raw_feature_columns"])
    all_ids = reference["partition"]["validation_race_ids"] + reference["partition"]["test_race_ids"]
    frame = load_finished_winner_rows(args.db, features)
    frame = frame.loc[frame["race_id"].isin(all_ids)].copy()
    rows = [] ; detail = {}
    for path, model, checkpoint in loaded:
        label = path.stem
        detail[label] = {}
        for split in ("validation", "test"):
            ids = checkpoint["partition"][f"{split}_race_ids"]
            part = frame.loc[frame["race_id"].isin(ids)].copy()
            raw = numeric_matrix(part, features)
            race_id_array = part["race_id"].to_numpy(dtype=np.int64)
            x = transform_raceformer(
                raw, race_id_array, features, checkpoint["zeroed_features"],
                checkpoint["preprocessing"],
            )
            y = part["is_winner"].to_numpy(dtype=np.float32)
            metrics, diagnostics, _ = evaluate_model(
                model, x, y, race_id_array, race_indices(race_id_array), part,
                args.races_per_batch, device,
            )
            detail[label][split] = {"metrics": metrics, "router_diagnostics": diagnostics}
            rows.append((label, checkpoint["model_config"]["model_type"], split, metrics))
    print("MODEL COMPARISON (complete chronological cohorts; Top-1 is primary)")
    print("model                         type       split        top1     top2     top3      mrr  logloss  avg_winner_p")
    for label, model_type, split, metric in rows:
        print(
            f"{label:<29} {model_type:<10} {split:<10} "
            f"{metric['top1_hit_rate']:>7.2%} {metric['top2_containment']:>8.2%} "
            f"{metric['top3_containment']:>8.2%} {metric['mrr']:>8.4f} "
            f"{metric['race_logloss']:>8.4f} {metric['average_winner_probability']:>13.4f}"
        )
    baseline_label = paths[0].stem
    baseline_validation = detail[baseline_label]["validation"]["metrics"]["top1_hit_rate"]
    baseline_test = detail[baseline_label]["test"]["metrics"]["top1_hit_rate"]
    print("\nMOE ROUTER / EXPERT VERDICT")
    for path in paths[1:]:
        label = path.stem
        validation = detail[label]["validation"]
        test = detail[label]["test"]
        warnings = collapse_warnings(
            validation["router_diagnostics"], args.collapse_threshold,
            args.correlation_threshold,
        ) + collapse_warnings(
            test["router_diagnostics"], args.collapse_threshold,
            args.correlation_threshold,
        )
        validation_delta = validation["metrics"]["top1_hit_rate"] - baseline_validation
        test_delta = test["metrics"]["top1_hit_rate"] - baseline_test
        print(
            f"{label}: validation_top1_delta={validation_delta:+.2%} "
            f"test_top1_delta={test_delta:+.2%}"
        )
        diagnostic = test["router_diagnostics"]
        for expert, description in enumerate(diagnostic["specialisation_descriptions"]):
            print(
                f"  Expert {expert}: overall_usage="
                f"{diagnostic['expert_usage_rate'][expert]:.1%}; {description}"
            )
        for warning in dict.fromkeys(warnings):
            print("  WARNING: " + warning)
        retain = validation_delta > 0 and test_delta >= 0 and not warnings
        print(
            "  recommendation=" + ("RETAIN for further controlled testing" if retain else "DISCARD as an improvement over baseline")
        )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(detail, indent=2) + "\n")
        print(f"output_json={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
