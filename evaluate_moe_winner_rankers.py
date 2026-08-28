#!/usr/bin/env python3
"""Compare baseline and MoE checkpoints on their identical chronological holdouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import DEFAULT_DB
from src.model.race_moe import build_race_winner_model
from src.race_moe_data import load_finished_winner_rows, numeric_matrix, race_indices
from src.race_moe_evaluation import collapse_warnings, evaluate_model
from src.raceformer_preprocessing import transform_raceformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--moe", "--challengers", type=Path, nargs="+", required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--races-per-batch", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--collapse-threshold", type=float, default=0.80)
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    return parser.parse_args()


def load(path: Path, device: torch.device):
    checkpoint = torch.load(path.resolve(), map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") != "race_winner_moe":
        raise ValueError(f"Unsupported checkpoint type in {path}")
    model = build_race_winner_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _mcnemar_exact(baseline_only: int, challenger_only: int) -> float:
    discordant = baseline_only + challenger_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, challenger_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def paired_comparison(
    baseline, challenger, samples: int, seed: int,
) -> dict[str, float | int | list[float]]:
    paired = baseline.merge(
        challenger, on="race_id", suffixes=("_baseline", "_challenger"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(challenger):
        raise ValueError("Paired comparison cohorts do not contain identical races")
    base_correct = paired["winner_rank_baseline"].eq(1).to_numpy(dtype=bool)
    challenge_correct = paired["winner_rank_challenger"].eq(1).to_numpy(dtype=bool)
    baseline_only = int(np.sum(base_correct & ~challenge_correct))
    challenger_only = int(np.sum(~base_correct & challenge_correct))
    differences = challenge_correct.astype(float) - base_correct.astype(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        "races": len(paired),
        "both_correct": int(np.sum(base_correct & challenge_correct)),
        "baseline_only_correct": baseline_only,
        "challenger_only_correct": challenger_only,
        "both_wrong": int(np.sum(~base_correct & ~challenge_correct)),
        "top1_difference": float(differences.mean()),
        "mcnemar_exact_p_value": _mcnemar_exact(baseline_only, challenger_only),
        "paired_bootstrap_samples": samples,
        "paired_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


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
            "expert_context_conditioning",
        ):
            if checkpoint["model_config"][key] != reference["model_config"][key]:
                raise ValueError(f"{path} differs from baseline architecture at {key}")
    features = list(reference["raw_feature_columns"])
    frame = load_finished_winner_rows(args.db, features)
    test_end = reference["partition"]["ranges"]["test"]["end_time"]
    test2_frame = frame.loc[frame["start_time_iso"] > test_end].copy()
    old_ids = set(
        reference["partition"]["training_race_ids"]
        + reference["partition"]["validation_race_ids"]
        + reference["partition"]["test_race_ids"]
    )
    test2_ids = [
        int(value) for value in test2_frame["race_id"].drop_duplicates()
        if int(value) not in old_ids
    ]
    cohorts = {
        "validation": reference["partition"]["validation_race_ids"],
        "test": reference["partition"]["test_race_ids"],
    }
    if test2_ids:
        cohorts["test2"] = test2_ids
        print(
            f"TEST-2 untouched_newer_races={len(test2_ids)} "
            f"start={test2_frame['start_time_iso'].min()} "
            f"end={test2_frame['start_time_iso'].max()} "
            "used_for_selection=no WARNING=small_cohort",
            flush=True,
        )
    rows = [] ; detail = {}
    for path, model, checkpoint in loaded:
        label = path.stem
        detail[label] = {}
        detail[label]["parameter_count"] = {
            "trainable": model.trainable_parameter_count(),
            "approximate_forward_active": model.approximate_active_parameter_count(),
        }
        for split, ids in cohorts.items():
            part = frame.loc[frame["race_id"].isin(ids)].copy()
            raw = numeric_matrix(part, features)
            race_id_array = part["race_id"].to_numpy(dtype=np.int64)
            x = transform_raceformer(
                raw, race_id_array, features, checkpoint["zeroed_features"],
                checkpoint["preprocessing"],
            )
            y = part["is_winner"].to_numpy(dtype=np.float32)
            metrics, diagnostics, predictions = evaluate_model(
                model, x, y, race_id_array, race_indices(race_id_array), part,
                args.races_per_batch, device,
            )
            detail[label][split] = {
                "metrics": metrics, "router_diagnostics": diagnostics,
                "predictions": predictions.to_dict(orient="records"),
            }
            rows.append((label, checkpoint["model_config"]["model_type"], split, metrics))
    print("MODEL COMPARISON (complete chronological cohorts; Top-1 is primary)")
    print("model                         type       split        params     active     top1     top2     top3      mrr  logloss  avg_winner_p")
    for label, model_type, split, metric in rows:
        count = detail[label]["parameter_count"]
        print(
            f"{label:<29} {model_type:<10} {split:<10} "
            f"{count['trainable']:>9,} {count['approximate_forward_active']:>10,} "
            f"{metric['top1_hit_rate']:>7.2%} {metric['top2_containment']:>8.2%} "
            f"{metric['top3_containment']:>8.2%} {metric['mrr']:>8.4f} "
            f"{metric['race_logloss']:>8.4f} {metric['average_winner_probability']:>13.4f}"
        )
    baseline_label = paths[0].stem
    baseline_validation = detail[baseline_label]["validation"]["metrics"]["top1_hit_rate"]
    baseline_test = detail[baseline_label]["test"]["metrics"]["top1_hit_rate"]
    paired = {}
    for path in paths[1:]:
        label = path.stem
        paired[label] = {}
        for split in cohorts:
            base_predictions = pd.DataFrame(detail[baseline_label][split]["predictions"])
            challenger_predictions = pd.DataFrame(detail[label][split]["predictions"])
            paired[label][split] = paired_comparison(
                base_predictions, challenger_predictions,
                args.bootstrap_samples, args.bootstrap_seed,
            )
    detail["paired_against_baseline"] = paired
    print("\nPAIRED TOP-1 COMPARISONS AGAINST BASELINE")
    for label, splits in paired.items():
        for split, result in splits.items():
            low, high = result["paired_bootstrap_95_ci"]
            print(
                f"{label} {split}: both_correct={result['both_correct']} "
                f"baseline_only={result['baseline_only_correct']} "
                f"challenger_only={result['challenger_only_correct']} "
                f"both_wrong={result['both_wrong']} "
                f"delta={result['top1_difference']:+.2%} "
                f"mcnemar_p={result['mcnemar_exact_p_value']:.6f} "
                f"bootstrap95=[{low:+.2%},{high:+.2%}]"
            )
    print("\nCHALLENGER ROUTER / EXPERT DIAGNOSTICS")
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
        if len(diagnostic["expert_usage_rate"]) > 1:
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
        for split in cohorts if len(diagnostic["expert_usage_rate"]) > 1 else ():
            diagnostic = detail[label][split]["router_diagnostics"]
            print(f"  {split} COMPLETE EXPERT MATRICES / UNIQUE PICKS")
            matrix_keys = (
                "pairwise_expert_logit_pearson",
                "pairwise_expert_logit_spearman",
                "pairwise_race_centred_expert_logit_pearson",
                "pairwise_race_centred_expert_logit_spearman",
                "mean_race_level_ranking_correlation",
                "top1_selection_agreement", "top1_selection_disagreement",
                "unique_top1_race_rate_per_expert",
                "unique_top1_winner_race_rate_per_expert",
                "winner_hit_rate_given_unique_top1_per_expert",
                "expert_specific_winner_hit_rate",
                "router_selected_expert_winner_hit_rate",
            )
            print(json.dumps({key: diagnostic.get(key) for key in matrix_keys}, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(detail, indent=2) + "\n")
        print(f"output_json={args.output_json.resolve()}")


if __name__ == "__main__":
    main()
