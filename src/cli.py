"""Command-line interface for TabFM training."""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from src.config import DEFAULT_CONTEXT, DEFAULT_DB, DEFAULT_FEATURES, DEFAULT_OUTPUT


def parse_args() -> argparse.Namespace:
    """Parse the original training command-line interface unchanged."""
    parser = argparse.ArgumentParser(
        description=(
            "Train model.TabFM on completed runners from the race_runners "
            "SQLite table."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--features-json", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--context-json", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--split-manifest", type=Path, default=None,
                        help="Required split-v2 runtime manifest for clean training.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help=(
            "Continue training an existing bundle using its features and preprocessing; "
            "the current database is_validation flags define the partition."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Start from scratch even when --output already exists. By default an "
            "existing output checkpoint is loaded and fine-tuned in place."
        ),
    )
    parser.add_argument(
        "--fine-tune-attention-head-only",
        action="store_true",
        help=(
            "Freeze the TabFM backbone and optimize only race_set_head. Requires "
            "resuming a self_attention checkpoint."
        ),
    )
    parser.add_argument(
        "--fine-tune-scope",
        choices=("full_model", "attention_head_only", "icl_and_race_head"),
        default=None,
        help=(
            "Parameters to optimize. icl_and_race_head adapts the ICL runner "
            "representation and race head while freezing lower feature encoders."
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument(
        "--probe-every-steps",
        type=int,
        default=10,
        help="Evaluate the deterministic fixed development probe every N optimizer steps.",
    )
    parser.add_argument(
        "--probe-races",
        type=int,
        default=20,
        help="Number of complete development races in the deterministic fixed probe.",
    )
    parser.add_argument(
        "--step-loss-window",
        type=int,
        default=10,
        help="Number of recent optimizer-step losses used for the rolling mean.",
    )
    parser.add_argument(
        "--auto-race-schedule",
        action="store_true",
        help=(
            "Derive query races per step and steps per epoch from the eligible "
            "training-race count. --query-races-per-step is used as the target "
            "maximum query-race batch size."
        ),
    )
    parser.add_argument(
        "--print-race-schedule",
        action="store_true",
        help="Print race_id:race_number values for fixed and per-step context/query races.",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=256,
        help="Deprecated compatibility option; complete-race batches have variable rows.",
    )
    parser.add_argument("--context-races-per-step", type=int, default=20)
    parser.add_argument("--query-races-per-step", type=int, default=80)
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=0.20,
        help="Deprecated; validation is selected by race_runners.is_validation = 1.",
    )
    parser.add_argument(
        "--zero-features",
        nargs="*",
        default=None,
        help="Set these columns to standardized zero after preprocessing.",
    )
    parser.add_argument(
        "--train-cutoff-iso",
        default=None,
        help=(
            "Deprecated and incompatible with flag-based validation; training uses "
            "race_runners.is_validation = 0."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "AdamW learning rate. Defaults to 3e-4 from scratch and 3e-5 when "
            "fine-tuning a checkpoint."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-valid-races",
        type=int,
        default=None,
        help=(
            "Compatibility check only; cohort-aware validation never truncates flagged "
            "races. A value below the full validation count is rejected."
        ),
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--race-context-mode", choices=("none", "self_attention"), default=None,
        help="Target-race interaction mode; resumed models inherit this when omitted.",
    )
    parser.add_argument("--race-context-dim", type=int, default=None)
    parser.add_argument("--race-context-layers", type=int, default=None)
    parser.add_argument("--race-context-heads", type=int, default=None)
    parser.add_argument("--race-context-ff-dim", type=int, default=None)
    parser.add_argument(
        "--race-context-residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add the race-head correction to base logits (default: true).",
    )
    parser.add_argument(
        "--encode-races-before-icl", action=argparse.BooleanOptionalAction,
        default=None,
        help="Experimental representation-level race encoder before ICL; requires retraining.",
    )
    parser.add_argument("--pairwise-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--attention-delta-pairwise-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Additional pairwise ranking loss applied directly to the race-head "
            "delta, preventing the residual base logits from carrying the ranking."
        ),
    )
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--stress-top3-recall-max-drop",
        type=float,
        default=0.5,
        help=(
            "Maximum absolute market-miss stress top3-recall drop allowed versus "
            "the best stress result observed during checkpoint selection."
        ),
    )
    parser.add_argument(
        "--min-race-number",
        type=int,
        default=None,
        help=(
            "Restrict optimizer-step context and query pools to complete races "
            "with race_number >= this value. Validation and the fixed validation "
            "context are unchanged."
        ),
    )
    parser.add_argument(
        "--progress-race-id",
        type=int,
        default=None,
        help="Race to rank after every epoch. Defaults to the first validation race.",
    )
    parser.add_argument(
        "--fine-tune-race-id",
        type=int,
        default=None,
        help=(
            "EXPERIMENT ONLY: move exactly this complete race into the optimizer "
            "pool while resuming a checkpoint; its labels are intentionally used."
        ),
    )
    parser.add_argument(
        "--classroom-overfit-all-races",
        action="store_true",
        help=(
            "EXPERIMENT ONLY: deliberately use every complete race exposed by the "
            "training-row view for both training and validation, including fixed "
            "context races. This leaks validation labels and produces no valid "
            "generalization evidence."
        ),
    )
    return parser.parse_args()
