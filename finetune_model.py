#!/usr/bin/env python3
"""Fine-tune an existing TabFM bundle without overwriting the source model."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train_model import DEFAULT_CONTEXT, DEFAULT_DB, DEFAULT_OUTPUT, ROOT


DEFAULT_FINETUNED_OUTPUT = ROOT / "outputs/tabfm_race_top3_finetuned.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_FINETUNED_OUTPUT)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--context-json", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--encode-races-before-icl", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--zero-features",
        nargs="*",
        default=None,
        help=(
            "Override features neutralized to standardized zero. "
            "Omit to inherit the source model; pass with no names to disable."
        ),
    )
    parser.add_argument("--progress-race-id", type=int, default=10733171)
    parser.add_argument("--fine-tune-race-id", type=int, required=True,
                        help="Experiment race whose labelled rows are used for adaptation.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model.resolve() == args.output.resolve():
        raise SystemExit("--output must differ from --model")
    command = [
        sys.executable,
        str(ROOT / "train_model.py"),
        "--resume-model", str(args.model),
        "--output", str(args.output),
        "--db", str(args.db),
        "--context-json", str(args.context_json),
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--batch-rows", str(args.batch_rows),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--early-stopping-patience", str(args.early_stopping_patience),
    ]
    if args.split_manifest is not None:
        command.extend(["--split-manifest", str(args.split_manifest)])
    if args.encode_races_before_icl is not None:
        command.append("--encode-races-before-icl" if args.encode_races_before_icl else "--no-encode-races-before-icl")
    if args.zero_features is not None:
        command.extend(["--zero-features", *args.zero_features])
    if args.progress_race_id is not None:
        command.extend(["--progress-race-id", str(args.progress_race_id)])
    command.extend(["--fine-tune-race-id", str(args.fine_tune_race_id)])
    if args.device is not None:
        command.extend(["--device", args.device])
    print(
        f"Fine-tuning {args.model.resolve()} -> {args.output.resolve()}",
        flush=True,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
