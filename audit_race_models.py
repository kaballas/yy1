#!/usr/bin/env python3
"""Audit saved per-race models, their feature counts, and artifact consistency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("models_test"))
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Defaults to per_race_models_manifest.json inside --models-dir.",
    )
    parser.add_argument(
        "--race-id",
        action="append",
        type=int,
        help="Audit only this race ID; repeat to select multiple races.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Defaults to model_feature_audit.csv inside --models-dir.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every model row; otherwise print the first --limit rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Rows printed when --show-all is absent (default: 50).",
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="Print only models with missing or feature-mismatched artifacts.",
    )
    return parser.parse_args()


def resolve_saved_path(raw_path: Any, models_dir: Path) -> Path:
    """Resolve an artifact path, tolerating a moved models directory."""
    path = Path(str(raw_path or ""))
    if path.is_file():
        return path.resolve()
    relocated = models_dir / path.name
    return relocated.resolve()


def xgboost_json_features(path: Path) -> tuple[list[str], int | None, str | None]:
    """Read feature metadata from an XGBoost JSON model without importing it."""
    if not path.is_file():
        return [], None, "missing_model_file"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        learner = payload["learner"]
        features = [str(feature) for feature in learner.get("feature_names", [])]
        raw_count = learner.get("learner_model_param", {}).get("num_feature")
        count = int(raw_count) if raw_count is not None else len(features)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return [], None, f"invalid_model_json:{type(exc).__name__}"
    return features, count, None


def load_model_feature_audit(
    manifest_path: Path,
    models_dir: Path,
) -> pd.DataFrame:
    """Return one feature-integrity row per logical race model."""
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.is_file():
        raise ValueError(f"Race-model manifest does not exist: {resolved_manifest}")
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    items = payload.get("models")
    if not isinstance(items, list):
        raise ValueError("Race-model manifest must contain a models list")

    records: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        sidecar_path = resolve_saved_path(item.get("features_file"), models_dir)
        sidecar: dict[str, Any] = {}
        if sidecar_path.is_file():
            try:
                loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
                sidecar = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                sidecar = {}
        features = details.get("input_features") or sidecar.get("input_features") or []
        features = [str(feature) for feature in features]
        configured_models = item.get("models")
        if not isinstance(configured_models, list) or not configured_models:
            configured_models = [item.get("model")]
        model_paths = [
            resolve_saved_path(path, models_dir) for path in configured_models
        ]
        member_audits = [xgboost_json_features(path) for path in model_paths]
        member_feature_lists = [audit[0] for audit in member_audits]
        member_feature_counts = [audit[1] for audit in member_audits]
        member_errors = [audit[2] for audit in member_audits if audit[2]]
        files_present = sum(path.is_file() for path in model_paths)
        features_match = bool(model_paths) and not member_errors and all(
            member_features == features for member_features in member_feature_lists
        )
        counts_match = bool(model_paths) and not member_errors and all(
            count == len(features) for count in member_feature_counts
        )
        signature = hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        gains = details.get("split_feature_gain") or sidecar.get(
            "split_feature_gain"
        ) or {}
        records.append({
            "model": str(item.get("name", f"race_{item.get('trained_on_race_id', '')}")),
            "trained_on_race_id": int(item.get("trained_on_race_id", 0)),
            "training_date_utc": str(item.get("training_date_utc", "")),
            "competition_id": details.get("competition_id", sidecar.get("competition_id")),
            "race_number": details.get("race_number", sidecar.get("race_number")),
            "race_name": str(details.get("race_name", "")),
            "feature_count": len(features),
            "features": ",".join(features),
            "feature_set_signature": signature,
            "nonzero_gain_features": len(gains) if isinstance(gains, dict) else 0,
            "gain_features": ",".join(map(str, gains)) if isinstance(gains, dict) else "",
            "ensemble_members": len(model_paths),
            "model_files_present": files_present,
            "artifact_feature_counts": ",".join(
                "" if count is None else str(count) for count in member_feature_counts
            ),
            "artifact_features_match_manifest": features_match,
            "artifact_counts_match_manifest": counts_match,
            "artifact_errors": ",".join(member_errors),
            "self_validation_winner_rank": details.get(
                "self_validation_winner_rank",
                sidecar.get("self_validation_winner_rank"),
            ),
            "model_paths": ",".join(str(path) for path in model_paths),
            "features_file": str(sidecar_path),
        })
    audit = pd.DataFrame(records)
    if audit.empty:
        raise ValueError(f"Race-model manifest contains no model entries: {resolved_manifest}")
    reuse = audit.groupby("feature_set_signature")["model"].transform("size")
    audit.insert(
        audit.columns.get_loc("feature_set_signature") + 1,
        "feature_set_reuse_count",
        reuse.astype(int),
    )
    return audit.sort_values(
        ["feature_count", "trained_on_race_id", "model"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )


def feature_count_distribution(audit: pd.DataFrame) -> pd.DataFrame:
    """Summarize logical models by saved manifest feature count."""
    return (
        audit.groupby("feature_count", as_index=False)
        .agg(models=("model", "size"), unique_feature_sets=("feature_set_signature", "nunique"))
        .sort_values("feature_count", kind="stable", ignore_index=True)
    )


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    models_dir = args.models_dir.resolve()
    manifest_path = (
        args.manifest or models_dir / "per_race_models_manifest.json"
    ).resolve()
    full_audit = load_model_feature_audit(manifest_path, models_dir)
    audit = full_audit
    if args.race_id:
        wanted = set(args.race_id)
        audit = audit.loc[audit["trained_on_race_id"].isin(wanted)].reset_index(drop=True)
        if audit.empty:
            raise ValueError("No manifest models match --race-id")
    output_path = (
        args.output_csv or models_dir / "model_feature_audit.csv"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_audit = audit if args.output_csv is not None else full_audit
    csv_audit.to_csv(output_path, index=False)

    errors = ~(
        audit["artifact_features_match_manifest"]
        & audit["artifact_counts_match_manifest"]
    )
    distribution = feature_count_distribution(audit)
    print("RACE MODEL FEATURE AUDIT")
    print(
        f"manifest={manifest_path}\n"
        f"models={len(audit):,} model_members={int(audit['ensemble_members'].sum()):,} "
        f"unique_feature_sets={audit['feature_set_signature'].nunique():,}\n"
        f"feature_count_min={int(audit['feature_count'].min())} "
        f"feature_count_median={float(audit['feature_count'].median()):.1f} "
        f"feature_count_max={int(audit['feature_count'].max())}\n"
        f"artifact_errors_or_mismatches={int(errors.sum()):,}\n"
        f"saved_csv={output_path}"
    )
    print("\nFEATURE COUNT DISTRIBUTION")
    print(distribution.to_string(index=False))

    displayed = audit.loc[errors] if args.only_errors else audit
    print("\nMODEL FEATURE COUNTS")
    if displayed.empty:
        print("No model artifact errors or mismatches")
        return
    displayed_total = len(displayed)
    if not args.show_all:
        displayed = displayed.head(args.limit)
    columns = [
        "model", "trained_on_race_id", "training_date_utc", "feature_count",
        "ensemble_members", "nonzero_gain_features", "feature_set_reuse_count",
        "artifact_feature_counts", "artifact_features_match_manifest",
        "artifact_counts_match_manifest",
        "self_validation_winner_rank", "features",
    ]
    print(displayed.loc[:, columns].to_string(index=False))
    if not args.show_all and len(displayed) < displayed_total:
        print(
            f"displayed={len(displayed):,}/{displayed_total:,}; use --show-all for "
            "every row. The CSV always contains the full audit."
        )


if __name__ == "__main__":
    main()
