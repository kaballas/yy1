"""Immutable, hash-verified row snapshots for winner-model experiments."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.race_moe_data import DIAGNOSTIC_COLUMNS, load_finished_winner_rows


SNAPSHOT_VERSION = 1
SNAPSHOT_BASE_COLUMNS = (
    "race_id", "runner_number", "start_time_iso", "competition_id",
    "is_winner", "finish_place", *DIAGNOSTIC_COLUMNS,
)


def snapshot_manifest_reference(manifest_path: Path, checkpoint_path: Path) -> str:
    """Return a manifest path portable with the checkpoint directory."""
    return Path(os.path.relpath(
        manifest_path.resolve(), start=checkpoint_path.resolve().parent,
    )).as_posix()


def resolve_snapshot_manifest(
    snapshot_metadata: Mapping[str, Any], checkpoint_path: Path,
) -> Path:
    """Resolve new relative references and legacy absolute checkpoint paths."""
    reference = Path(str(snapshot_metadata["manifest"]))
    checkpoint_directory = checkpoint_path.resolve().parent
    candidates: list[Path] = []
    if reference.is_absolute():
        candidates.append(reference)
    else:
        candidates.extend((checkpoint_directory / reference, Path.cwd() / reference))

    # Legacy checkpoints stored the creator's absolute workspace path. Their
    # snapshots were saved beside the checkpoints in a `snapshot` directory.
    candidates.append(checkpoint_directory / "snapshot" / reference.name)
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if str(resolved) in checked:
            continue
        checked.append(str(resolved))
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "Immutable snapshot manifest is unavailable. Checked: " + ", ".join(checked)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_columns(features: Sequence[str]) -> list[str]:
    return list(dict.fromkeys([*SNAPSHOT_BASE_COLUMNS, *features]))


def canonical_snapshot_bytes(frame: pd.DataFrame, features: Sequence[str]) -> bytes:
    columns = snapshot_columns(features)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError("Snapshot frame is missing columns: " + ", ".join(missing))
    ordered = frame.loc[:, columns].sort_values(
        ["start_time_iso", "race_id", "runner_number"], kind="stable"
    )
    return ordered.to_csv(
        index=False, lineterminator="\n", na_rep="__NULL__", float_format="%.9g",
    ).encode("utf-8")


def _race_id_sha256(frame: pd.DataFrame) -> str:
    values = "\n".join(map(str, frame["race_id"].drop_duplicates())) + "\n"
    return _sha256_bytes(values.encode("ascii"))


def create_split_snapshot(
    directory: Path, frames: Mapping[str, pd.DataFrame], features: Sequence[str],
    *, database: Path, excluded_features: Sequence[str],
) -> Path:
    """Create a new snapshot directory; existing manifests are never overwritten."""
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Immutable snapshot already exists: {manifest_path}; choose a new directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    split_metadata: dict[str, Any] = {}
    created_files: list[Path] = []
    try:
        required = ("training", "validation", "test")
        for split in required:
            if split not in frames or frames[split].empty:
                raise ValueError(f"Snapshot split {split} is empty")
        ordered_splits = [*required, *(name for name in frames if name not in required)]
        for split in ordered_splits:
            if frames[split].empty:
                continue
            frame = frames[split]
            content = canonical_snapshot_bytes(frame, features)
            path = directory / f"{split}.csv.gz"
            temporary = directory / f".{split}.csv.gz.tmp"
            ordered = frame.loc[:, snapshot_columns(features)].sort_values(
                ["start_time_iso", "race_id", "runner_number"], kind="stable"
            )
            ordered.to_csv(
                temporary, index=False, na_rep="__NULL__", float_format="%.9g",
                compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
            )
            os.replace(temporary, path)
            created_files.append(path)
            split_metadata[split] = {
                "path": path.name,
                "rows": len(ordered),
                "races": int(ordered["race_id"].nunique()),
                "first_start_time": str(ordered["start_time_iso"].iloc[0]),
                "last_start_time": str(ordered["start_time_iso"].iloc[-1]),
                "file_sha256": file_sha256(path),
                "content_sha256": _sha256_bytes(content),
                "race_ids_sha256": _race_id_sha256(ordered),
            }
        manifest = {
            "snapshot_type": "race_winner_feature_snapshot",
            "snapshot_version": SNAPSHOT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_database": str(database.resolve()),
            "feature_columns": list(features),
            "excluded_features": list(excluded_features),
            "identity_columns": ["race_id", "runner_number"],
            "label_column": "is_winner",
            "columns": snapshot_columns(features),
            "splits": split_metadata,
        }
        temporary_manifest = directory / ".manifest.json.tmp"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary_manifest, manifest_path)
    except Exception:
        # Leave successfully written files in place for forensic inspection, but
        # without a manifest they can never be consumed as a valid snapshot.
        raise
    return manifest_path


def load_split_snapshot(
    manifest_path: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("snapshot_type") != "race_winner_feature_snapshot"
        or int(manifest.get("snapshot_version", 0)) != SNAPSHOT_VERSION
    ):
        raise ValueError(f"Unsupported snapshot manifest: {manifest_path}")
    frames: dict[str, pd.DataFrame] = {}
    for split, metadata in manifest["splits"].items():
        path = manifest_path.parent / metadata["path"]
        actual_hash = file_sha256(path)
        if actual_hash != metadata["file_sha256"]:
            raise ValueError(
                f"IMMUTABLE SNAPSHOT HASH MISMATCH split={split} "
                f"expected={metadata['file_sha256']} actual={actual_hash} path={path}"
            )
        frame = pd.read_csv(path, na_values=["__NULL__"], keep_default_na=True)
        if len(frame) != int(metadata["rows"]):
            raise ValueError(f"Snapshot row count changed for {split}")
        if int(frame["race_id"].nunique()) != int(metadata["races"]):
            raise ValueError(f"Snapshot race count changed for {split}")
        if frame.duplicated(["race_id", "runner_number"]).any():
            raise ValueError(f"Snapshot has duplicate runner identities in {split}")
        if _race_id_sha256(frame) != metadata["race_ids_sha256"]:
            raise ValueError(f"Snapshot race identity hash changed for {split}")
        actual_content_hash = _sha256_bytes(
            canonical_snapshot_bytes(frame, manifest["feature_columns"])
        )
        if actual_content_hash != metadata["content_sha256"]:
            raise ValueError(
                f"IMMUTABLE SNAPSHOT CONTENT HASH MISMATCH split={split} "
                f"expected={metadata['content_sha256']} "
                f"actual={actual_content_hash} path={path}"
            )
        frames[split] = frame
    return frames, manifest


def audit_live_database_against_snapshot(
    manifest_path: Path, database: Path,
) -> None:
    frames, manifest = load_split_snapshot(manifest_path)
    features = list(manifest["feature_columns"])
    live = load_finished_winner_rows(database, features)
    for split, snapshot in frames.items():
        race_ids = set(map(int, snapshot["race_id"].unique()))
        current = live.loc[live["race_id"].isin(race_ids)].copy()
        actual = _sha256_bytes(canonical_snapshot_bytes(current, features))
        expected = manifest["splits"][split]["content_sha256"]
        if actual != expected:
            raise ValueError(
                f"LIVE DATABASE DRIFT DETECTED split={split} "
                f"expected_content_sha256={expected} actual_content_sha256={actual}"
            )
