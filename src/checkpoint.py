"""TabFM training checkpoint helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_resume_model_path(
    output: Path,
    resume_model: Path | None,
    overwrite_existing: bool,
) -> tuple[Path | None, bool]:
    """Return the effective resume path and whether it was inferred from output."""
    if resume_model is not None and overwrite_existing:
        raise ValueError("--resume-model and --overwrite-existing cannot be combined")
    if resume_model is not None:
        if not resume_model.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_model}")
        return resume_model, False
    if output.exists() and not output.is_file():
        raise ValueError(f"Output path exists but is not a file: {output}")
    if output.is_file() and not overwrite_existing:
        return output, True
    return None, False
