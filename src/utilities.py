"""TabFM training utilities helpers."""

from __future__ import annotations

import datetime as dt
import math
import numpy as np
import torch
from src.model import TabFM


def resolve_learning_rate(
    requested_learning_rate: float | None, fine_tuning: bool
) -> float:
    """Use a conservative default for small-data checkpoint fine-tuning."""
    learning_rate = (
        requested_learning_rate
        if requested_learning_rate is not None
        else 3e-5 if fine_tuning else 3e-4
    )
    if learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    return learning_rate


def resolve_races_per_step(
    available_races: int,
    requested_context_races: int,
    query_races: int,
) -> tuple[int, int]:
    """Fit the context size to the pool while preserving disjoint query races."""
    if available_races <= query_races:
        raise ValueError(
            f"Need at least {query_races + 1} training races for "
            f"{query_races} query races and one context race, but only "
            f"{available_races} are available"
        )
    return min(requested_context_races, available_races - query_races), query_races


def resolve_training_schedule(
    training_race_count: int,
    requested_steps_per_epoch: int,
    requested_query_races_per_step: int,
    auto: bool,
) -> tuple[int, int]:
    """Resolve optimizer steps and query races for one epoch.

    In automatic mode, the requested query size is treated as a target maximum
    and the number of steps is increased as necessary so every eligible race is
    scheduled as a query at least once per epoch.  The final schedule may have
    a small number of repeated query slots when the count is not divisible by
    the target batch size; the schedule builder cycles deterministically.
    """
    if training_race_count < 1:
        raise ValueError("training_race_count must be positive")
    if requested_steps_per_epoch < 1:
        raise ValueError("requested_steps_per_epoch must be positive")
    if requested_query_races_per_step < 1:
        raise ValueError("requested_query_races_per_step must be positive")
    if not auto:
        return requested_steps_per_epoch, requested_query_races_per_step
    query_races = min(requested_query_races_per_step, training_race_count)
    steps = math.ceil(training_race_count / query_races)
    return steps, query_races


def parse_iso_timestamp(value: str) -> dt.datetime:
    """Parse a timezone-aware timestamp and normalize it to UTC."""
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp has no timezone")
    except ValueError as error:
        raise ValueError(f"Invalid timezone-aware ISO-8601 timestamp: {value!r}") from error
    return parsed.astimezone(dt.timezone.utc)


def normalize_cutoff_iso(value: str) -> str:
    """Normalize a date or timezone-aware cutoff to UTC ISO format."""
    try:
        if len(value) == 10:
            parsed = dt.datetime.combine(
                dt.date.fromisoformat(value), dt.time(), tzinfo=dt.timezone.utc
            )
        else:
            parsed = parse_iso_timestamp(value)
    except ValueError as error:
        raise ValueError(
            "--train-cutoff-iso must be YYYY-MM-DD or a timezone-aware ISO-8601 timestamp"
        ) from error
    return parsed.astimezone(dt.timezone.utc).isoformat()


def initialize_fourier_frequencies(model: TabFM, seed: int) -> None:
    """Give a from-scratch TabFM a non-constant numeric input embedding."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    embedder = model.cell_embedder
    shape = embedder.fourier_frequencies.shape
    scales = torch.logspace(-1.0, 1.0, shape[1], dtype=torch.float32)
    numeric = torch.randn(shape, generator=generator) * scales.unsqueeze(0)
    categorical = torch.randn(shape, generator=generator) * scales.unsqueeze(0)
    with torch.no_grad():
        embedder.fourier_frequencies.copy_(
            numeric.to(embedder.fourier_frequencies.device)
        )
        embedder.fourier_frequencies_cat.copy_(
            categorical.to(embedder.fourier_frequencies_cat.device)
        )
    if not torch.count_nonzero(embedder.fourier_frequencies):
        raise RuntimeError("Fourier frequency initialization produced an all-zero buffer")
