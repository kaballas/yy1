"""TabFM training prediction helpers."""

from __future__ import annotations

import numpy as np
import torch
from src.model import TabFM
from src.sampling import build_race_group_ids


def predict(
    model: TabFM, context_x: np.ndarray, context_y: np.ndarray,
    query_x: np.ndarray, query_race_ids: np.ndarray,
    context_rows: int, device: torch.device,
    context_race_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Run race-aware model prediction using the original tensor layout."""
    if len(query_x) != len(query_race_ids):
        raise ValueError("query_x and query_race_ids must have the same row count")
    context_rows = min(context_rows, len(context_y))
    combined_x = torch.from_numpy(
        np.concatenate([context_x[-context_rows:], query_x])[None, ...]
    ).to(device)
    combined_y = torch.from_numpy(
        np.concatenate([context_y[-context_rows:], np.full(len(query_x), -100)])[None, ...]
    ).to(device)
    train_size = torch.tensor([context_rows], device=device)
    if model.encode_races_before_icl:
        if context_race_ids is None:
            raise ValueError("context_race_ids are required for pre-ICL race encoding")
        context_race_ids = np.asarray(context_race_ids)[-context_rows:]
    race_group_ids = build_race_group_ids(
        query_race_ids, context_rows,
        context_race_ids=context_race_ids if model.encode_races_before_icl else None,
    ).to(device)
    valid_row_mask = torch.ones(
        combined_x.shape[:2], dtype=torch.bool, device=device
    )
    with torch.no_grad():
        logits = model(
            combined_x, combined_y, train_size, race_group_ids=race_group_ids,
            valid_row_mask=valid_row_mask,
        )
    query_logits = logits[0, context_rows:, :2]
    query_top3_probability = torch.softmax(query_logits, dim=-1)[:, 1]
    if not torch.isfinite(query_top3_probability).all():
        raise RuntimeError("Prediction produced non-finite top-three probabilities")
    if torch.any((query_top3_probability < 0) | (query_top3_probability > 1)):
        raise RuntimeError("Top-three probabilities must be in [0, 1]")
    return query_top3_probability.cpu().numpy()


def predict_with_chronological_context(
    model: TabFM,
    context_x: np.ndarray,
    context_y: np.ndarray,
    context_race_indices: dict[int, np.ndarray],
    race_time_by_id: dict[int, object],
    query_x: np.ndarray,
    query_race_indices: dict[int, np.ndarray],
    context_races_per_prediction: int,
    device: torch.device,
) -> np.ndarray:
    """Predict complete races using only the most recent earlier context races."""
    if context_races_per_prediction < 1:
        raise ValueError("context_races_per_prediction must be positive")
    if not query_race_indices:
        raise ValueError("At least one query race is required")

    ordered_context_races = sorted(
        context_race_indices,
        key=lambda race_id: (race_time_by_id[race_id], race_id),
    )
    result = np.empty(len(query_x), dtype=np.float32)
    for query_race_id, query_indices in query_race_indices.items():
        query_time = race_time_by_id[query_race_id]
        eligible_context_races = [
            race_id
            for race_id in ordered_context_races
            if race_time_by_id[race_id] < query_time
        ]
        if len(eligible_context_races) < context_races_per_prediction:
            raise ValueError(
                f"Query race {query_race_id} has only "
                f"{len(eligible_context_races)} earlier context races; "
                f"{context_races_per_prediction} are required"
            )
        selected_context_races = eligible_context_races[
            -context_races_per_prediction:
        ]
        context_indices = np.concatenate(
            [context_race_indices[race_id] for race_id in selected_context_races]
        )
        context_row_race_ids = np.concatenate(
            [
                np.full(
                    len(context_race_indices[race_id]), race_id, dtype=np.int64
                )
                for race_id in selected_context_races
            ]
        )
        result[query_indices] = predict(
            model,
            context_x[context_indices],
            context_y[context_indices],
            query_x[query_indices],
            np.full(len(query_indices), query_race_id, dtype=np.int64),
            len(context_indices),
            device,
            context_race_ids=context_row_race_ids,
        )
    return result


def market_rank_scores(prices: np.ndarray) -> np.ndarray:
    """Return a higher-is-better ranking score from positive market prices."""
    valid = np.isfinite(prices) & (prices > 0)
    scores = np.full(prices.shape, -np.inf, dtype=np.float64)
    scores[valid] = 1.0 / prices[valid]
    return scores
