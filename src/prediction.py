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
    return logits[0, context_rows:, :2].softmax(-1)[:, 1].cpu().numpy()


def market_rank_scores(prices: np.ndarray) -> np.ndarray:
    """Return a higher-is-better ranking score from positive market prices."""
    valid = np.isfinite(prices) & (prices > 0)
    scores = np.full(prices.shape, -np.inf, dtype=np.float64)
    scores[valid] = 1.0 / prices[valid]
    return scores
