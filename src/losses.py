"""TabFM training losses helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def grouped_race_losses(
    query_logits: torch.Tensor,
    query_targets: torch.Tensor,
    query_row_race_ids: torch.Tensor,
    class_weights: torch.Tensor,
    pairwise_loss_weight: float = 0.0,
    cardinality_loss_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equal-per-race classification, ranking, and top-three cardinality loss."""
    row_losses = F.cross_entropy(
        query_logits, query_targets, weight=class_weights, reduction="none"
    )
    classification_losses = []
    pairwise_losses = []
    cardinality_losses = []
    scores = query_logits[:, 1] - query_logits[:, 0]
    probabilities = query_logits.softmax(-1)[:, 1]
    for race_id in torch.unique(query_row_race_ids):
        race_mask = query_row_race_ids == race_id
        race_targets = query_targets[race_mask]
        classification_losses.append(row_losses[race_mask].mean())
        if pairwise_loss_weight > 0:
            positive_scores = scores[race_mask][race_targets == 1]
            negative_scores = scores[race_mask][race_targets == 0]
            if positive_scores.numel() == 0 or negative_scores.numel() == 0:
                raise ValueError(
                    "Pairwise loss requires positive and negative runners per race"
                )
            pairwise_losses.append(
                F.softplus(
                    -(positive_scores[:, None] - negative_scores[None, :])
                ).mean()
            )
        else:
            pairwise_losses.append(scores.new_zeros(()))
        if cardinality_loss_weight > 0:
            cardinality_losses.append(
                ((probabilities[race_mask].sum() - 3.0) / 3.0).square()
            )
        else:
            cardinality_losses.append(scores.new_zeros(()))
    classification = torch.stack(classification_losses).mean()
    pairwise = torch.stack(pairwise_losses).mean()
    cardinality = torch.stack(cardinality_losses).mean()
    total = (
        classification
        + pairwise_loss_weight * pairwise
        + cardinality_loss_weight * cardinality
    )
    return total, classification, pairwise, cardinality


def grouped_pairwise_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    row_race_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean positive-vs-negative softplus ranking loss, weighted equally by race."""
    scores = logits[:, 1] - logits[:, 0]
    losses = []
    for race_id in torch.unique(row_race_ids):
        race_mask = row_race_ids == race_id
        race_targets = targets[race_mask]
        positive_scores = scores[race_mask][race_targets == 1]
        negative_scores = scores[race_mask][race_targets == 0]
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            raise ValueError("Pairwise loss requires positive and negative runners per race")
        losses.append(
            F.softplus(-(positive_scores[:, None] - negative_scores[None, :])).mean()
        )
    return torch.stack(losses).mean()
