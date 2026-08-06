"""TabFM training losses helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def query_row_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    train_size: torch.Tensor,
    valid_row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over valid query rows only; context labels are excluded."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, T, classes].")
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match logits batch and row dimensions.")
    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    query_mask = positions >= train_size.unsqueeze(1)
    if valid_row_mask is not None:
        if (valid_row_mask.shape != logits.shape[:2]
                or valid_row_mask.dtype != torch.bool
                or valid_row_mask.device != logits.device):
            raise ValueError("valid_row_mask must match logits and be a bool tensor on its device.")
        query_mask &= valid_row_mask
    query_mask &= targets != -100
    if not torch.any(query_mask):
        raise ValueError("Batch contains no valid query rows for loss calculation.")
    return F.cross_entropy(logits[query_mask], targets[query_mask].long())


def race_winner_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    train_size: torch.Tensor,
    race_group_ids: torch.Tensor,
    valid_row_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a one-winner softmax loss independently within each query race."""
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError(
            "race_winner_cross_entropy requires binary logits with shape [B, T, 2]."
        )
    if targets.shape != logits.shape[:2]:
        raise ValueError("targets must match logits batch and row dimensions.")
    if race_group_ids.shape != logits.shape[:2]:
        raise ValueError("race_group_ids must match logits batch and row dimensions.")
    if train_size.shape != (logits.shape[0],):
        raise ValueError("train_size must match the logits batch dimension.")
    if valid_row_mask is not None and (
        valid_row_mask.shape != logits.shape[:2]
        or valid_row_mask.dtype != torch.bool
        or valid_row_mask.device != logits.device
    ):
        raise ValueError("valid_row_mask must match logits and be a bool tensor on its device.")

    positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    query_mask = positions >= train_size.unsqueeze(1)
    query_mask &= targets != -100
    query_mask &= race_group_ids >= 0
    if valid_row_mask is not None:
        query_mask &= valid_row_mask

    winner_scores = logits[..., 1] - logits[..., 0]
    race_losses = []
    for batch_index in range(logits.shape[0]):
        batch_groups = torch.unique(race_group_ids[batch_index][query_mask[batch_index]])
        for group_id in batch_groups:
            race_mask = query_mask[batch_index] & (race_group_ids[batch_index] == group_id)
            race_targets = targets[batch_index][race_mask].long()
            winner_positions = torch.nonzero(race_targets == 1, as_tuple=False).flatten()
            if winner_positions.numel() != 1:
                raise ValueError(
                    f"Race group {int(group_id)} in batch {batch_index} must contain exactly one winner, "
                    f"found {winner_positions.numel()}."
                )
            race_scores = winner_scores[batch_index][race_mask]
            race_losses.append(F.cross_entropy(race_scores.unsqueeze(0), winner_positions))
    if not race_losses:
        raise ValueError("Batch contains no complete query races.")
    return torch.stack(race_losses).mean()


def grouped_race_losses(
    query_logits: torch.Tensor,
    query_targets: torch.Tensor,
    query_row_race_ids: torch.Tensor,
    class_weights: torch.Tensor,
    pairwise_loss_weight: float = 0.0,
    cardinality_loss_weight: float = 0.0,
    classification_loss_weight: float = 1.0,
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
        classification_loss_weight * classification
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
