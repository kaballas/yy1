"""Current-race-only models for top-three runner prediction."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class RunnerFeatureEncoder(nn.Module):
    """Encode one flat pre-race feature vector per runner."""

    def __init__(
        self, feature_count: int, hidden_dim: int = 256, model_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RaceFormerTop3(nn.Module):
    """Predict runners from the current field without labelled historical context.

    Variants form a controlled experiment:
      * ``independent``: runner MLP only (Model A)
      * ``transformer``: runner MLP plus field self-attention (Model B)
      * ``race_token``: field self-attention plus a learned race summary (Model C)
      * ``market_residual``: Model C learns a correction to a fixed market anchor
    """

    VARIANTS = {"independent", "transformer", "race_token", "market_residual"}

    def __init__(
        self,
        feature_count: int,
        variant: str = "race_token",
        hidden_dim: int = 256,
        model_dim: int = 128,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        market_feature_index: int | None = None,
        market_anchor_bias: float = 0.0,
        market_anchor_scale: float = 1.0,
        market_residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown RaceFormer variant {variant!r}")
        if model_dim < 1 or hidden_dim < 1 or feedforward_dim < 1:
            raise ValueError("model dimensions must be positive")
        if heads < 1 or model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        if layers < 1 and variant != "independent":
            raise ValueError("transformer variants require at least one layer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if variant == "market_residual":
            if market_feature_index is None or not 0 <= market_feature_index < feature_count:
                raise ValueError(
                    "market_residual requires a valid market_feature_index"
                )
            if market_anchor_scale <= 0:
                raise ValueError("market_anchor_scale must be positive")
            if market_residual_scale <= 0:
                raise ValueError("market_residual_scale must be positive")

        self.feature_count = int(feature_count)
        self.variant = variant
        self.hidden_dim = int(hidden_dim)
        self.model_dim = int(model_dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.feedforward_dim = int(feedforward_dim)
        self.dropout = float(dropout)
        self.market_feature_index = market_feature_index
        self.market_anchor_bias = float(market_anchor_bias)
        self.market_anchor_scale = float(market_anchor_scale)
        self.market_residual_scale = float(market_residual_scale)
        self.feature_encoder = RunnerFeatureEncoder(
            feature_count, hidden_dim, model_dim, dropout
        )

        self.race_transformer: nn.Module | None = None
        self.race_token: nn.Parameter | None = None
        if variant != "independent":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )
            self.race_transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=layers, norm=nn.LayerNorm(model_dim),
                enable_nested_tensor=False,
            )
        if variant in {"race_token", "market_residual"}:
            self.race_token = nn.Parameter(torch.empty(1, 1, model_dim))
            nn.init.normal_(self.race_token, std=1.0 / math.sqrt(model_dim))

        prediction_width = model_dim * (
            2 if variant in {"race_token", "market_residual"} else 1
        )
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(prediction_width),
            nn.Linear(prediction_width, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )
        if variant == "market_residual":
            # Start from the fitted market exactly. The network must learn every
            # departure from that ordering from zero.
            final = self.prediction_head[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def config(self) -> dict[str, int | float | str | None]:
        result: dict[str, int | float | str | None] = {
            "feature_count": self.feature_count,
            "variant": self.variant,
            "hidden_dim": self.hidden_dim,
            "model_dim": self.model_dim,
            "heads": self.heads,
            "layers": self.layers,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
        }
        if self.variant == "market_residual":
            result.update({
                "market_feature_index": self.market_feature_index,
                "market_anchor_bias": self.market_anchor_bias,
                "market_anchor_scale": self.market_anchor_scale,
                "market_residual_scale": self.market_residual_scale,
            })
        return result

    def forward_parts(
        self, x: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final logits, fixed anchor logits, and residual corrections."""
        if x.ndim != 3 or x.shape[-1] != self.feature_count:
            raise ValueError(
                f"x must have shape [batch, runners, {self.feature_count}]"
            )
        if valid_mask.shape != x.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [batch, runners]")
        if torch.any(valid_mask.sum(dim=1) < 1):
            raise ValueError("every batch item must contain at least one runner")

        runner = self.feature_encoder(x)
        if self.variant == "independent":
            contextual = runner
        elif self.variant == "transformer":
            assert self.race_transformer is not None
            contextual = self.race_transformer(
                runner, src_key_padding_mask=~valid_mask
            )
        else:
            assert self.race_transformer is not None and self.race_token is not None
            token = self.race_token.expand(x.shape[0], -1, -1)
            sequence = torch.cat((token, runner), dim=1)
            token_valid = torch.ones(
                (x.shape[0], 1), dtype=torch.bool, device=x.device
            )
            sequence_valid = torch.cat((token_valid, valid_mask), dim=1)
            encoded = self.race_transformer(
                sequence, src_key_padding_mask=~sequence_valid
            )
            summary = encoded[:, :1].expand(-1, x.shape[1], -1)
            contextual = torch.cat((encoded[:, 1:], summary), dim=-1)

        raw_prediction = self.prediction_head(contextual).squeeze(-1)
        if self.variant == "market_residual":
            assert self.market_feature_index is not None
            anchor = (
                self.market_anchor_bias
                - self.market_anchor_scale * x[..., self.market_feature_index]
            )
            correction = self.market_residual_scale * raw_prediction
            logits = anchor + correction
        else:
            anchor = torch.zeros_like(raw_prediction)
            correction = raw_prediction
            logits = raw_prediction
        return tuple(
            value.masked_fill(~valid_mask, 0.0)
            for value in (logits, anchor, correction)
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_parts(x, valid_mask)[0]


def raceformer_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    ranking_weight: float = 0.5,
    cardinality_weight: float = 0.1,
    listwise_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equal-per-race classification, pairwise, cardinality, and listwise losses."""
    if logits.shape != targets.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, targets, and valid_mask must have equal shapes")
    if ranking_weight < 0 or cardinality_weight < 0 or listwise_weight < 0:
        raise ValueError("loss weights must be non-negative")
    bce_losses = []
    ranking_losses = []
    cardinality_losses = []
    listwise_losses = []
    for batch_index in range(logits.shape[0]):
        mask = valid_mask[batch_index]
        race_logits = logits[batch_index][mask]
        race_targets = targets[batch_index][mask].float()
        positives = race_logits[race_targets == 1]
        negatives = race_logits[race_targets == 0]
        if len(race_logits) < 4 or len(positives) != 3 or len(negatives) < 1:
            raise ValueError(
                "each race must have at least four runners and exactly three positives"
            )
        bce_losses.append(F.binary_cross_entropy_with_logits(race_logits, race_targets))
        ranking_losses.append(
            F.softplus(-(positives[:, None] - negatives[None, :])).mean()
        )
        cardinality_losses.append(
            ((torch.sigmoid(race_logits).sum() - 3.0) / 3.0).square()
        )
        target_distribution = race_targets / race_targets.sum()
        listwise_losses.append(
            -(target_distribution * F.log_softmax(race_logits, dim=0)).sum()
        )
    components = {
        "bce": torch.stack(bce_losses).mean(),
        "ranking": torch.stack(ranking_losses).mean(),
        "cardinality": torch.stack(cardinality_losses).mean(),
        "listwise": torch.stack(listwise_losses).mean(),
    }
    total = (
        components["bce"]
        + ranking_weight * components["ranking"]
        + cardinality_weight * components["cardinality"]
        + listwise_weight * components["listwise"]
    )
    return total, components
