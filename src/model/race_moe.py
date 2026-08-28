"""Market-blind race-level winner MLP and mixture-of-experts models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RaceWinnerModelConfig:
    feature_count: int
    model_type: str = "moe"
    encoder_hidden_dim: int = 128
    representation_dim: int = 64
    dropout: float = 0.20
    num_experts: int = 4
    top_k: int | None = 2
    gate_temperature: float = 1.0
    expert_hidden_dims: tuple[int, ...] = (64,)
    router_hidden_dim: int = 64
    expert_context_conditioning: bool = False


class MoEExpert(nn.Module):
    """A small MLP that emits one unnormalised ranking logit per runner."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        widths = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for source, target in zip(widths, widths[1:]):
            layers.extend((nn.Linear(source, target), nn.GELU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(widths[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.network(representation).squeeze(-1)


class MoERouter(nn.Module):
    """Route runners using their embedding and permutation-invariant race context."""

    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class RunnerEncoder(nn.Module):
    def __init__(
        self, feature_count: int, hidden_dim: int, representation_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, representation_dim), nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def masked_mean_max(
    representation: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return concatenated mean/max field embeddings without using runner order."""
    numeric_mask = valid_mask.unsqueeze(-1).to(representation.dtype)
    mean = (representation * numeric_mask).sum(dim=1) / numeric_mask.sum(dim=1).clamp_min(1)
    floor = torch.finfo(representation.dtype).min
    maximum = representation.masked_fill(~valid_mask.unsqueeze(-1), floor).max(dim=1).values
    return torch.cat((mean, maximum), dim=-1)


class RaceMixtureOfExperts(nn.Module):
    """Shared runner encoder followed by learned race-aware sparse routing."""

    def __init__(self, config: RaceWinnerModelConfig):
        super().__init__()
        if config.model_type not in {"baseline", "moe"}:
            raise ValueError("model_type must be baseline or moe")
        if config.feature_count < 1 or config.num_experts < 1:
            raise ValueError("feature_count and num_experts must be positive")
        if config.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        top_k = config.num_experts if config.top_k is None else config.top_k
        if not 1 <= top_k <= config.num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        if config.model_type == "baseline" and config.num_experts != 1:
            raise ValueError("baseline must use exactly one expert")
        self.model_config = config
        self.encoder = RunnerEncoder(
            config.feature_count, config.encoder_hidden_dim,
            config.representation_dim, config.dropout,
        )
        expert_input_dim = config.representation_dim * (
            3 if config.expert_context_conditioning else 1
        )
        self.experts = nn.ModuleList([
            MoEExpert(expert_input_dim, config.expert_hidden_dims, config.dropout)
            for _ in range(config.num_experts)
        ])
        router_input_dim = 3 * config.representation_dim
        self.router = MoERouter(
            router_input_dim, config.router_hidden_dim, config.num_experts
        ) if config.model_type == "moe" else None

    def config(self) -> dict[str, Any]:
        result = asdict(self.model_config)
        result["expert_hidden_dims"] = list(self.model_config.expert_hidden_dims)
        return result

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor,
        *, return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.model_config.feature_count:
            raise ValueError("features must be [batch, runners, feature_count]")
        if valid_mask.shape != features.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [batch, runners]")
        representation = self.encoder(features)
        race_context = masked_mean_max(representation, valid_mask)
        expanded_context = race_context.unsqueeze(1).expand(-1, features.shape[1], -1)
        expert_input = (
            torch.cat((representation, expanded_context), dim=-1)
            if self.model_config.expert_context_conditioning else representation
        )
        expert_logits = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=-1
        )

        if self.router is None:
            router_logits = torch.zeros_like(expert_logits)
            dense_weights = torch.ones_like(expert_logits)
            router_weights = dense_weights
            selected = torch.ones_like(expert_logits, dtype=torch.bool)
        else:
            router_input = torch.cat((representation, expanded_context), dim=-1)
            router_logits = self.router(router_input)
            dense_weights = F.softmax(
                router_logits / self.model_config.gate_temperature, dim=-1
            )
            top_k = (
                self.model_config.num_experts if self.model_config.top_k is None
                else self.model_config.top_k
            )
            if top_k < self.model_config.num_experts:
                indices = dense_weights.topk(top_k, dim=-1).indices
                selected = torch.zeros_like(dense_weights, dtype=torch.bool)
                selected.scatter_(-1, indices, True)
                sparse_weights = dense_weights * selected
                sparse_weights = sparse_weights / sparse_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(sparse_weights.dtype).eps)
                # The forward value is exactly sparse. The straight-through
                # dense term gives top-k=1 a useful ranking gradient; without
                # it, a renormalised single selected weight is the constant 1.
                router_weights = (
                    sparse_weights + dense_weights - dense_weights.detach()
                )
            else:
                selected = torch.ones_like(dense_weights, dtype=torch.bool)
                router_weights = dense_weights
        final_logits = (router_weights * expert_logits).sum(dim=-1)
        final_logits = final_logits.masked_fill(~valid_mask, 0.0)
        if not return_diagnostics:
            return final_logits
        return {
            "logits": final_logits,
            "expert_logits": expert_logits.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "router_logits": router_logits.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "dense_router_weights": dense_weights.masked_fill(
                ~valid_mask.unsqueeze(-1), 0.0
            ),
            "router_weights": router_weights.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "selected_experts": selected & valid_mask.unsqueeze(-1),
            "representation": representation.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "race_context": race_context,
        }


def race_softmax_nll(
    logits: torch.Tensor, is_winner: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean negative winner log-probability, weighting every race equally."""
    if logits.shape != is_winner.shape or logits.shape != valid_mask.shape:
        raise ValueError("logits, is_winner and valid_mask must have equal shapes")
    losses = []
    for race_index in range(logits.shape[0]):
        mask = valid_mask[race_index]
        targets = is_winner[race_index][mask]
        if int(targets.sum().item()) != 1:
            raise ValueError("each race must contain exactly one winner")
        winner_index = torch.nonzero(targets > 0, as_tuple=False).squeeze(-1)
        losses.append(F.cross_entropy(logits[race_index][mask].unsqueeze(0), winner_index))
    return torch.stack(losses).mean()


def router_balance_loss(
    router_weights: torch.Tensor, valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Gentle load CV penalty: zero for uniform mean routing, positive on collapse."""
    weights = router_weights[valid_mask]
    if weights.shape[-1] <= 1:
        return weights.new_zeros(())
    mean_load = weights.mean(dim=0)
    return weights.shape[-1] * mean_load.square().sum() - 1.0


def build_race_winner_model(config: dict[str, Any]) -> RaceMixtureOfExperts:
    clean = dict(config)
    clean["expert_hidden_dims"] = tuple(clean.get("expert_hidden_dims", (64,)))
    return RaceMixtureOfExperts(RaceWinnerModelConfig(**clean))
