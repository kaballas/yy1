"""Race winner MoE with per-expert feature allowlists loaded from JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.model.race_moe import MoERouter, masked_mean_max

RELATIVE_SUFFIX = "__race_percentile"


@dataclass(frozen=True)
class FeatureMappedRaceWinnerConfig:
    feature_count: int
    num_experts: int = 4
    top_k: int | None = 2
    gate_temperature: float = 1.0
    expert_hidden_dims: tuple[int, ...] = (64,)
    router_hidden_dim: int = 64
    dropout: float = 0.20
    routing_mode: str = "learned"
    feature_map: tuple[tuple[int, ...], ...] = ()
    router_feature_indices: tuple[int, ...] = ()


def _normalise_expert_map(raw: Any, feature_names: list[str], num_experts: int) -> tuple[tuple[int, ...], ...]:
    """Return an explicit expert-to-feature mapping.

    Each expert must explicitly list at least one available feature. Features
    may be omitted from every expert, allowing the JSON file to act as a manual
    feature selector. There is no fallback to all features.

    Supported JSON shapes:
      {"experts": {"0": ["feat_a", "feat_b"], "1": ["feat_c"]}}
      {"expert_0": ["feat_a"], "expert_1": ["feat_b"]}
      {"experts": [["feat_a", "feat_b"], ["feat_c"]]}
      {"shared_features": ["feat_x"]}  # same features duplicated across all experts
    """
    if raw is None:
        raise ValueError("A feature-to-expert JSON map is required; no fallback is allowed")

    payload = json.loads(Path(raw).read_text()) if isinstance(raw, (str, Path)) else raw
    if not isinstance(payload, dict):
        raise ValueError("feature expert map JSON must be an object")

    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    map_by_expert: dict[int, set[int]] = {expert: set() for expert in range(num_experts)}

    shared = payload.get("shared_features", [])
    if isinstance(shared, list):
        for name in shared:
            idx = feature_to_index.get(name)
            if idx is not None:
                for expert in range(num_experts):
                    map_by_expert[expert].add(idx)

    experts_dict = payload.get("experts", payload)
    if isinstance(experts_dict, dict):
        for key, value in experts_dict.items():
            if key == "shared_features":
                continue
            expert_id = int(str(key).replace("expert_", "")) if str(key).startswith("expert_") else int(key)
            if expert_id < 0 or expert_id >= num_experts:
                raise ValueError(f"JSON expert id {expert_id} is outside 0..{num_experts - 1}")
            if not isinstance(value, list):
                raise ValueError(f"expert {expert_id} feature list must be a list")
            for name in value:
                idx = feature_to_index.get(name)
                if idx is not None:
                    map_by_expert[expert_id].add(idx)
    elif isinstance(experts_dict, list):
        for expert_id, value in enumerate(experts_dict):
            if expert_id >= num_experts:
                raise ValueError(f"JSON contains more experts than configured: {expert_id + 1} > {num_experts}")
            if not isinstance(value, list):
                raise ValueError(f"expert {expert_id} feature list must be a list")
            for name in value:
                idx = feature_to_index.get(name)
                if idx is not None:
                    map_by_expert[expert_id].add(idx)
    else:
        raise ValueError("feature map JSON must contain an 'experts' object or list")

    for expert_id in range(num_experts):
        if not map_by_expert[expert_id]:
            raise ValueError(f"expert {expert_id} has no explicit features; every expert must list at least one feature")

    return tuple(tuple(sorted(indices)) for indices in (map_by_expert[expert] for expert in range(num_experts)))


def load_feature_expert_map(path: str | Path | None, feature_names: list[str], num_experts: int) -> tuple[tuple[int, ...], ...]:
    return _normalise_expert_map(path, feature_names, num_experts)


def load_router_feature_indices(
    path: str | Path, feature_names: list[str],
) -> tuple[int, ...]:
    """Load the explicit raw-feature allowlist used by the learned router."""
    payload = json.loads(Path(path).read_text())
    router_features = payload.get("router_features")
    if not isinstance(router_features, list) or not router_features:
        raise ValueError(
            "feature map JSON must contain a non-empty 'router_features' list"
        )
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}
    unknown = [name for name in router_features if name not in feature_to_index]
    if unknown:
        raise ValueError(
            "router_features contains unavailable features: " + ", ".join(unknown)
        )
    return tuple(
        dict.fromkeys(feature_to_index[name] for name in router_features)
    )


def expand_feature_map_to_model_features(
    raw_map: tuple[tuple[int, ...], ...],
    raw_feature_names: list[str],
    model_feature_names: list[str],
) -> tuple[tuple[int, ...], ...]:
    model_index_by_name = {name: idx for idx, name in enumerate(model_feature_names)}
    expanded: list[tuple[int, ...]] = []
    for expert_indices in raw_map:
        indices: set[int] = set()
        for raw_idx in expert_indices:
            raw_name = raw_feature_names[raw_idx]
            if raw_name in model_index_by_name:
                indices.add(model_index_by_name[raw_name])
            suffixed = f"{raw_name}{RELATIVE_SUFFIX}"
            if suffixed in model_index_by_name:
                indices.add(model_index_by_name[suffixed])
        if not indices:
            raise ValueError(
                f"expert feature map resolves to no model features for raw names {tuple(raw_feature_names[idx] for idx in expert_indices)}"
            )
        expanded.append(tuple(sorted(indices)))
    return tuple(expanded)


def expand_feature_indices_to_model_features(
    raw_indices: tuple[int, ...],
    raw_feature_names: list[str],
    model_feature_names: list[str],
) -> tuple[int, ...]:
    """Expand an allowlist to include each configured race-relative feature."""
    return expand_feature_map_to_model_features(
        (raw_indices,), raw_feature_names, model_feature_names,
    )[0]


class FeatureMappedExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        widths = (input_dim, *hidden_dims)
        layers: list[nn.Module] = []
        for source, target in zip(widths, widths[1:]):
            layers.extend((nn.Linear(source, target), nn.GELU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(widths[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class RaceMixtureOfExpertsFeatureMap(nn.Module):
    """Mixture of experts with an explicit feature allowlist per expert."""

    def __init__(self, config: FeatureMappedRaceWinnerConfig):
        super().__init__()
        self.model_config = config
        self.num_experts = config.num_experts
        self.feat_to_expert = list(config.feature_map)
        self.has_feature_map = bool(config.feature_map)
        self.router_feature_indices = (
            config.router_feature_indices
            or tuple(range(config.feature_count))
        )

        self.experts = nn.ModuleList([
            FeatureMappedExpert(len(indices), config.expert_hidden_dims, config.dropout)
            for indices in self.feat_to_expert
        ])
        router_input_dim = 3 * len(self.router_feature_indices)
        self.router = (
            MoERouter(router_input_dim, config.router_hidden_dim, config.num_experts)
            if config.routing_mode == "learned" else None
        )

    @property
    def feature_map(self) -> tuple[tuple[int, ...], ...]:
        return self.feat_to_expert

    def forward(
        self, features: torch.Tensor, valid_mask: torch.Tensor, *, return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.model_config.feature_count:
            raise ValueError("features must be [batch, runners, feature_count]")
        if valid_mask.shape != features.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [batch, runners]")

        router_features = features[..., self.router_feature_indices]
        race_context = masked_mean_max(router_features, valid_mask)
        expert_logits = []
        for expert_index, feature_indices in enumerate(self.feat_to_expert):
            expert_features = features[..., feature_indices]
            expert_logits.append(self.experts[expert_index](expert_features))
        expert_logits_tensor = torch.stack(expert_logits, dim=-1)

        if self.router is None:
            router_logits = torch.zeros_like(expert_logits_tensor)
            dense_weights = torch.full_like(
                expert_logits_tensor, 1.0 / self.model_config.num_experts
            )
            router_weights = dense_weights
            selected = torch.ones_like(dense_weights, dtype=torch.bool)
        else:
            expanded_context = race_context.unsqueeze(1).expand(-1, features.shape[1], -1)
            router_input = torch.cat((router_features, expanded_context), dim=-1)
            router_logits = self.router(router_input)
            dense_weights = F.softmax(router_logits / self.model_config.gate_temperature, dim=-1)

            top_k = self.model_config.num_experts if self.model_config.top_k is None else self.model_config.top_k
            if top_k < self.model_config.num_experts:
                indices = dense_weights.topk(top_k, dim=-1).indices
                selected = torch.zeros_like(dense_weights, dtype=torch.bool)
                selected.scatter_(-1, indices, True)
                sparse_weights = dense_weights * selected
                sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(sparse_weights.dtype).eps)
                router_weights = sparse_weights + dense_weights - dense_weights.detach()
            else:
                selected = torch.ones_like(dense_weights, dtype=torch.bool)
                router_weights = dense_weights
        final_logits = (router_weights * expert_logits_tensor).sum(dim=-1)
        final_logits = final_logits.masked_fill(~valid_mask, 0.0)

        if not return_diagnostics:
            return final_logits

        return {
            "logits": final_logits,
            "expert_logits": expert_logits_tensor.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "router_logits": router_logits.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "dense_router_weights": dense_weights.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "router_weights": router_weights.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "selected_experts": selected & valid_mask.unsqueeze(-1),
            "race_context": race_context,
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def contributing_parameter_count(self) -> int:
        expert_counts = [sum(p.numel() for p in expert.parameters()) for expert in self.experts]
        if self.router is None or self.model_config.top_k is None:
            active = expert_counts
        else:
            active = sorted(expert_counts, reverse=True)[: self.model_config.top_k]
        router = 0 if self.router is None else sum(p.numel() for p in self.router.parameters())
        return router + sum(active)

    def executed_parameter_count(self) -> int:
        return self.trainable_parameter_count()


def build_feature_mapped_model(config: FeatureMappedRaceWinnerConfig) -> RaceMixtureOfExpertsFeatureMap:
    return RaceMixtureOfExpertsFeatureMap(config)
