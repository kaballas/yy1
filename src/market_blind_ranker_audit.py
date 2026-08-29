"""Validation-only controls for the simple market-blind winner ranker."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.advanced_racing_features import ADVANCED_FEATURE_NAMES
from src.derived_racing_features import DERIVED_FEATURE_NAMES
from src.model.race_moe import (
    RaceMixtureOfExperts, RaceWinnerModelConfig, race_softmax_nll,
)
from src.race_moe_data import (
    IDENTIFIER_FEATURES, batches, is_market_feature, market_blind_features,
    numeric_matrix, pad_batch, race_indices,
)
from src.race_moe_evaluation import evaluate_model
from src.race_moe_snapshot import load_split_snapshot
from src.raceformer_preprocessing import (
    fit_raceformer_preprocessor, model_feature_columns, transform_raceformer,
)
from src.winner_ranker import (
    CURRENT_MARKET_EXACT, IDENTIFIER_COLUMNS, is_current_market_feature,
)
from update_derived_racing_features import (
    PREPARATION_FEATURE_NAMES, RACE_AGGREGATE_FEATURE_NAMES,
)


DEFAULT_SEEDS = (11, 29, 42, 73, 101)
GROUP_NAMES = (
    "recent_form", "sectionals_speed", "career_profile", "distance",
    "track_condition", "class", "weight_barrier", "connections",
    "freshness_preparation", "race_context_relative", "prize_money_profile",
    "other_profile",
)
RACE_RELATIVE_MARKERS = (
    "_rank", "_rank_in_race", "_pct_in_race", "_minus_race_",
    "_zscore_in_race", "_gap_to_best", "race_field_", "_vs_field_",
)
KNOWN_DERIVED = set(DERIVED_FEATURE_NAMES) | set(ADVANCED_FEATURE_NAMES)
KNOWN_DERIVED |= set(PREPARATION_FEATURE_NAMES) | set(RACE_AGGREGATE_FEATURE_NAMES)


@dataclass(frozen=True)
class NeuralTrainingConfig:
    epochs: int = 15
    races_per_batch: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 4
    dropout: float = 0.20
    standardized_clip: float = 5.0


def git_commit_sha(workdir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workdir, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_market_blind_contract(features: Sequence[str]) -> None:
    retained, rejected = market_blind_features(features)
    forbidden = sorted(set(features) - set(retained))
    production_forbidden = sorted(
        feature for feature in features
        if feature in IDENTIFIER_COLUMNS
        or feature in CURRENT_MARKET_EXACT
        or is_current_market_feature(feature)
        or is_market_feature(feature)
    )
    if rejected or forbidden or production_forbidden:
        raise ValueError(
            "MARKET-BLIND CONTRACT VIOLATION: "
            + ", ".join(sorted(set(rejected + forbidden + production_forbidden)))
        )


def load_development_snapshot(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load training/validation only; inspected and sealed cohorts stay unopened."""
    frames, manifest = load_split_snapshot(
        manifest_path, splits=("training", "validation"),
    )
    if set(frames) != {"training", "validation"}:
        raise AssertionError("Development snapshot exposed a prohibited split")
    return frames, manifest


def feature_group(feature: str) -> str:
    """Assign exactly one deterministic primary ablation group."""
    name = feature.casefold()
    # Prize-money measures remain one semantic group, including their
    # race-relative forms, as declared by the experiment protocol.
    if "prize" in name:
        return "prize_money_profile"
    if any(marker in name for marker in RACE_RELATIVE_MARKERS) or name in {
        "field_size", "active_field_size", "num_scratchings", "race_number",
        "current_form_strength",
    }:
        return "race_context_relative"
    if any(token in name for token in (
        "sectional", "last600", "closing_speed", "speed_mps", "speed_rating",
        "run_quality", "hidden_",
    )):
        return "sectionals_speed"
    if any(token in name for token in (
        "jockey", "trainer", "partnership", "synergy",
    )):
        return "connections"
    if any(token in name for token in (
        "days_since", "first_up", "second_up", "third_up", "preparation",
        "best_run_recency", "second_best_run_recency", "fresh", "run_is_last",
        "run_within_last", "runs_this_preparation",
    )):
        return "freshness_preparation"
    if any(token in name for token in (
        "weight", "barrier", "draw_number",
    )):
        return "weight_barrier"
    if any(token in name for token in ("class", "grade")):
        return "class"
    if "distance" in name or "step_up" in name or "step_down" in name:
        return "distance"
    if any(token in name for token in (
        "track", "condition", "good", "soft", "heavy", "wet", "dry",
        "groundpro",
    )):
        return "track_condition"
    if any(token in name for token in (
        "career", "age", "familiarity", "historical_", "win_percentage",
        "place_percentage", "starts", "wins", "seconds", "thirds",
    )):
        return "career_profile"
    if any(token in name for token in (
        "recent", "margin", "finish", "form", "podium", "hotstreak",
        "ceiling", "consistency", "average", "best_place",
    )):
        return "recent_form"
    return "other_profile"


def assign_feature_groups(features: Sequence[str]) -> dict[str, list[str]]:
    groups = {name: [] for name in GROUP_NAMES}
    for feature in features:
        groups[feature_group(feature)].append(feature)
    assigned = [feature for names in groups.values() for feature in names]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(features):
        raise AssertionError("Every snapshot feature must be assigned exactly once")
    return groups


def feature_provenance(feature: str) -> dict[str, str]:
    """Conservative provenance: unknown inputs are never silently called safe."""
    if feature in IDENTIFIER_FEATURES or feature in IDENTIFIER_COLUMNS:
        return {
            "status": "SUSPECT", "classification": "identifier",
            "source": "src/winner_ranker.py::IDENTIFIER_COLUMNS",
            "reason": "Identifier fields cannot be model inputs.",
        }
    if feature in CURRENT_MARKET_EXACT or is_current_market_feature(feature) or is_market_feature(feature):
        return {
            "status": "SUSPECT", "classification": "market_or_price",
            "source": "src/winner_ranker.py and src/race_moe_data.py",
            "reason": "Known market-derived field.",
        }
    if feature in DERIVED_FEATURE_NAMES:
        return {
            "status": "VERIFIED_PRE_RACE", "classification": "six_prior_starts",
            "source": "src/derived_racing_features.py::derive_racing_features",
            "reason": "Generator explicitly uses stored pre-race starts and current declared conditions.",
        }
    if feature in ADVANCED_FEATURE_NAMES:
        return {
            "status": "VERIFIED_PRE_RACE", "classification": "causal_derived",
            "source": "src/advanced_racing_features.py",
            "reason": "Derived registry uses prior starts or chronological shifted entity history.",
        }
    if feature in RACE_AGGREGATE_FEATURE_NAMES:
        return {
            "status": "VERIFIED_PRE_RACE", "classification": "current_race_declared_context",
            "source": "update_derived_racing_features.py::add_race_aggregate_features",
            "reason": "Calculated from declared field, weights, ratings and prize money before the race.",
        }
    if feature in PREPARATION_FEATURE_NAMES:
        return {
            "status": "VERIFIED_PRE_RACE", "classification": "prior_race_dates",
            "source": "update_derived_racing_features.py::add_preparation_features",
            "reason": "Calculated from target start time and strictly prior stored race dates.",
        }
    return {
        "status": "UNKNOWN", "classification": "database_source_field",
        "source": "race_runners snapshot; population code not proven in repository",
        "reason": "Semantics may be pre-race, but the population path was not verified.",
    }


def _numeric_summary(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    present = numeric.dropna()
    counts = present.value_counts(dropna=False)
    dominant = float(counts.iloc[0] / len(numeric)) if len(counts) else 0.0
    zero_dominance = float((present == 0).sum() / len(numeric)) if len(numeric) else 0.0
    return {
        "coverage": float(numeric.notna().mean()),
        "missing_pct": float(numeric.isna().mean() * 100.0),
        "unique_values": int(present.nunique()),
        "mean": float(present.mean()) if len(present) else None,
        "std": float(present.std()) if len(present) else None,
        "min": float(present.min()) if len(present) else None,
        "max": float(present.max()) if len(present) else None,
        "constant": int(present.nunique()) <= 1,
        "near_constant": dominant >= 0.995,
        "zero_dominance_pct": zero_dominance * 100.0,
        "dominant_value_pct": dominant * 100.0,
    }


def audit_features(
    training: pd.DataFrame, validation: pd.DataFrame, features: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, list[str]]]:
    assert_market_blind_contract(features)
    groups = assign_feature_groups(features)
    group_by_feature = {
        feature: group for group, members in groups.items() for feature in members
    }
    rows, provenance = [], {}
    for feature in features:
        train_summary = _numeric_summary(training[feature])
        validation_summary = _numeric_summary(validation[feature])
        source = feature_provenance(feature)
        provenance[feature] = source
        rows.append({
            "feature": feature, "feature_group": group_by_feature[feature],
            "dtype": str(training[feature].dtype),
            "training_coverage": train_summary["coverage"],
            "validation_coverage": validation_summary["coverage"],
            "training_missing_pct": train_summary["missing_pct"],
            "validation_missing_pct": validation_summary["missing_pct"],
            "unique_values": train_summary["unique_values"],
            "mean": train_summary["mean"], "std": train_summary["std"],
            "min": train_summary["min"], "max": train_summary["max"],
            "constant_flag": train_summary["constant"],
            "near_constant_flag": train_summary["near_constant"],
            "zero_dominance_pct": train_summary["zero_dominance_pct"],
            "dominant_value_pct": train_summary["dominant_value_pct"],
            "source_provenance_classification": source["classification"],
            "pre_race_availability_status": source["status"],
            "provenance_source": source["source"],
            "provenance_reason": source["reason"],
        })
    return pd.DataFrame(rows), provenance, groups


def _connected_components(nodes: Sequence[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    parent = {node: node for node in nodes}

    def root(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in edges:
        a, b = root(left), root(right)
        if a != b:
            parent[b] = a
    components: dict[str, list[str]] = {}
    for node in nodes:
        components.setdefault(root(node), []).append(node)
    return [members for members in components.values() if len(members) > 1]


def _pairwise_complete_correlation(values: np.ndarray) -> np.ndarray:
    """Exact pairwise-complete Pearson matrix via sufficient statistics."""
    data = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(data).astype(np.float64)
    filled = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    counts = valid.T @ valid
    sums = filled.T @ valid
    sums_squared = np.square(filled).T @ valid
    cross = filled.T @ filled
    safe_counts = np.maximum(counts, 1.0)
    covariance = cross - sums * sums.T / safe_counts
    left_variance = sums_squared - np.square(sums) / safe_counts
    right_variance = left_variance.T
    denominator = np.sqrt(np.maximum(left_variance * right_variance, 0.0))
    correlation = np.divide(
        covariance, denominator, out=np.full_like(covariance, np.nan),
        where=(counts >= 20) & (denominator > 0),
    )
    np.fill_diagonal(correlation, 1.0)
    return correlation


def redundant_feature_clusters(
    training: pd.DataFrame, features: Sequence[str], threshold: float = 0.995,
) -> dict[str, Any]:
    numeric = training.loc[:, features].apply(pd.to_numeric, errors="coerce")
    fingerprints: dict[str, list[str]] = {}
    for feature in features:
        hashed = pd.util.hash_pandas_object(numeric[feature], index=False).to_numpy().tobytes()
        fingerprints.setdefault(hashlib.sha256(hashed).hexdigest(), []).append(feature)
    exact = [members for members in fingerprints.values() if len(members) > 1]
    constant = [feature for feature in features if numeric[feature].nunique(dropna=True) <= 1]
    near_constant = [
        feature for feature in features
        if len(numeric[feature].dropna())
        and numeric[feature].value_counts(dropna=True).iloc[0] / len(numeric) >= 0.995
    ]
    pearson = _pairwise_complete_correlation(numeric.to_numpy(dtype=float))
    ranks = numeric.rank(method="average", na_option="keep").to_numpy(dtype=float)
    spearman = _pairwise_complete_correlation(ranks)
    pearson_edges, spearman_edges = [], []
    equivalent = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1:]:
            right_index = features.index(right)
            p = pearson[left_index, right_index]
            s = spearman[left_index, right_index]
            if np.isfinite(p) and abs(float(p)) >= threshold:
                pearson_edges.append((left, right))
            if np.isfinite(s) and abs(float(s)) >= threshold:
                spearman_edges.append((left, right))
            a, b = numeric[left], numeric[right]
            shared = a.notna() & b.notna()
            if shared.any() and a.isna().equals(b.isna()) and np.allclose(
                a[shared], b[shared], rtol=1e-7, atol=1e-9,
            ):
                equivalent.append((left, right))
    all_edges = set(pearson_edges + spearman_edges + equivalent)
    clusters = []
    for index, members in enumerate(_connected_components(features, all_edges), 1):
        clusters.append({
            "cluster": index, "features": members,
            "contains_race_relative": any(
                any(marker in feature for marker in RACE_RELATIVE_MARKERS)
                for feature in members
            ),
        })
    return {
        "investigation_threshold": threshold,
        "exact_duplicate_columns": exact,
        "numerically_equivalent_pairs": [list(pair) for pair in equivalent],
        "constant_columns": constant, "near_constant_columns": near_constant,
        "pearson_pairs": [
            {"left": left, "right": right,
             "correlation": float(pearson[features.index(left), features.index(right)])}
            for left, right in pearson_edges
        ],
        "spearman_pairs": [
            {"left": left, "right": right,
             "correlation": float(spearman[features.index(left), features.index(right)])}
            for left, right in spearman_edges
        ],
        "candidate_redundant_clusters": clusters,
    }


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(width, width), nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class ResidualRaceMLP(nn.Module):
    """Small per-runner residual MLP emitting one ranking logit."""

    def __init__(self, feature_count: int, width: int = 128, blocks: int = 2, dropout: float = 0.2):
        super().__init__()
        self.feature_count = feature_count
        self.width = width
        self.blocks_count = blocks
        self.dropout = dropout
        self.projection = nn.Sequential(nn.Linear(feature_count, width), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(width, dropout) for _ in range(blocks)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def config(self) -> dict[str, Any]:
        return {
            "model_type": "residual_mlp", "feature_count": self.feature_count,
            "width": self.width, "blocks": self.blocks_count, "dropout": self.dropout,
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor, *, return_diagnostics: bool = False):
        value = self.projection(features)
        for block in self.blocks:
            value = block(value)
        logits = self.output(value).squeeze(-1).masked_fill(~valid_mask, 0.0)
        if not return_diagnostics:
            return logits
        expert_logits = logits.unsqueeze(-1)
        weights = valid_mask.unsqueeze(-1).to(logits.dtype)
        return {
            "logits": logits, "expert_logits": expert_logits,
            "router_logits": torch.zeros_like(expert_logits),
            "dense_router_weights": weights, "router_weights": weights,
            "selected_experts": valid_mask.unsqueeze(-1),
            "representation": value.masked_fill(~valid_mask.unsqueeze(-1), 0.0),
            "race_context": value.new_zeros((value.shape[0], 0)),
        }


def build_neural_model(architecture: str, feature_count: int) -> nn.Module:
    if architecture == "current_mlp":
        return RaceMixtureOfExperts(RaceWinnerModelConfig(
            feature_count=feature_count, model_type="baseline", num_experts=1,
            top_k=1, encoder_hidden_dim=128, representation_dim=64,
            expert_hidden_dims=(64,), dropout=0.20,
        ))
    if architecture == "wider_mlp":
        return RaceMixtureOfExperts(RaceWinnerModelConfig(
            feature_count=feature_count, model_type="baseline", num_experts=1,
            top_k=1, encoder_hidden_dim=192, representation_dim=128,
            expert_hidden_dims=(128,), dropout=0.20,
        ))
    if architecture == "residual_mlp":
        return ResidualRaceMLP(feature_count, width=128, blocks=2, dropout=0.20)
    raise ValueError(f"Unsupported neural architecture: {architecture}")


def prepare_feature_matrices(
    training: pd.DataFrame, validation: pd.DataFrame, features: Sequence[str],
    clip: float = 5.0, *, zero_features: Sequence[str] = (),
    preprocessing: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[str]]:
    raw_training = numeric_matrix(training, features)
    raw_validation = numeric_matrix(validation, features)
    if preprocessing is None:
        preprocessing = fit_raceformer_preprocessor(
            raw_training, features, clip=clip, layoff_bucket_mode="none",
        )
    preprocessing = dict(preprocessing)
    values = {
        "training": transform_raceformer(
            raw_training, training["race_id"].to_numpy(np.int64), features,
            zero_features, preprocessing,
        ),
        "validation": transform_raceformer(
            raw_validation, validation["race_id"].to_numpy(np.int64), features,
            zero_features, preprocessing,
        ),
    }
    return values, preprocessing, model_feature_columns(features, preprocessing)


def subset_preprocessing_contract(
    reference: Mapping[str, Any], reference_features: Sequence[str],
    selected_features: Sequence[str],
) -> dict[str, Any]:
    """Project the frozen per-column preprocessing contract onto an ablation."""
    indices = [reference_features.index(feature) for feature in selected_features]
    result = dict(reference)
    result["median"] = np.asarray(reference["median"], dtype=np.float32)[indices]
    result["scale"] = np.asarray(reference["scale"], dtype=np.float32)[indices]
    selected = set(selected_features)
    result["log1p_features"] = [
        feature for feature in reference.get("log1p_features", []) if feature in selected
    ]
    result["relative_features"] = [
        feature for feature in reference.get("relative_features", []) if feature in selected
    ]
    return result


def _selection(metrics: Mapping[str, float | int]) -> tuple[float, float, float]:
    return (
        float(metrics["top1_hit_rate"]), float(metrics["mrr"]),
        -float(metrics["race_logloss"]),
    )


def train_neural_seed(
    architecture: str, seed: int, features: Sequence[str],
    training: pd.DataFrame, validation: pd.DataFrame,
    values: Mapping[str, np.ndarray], preprocessing: Mapping[str, Any],
    transformed_features: Sequence[str], config: NeuralTrainingConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, torch.Tensor]]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = build_neural_model(architecture, len(transformed_features)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
    )
    train_y = training["is_winner"].to_numpy(np.float32)
    validation_y = validation["is_winner"].to_numpy(np.float32)
    train_ids = training["race_id"].to_numpy(np.int64)
    validation_ids = validation["race_id"].to_numpy(np.int64)
    train_groups = race_indices(train_ids)
    validation_groups = race_indices(validation_ids)
    rng = np.random.default_rng(seed)
    best_state, best_selection, best_epoch, stale, history = None, None, 0, 0, []
    for epoch in range(1, config.epochs + 1):
        model.train(); losses = []
        for groups in batches(train_groups, config.races_per_batch, rng):
            bx, by, valid = pad_batch(values["training"], train_y, groups, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx, valid)
            loss = race_softmax_nll(logits, by, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step(); losses.append(float(loss.detach()))
        metrics, _, _ = evaluate_model(
            model, values["validation"], validation_y, validation_ids,
            validation_groups, validation, config.races_per_batch, device,
        )
        selection = _selection(metrics)
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch; stale = 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "training_ranking_loss": float(np.mean(losses)),
            "validation": metrics, "best": improved,
        })
        print(
            f"  epoch={epoch:02d}/{config.epochs} seed={seed} "
            f"loss={np.mean(losses):.5f} validation_top1={metrics['top1_hit_rate']:.2%} "
            f"mrr={metrics['mrr']:.4f} logloss={metrics['race_logloss']:.4f} "
            f"best={'yes' if improved else 'no'} stale={stale}/{config.early_stopping_patience}",
            flush=True,
        )
        if stale >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError("Neural training produced no best state")
    model.load_state_dict(best_state)
    metrics, _, predictions = evaluate_model(
        model, values["validation"], validation_y, validation_ids,
        validation_groups, validation, config.races_per_batch, device,
    )
    model_config = model.config()
    result = {
        "architecture": architecture, "seed": seed, "best_epoch": best_epoch,
        "metrics": metrics, "trainable_parameters": model.trainable_parameter_count(),
        "raw_feature_count": len(features),
        "transformed_feature_count": len(transformed_features),
        "raw_features": list(features), "transformed_features": list(transformed_features),
        "model_configuration": model_config,
        "preprocessing_configuration": {
            key: value for key, value in preprocessing.items()
            if key not in {"median", "scale"}
        },
        "preprocessing_median": np.asarray(preprocessing["median"]).tolist(),
        "preprocessing_scale": np.asarray(preprocessing["scale"]).tolist(),
        "training_parameters": asdict(config), "history": history,
    }
    return result, predictions, best_state


def metrics_from_scores(frame: pd.DataFrame, scores: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    records = []
    for race_id, rows in frame.groupby("race_id", sort=False).groups.items():
        positions = np.asarray(list(rows), dtype=np.int64)
        race_scores = np.asarray(scores)[positions]
        probabilities = np.exp(race_scores - np.max(race_scores))
        probabilities /= probabilities.sum()
        winners = np.flatnonzero(frame.iloc[positions]["is_winner"].to_numpy() == 1)
        if len(winners) != 1:
            raise ValueError("Each XGBoost validation race must have exactly one winner")
        winner = int(winners[0]); order = np.argsort(-race_scores, kind="stable")
        winner_rank = int(np.flatnonzero(order == winner)[0]) + 1
        winner_probability = float(probabilities[winner])
        records.append({
            "race_id": int(race_id), "field_size": len(positions),
            "winner_rank": winner_rank, "winner_probability": winner_probability,
            "predicted_runner_number": int(frame.iloc[positions[order[0]]]["runner_number"]),
            "winner_runner_number": int(frame.iloc[positions[winner]]["runner_number"]),
            "race_logloss": -math.log(max(winner_probability, 1e-12)),
        })
    predictions = pd.DataFrame(records)
    metrics = {
        "races": len(predictions),
        "top1_hit_rate": float(predictions["winner_rank"].eq(1).mean()),
        "top2_containment": float(predictions["winner_rank"].le(2).mean()),
        "top3_containment": float(predictions["winner_rank"].le(3).mean()),
        "mrr": float((1 / predictions["winner_rank"]).mean()),
        "race_logloss": float(predictions["race_logloss"].mean()),
        "average_winner_probability": float(predictions["winner_probability"].mean()),
    }
    return metrics, predictions


def aggregate_seed_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top1 = np.asarray([row["metrics"]["top1_hit_rate"] for row in results], dtype=float)
    return {
        "seeds": [int(row["seed"]) for row in results],
        "mean_top1": float(top1.mean()), "std_top1": float(top1.std(ddof=1)),
        "minimum_top1": float(top1.min()), "maximum_top1": float(top1.max()),
        "range_top1": float(top1.max() - top1.min()),
        "mean_top2": float(np.mean([row["metrics"]["top2_containment"] for row in results])),
        "mean_top3": float(np.mean([row["metrics"]["top3_containment"] for row in results])),
        "mean_mrr": float(np.mean([row["metrics"]["mrr"] for row in results])),
        "mean_logloss": float(np.mean([row["metrics"]["race_logloss"] for row in results])),
        "mean_winner_probability": float(np.mean([
            row["metrics"]["average_winner_probability"] for row in results
        ])),
    }


def aggregate_multiseed_bootstrap(
    baseline_predictions: Mapping[int, pd.DataFrame],
    challenger_predictions: Mapping[int, pd.DataFrame],
    seeds: Sequence[int], samples: int = 20000, bootstrap_seed: int = 20260828,
) -> dict[str, Any]:
    if set(baseline_predictions) != set(seeds) or set(challenger_predictions) != set(seeds):
        raise ValueError("Baseline and challenger predictions must use identical seeds")
    base_columns, challenge_columns, race_ids = [], [], None
    for seed in seeds:
        base = baseline_predictions[seed].sort_values("race_id")
        challenge = challenger_predictions[seed].sort_values("race_id")
        if not np.array_equal(base["race_id"].to_numpy(), challenge["race_id"].to_numpy()):
            raise ValueError(f"Paired comparison race IDs differ for seed {seed}")
        if race_ids is None:
            race_ids = base["race_id"].to_numpy(np.int64)
        elif not np.array_equal(race_ids, base["race_id"].to_numpy()):
            raise ValueError("Race IDs differ across seeds")
        base_columns.append(base["winner_rank"].eq(1).to_numpy(float))
        challenge_columns.append(challenge["winner_rank"].eq(1).to_numpy(float))
    differences = np.column_stack(challenge_columns).mean(axis=1) - np.column_stack(base_columns).mean(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap = np.empty(samples, dtype=float)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        bootstrap[start:start + count] = differences[indices].mean(axis=1)
    return {
        "races": int(len(differences)), "seeds": list(map(int, seeds)),
        "mean_top1_difference": float(differences.mean()),
        "paired_race_bootstrap_samples": samples,
        "paired_race_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975)),
        ],
    }


def xgboost_group_contract(training: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    train_ids = training["race_id"].to_numpy(np.int64)
    validation_ids = validation["race_id"].to_numpy(np.int64)
    if set(train_ids) & set(validation_ids):
        raise ValueError("Target validation races entered XGBoost training")
    for label, ids in (("training", train_ids), ("validation", validation_ids)):
        seen: set[int] = set()
        previous: int | None = None
        for value in map(int, ids):
            if value != previous:
                if value in seen:
                    raise ValueError(f"{label} XGBoost race rows are not contiguous")
                seen.add(value); previous = value
    return {
        "training_races": int(pd.Series(train_ids).nunique()),
        "validation_races": int(pd.Series(validation_ids).nunique()),
        "overlapping_races": 0,
        "training_group_sizes": training.groupby("race_id", sort=False).size().astype(int).tolist(),
        "validation_group_sizes": validation.groupby("race_id", sort=False).size().astype(int).tolist(),
    }


def validate_final_selection(selection: Mapping[str, Any]) -> None:
    challengers = selection.get("selected_challengers", [])
    if len(challengers) > 1:
        raise ValueError("Final selection cannot contain more than one challenger")
    if selection.get("status") == "LOCKED" and not selection.get("configuration_sha256"):
        raise ValueError("Locked selection requires a configuration SHA256")
