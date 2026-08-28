import numpy as np
import pandas as pd
import pytest
import torch

from src.model.race_moe import (
    RaceMixtureOfExperts, RaceWinnerModelConfig, build_race_winner_model,
    race_softmax_nll, router_balance_loss,
)
from src.race_moe_data import chronological_race_ids, market_blind_features
from src.race_moe_evaluation import collapse_warnings, routing_diagnostics


def test_moe_forward_contract_and_sparse_top_k():
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=5, num_experts=4, top_k=2, dropout=0.0,
    )).eval()
    x = torch.randn(2, 6, 5)
    valid = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
    ])
    output = model(x, valid, return_diagnostics=True)
    assert set(output) == {
        "logits", "expert_logits", "router_logits", "router_weights",
        "dense_router_weights", "selected_experts", "representation", "race_context",
    }
    assert output["logits"].shape == (2, 6)
    assert torch.all(output["selected_experts"][valid].sum(dim=-1) == 2)
    assert torch.allclose(
        output["router_weights"][valid].sum(dim=-1), torch.ones(10), atol=1e-6
    )
    assert torch.all(output["router_weights"][~valid] == 0)


def test_moe_is_permutation_equivariant_with_race_context():
    torch.manual_seed(3)
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=3, num_experts=4, top_k=2, dropout=0.0,
        expert_context_conditioning=True,
    )).eval()
    x = torch.randn(1, 5, 3); valid = torch.ones((1, 5), dtype=torch.bool)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = model(x, valid, return_diagnostics=True)
    shuffled = model(x[:, permutation], valid, return_diagnostics=True)
    assert torch.allclose(shuffled["logits"], original["logits"][:, permutation], atol=1e-6)
    assert torch.allclose(
        shuffled["router_weights"], original["router_weights"][:, permutation], atol=1e-6
    )


def test_top1_router_receives_ranking_gradient_through_straight_through_gate():
    torch.manual_seed(9)
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=3, num_experts=2, top_k=1, dropout=0.0,
    ))
    x = torch.randn(2, 4, 3)
    valid = torch.ones((2, 4), dtype=torch.bool)
    winners = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    output = model(x, valid, return_diagnostics=True)
    race_softmax_nll(output["logits"], winners, valid).backward()
    assert model.router is not None
    gradients = [p.grad for p in model.router.parameters()]
    assert any(value is not None and torch.count_nonzero(value) for value in gradients)


def test_race_softmax_nll_is_equal_per_race_and_exact():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 9.0]])
    winners = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    valid = torch.ones_like(winners, dtype=torch.bool)
    expected = (
        -torch.log_softmax(logits[0], dim=0)[0]
        -torch.log_softmax(logits[1], dim=0)[1]
    ) / 2
    assert race_softmax_nll(logits, winners, valid) == pytest.approx(float(expected))


def test_router_balance_loss_detects_collapse_without_forcing_baseline():
    valid = torch.ones((1, 8), dtype=torch.bool)
    uniform = torch.full((1, 8, 4), 0.25)
    collapsed = torch.zeros((1, 8, 4)); collapsed[..., 2] = 1.0
    assert router_balance_loss(uniform, valid) == pytest.approx(0.0)
    assert router_balance_loss(collapsed, valid) == pytest.approx(3.0)
    assert router_balance_loss(torch.ones((1, 8, 1)), valid) == pytest.approx(0.0)


def test_config_round_trip_reconstructs_model():
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=7, num_experts=2, top_k=None, expert_hidden_dims=(32, 16),
    ))
    rebuilt = build_race_winner_model(model.config())
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.config() == model.config()


def test_market_blind_filter_rejects_direct_and_indirect_prices():
    retained, excluded = market_blind_features([
        "age", "marketWinPrice", "recent_1_starting_price", "fluc2",
        "market_implied_prob_change_open_to_fluc2", "career_starts",
    ])
    assert retained == ["age", "career_starts"]
    assert set(excluded) == {
        "marketWinPrice", "recent_1_starting_price", "fluc2",
        "market_implied_prob_change_open_to_fluc2",
    }


def test_chronological_split_is_consecutive_and_sealed():
    frame = pd.DataFrame({"race_id": np.repeat(np.arange(1, 11), 4)})
    train, validation, test = chronological_race_ids(frame, 3, 2)
    assert train == [1, 2, 3, 4, 5]
    assert validation == [6, 7, 8]
    assert test == [9, 10]


def test_diagnostics_warn_for_router_and_output_collapse():
    rows = 20
    weights = np.zeros((rows, 3)); weights[:, 1] = 1.0
    selected = weights.astype(bool)
    base = np.arange(rows, dtype=float)
    expert_logits = np.column_stack((base, base * 2, -base))
    frame = pd.DataFrame({
        "race_id": np.repeat(np.arange(4), 5), "distance_m": 1200,
        "class_name": "BM", "field_size": 5, "active_field_size": 5,
        "track_status": "Good", "career_starts": 3,
    })
    diagnostics = routing_diagnostics(weights, selected, expert_logits, frame)
    warnings = collapse_warnings(diagnostics, 0.8, 0.98)
    assert diagnostics["dominant_expert_rate"] == 1.0
    assert len(warnings) == 2
