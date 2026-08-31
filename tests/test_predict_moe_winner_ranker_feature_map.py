import pandas as pd
import numpy as np

from predict_moe_winner_ranker_feature_map import (
    build_model_from_checkpoint_config,
    expert_influence_line,
    expert_usage_line,
    prediction_view,
)


def test_default_prediction_view_only_shows_rank_number_and_name():
    result = pd.DataFrame({
        "runner_number": [4],
        "runner_name": ["Nankeen"],
        "ranking_logit": [0.42],
        "winner_probability": [0.14],
        "rank": [1],
        "expert_0_gate": [0.5],
    })

    view = prediction_view(result, diagnostics=False)

    assert view.columns.tolist() == ["rank", "runner_number", "runner_name"]
    assert view.iloc[0].tolist() == [1, 4, "Nankeen"]


def test_diagnostic_prediction_view_keeps_all_columns():
    result = pd.DataFrame({"rank": [1], "expert_0_gate": [0.5]})

    assert prediction_view(result, diagnostics=True) is result


def test_expert_usage_line_reports_mean_gate_percentages():
    weights = np.array([[0.25, 0.75], [0.5, 0.5]])

    assert expert_usage_line(weights) == (
        "expert_usage_mean_gate: expert_0=37.50% expert_1=62.50%"
    )


def test_expert_influence_line_reports_weighted_score_variation():
    weights = np.full((2, 2), 0.5)
    logits = np.array([[0.0, 0.0], [2.0, 6.0]])

    assert expert_influence_line(weights, logits) == (
        "expert_influence_score_variation: expert_0=25.00% expert_1=75.00%"
    )


def test_checkpoint_model_builder_restores_fixed_uniform_routing():
    model_config = {
        "feature_count": 3,
        "num_experts": 2,
        "top_k": 1,
        "gate_temperature": 1.0,
        "expert_hidden_dims": [4],
        "router_hidden_dim": 8,
        "routing_mode": "fixed_uniform",
        "feature_expert_map": {"0": [0, 1], "1": [1, 2]},
    }

    model = build_model_from_checkpoint_config(model_config)

    assert model.model_config.routing_mode == "fixed_uniform"
    assert model.router is None
    assert not any(key.startswith("router.") for key in model.state_dict())
