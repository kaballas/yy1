"""Focused contracts for the current-race-only RaceFormerTop3 model."""

import pytest

torch = pytest.importorskip("torch")

from src.model.raceformer import RaceFormerTop3, raceformer_losses
from finetune_raceformer import configure_scope
from train_raceformer import fit_market_anchor


@pytest.mark.parametrize("variant", ["independent", "transformer", "race_token"])
def test_raceformer_preserves_runner_permutation_equivariance(variant):
    torch.manual_seed(7)
    model = RaceFormerTop3(
        feature_count=5, variant=variant, hidden_dim=12, model_dim=8,
        heads=2, layers=1, feedforward_dim=16, dropout=0.0,
    ).eval()
    x = torch.randn(1, 6, 5)
    valid = torch.ones(1, 6, dtype=torch.bool)
    order = torch.tensor([3, 0, 5, 2, 1, 4])
    inverse = torch.argsort(order)

    with torch.inference_mode():
        baseline = model(x, valid)
        permuted = model(x[:, order], valid[:, order])[:, inverse]

    assert torch.allclose(baseline, permuted, atol=1e-6)


@pytest.mark.parametrize("variant", ["transformer", "race_token"])
def test_padding_does_not_change_real_runner_predictions(variant):
    torch.manual_seed(11)
    model = RaceFormerTop3(
        feature_count=4, variant=variant, hidden_dim=12, model_dim=8,
        heads=2, layers=1, feedforward_dim=16, dropout=0.0,
    ).eval()
    race = torch.randn(1, 5, 4)
    padded = torch.cat((race, torch.randn(1, 3, 4) * 100), dim=1)
    race_valid = torch.ones(1, 5, dtype=torch.bool)
    padded_valid = torch.tensor([[True] * 5 + [False] * 3])

    with torch.inference_mode():
        expected = model(race, race_valid)
        actual = model(padded, padded_valid)[:, :5]

    assert torch.allclose(expected, actual, atol=1e-6)


def test_independent_variant_does_not_use_other_runners():
    torch.manual_seed(13)
    model = RaceFormerTop3(
        feature_count=3, variant="independent", hidden_dim=8, model_dim=4,
        heads=1, layers=1, feedforward_dim=8, dropout=0.0,
    ).eval()
    x = torch.randn(1, 5, 3)
    changed = x.clone()
    changed[:, 1:] += 100
    valid = torch.ones(1, 5, dtype=torch.bool)

    with torch.inference_mode():
        assert torch.allclose(model(x, valid)[:, 0], model(changed, valid)[:, 0])


def test_market_residual_starts_at_exact_market_anchor():
    model = RaceFormerTop3(
        feature_count=3, variant="market_residual", hidden_dim=8, model_dim=4,
        heads=1, layers=1, feedforward_dim=8, dropout=0.0,
        market_feature_index=1, market_anchor_bias=-0.4,
        market_anchor_scale=1.7, market_residual_scale=0.25,
    ).eval()
    x = torch.tensor([[[0.0, -1.0, 2.0], [0.0, 0.5, 1.0], [0.0, 2.0, 0.0]]])
    valid = torch.ones(1, 3, dtype=torch.bool)

    with torch.inference_mode():
        logits, anchor, residual = model.forward_parts(x, valid)

    assert torch.allclose(anchor, -0.4 - 1.7 * x[..., 1])
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.equal(logits, anchor)
    assert torch.equal(torch.argsort(logits, descending=True), torch.tensor([[0, 1, 2]]))


def test_market_residual_is_permutation_equivariant_and_padding_safe():
    torch.manual_seed(17)
    model = RaceFormerTop3(
        feature_count=4, variant="market_residual", hidden_dim=12, model_dim=8,
        heads=2, layers=1, feedforward_dim=16, dropout=0.0,
        market_feature_index=2, market_anchor_bias=-0.2,
        market_anchor_scale=1.3, market_residual_scale=0.25,
    ).eval()
    x = torch.randn(1, 6, 4)
    valid = torch.ones(1, 6, dtype=torch.bool)
    order = torch.tensor([3, 0, 5, 2, 1, 4])
    padded = torch.cat((x, torch.randn(1, 2, 4) * 100), dim=1)
    padded_valid = torch.tensor([[True] * 6 + [False] * 2])

    with torch.inference_mode():
        expected = model(x, valid)
        permuted = model(x[:, order], valid[:, order])[:, torch.argsort(order)]
        padded_result = model(padded, padded_valid)[:, :6]

    assert torch.allclose(expected, permuted, atol=1e-6)
    assert torch.allclose(expected, padded_result, atol=1e-6)


def test_market_residual_requires_valid_anchor_configuration():
    with pytest.raises(ValueError, match="market_feature_index"):
        RaceFormerTop3(feature_count=3, variant="market_residual")
    with pytest.raises(ValueError, match="market_anchor_scale"):
        RaceFormerTop3(
            feature_count=3, variant="market_residual", market_feature_index=1,
            market_anchor_scale=0.0,
        )


def test_fitted_market_anchor_keeps_a_meaningful_positive_slope():
    x = torch.tensor([
        [-2.0], [-1.0], [0.0], [1.0], [2.0],
        [-2.0], [-1.0], [0.0], [1.0], [2.0],
    ]).numpy()
    y = torch.tensor([1, 1, 1, 0, 0, 0, 0, 1, 1, 1]).numpy()
    race_ids = torch.tensor([1] * 5 + [2] * 5).numpy()
    groups = {1: torch.arange(5).numpy(), 2: torch.arange(5, 10).numpy()}

    bias, scale, summary = fit_market_anchor(
        x, y, race_ids, groups, 0, 0.5, 0.1, 0.5
    )

    assert torch.isfinite(torch.tensor(bias))
    assert scale >= 0.25
    assert summary["mean_race_probability_sum"] == pytest.approx(3.0, abs=0.1)


def test_losses_are_equal_per_race_and_differentiable():
    logits = torch.tensor(
        [[2.0, 1.0, 0.5, -1.0, 0.0], [0.1, 0.2, 0.3, -0.1, 99.0]],
        requires_grad=True,
    )
    targets = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    valid = torch.tensor([[True] * 5, [True] * 4 + [False]])

    loss, components = raceformer_losses(logits, targets, valid, 0.5, 0.1)
    loss.backward()

    assert torch.isfinite(loss)
    assert set(components) == {"bce", "ranking", "cardinality", "listwise"}
    assert logits.grad is not None
    assert logits.grad[1, 4] == 0


def test_losses_reject_incomplete_top_three_race():
    with pytest.raises(ValueError, match="exactly three positives"):
        raceformer_losses(
            torch.zeros(1, 4),
            torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
            torch.ones(1, 4, dtype=torch.bool),
        )


def test_fine_tune_head_only_freezes_backbone():
    model = RaceFormerTop3(
        feature_count=3, variant="race_token", hidden_dim=8, model_dim=4,
        heads=1, layers=1, feedforward_dim=8, dropout=0.0,
    )
    names = configure_scope(model, "head_only")
    assert names
    assert all(name.startswith("prediction_head.") for name in names)
    assert not any(parameter.requires_grad for parameter in model.feature_encoder.parameters())


def test_fine_tune_transformer_scope_includes_race_token_but_not_feature_encoder():
    model = RaceFormerTop3(
        feature_count=3, variant="race_token", hidden_dim=8, model_dim=4,
        heads=1, layers=1, feedforward_dim=8, dropout=0.0,
    )
    names = configure_scope(model, "transformer_and_head")
    assert "race_token" in names
    assert any(name.startswith("race_transformer.") for name in names)
    assert not any(name.startswith("feature_encoder.") for name in names)
