"""Focused tests for query-only and race-group winner training contracts."""

import pytest

torch = pytest.importorskip("torch")

from src.losses import query_row_cross_entropy, race_winner_cross_entropy
from src.sampling import validate_complete_race_batch


def _winner_batch():
  logits = torch.tensor([[[3.0, 0.0], [4.0, 1.0], [0.0, 2.0],
                         [2.0, 0.0], [0.0, 1.0], [9.0, -9.0]]], requires_grad=True)
  targets = torch.tensor([[-100, -100, 1, 0, 1, -100]])
  groups = torch.tensor([[-1, -1, 0, 0, 1, -1]])
  train_size = torch.tensor([2])
  valid = torch.tensor([[True, True, True, True, True, False]])
  return logits, targets, groups, train_size, valid


def test_race_loss_ignores_context_and_padding_logits():
  logits, targets, groups, train_size, valid = _winner_batch()
  baseline = race_winner_cross_entropy(logits, targets, train_size, groups, valid)
  changed = logits.detach().clone()
  changed[:, :2] += 1000
  changed[:, 5] += 7
  assert torch.equal(
      baseline,
      race_winner_cross_entropy(changed, targets, train_size, groups, valid),
  )
  query_changed = logits.detach().clone()
  query_changed[:, 2, 1] += 7
  assert not torch.equal(
      baseline,
      race_winner_cross_entropy(query_changed, targets, train_size, groups, valid),
  )


def test_query_row_loss_ignores_context_and_padding_logits():
  logits = torch.zeros(1, 5, 2)
  targets = torch.tensor([[0, 1, 1, 0, -100]])
  train_size = torch.tensor([2])
  valid = torch.tensor([[True, True, True, True, False]])
  baseline = query_row_cross_entropy(logits, targets, train_size, valid)
  changed = logits.clone()
  changed[:, :2] = 100
  changed[:, 4] = -100
  assert torch.equal(baseline, query_row_cross_entropy(changed, targets, train_size, valid))
  changed[:, 2, 1] = 5
  assert not torch.equal(baseline, query_row_cross_entropy(changed, targets, train_size, valid))


def test_race_loss_is_order_invariant_and_rejects_bad_winner_counts():
  logits, targets, groups, train_size, valid = _winner_batch()
  baseline = race_winner_cross_entropy(logits, targets, train_size, groups, valid)
  order = torch.tensor([1, 0, 3, 2, 4])
  assert torch.allclose(
      baseline,
      race_winner_cross_entropy(logits[:, order], targets[:, order], groups[:, order], train_size, valid[:, order]),
  )
  with pytest.raises(ValueError, match="exactly one winner"):
    race_winner_cross_entropy(logits, targets.clone().fill_(-100), train_size, groups, valid)


def test_complete_race_validation_rejects_context_overlap_and_bad_padding():
  canonical = [[10, 10, 20, 20]]
  groups = [[-1, -1, 0, 0]]
  targets = [[-100, -100, 0, 1]]
  valid = [[True, True, True, True]]
  validate_complete_race_batch(canonical, groups, [2], valid, targets,
                               expected_race_row_counts={10: 2, 20: 2})
  with pytest.raises(ValueError, match="both context and query"):
    validate_complete_race_batch([[10, 20, 10, 20]], groups, [2], valid, targets)
  with pytest.raises(ValueError, match="padding rows"):
    validate_complete_race_batch(canonical, [[-1, -1, 0, 1]], [2],
                                 [[True, True, True, False]], targets)
  with pytest.raises(ValueError, match="incomplete"):
    validate_complete_race_batch(canonical, groups, [2], valid, targets,
                                 expected_race_row_counts={20: 3})
  with pytest.raises(ValueError, match="non-negative race_group_id"):
    validate_complete_race_batch(canonical, [[-1, -1, -1, 0]], [2], valid, targets)
