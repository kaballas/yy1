"""Parity checks for the modular TabFM implementation and legacy baseline."""

import importlib.util
import io
import pickle
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import model


def _load_legacy():
  path = Path(__file__).parents[1] / "model_legacy.py"
  if not path.exists():
    pytest.skip("temporary legacy baseline is not present")
  spec = importlib.util.spec_from_file_location("model_legacy", path)
  legacy = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(legacy)
  return legacy


def _args(**overrides):
  args = dict(
      embed_dim=8, max_classes=3, col_num_blocks=1, col_nhead=2,
      col_num_inds=2, row_num_blocks=1, row_nhead=2, row_num_cls=2,
      icl_num_blocks=1, icl_nhead=2, ff_factor=2, feature_group_size=3,
      num_freq=4, decoder_hidden=6,
  )
  args.update(overrides)
  return args


def _inputs(regression=False, race=False):
  torch.manual_seed(42)
  x = torch.randn(2, 4, 5)
  if regression:
    y = torch.tensor([[1.0, 2.0, -100.0, -100.0],
                      [3.0, 4.0, 5.0, -100.0]])
  else:
    y = torch.tensor([[0.0, 1.0, -100.0, -100.0],
                      [1.0, 2.0, 0.0, -100.0]])
  train_size = torch.tensor([2, 3], dtype=torch.long)
  d = torch.tensor([3, 5], dtype=torch.long)
  cat_mask = torch.tensor([[True, False, True, False, False],
                           [False, True, False, True, False]])
  kwargs = dict(d=d, cat_mask=cat_mask)
  if race:
    kwargs.update(
        race_group_ids=torch.tensor([[-1, -1, 10, 10],
                                      [-1, -1, -1, 20]]),
        valid_row_mask=torch.ones(2, 4, dtype=torch.bool),
    )
  return x, y, train_size, kwargs


def _assert_nested_equal(left, right):
  if isinstance(left, torch.Tensor):
    torch.testing.assert_close(left, right, rtol=0, atol=0)
  elif isinstance(left, dict):
    assert left.keys() == right.keys()
    for key in left:
      _assert_nested_equal(left[key], right[key])
  elif isinstance(left, (list, tuple)):
    assert len(left) == len(right)
    for a, b in zip(left, right):
      _assert_nested_equal(a, b)
  elif isinstance(left, model.QuantizedTensor):
    _assert_nested_equal(left.data, right.data)
    _assert_nested_equal(left.scale, right.scale)
  elif hasattr(left, "__dataclass_fields__"):
    for name in left.__dataclass_fields__:
      _assert_nested_equal(getattr(left, name), getattr(right, name))
  else:
    assert left == right


def test_import_compatibility_and_state_dict_order():
  from model import (CellEmbedder, ICLearningCache, MultiheadAttention,
                     RMSNorm, RaceSetEncoder, TabFM, detach_cache,
                     move_cache_to_device)
  assert all((CellEmbedder, ICLearningCache, MultiheadAttention, RMSNorm,
              RaceSetEncoder, TabFM, detach_cache, move_cache_to_device))

  legacy = _load_legacy()
  torch.manual_seed(7)
  old = legacy.TabFM(**_args())
  torch.manual_seed(7)
  new = model.TabFM(**_args())
  assert list(old.state_dict()) == list(new.state_dict())
  new.load_state_dict(old.state_dict(), strict=True)
  old.load_state_dict(new.state_dict(), strict=True)


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("race_mode", ["none", "self_attention"])
def test_full_forward_parity(regression, race_mode):
  legacy = _load_legacy()
  args = _args(is_classifier=not regression, race_context_mode=race_mode)
  if race_mode != "none":
    args.update(race_context_dim=8, race_context_heads=2, race_context_ff_dim=12)
  torch.manual_seed(11)
  old = legacy.TabFM(**args).eval()
  torch.manual_seed(11)
  new = model.TabFM(**args).eval()
  new.load_state_dict(old.state_dict(), strict=True)
  x, y, train_size, kwargs = _inputs(regression, race_mode != "none")
  with torch.no_grad():
    expected = old(x, y, train_size, **kwargs)
    actual = new(x, y, train_size, **kwargs)
  _assert_nested_equal(expected, actual)


def test_prefill_decode_quantized_cache_and_utilities():
  legacy = _load_legacy()
  args = _args()
  torch.manual_seed(13)
  old = legacy.TabFM(**args).eval()
  torch.manual_seed(13)
  new = model.TabFM(**args).eval()
  new.load_state_dict(old.state_dict(), strict=True)
  x, y, _, kwargs = _inputs()
  with torch.no_grad():
    cache_args = dict(d=kwargs["d"], cat_mask=kwargs["cat_mask"],
                      feature_schema_hash="schema-v1",
                      preprocessing_version="prep-v1")
    old_prefill, old_cache = old.prefill(x, y, **cache_args)
    new_prefill, new_cache = new.prefill(x, y, **cache_args)
    _assert_nested_equal(old_prefill, new_prefill)
    _assert_nested_equal(old_cache, new_cache)
    query = x[:, :2] + 0.1
    old_decoded = old.decode(query, old_cache, **cache_args)
    new_decoded = new.decode(query, new_cache, **cache_args)
    _assert_nested_equal(old_decoded, new_decoded)
    old_q = old_cache["icl"].quantize(torch.int8)
    new_q = new_cache["icl"].quantize(torch.int8)
    old_cache["icl"] = old_q
    new_cache["icl"] = new_q
    _assert_nested_equal(old.decode(query, old_cache, **cache_args),
                         new.decode(query, new_cache, **cache_args))

  nested = model.ICLearningCache(
      [(model.QuantizedTensor(torch.ones(2), torch.ones(())), torch.ones(2))],
      torch.ones(1, requires_grad=True),
  )
  moved = model.move_cache_to_device({"nested": [nested]}, "cpu")
  detached = model.detach_cache(moved)
  assert not detached["nested"][0].prefill_train_size.requires_grad
  assert detached["nested"][0].layer_caches[0][0].data.device.type == "cpu"


def test_pickle_and_torch_serialization_round_trip():
  instance = model.TabFM(**_args()).eval()
  restored = pickle.loads(pickle.dumps(instance))
  assert list(restored.state_dict()) == list(instance.state_dict())
  buffer = io.BytesIO()
  torch.save(instance, buffer)
  buffer.seek(0)
  loaded = torch.load(buffer, weights_only=False)
  assert list(loaded.state_dict()) == list(instance.state_dict())


def test_validation_exception_compatibility():
  legacy = _load_legacy()
  cases = [
      (lambda m: m(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1])),
       ValueError),
      (lambda m: m(torch.zeros(1, 2, 2), torch.zeros(1, 2), torch.tensor([1.0])),
       ValueError),
  ]
  for invoke, exception in cases:
    for cls in (legacy.TabFM, model.TabFM):
      with pytest.raises(exception) as caught:
        invoke(cls(**_args()))
      if cls is legacy.TabFM:
        message = str(caught.value)
      else:
        assert str(caught.value) == message


def test_regression_prefill_accepts_explicit_context_with_minus_100_target():
  instance = model.TabFM(**_args(is_classifier=False)).eval()
  x = torch.randn(1, 3, 4)
  y = torch.tensor([[-100.0, 1.5, -100.0]])
  with pytest.raises(ValueError, match="regression prefill requires explicit"):
    instance.prefill(x, y, feature_schema_hash="schema-v1",
                     preprocessing_version="prep-v1")
  logits, cache = instance.prefill(
      x, y, train_size=torch.tensor([2]), feature_schema_hash="schema-v1",
      preprocessing_version="prep-v1")
  assert logits.shape[:2] == y.shape
  assert cache["metadata"]["format_version"] == 3


def test_decode_accepts_masked_padded_race_rows():
  instance = model.TabFM(
      **_args(race_context_mode="self_attention", race_context_dim=8,
              race_context_heads=2, race_context_ff_dim=12)).eval()
  x = torch.randn(1, 2, 4)
  y = torch.tensor([[0.0, 1.0]])
  _, cache = instance.prefill(
      x, y, feature_schema_hash="schema-v1", preprocessing_version="prep-v1")
  query = torch.randn(1, 3, 4)
  groups = torch.tensor([[7, 7, -1]])
  valid = torch.tensor([[True, True, False]])
  output = instance.decode(
      query, cache, race_group_ids=groups, valid_row_mask=valid,
      feature_schema_hash="schema-v1", preprocessing_version="prep-v1")
  assert torch.equal(output[:, 2], torch.zeros_like(output[:, 2]))
