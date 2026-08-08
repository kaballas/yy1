"""Feature-manifest contracts for training inputs and neutralized columns."""

import json

import pytest

from src.dataset import load_feature_columns, load_feature_manifest


def _write_manifest(tmp_path, **updates):
    payload = {"features": ["speed", "weight"], "zeroed_features": ["weight"]}
    payload.update(updates)
    path = tmp_path / "features.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_feature_manifest_returns_zeroed_features(tmp_path):
    path = _write_manifest(tmp_path)

    assert load_feature_manifest(path) == (["speed", "weight"], ["weight"])
    assert load_feature_columns(path) == ["speed", "weight"]


def test_load_feature_manifest_defaults_zeroed_features_to_empty(tmp_path):
    path = _write_manifest(tmp_path, zeroed_features=[])

    assert load_feature_manifest(path) == (["speed", "weight"], [])


@pytest.mark.parametrize(
    ("zeroed_features", "message"),
    [
        ("weight", "must contain a 'zeroed_features' list"),
        (["weight", "weight"], "duplicate zeroed features"),
        (["odds"], "zeroed features are absent from 'features': odds"),
        ([""], "invalid zeroed feature name"),
    ],
)
def test_load_feature_manifest_rejects_invalid_zeroed_features(
    tmp_path, zeroed_features, message
):
    path = _write_manifest(tmp_path, zeroed_features=zeroed_features)

    with pytest.raises(ValueError, match=message):
        load_feature_manifest(path)
