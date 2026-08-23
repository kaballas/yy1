import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "detect_duplicate_winner_models.py"
SPEC = importlib.util.spec_from_file_location("duplicate_winner_models", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
duplicate_groups = MODULE.duplicate_groups
load_feature_groups = MODULE.load_feature_groups


def test_duplicate_groups_separates_exact_and_reordered_models():
    exact, reordered = duplicate_groups({
        "x1": ("a", "b"),
        "x2": ("a", "b"),
        "x3": ("b", "a"),
        "x4": ("c",),
    })

    assert exact == [("x1", "x2")]
    assert reordered == [("x1", "x2", "x3")]


def test_load_feature_groups_rejects_duplicate_features_within_a_model(tmp_path):
    manifest = tmp_path / "winner_ranker_features.json"
    manifest.write_text(json.dumps({
        "models": {"x1": {"features": ["a", "a"]}},
    }))

    with pytest.raises(ValueError, match="contains duplicates"):
        load_feature_groups(manifest)
