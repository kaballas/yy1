import pytest

from make_each_feature_moe_expert import one_feature_per_expert


def test_every_unique_feature_becomes_one_expert_by_default():
    result = one_feature_per_expert({
        "shared_features": ["field_size", "distance"],
        "experts": {
            "0": ["speed", "field_size"],
            "1": ["form", "speed"],
        },
    })

    assert result == {
        "shared_features": [],
        "experts": {
            "0": ["field_size"],
            "1": ["distance"],
            "2": ["speed"],
            "3": ["form"],
        },
    }


def test_keep_shared_only_creates_experts_for_non_shared_features():
    result = one_feature_per_expert(
        {
            "shared_features": ["field_size"],
            "experts": {"0": ["speed"], "1": ["form"]},
        },
        keep_shared=True,
    )

    assert result == {
        "shared_features": ["field_size"],
        "experts": {"0": ["speed"], "1": ["form"]},
    }


def test_rejects_invalid_expert_feature_lists():
    with pytest.raises(ValueError, match="experts.0 must be a list"):
        one_feature_per_expert({
            "shared_features": [],
            "experts": {"0": "speed"},
        })


def test_excluded_features_are_removed_and_experts_are_reindexed():
    result = one_feature_per_expert(
        {
            "shared_features": ["field_size"],
            "experts": {
                "0": ["unavailable"],
                "7": ["speed"],
                "300": ["form"],
            },
        },
        excluded_features={"unavailable"},
    )

    assert result == {
        "shared_features": [],
        "experts": {
            "0": ["field_size"],
            "1": ["speed"],
            "2": ["form"],
        },
    }


def test_allowed_features_remove_names_absent_from_training_manifest():
    result = one_feature_per_expert(
        {
            "shared_features": ["field_size"],
            "experts": {
                "0": ["speed"],
                "1": ["not_in_manifest"],
            },
        },
        allowed_features={"field_size", "speed"},
    )

    assert result == {
        "shared_features": [],
        "experts": {"0": ["field_size"], "1": ["speed"]},
    }
