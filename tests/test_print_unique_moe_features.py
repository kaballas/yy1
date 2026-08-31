import pytest

from print_unique_moe_features import unique_features


def test_unique_features_deduplicates_all_feature_groups_in_first_seen_order():
    payload = {
        "shared_features": ["shared", "duplicate"],
        "router_features": ["router", "duplicate"],
        "experts": {
            "0": ["expert_zero", "shared"],
            "1": ["expert_one", "expert_zero"],
        },
    }

    assert unique_features(payload) == [
        "shared", "duplicate", "router", "expert_zero", "expert_one",
    ]


def test_unique_features_rejects_non_list_feature_group():
    with pytest.raises(ValueError, match="shared_features.*must be a list"):
        unique_features({"shared_features": "field_size", "experts": {}})
