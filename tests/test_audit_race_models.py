import json

from audit_race_models import (
    feature_count_distribution,
    load_model_feature_audit,
    xgboost_json_features,
)


def write_model(path, features):
    path.write_text(json.dumps({
        "learner": {
            "feature_names": features,
            "learner_model_param": {"num_feature": str(len(features))},
        },
    }))


def test_audit_reports_feature_counts_reuse_and_artifact_mismatch(tmp_path):
    first = tmp_path / "race_1.json"
    second = tmp_path / "race_2.json"
    write_model(first, ["speed", "form"])
    write_model(second, ["speed"])
    manifest = tmp_path / "per_race_models_manifest.json"
    manifest.write_text(json.dumps({
        "models": [
            {
                "name": "race_1",
                "trained_on_race_id": 1,
                "model": str(first),
                "details": {
                    "input_features": ["speed", "form"],
                    "split_feature_gain": {"speed": 1.0},
                },
            },
            {
                "name": "race_2",
                "trained_on_race_id": 2,
                "model": str(second),
                "details": {"input_features": ["speed", "form"]},
            },
        ],
    }))

    audit = load_model_feature_audit(manifest, tmp_path).set_index("model")

    assert audit.loc["race_1", "feature_count"] == 2
    assert audit.loc["race_1", "nonzero_gain_features"] == 1
    assert audit.loc["race_1", "feature_set_reuse_count"] == 2
    assert audit.loc["race_1", "artifact_features_match_manifest"]
    assert not audit.loc["race_2", "artifact_features_match_manifest"]
    distribution = feature_count_distribution(audit.reset_index())
    assert distribution.to_dict("records") == [{
        "feature_count": 2, "models": 2, "unique_feature_sets": 1,
    }]


def test_xgboost_json_feature_reader_reports_missing_file(tmp_path):
    features, count, error = xgboost_json_features(tmp_path / "missing.json")
    assert features == []
    assert count is None
    assert error == "missing_model_file"
