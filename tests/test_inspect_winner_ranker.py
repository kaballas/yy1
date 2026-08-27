import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from inspect_winner_ranker import (
    attach_oof_scores,
    build_reprediction_command,
    build_training_command,
    combined_global_gain_table,
    combined_winner_backing_table,
    comparison_indices,
    contribution_delta_table,
    ensemble_member_runner_diagnostics,
    load_finished_race,
    load_suggested_feature_values,
    model_output_path,
    model_features_from_bundle,
    parse_args,
    print_suggested_feature_values,
    runner_vs_field_contribution_table,
    strict_winner_features,
    update_feature_manifest_model,
    validate_derived_feature_version,
    validate_extra_trainer_arguments,
    winner_backing_feature_table,
)


def test_train_and_repredict_arguments_accept_passthrough_trainer_options():
    args = parse_args([
        "--race-id", "10842431",
        "--train-and-repredict",
        "--training-competition-id", "999",
        "--trainer-args", "--folds", "5", "--ranker-diagnostics",
    ])

    assert args.train_and_repredict
    assert args.training_competition_id == 999
    assert args.trainer_args == ["--folds", "5", "--ranker-diagnostics"]


def test_training_command_uses_final_model_and_inherits_bundle_settings(tmp_path):
    command = build_training_command(
        database=tmp_path / "races.sqlite",
        output_dir=tmp_path / "trained",
        manifest_path=tmp_path / "features.json",
        model_name="winner_backing",
        bundle={
            "models": {"a0": ["one.json", "two.json", "three.json"]},
            "all_finished_crossfit": {
                "objective": "top3",
                "crossfit_folds": 5,
                "tree_count_max_estimators": 700,
                "tree_count_early_stopping_rounds": 60,
                "tree_count_maximum_inner_validation_races": 1000,
                "minimum_feature_coverage": 0.01,
            },
        },
        competition_id=999,
        training_weekday="Wednesday",
        extra_arguments=["--ranker-diagnostics"],
        python_executable="python-test",
    )

    assert command[0] == "python-test"
    assert command[command.index("--models") + 1] == "winner_backing"
    assert command[command.index("--competition-id") + 1] == "999"
    assert command[command.index("--training-weekday") + 1] == "Wednesday"
    assert command[command.index("--objective") + 1] == "top3"
    assert command[command.index("--folds") + 1] == "5"
    assert command[command.index("--ensemble-size") + 1] == "3"
    assert command[command.index("--minimum-feature-coverage") + 1] == "0.01"
    assert command[-1] == "--ranker-diagnostics"


def test_training_command_rejects_passthrough_model_or_path_overrides():
    with pytest.raises(ValueError, match="cannot override.*--models"):
        validate_extra_trainer_arguments(["--models", "other"])


def test_reprediction_command_uses_new_bundle_and_disables_manifest_update(
    tmp_path,
):
    command = build_reprediction_command(
        database=tmp_path / "races.sqlite",
        output_dir=tmp_path / "trained",
        manifest_path=tmp_path / "features.json",
        model_name="winner_backing",
        race_id=10842431,
        top_features=10,
        python_executable="python-test",
    )

    assert command[command.index("--model") + 1] == "winner_backing"
    assert command[command.index("--race-id") + 1] == "10842431"
    assert command[command.index("--bundle") + 1].endswith(
        "/trained/winner_ranker_bundle.json"
    )
    assert command[command.index("--oof-predictions") + 1].endswith(
        "/trained/all_finished_oof_predictions.csv"
    )
    assert "--no-update-feature-manifest" in command


def test_model_features_use_exact_configured_schema():
    bundle = {
        "form_features": ["legacy"],
        "model_features": {"x1": ["speed", "weight"]},
    }

    assert model_features_from_bundle(bundle, "x1") == ["speed", "weight"]


def test_all_model_outputs_get_distinct_model_suffixes(tmp_path):
    path = tmp_path / "shap.csv"

    assert model_output_path(path, "x1", False) == path
    assert model_output_path(path, "x1", True) == tmp_path / "shap_x1.csv"
    assert model_output_path(None, "x1", True) is None


def test_feature_version_mismatch_warns_but_does_not_fail():
    frame = pd.DataFrame({"derived_racing_features_version": ["2026-08-22-1"]})

    with pytest.warns(UserWarning, match="Race feature version '2026-08-22-1' was not used by this bundle"):
        validate_derived_feature_version(
            frame,
            ["2026-08-20-v6"],
            target="bundle",
            refresh_hint="Rerun the training pipeline.",
        )


def test_combined_gain_averages_across_models_with_missing_features_as_zero():
    tables = {
        "x1": pd.DataFrame({
            "feature": ["speed", "weight"],
            "mean_gain": [10.0, 4.0],
            "members_using_feature": [3, 2],
        }),
        "x2": pd.DataFrame({
            "feature": ["speed", "box"],
            "mean_gain": [6.0, 2.0],
            "members_using_feature": [2, 1],
        }),
    }

    result = combined_global_gain_table(tables).set_index("feature")

    assert result.loc["speed", "mean_gain_across_models"] == 8.0
    assert result.loc["weight", "mean_gain_across_models"] == 2.0
    assert result.loc["speed", "models_using_feature"] == 2
    assert result.loc["speed", "members_using_feature"] == 5


def test_comparison_uses_winner_for_wrong_pick_and_runner_up_for_correct_pick():
    target = np.asarray([0, 1, 0])

    assert comparison_indices(target, np.asarray([0.9, 0.8, 0.1])) == (
        0, 1, "selected_vs_actual_winner"
    )
    assert comparison_indices(target, np.asarray([0.8, 0.9, 0.1])) == (
        1, 0, "correct_winner_vs_model_runner_up"
    )


def test_contribution_delta_reports_direction_and_member_agreement():
    matrix = pd.DataFrame({"speed": [8.0, 5.0], "weight": [55.0, 57.0]})
    # Two members, two runners, two features plus bias.
    contributions = np.asarray([
        [[2.0, -0.5, 0.1], [0.5, 0.5, 0.1]],
        [[1.0, -0.2, 0.2], [0.0, 0.2, 0.2]],
    ])

    result = contribution_delta_table(matrix, contributions, 0, 1)

    speed = result.set_index("feature").loc["speed"]
    assert speed["shap_delta_mean"] == pytest.approx(1.25)
    assert speed["member_sign_agreement"] == 1.0
    assert result.iloc[0]["feature"] == "speed"


def test_winner_backing_table_marks_winner_favoring_features():
    matrix = pd.DataFrame({
        "speed": [8.0, 9.0, 5.0],
        "weight": [55.0, 57.0, 56.0],
    })
    # Members agree: speed helps runner 1 (the winner), weight helps runner 0.
    contributions = np.asarray([
        [[-1.0, 0.5, 0.1], [2.0, -0.5, 0.1], [-1.0, 0.0, 0.1]],
        [[-0.5, 0.4, 0.2], [1.5, -0.4, 0.2], [-0.5, 0.0, 0.2]],
    ])

    result = winner_backing_feature_table(matrix, contributions, 1, 0)

    assert result["feature"].tolist() == ["speed", "weight"]
    row = result.set_index("feature").loc["speed"]
    assert row["winner_shap_minus_other"] == pytest.approx(2.5)
    assert row["winner_field_rank"] == 1
    assert row["winner_top_tie_count"] == 1
    assert row["field_unique_values"] == 3
    assert bool(row["solo_pick_correct"])
    assert bool(row["backs_winner"])
    assert not bool(result.set_index("feature").loc["weight", "backs_winner"])


def test_winner_backing_tied_first_is_not_a_correct_solo_pick():
    matrix = pd.DataFrame({"signal": [2.0, 2.0, 1.0]})
    contributions = np.asarray([
        [[2.0, 0.0], [1.5, 0.0], [0.0, 0.0]],
    ])

    result = winner_backing_feature_table(matrix, contributions, 0, 2)

    row = result.iloc[0]
    assert row["winner_field_rank"] == 1
    assert row["winner_top_tie_count"] == 2
    assert not bool(row["solo_pick_correct"])


def test_combined_winner_backing_aggregates_across_models():
    tables = {
        "x1": pd.DataFrame({
            "feature": ["speed", "weight"],
            "winner_value": [9.0, 55.0],
            "other_value": [8.0, 57.0],
            "winner_shap_minus_other": [2.0, 0.5],
            "winner_field_rank": [1.0, 3.0],
            "field_unique_values": [3, 3],
            "winner_top_tie_count": [1, 0],
            "solo_pick_correct": [True, False],
        }),
        "x2": pd.DataFrame({
            "feature": ["speed"],
            "winner_value": [9.0],
            "other_value": [8.0],
            "winner_shap_minus_other": [1.0],
            "winner_field_rank": [np.nan],
            "field_unique_values": [3],
            "winner_top_tie_count": [0],
            "solo_pick_correct": [False],
        }),
    }

    result = combined_winner_backing_table(
        tables, {"x1": ["speed", "weight"], "x2": ["speed"]}
    ).set_index("feature")

    assert set(result.index) == {"speed", "weight"}
    speed = result.loc["speed"]
    assert speed["mean_winner_shap_minus_other"] == pytest.approx(1.5)
    assert speed["models_using_feature"] == 2
    assert speed["models_backing_winner"] == 2
    assert speed["unique_solo_pick_models"] == 1
    assert not bool(speed["strictly_eligible"])
    assert speed["mean_winner_field_rank"] == pytest.approx(1.0)
    weight = result.loc["weight"]
    # Absent from x2, so it counts as zero evidence there.
    assert weight["mean_winner_shap_minus_other"] == pytest.approx(0.25)
    assert weight["models_backing_winner"] == 1
    assert weight["strict_rejection_reason"] == "insufficient_model_support"


def test_strict_features_require_unique_success_in_every_model_using_feature():
    table = pd.DataFrame({
        "feature": ["reliable", "tied", "constant"],
        "strictly_eligible": [True, False, False],
    })

    assert strict_winner_features(table, 10) == ["reliable"]


def test_update_feature_manifest_model_creates_and_replaces_group(tmp_path):
    manifest = tmp_path / "winner_ranker_features.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "models": {"a0": {"features": ["distance_m"]}},
    }))

    update_feature_manifest_model(
        manifest, "winner_backing", ["speed", "weight", "speed"],
        race_id=7, top_features=3,
    )
    payload = json.loads(manifest.read_text())
    assert payload["models"]["a0"] == {"features": ["distance_m"]}
    entry = payload["models"]["winner_backing"]
    assert entry["features"] == ["speed", "weight"]
    assert entry["selection"]["race_id"] == 7
    assert entry["selection"]["minimum_model_groups"] == 2
    assert entry["selection"]["requires_unique_solo_winner_in_every_using_model"]

    update_feature_manifest_model(
        manifest, "winner_backing", ["box"], race_id=8, top_features=1,
    )
    payload = json.loads(manifest.read_text())
    assert payload["models"]["winner_backing"]["features"] == ["box"]


def test_update_feature_manifest_model_rejects_empty_selection(tmp_path):
    manifest = tmp_path / "winner_ranker_features.json"
    manifest.write_text(json.dumps({"models": {}}))

    with pytest.raises(ValueError, match="no selected features"):
        update_feature_manifest_model(
            manifest, "winner_backing", [], race_id=7, top_features=20,
        )


def test_winner_backing_table_omits_rank_when_direction_is_unknown():
    matrix = pd.DataFrame({"constant": [1.0, 1.0], "speed": [8.0, 5.0]})
    contributions = np.asarray([
        [[0.5, 1.0, 0.1], [0.3, 0.0, 0.1]],
        [[0.4, 1.0, 0.2], [0.2, 0.0, 0.2]],
    ])

    result = winner_backing_feature_table(
        matrix, contributions, 0, 1
    ).set_index("feature")

    assert np.isnan(result.loc["constant", "winner_field_rank"])
    assert not bool(result.loc["constant", "solo_pick_correct"])


def test_runner_vs_field_diagnosis_separates_negative_and_positive_reasons():
    matrix = pd.DataFrame({"speed": [8.0, 5.0], "weight": [55.0, 57.0]})
    contributions = np.asarray([
        [[2.0, -0.5, 0.1], [0.0, 0.5, 0.1]],
        [[1.0, -0.3, 0.2], [0.0, 0.3, 0.2]],
    ])

    result = runner_vs_field_contribution_table(matrix, contributions, 1).set_index(
        "feature"
    )

    assert result.loc["speed", "winner_vs_field_shap"] == pytest.approx(-0.75)
    assert result.loc["weight", "winner_vs_field_shap"] == pytest.approx(0.4)


def test_member_diagnosis_exposes_ensemble_rank_disagreement():
    # Two members, three runners, one feature plus bias.
    contributions = np.asarray([
        [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [3.0, 0.0], [2.0, 0.0]],
    ])

    result = ensemble_member_runner_diagnostics(contributions, 0)

    assert result["winner_rank"].tolist() == [1, 3]
    assert result["winner_rank_score"].tolist() == pytest.approx([1.0, 0.0])


def test_oof_scores_attach_by_runner_identity(tmp_path):
    path = tmp_path / "oof.csv"
    pd.DataFrame({
        "race_id": [7, 7],
        "runner_number": [2, 1],
        "x1_score": [0.2, 0.8],
    }).to_csv(path, index=False)
    output = pd.DataFrame({"runner_number": [1, 2], "runner_name": ["A", "B"]})

    result = attach_oof_scores(output, path, 7, "x1")

    assert result["x1_oof_score"].tolist() == [0.8, 0.2]
    assert result["x1_oof_rank"].tolist() == [1, 2]


def test_finished_race_loader_keeps_only_active_runners(tmp_path, capsys):
    path = tmp_path / "races.sqlite"
    metadata_types = {
        "race_id": "INTEGER", "start_time_iso": "TEXT",
        "competition_id": "INTEGER", "competition_name": "TEXT",
        "race_number": "INTEGER", "race_name": "TEXT",
        "runner_number": "INTEGER", "runner_name": "TEXT",
        "runner_mask": "INTEGER", "status": "TEXT",
        "source_betting_status": "TEXT", "active_field_size": "INTEGER",
        "fluc2": "REAL", "derived_racing_features_version": "TEXT",
        "is_winner": "INTEGER", "speed": "REAL",
    }
    with sqlite3.connect(path) as connection:
        columns = ", ".join(
            f'"{name}" {kind}' for name, kind in metadata_types.items()
        )
        connection.execute(f"CREATE TABLE race_runners ({columns})")
        keys = list(metadata_types)
        placeholders = ", ".join("?" for _ in keys)
        rows = []
        for runner, active, winner in ((1, 1, 1), (2, 1, 0), (3, 0, 0)):
            values = {
                "race_id": 7, "start_time_iso": "2026-01-01Z",
                "competition_id": 580, "competition_name": "Track",
                "race_number": 1, "race_name": "Race", "runner_number": runner,
                "runner_name": chr(64 + runner), "runner_mask": active,
                "status": "finished", "source_betting_status": "RESULTED",
                "active_field_size": 2, "fluc2": float(runner + 1),
                "derived_racing_features_version": "v3",
                "is_winner": winner, "speed": float(10 - runner),
            }
            rows.append(tuple(values[key] for key in keys))
        connection.executemany(
            f"INSERT INTO race_runners VALUES ({placeholders})", rows
        )

    result = load_finished_race(path, 7, ["speed"])

    assert result["runner_number"].tolist() == [1, 2]
    assert result["is_winner"].tolist() == [1, 0]

    values = load_suggested_feature_values(
        path, 7, ["speed", "current_market_log_price"]
    )

    assert values.columns.tolist() == [
        "runner_number", "runner_name", "is_winner", "speed",
        "current_market_log_price",
    ]
    assert values["speed"].tolist() == [9.0, 8.0]
    assert values["current_market_log_price"].tolist() == pytest.approx(
        np.log([2.0, 3.0])
    )

    print_suggested_feature_values(path, 7, ["speed"])

    output = capsys.readouterr().out
    assert "SUGGESTED FEATURE VALUES FROM DATABASE" in output
    assert 'database_features=["speed"]' in output
    assert "runner_number runner_name  is_winner" in output
