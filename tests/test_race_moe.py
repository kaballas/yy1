import json

import numpy as np
import pandas as pd
import pytest
import torch

from evaluate_moe_winner_rankers import paired_comparison
from src.model.race_moe import (
    RaceMixtureOfExperts, RaceWinnerModelConfig, build_race_winner_model,
    race_softmax_nll, router_balance_loss,
)
from src.model.race_moe_feature_map import (
    FeatureMappedRaceWinnerConfig,
    RaceMixtureOfExpertsFeatureMap,
    load_feature_expert_map,
    load_router_feature_indices,
)
from src.race_moe_data import chronological_race_ids, market_blind_features
from src.race_moe_evaluation import collapse_warnings, routing_diagnostics
from src.race_moe_snapshot import (
    create_split_snapshot, load_split_snapshot, resolve_snapshot_manifest,
    snapshot_manifest_reference,
)
from train_moe_winner_ranker_feature_map import (
    _competition_population,
    _selection,
    available_trainable_components,
    fine_tune_command,
    freeze_base_model_for_judge,
    initial_training_command,
    main as feature_map_main,
    next_fine_tuned_output,
    parse_args,
    previous_fine_tune_checkpoint,
    set_trainable_components,
    training_command,
    training_arguments,
    validate_args,
)


def test_moe_forward_contract_and_sparse_top_k():
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=5, num_experts=4, top_k=2, dropout=0.0,
    )).eval()
    x = torch.randn(2, 6, 5)
    valid = torch.tensor([
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
    ])
    output = model(x, valid, return_diagnostics=True)
    assert set(output) == {
        "logits", "expert_logits", "router_logits", "router_weights",
        "dense_router_weights", "selected_experts", "representation", "race_context",
    }
    assert output["logits"].shape == (2, 6)
    assert torch.all(output["selected_experts"][valid].sum(dim=-1) == 2)
    assert torch.allclose(
        output["router_weights"][valid].sum(dim=-1), torch.ones(10), atol=1e-6
    )
    assert torch.all(output["router_weights"][~valid] == 0)


def test_manual_feature_map_can_intentionally_omit_global_features():
    mapping = load_feature_expert_map(
        {"experts": {"0": ["speed"], "1": ["form"]}},
        ["speed", "form", "unused"],
        2,
    )
    assert mapping == ((0,), (1,))


def test_manual_feature_map_still_rejects_an_empty_expert():
    with pytest.raises(ValueError, match="expert 1 has no explicit features"):
        load_feature_expert_map(
            {"experts": {"0": ["speed"], "1": ["not_available"]}},
            ["speed", "form"],
            2,
        )


def test_router_feature_allowlist_is_explicit_and_rejects_unknown_features(tmp_path):
    path = tmp_path / "map.json"
    path.write_text('{"router_features": ["form", "speed", "form"]}')
    assert load_router_feature_indices(path, ["speed", "form"]) == (1, 0)

    path.write_text('{"router_features": ["missing"]}')
    with pytest.raises(ValueError, match="unavailable features"):
        load_router_feature_indices(path, ["speed", "form"])


def test_feature_mapped_router_uses_only_allowlisted_features():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
        router_feature_indices=(1,),
    ))

    assert model.router is not None
    assert model.router.network[1].in_features == 3


def test_feature_mapped_judge_only_uses_moe_outputs_and_starts_as_identity():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
        router_feature_indices=(1,),
        judge_hidden_dims=(8,),
    )).eval()
    x = torch.randn(1, 4, 3)
    valid = torch.tensor([[True, True, True, False]])

    output = model(x, valid, return_diagnostics=True)

    assert model.judge is not None
    assert model.judge.hidden[0].in_features == 5
    assert torch.allclose(output["logits"], output["base_logits"])
    assert torch.count_nonzero(output["judge_adjustment"]) == 0


def test_freezing_base_model_leaves_only_judge_trainable():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
        judge_hidden_dims=(8,),
    ))

    freeze_base_model_for_judge(model)

    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("judge.")
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("judge.")
    )


def test_trainable_component_selection_supports_individual_experts():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
        judge_hidden_dims=(8,),
    ))

    assert available_trainable_components(model) == (
        "expert_0", "expert_1", "router", "judge",
    )
    assert set_trainable_components(model, ["expert_1", "judge"]) == (
        "expert_1", "judge",
    )
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("experts.1.") or name.startswith("judge.")
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("experts.0.") or name.startswith("router.")
    )
    with pytest.raises(ValueError, match="Unknown"):
        set_trainable_components(model, ["missing"])


def test_trainable_component_selection_accepts_comma_separated_components():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
        judge_hidden_dims=(8,),
    ))

    selected = set_trainable_components(model, ["router,judge,expert_1"])

    assert selected == ("router", "judge", "expert_1")
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if (
            name.startswith("router.")
            or name.startswith("judge.")
            or name.startswith("experts.1.")
        )
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith("experts.0.")
    )


def test_training_arguments_include_defaults_and_are_json_safe():
    args = parse_args([
        "--feature-map-json", "map.json",
        "--moe-expert-hidden-dims", "64,32",
    ])

    values = training_arguments(args)

    assert values["feature_map_json"] == "map.json"
    assert values["moe_expert_hidden_dims"] == [64, 32]
    assert values["learning_rate"] == pytest.approx(3e-4)


def test_trainer_rejects_overwriting_initial_checkpoint(tmp_path):
    checkpoint = tmp_path / "same.pt"
    args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--initial-checkpoint", str(checkpoint),
        "--output", str(checkpoint),
    ])

    with pytest.raises(ValueError, match="must differ"):
        validate_args(args)


def test_fine_tune_command_preserves_architecture_and_uses_safe_output(tmp_path):
    args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--db", str(tmp_path / "races.sqlite"),
        "--features-json", str(tmp_path / "features.json"),
        "--train-competition-id", "7", "8",
        "--validation-competition-id", "7", "8",
        "--validation-races", "400",
        "--test-races", "50",
        "--moe-num-experts", "4",
        "--moe-top-k", "2",
        "--moe-router-balance-weight", "0.05",
        "--moe-gate-temperature", "1.25",
        "--epochs", "2000",
        "--learning-rate", "0.0003",
    ])
    checkpoint = tmp_path / "winner.pt"

    command, output, learning_rate, epochs = fine_tune_command(
        args,
        checkpoint,
        ("expert_0", "expert_1", "expert_2", "expert_3", "router", "judge"),
    )

    assert output == tmp_path / "winner_finetuned.pt"
    assert learning_rate == pytest.approx(0.0001)
    assert epochs == 200
    assert "--initial-checkpoint" in command
    assert str(checkpoint.resolve()) in command
    assert "--output" in command
    assert str(output) in command
    assert "--moe-top-k 2" in command
    assert "--moe-router-balance-weight 0.05" in command
    assert "--train-competition-id 7 8" in command
    assert (
        "--trainable-components "
        "expert_0,expert_1,expert_2,expert_3,router,judge"
    ) in command


def test_fine_tune_output_uses_versions_instead_of_repeated_suffixes(tmp_path):
    assert next_fine_tuned_output(tmp_path / "winner.pt") == (
        tmp_path / "winner_finetuned.pt"
    )
    assert next_fine_tuned_output(tmp_path / "winner_finetuned.pt") == (
        tmp_path / "winner_finetuned_2.pt"
    )
    assert next_fine_tuned_output(
        tmp_path / "winner_finetuned_finetuned.pt"
    ) == (tmp_path / "winner_finetuned_3.pt")
    assert next_fine_tuned_output(tmp_path / "winner_finetuned_3.pt") == (
        tmp_path / "winner_finetuned_4.pt"
    )


def test_initial_training_command_traces_to_root_without_initializer(tmp_path):
    root_checkpoint = tmp_path / "root.pt"
    child_checkpoint = tmp_path / "child_finetuned.pt"
    current_output = tmp_path / "current_finetuned.pt"
    root_args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--db", str(tmp_path / "races.sqlite"),
        "--features-json", str(tmp_path / "features.json"),
        "--learning-rate", "0.0003",
        "--output", str(root_checkpoint),
    ])
    root_checkpoint.with_suffix(".report.json").write_text(json.dumps({
        "initial_checkpoint": None,
        "training_config": training_arguments(root_args),
        "trainable_components": [
            "expert_0", "expert_1", "expert_2", "expert_3", "router", "judge",
        ],
    }))
    child_checkpoint.with_suffix(".report.json").write_text(json.dumps({
        "initial_checkpoint": str(root_checkpoint),
        "training_config": training_arguments(root_args),
        "trainable_components": ["router", "judge"],
    }))
    current_args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--initial-checkpoint", str(child_checkpoint),
        "--learning-rate", "0.00003",
        "--output", str(current_output),
    ])

    command, checkpoint, depth = initial_training_command(
        current_args, current_output, ("router", "judge"),
    )

    assert checkpoint == root_checkpoint
    assert depth == 1
    assert "--learning-rate 0.0003" in command
    assert "--initial-checkpoint" not in command
    assert f"--output {root_checkpoint}" in command


def test_initial_training_command_recovers_legacy_self_referencing_report(tmp_path):
    root_checkpoint = tmp_path / "winner.pt"
    child_checkpoint = tmp_path / "winner_finetuned.pt"
    root_args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--output", str(root_checkpoint),
    ])
    root_checkpoint.with_suffix(".report.json").write_text(json.dumps({
        "initial_checkpoint": None,
        "training_config": training_arguments(root_args),
        "trainable_components": [
            "expert_0", "expert_1", "expert_2", "expert_3", "router", "judge",
        ],
    }))
    child_checkpoint.with_suffix(".report.json").write_text(json.dumps({
        "initial_checkpoint": str(child_checkpoint),
        "training_config": training_arguments(root_args),
        "trainable_components": ["router", "judge"],
    }))
    current_args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--initial-checkpoint", str(child_checkpoint),
    ])

    command, checkpoint, depth = initial_training_command(
        current_args, tmp_path / "current.pt", ("router", "judge"),
    )

    assert previous_fine_tune_checkpoint(child_checkpoint) == root_checkpoint
    assert checkpoint == root_checkpoint
    assert depth == 1
    assert "--initial-checkpoint" not in command


def test_training_command_prints_complete_reproducible_invocation(tmp_path):
    args = parse_args([
        "--feature-map-json", str(tmp_path / "map.json"),
        "--db", str(tmp_path / "races.sqlite"),
        "--features-json", str(tmp_path / "features.json"),
        "--competition-id", "999",
        "--validation-competition-id", "999",
        "--moe-top-k", "all",
        "--no-include-market-features",
        "--initial-checkpoint", str(tmp_path / "initial.pt"),
        "--trainable-components", "router,judge",
        "--output", str(tmp_path / "trained.pt"),
    ])

    command = training_command(
        args, args.output, ("router", "judge"),
    )

    assert "--train-competition-id 999" in command
    assert "--validation-competition-id 999" in command
    assert "--moe-top-k all" in command
    assert "--no-include-market-features" in command
    assert f"--initial-checkpoint {(tmp_path / 'initial.pt').resolve()}" in command
    assert "--trainable-components router,judge" in command
    assert f"--output {(tmp_path / 'trained.pt').resolve()}" in command


def test_feature_mapped_fixed_uniform_routing_bypasses_router():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=None,
        routing_mode="fixed_uniform",
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
    )).eval()
    x = torch.randn(1, 4, 3)
    valid = torch.tensor([[True, True, True, False]])

    output = model(x, valid, return_diagnostics=True)

    assert model.router is None
    assert torch.allclose(
        output["router_weights"][valid],
        torch.full((3, 2), 0.5),
    )
    assert torch.allclose(output["logits"], output["expert_logits"].mean(dim=-1))


def test_feature_mapped_moe_has_no_disconnected_encoder():
    model = RaceMixtureOfExpertsFeatureMap(FeatureMappedRaceWinnerConfig(
        feature_count=3,
        num_experts=2,
        top_k=1,
        dropout=0.0,
        feature_map=((0, 1), (1, 2)),
    )).eval()
    x = torch.randn(1, 4, 3)
    valid = torch.tensor([[True, True, True, False]])

    output = model(x, valid, return_diagnostics=True)

    assert not hasattr(model, "encoder")
    assert "representation" not in output
    assert model.contributing_parameter_count() < model.trainable_parameter_count()


def test_moe_is_permutation_equivariant_with_race_context():
    torch.manual_seed(3)
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=3, num_experts=4, top_k=2, dropout=0.0,
        expert_context_conditioning=True,
    )).eval()
    x = torch.randn(1, 5, 3); valid = torch.ones((1, 5), dtype=torch.bool)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = model(x, valid, return_diagnostics=True)
    shuffled = model(x[:, permutation], valid, return_diagnostics=True)
    assert torch.allclose(shuffled["logits"], original["logits"][:, permutation], atol=1e-6)
    assert torch.allclose(
        shuffled["router_weights"], original["router_weights"][:, permutation], atol=1e-6
    )


def test_top1_router_receives_ranking_gradient_through_straight_through_gate():
    torch.manual_seed(9)
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=3, num_experts=2, top_k=1, dropout=0.0,
    ))
    x = torch.randn(2, 4, 3)
    valid = torch.ones((2, 4), dtype=torch.bool)
    winners = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    output = model(x, valid, return_diagnostics=True)
    race_softmax_nll(output["logits"], winners, valid).backward()
    assert model.router is not None
    gradients = [p.grad for p in model.router.parameters()]
    assert any(value is not None and torch.count_nonzero(value) for value in gradients)


def test_race_softmax_nll_is_equal_per_race_and_exact():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 9.0]])
    winners = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    valid = torch.ones_like(winners, dtype=torch.bool)
    expected = (
        -torch.log_softmax(logits[0], dim=0)[0]
        -torch.log_softmax(logits[1], dim=0)[1]
    ) / 2
    assert race_softmax_nll(logits, winners, valid) == pytest.approx(float(expected))


def test_router_balance_loss_detects_collapse_without_forcing_baseline():
    valid = torch.ones((1, 8), dtype=torch.bool)
    uniform = torch.full((1, 8, 4), 0.25)
    collapsed = torch.zeros((1, 8, 4)); collapsed[..., 2] = 1.0
    assert router_balance_loss(uniform, valid) == pytest.approx(0.0)
    assert router_balance_loss(collapsed, valid) == pytest.approx(3.0)
    assert router_balance_loss(torch.ones((1, 8, 1)), valid) == pytest.approx(0.0)


def test_config_round_trip_reconstructs_model():
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=7, num_experts=2, top_k=None, expert_hidden_dims=(32, 16),
    ))
    rebuilt = build_race_winner_model(model.config())
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    assert rebuilt.config() == model.config()


def test_fixed_uniform_routing_averages_all_experts_without_router():
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=4, model_type="moe", num_experts=4, top_k=None,
        routing_mode="fixed_uniform", dropout=0.0,
    )).eval()
    x = torch.randn(1, 5, 4); valid = torch.ones((1, 5), dtype=torch.bool)
    output = model(x, valid, return_diagnostics=True)
    assert model.router is None
    assert torch.allclose(output["router_weights"], torch.full((1, 5, 4), 0.25))
    assert torch.allclose(output["logits"], output["expert_logits"].mean(dim=-1))
    assert model.executed_parameter_count() == model.trainable_parameter_count()
    assert model.contributing_parameter_count() == model.trainable_parameter_count()


def test_market_blind_filter_rejects_direct_and_indirect_prices():
    retained, excluded = market_blind_features([
        "age", "marketWinPrice", "recent_1_starting_price", "fluc2",
        "market_implied_prob_change_open_to_fluc2", "career_starts",
    ])
    assert retained == ["age", "career_starts"]
    assert set(excluded) == {
        "marketWinPrice", "recent_1_starting_price", "fluc2",
        "market_implied_prob_change_open_to_fluc2",
    }


def test_market_blind_filter_rejects_consensus_overlay_signal_and_identifiers():
    forbidden = [
        "race_consensus_score", "race_consensus_rank", "race_overlay_score",
        "race_overlay_rank", "race_signal_agreement_score",
        "race_signal_agreement_rank", "competition_id",
    ]
    retained, excluded = market_blind_features(["age", *forbidden, "career_starts"])
    assert retained == ["age", "career_starts"]
    assert excluded == forbidden


def test_top2_reports_all_parameters_executed_but_only_two_experts_contributing():
    model = RaceMixtureOfExperts(RaceWinnerModelConfig(
        feature_count=10, model_type="moe", num_experts=4, top_k=2,
    ))
    assert model.executed_parameter_count() == model.trainable_parameter_count()
    assert model.contributing_parameter_count() < model.executed_parameter_count()


def test_snapshot_hash_verification_rejects_changed_file(tmp_path):
    rows = []
    for race_id in range(1, 4):
        for runner in range(1, 5):
            rows.append({
                "race_id": race_id, "runner_number": runner,
                "start_time_iso": f"2026-01-0{race_id}T00:00:00+00:00",
                "competition_id": 99, "is_winner": int(runner == 1),
                "finish_place": runner, "distance_m": 1200,
                "class_name": "BM", "field_size": 4, "active_field_size": 4,
                "track_status": "Good", "career_starts": 3,
                "runner_name": f"R{runner}", "age": float(runner),
            })
    frame = pd.DataFrame(rows)
    frames = {name: frame.loc[frame["race_id"] == race].copy() for name, race in (
        ("training", 1), ("validation", 2), ("test", 3),
    )}
    manifest = create_split_snapshot(
        tmp_path / "snapshot", frames, ["age"], database=tmp_path / "db.sqlite",
        excluded_features=["competition_id"],
    )
    loaded, metadata = load_split_snapshot(manifest)
    assert metadata["identity_columns"] == ["race_id", "runner_number"]
    assert len(loaded["validation"]) == 4
    test_file = manifest.parent / metadata["splits"]["test"]["path"]
    test_file.write_bytes(test_file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="HASH MISMATCH"):
        load_split_snapshot(manifest)


def test_snapshot_reference_is_checkpoint_relative_and_legacy_paths_fall_back(tmp_path):
    checkpoint = tmp_path / "experiment" / "baseline.pt"
    manifest = checkpoint.parent / "snapshot" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    assert snapshot_manifest_reference(manifest, checkpoint) == "snapshot/manifest.json"
    assert resolve_snapshot_manifest(
        {"manifest": "snapshot/manifest.json"}, checkpoint,
    ) == manifest.resolve()
    assert resolve_snapshot_manifest(
        {"manifest": "/old/machine/project/outputs/run/snapshot/manifest.json"},
        checkpoint,
    ) == manifest.resolve()


def test_chronological_split_is_consecutive_and_sealed():
    frame = pd.DataFrame({"race_id": np.repeat(np.arange(1, 11), 4)})
    train, validation, test = chronological_race_ids(frame, 3, 2)
    assert train == [1, 2, 3, 4, 5]
    assert validation == [6, 7, 8]
    assert test == [9, 10]


def test_competition_population_filter_preserves_row_order():
    frame = pd.DataFrame({
        "race_id": [10, 11, 12, 13, 14],
        "competition_id": [7, 8, 7, 9, 8],
    })
    filtered = _competition_population(frame, [8, 7])
    assert filtered["race_id"].tolist() == [10, 11, 12, 14]
    assert _competition_population(frame, None).equals(frame)


def test_feature_map_checkpoint_selection_prioritizes_logloss():
    lower_logloss = {
        "top1_hit_rate": 0.20,
        "mrr": 0.40,
        "race_logloss": 1.20,
    }
    higher_top1 = {
        "top1_hit_rate": 0.30,
        "mrr": 0.50,
        "race_logloss": 1.30,
    }
    assert _selection(lower_logloss) > _selection(higher_top1)


def test_competition_cli_accepts_training_alias_and_validation_ids():
    args = parse_args([
        "--feature-map-json", "map.json",
        "--competition-id", "7", "8",
        "--validation-competition-id", "9", "10",
    ])
    assert args.train_competition_ids == [7, 8]
    assert args.validation_competition_ids == [9, 10]


def test_feature_map_trainer_rejects_mismatched_competition_populations():
    with pytest.raises(ValueError, match="must match"):
        feature_map_main([
            "--feature-map-json", "map.json",
            "--train-competition-id", "7",
            "--validation-competition-id", "8",
        ])


def test_diagnostics_warn_for_router_and_output_collapse():
    rows = 20
    weights = np.zeros((rows, 3)); weights[:, 1] = 1.0
    selected = weights.astype(bool)
    base = np.arange(rows, dtype=float)
    expert_logits = np.column_stack((base, base * 2, -base))
    frame = pd.DataFrame({
        "race_id": np.repeat(np.arange(4), 5), "distance_m": 1200,
        "class_name": "BM", "field_size": 5, "active_field_size": 5,
        "track_status": "Good", "career_starts": 3,
    })
    diagnostics = routing_diagnostics(weights, selected, expert_logits, frame)
    warnings = collapse_warnings(diagnostics, 0.8, 0.98)
    assert diagnostics["dominant_expert_rate"] == 1.0
    assert len(warnings) == 2


def test_paired_comparison_counts_discordance_and_bootstrap_interval():
    baseline = pd.DataFrame({
        "race_id": [1, 2, 3, 4], "winner_rank": [1, 1, 2, 2],
    })
    challenger = pd.DataFrame({
        "race_id": [1, 2, 3, 4], "winner_rank": [1, 2, 1, 2],
    })
    result = paired_comparison(baseline, challenger, samples=500, seed=4)
    assert result["both_correct"] == 1
    assert result["baseline_only_correct"] == 1
    assert result["challenger_only_correct"] == 1
    assert result["both_wrong"] == 1
    assert result["top1_difference"] == 0.0
    assert result["mcnemar_exact_p_value"] == 1.0
