"""TabFM optimiser configuration and behaviour-preserving training loop."""

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import random
import numpy as np
import torch
from src.checkpoint import resolve_resume_model_path
from src.config import DEFAULT_FEATURES
from src.constants import (
    MIN_CHECKPOINT_SELECTION_RACES,
    TRAINING_ROWS_VIEW,
    VALIDATION_ROWS_VIEW,
)
from src.context import load_context_race_ids
from src.database import (
    export_rows_to_csv,
    load_race_number_eligible_ids,
    load_race_numbers,
    load_rows_from_csv,
    load_validation_cohorts,
    print_race_selection_logic,
    validate_feature_columns,
)
from src.dataset import load_feature_manifest
from src.losses import (
    context_dependence_margin_loss,
    grouped_pairwise_loss,
    grouped_race_losses,
)
from src.metrics import (checkpoint_selection_improves, cohort_checkpoint_selection, context_permutation_is_ineffective, fixed_probe_has_material_regression, format_metric_line, pre_update_training_batch_metrics, prediction_change_metrics, probability_metrics, race_top3_metrics, select_fixed_probe_race_ids, stress_guardrail_passes, validation_metrics_by_cohort)
from src.model import TabFM
from src.prediction import (
    market_rank_scores,
    predict_with_chronological_context,
)
from src.preprocessing import fit_preprocessor, transform, zero_feature_columns
from src.progress import load_progress_race, print_progress_race
from src.sampling import (
    build_query_race_schedule,
    eligible_query_race_ids_from_context,
    eligible_query_race_ids,
    sample_independent_race_batch,
    permute_context_labels,
)
from src.utilities import (
    initialize_fourier_frequencies,
    resolve_learning_rate,
    resolve_races_per_step,
    resolve_training_schedule,
)
from tabfm_split.runtime import load_and_validate_runtime_manifest
from src.validation import build_race_indices, exclude_invalid_races


MAX_SAFE_FINE_TUNE_LEARNING_RATE = 3e-5
CONTEXT_STOP_WARNING_MIN_STEP = 100
CONTEXT_STOP_WARNING_PROBES = 5
FIXED_PROBE_REGRESSION_WARNING_MIN_STEP = 50
FIXED_PROBE_STOP_WARNING_PROBES = 2


def timestamped_best_checkpoint_path(
    output: Path, epoch: int, saved_at: datetime | None = None
) -> tuple[Path, str]:
    """Return a unique, sortable path for an improved epoch checkpoint."""
    saved_at = saved_at or datetime.now().astimezone()
    if saved_at.tzinfo is None:
        raise ValueError("Checkpoint timestamp must be timezone-aware")
    timestamp = saved_at.strftime("%Y%m%dT%H%M%S%f%z")
    suffix = output.suffix or ".pt"
    stem = output.stem if output.suffix else output.name
    path = output.with_name(
        f"{stem}.best-epoch-{epoch:03d}.{timestamp}{suffix}"
    )
    return path, saved_at.isoformat()


def save_best_epoch_checkpoint(
    *,
    output: Path,
    epoch: int,
    state_dict: dict[str, torch.Tensor],
    model_kwargs: dict[str, object],
    feature_columns: list[str],
    median: np.ndarray,
    scale: np.ndarray,
    context_races_per_step: int,
    zeroed_features: list[str],
    metrics: dict[str, object],
    metrics_by_cohort: dict[str, object],
) -> Path:
    """Atomically retain a standalone checkpoint whenever validation improves."""
    path, saved_at = timestamped_best_checkpoint_path(output, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": state_dict,
        "model_kwargs": model_kwargs,
        "feature_columns": feature_columns,
        "median": median,
        "scale": scale,
        "context_races_per_step": context_races_per_step,
        "validation_context_races_per_prediction": context_races_per_step,
        "zeroed_features": zeroed_features,
        "label": "top3_mask",
        "best_epoch": epoch,
        "best_metrics": metrics,
        "best_metrics_by_validation_cohort": metrics_by_cohort,
        "checkpoint_saved_at": saved_at,
        "checkpoint_kind": "best_epoch_snapshot",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    return path


def configure_trainable_parameters(
    model: TabFM, training_scope: str = "full_model", **legacy_kwargs: object
) -> tuple[list[torch.nn.Parameter], int, int]:
    """Configure optimizer parameters and return trainable and total counts."""
    if "attention_head_only" in legacy_kwargs:
        training_scope = (
            "attention_head_only" if legacy_kwargs["attention_head_only"] else "full_model"
        )
    valid_scopes = {
        "full_model",
        "attention_head_only",
        "decoder_and_race_head",
        "icl_and_race_head",
    }
    if training_scope not in valid_scopes:
        raise ValueError(f"Unknown training scope: {training_scope}")

    for parameter in model.parameters():
        parameter.requires_grad_(training_scope == "full_model")

    if training_scope != "full_model":
        if model.race_set_head is None:
            raise ValueError(
                f"--fine-tune-scope={training_scope} requires "
                "race_context_mode=self_attention"
            )
        for parameter in model.race_set_head.parameters():
            parameter.requires_grad_(True)
        if getattr(model, "context_prototype_head", None) is not None:
            for parameter in model.context_prototype_head.parameters():
                parameter.requires_grad_(True)
        if training_scope == "decoder_and_race_head":
            for parameter in model.icl_predictor.decoder.parameters():
                parameter.requires_grad_(True)
        elif training_scope == "icl_and_race_head":
            for parameter in model.icl_predictor.parameters():
                parameter.requires_grad_(True)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if not parameters:
        raise RuntimeError("No trainable model parameters are available")

    if training_scope == "decoder_and_race_head":
        names = trainable_parameter_names(model)
        unexpected = [
            name for name in names
            if not (
                name.startswith("icl_predictor.decoder.")
                or name.startswith("race_set_head.")
                or name.startswith("context_prototype_head.")
            )
        ]
        if unexpected:
            raise RuntimeError(
                "decoder_and_race_head unexpectedly enabled parameters: "
                + ", ".join(unexpected)
            )
        if not any(name.startswith("icl_predictor.decoder.") for name in names):
            raise RuntimeError("ICL decoder has no trainable parameters")
        if not any(name.startswith("race_set_head.") for name in names):
            raise RuntimeError("Race head has no trainable parameters")
    return parameters, trainable_count, total_count


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    """Return names of parameters enabled for the current optimizer."""
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def early_stopping_is_enabled(representative_races: int) -> bool:
    """Enable patience-based stopping only for a sufficiently sized cohort."""
    return representative_races >= MIN_CHECKPOINT_SELECTION_RACES


def resolve_training_scope(
    requested_scope: str | None,
    attention_head_only: bool,
    fine_tuning: bool,
    source_race_mode: str,
) -> str:
    """Choose a conservative default while preserving explicit scope requests."""
    if requested_scope is not None:
        return requested_scope
    if attention_head_only:
        return "attention_head_only"
    if fine_tuning and source_race_mode == "self_attention":
        return "icl_and_race_head"
    return "full_model"


def validate_fine_tune_learning_rate(
    learning_rate: float,
    fine_tuning: bool,
    allow_high_rate: bool,
) -> None:
    """Reject destructive checkpoint learning rates unless explicitly unlocked."""
    if (
        fine_tuning
        and learning_rate > MAX_SAFE_FINE_TUNE_LEARNING_RATE
        and not allow_high_rate
    ):
        raise ValueError(
            f"Fine-tuning learning rate {learning_rate:g} exceeds the safe ceiling "
            f"{MAX_SAFE_FINE_TUNE_LEARNING_RATE:g}. Use 3e-5 or lower, or pass "
            "--allow-high-fine-tune-learning-rate for a deliberate experiment."
        )


def run_training(args: argparse.Namespace) -> int:
    """Execute the original training, evaluation, and checkpoint workflow."""
    if args.epochs < 1 or args.steps_per_epoch < 1:
        raise SystemExit("--epochs and --steps-per-epoch must be positive")
    if (args.context_races_per_step is not None and args.context_races_per_step < 1) or args.query_races_per_step < 1:
        raise SystemExit(
            "--context-races-per-step and --query-races-per-step must be positive"
        )
    if args.probe_every_steps < 1:
        raise SystemExit("--probe-every-steps must be positive")
    if args.probe_races < 1:
        raise SystemExit("--probe-races must be positive")
    if args.step_loss_window < 1:
        raise SystemExit("--step-loss-window must be positive")
    if args.train_cutoff_iso is not None:
        raise SystemExit(
            "--train-cutoff-iso cannot be used when validation is selected by "
            "race_runners.is_validation"
        )
    if args.max_valid_races is not None and args.max_valid_races < 1:
        raise SystemExit("--max-valid-races must be positive")
    if args.max_grad_norm <= 0:
        raise SystemExit("--max-grad-norm must be positive")
    if args.early_stopping_patience < 1:
        raise SystemExit("--early-stopping-patience must be positive")
    if args.min_race_number is not None and args.min_race_number < 1:
        raise SystemExit("--min-race-number must be positive")
    if (
        args.classification_loss_weight < 0
        or args.auxiliary_row_loss_weight < 0
        or args.pairwise_loss_weight < 0
        or args.attention_delta_pairwise_loss_weight < 0
        or (
            args.context_prototype_loss_weight is not None
            and args.context_prototype_loss_weight < 0
        )
        or args.cardinality_loss_weight < 0
        or args.context_dependence_loss_weight < 0
    ):
        raise SystemExit("Loss weights must be non-negative")
    if args.context_dependence_margin < 0:
        raise SystemExit("--context-dependence-margin must be non-negative")
    if args.context_prototype_dim is not None and args.context_prototype_dim < 1:
        raise SystemExit("--context-prototype-dim must be positive")
    if (
        args.context_prototype_max_correction is not None
        and args.context_prototype_max_correction <= 0
    ):
        raise SystemExit("--context-prototype-max-correction must be positive")
    if args.fine_tune_attention_head_only and args.fine_tune_scope not in (
        None, "attention_head_only"
    ):
        raise SystemExit(
            "--fine-tune-attention-head-only conflicts with --fine-tune-scope="
            f"{args.fine_tune_scope}"
        )
    if not 0.0 <= args.stress_top3_recall_max_drop <= 1.0:
        raise SystemExit("--stress-top3-recall-max-drop must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    resume_model_path, auto_resumed_output = resolve_resume_model_path(
        args.output, args.resume_model, args.overwrite_existing
    )
    if (
        args.resume_model is not None
        and resume_model_path is not None
        and resume_model_path.resolve() == args.output.resolve()
        and not args.allow_in_place_fine_tune
    ):
        raise ValueError(
            "Explicit in-place fine-tuning would overwrite the source checkpoint. "
            "Choose a different --output, or pass --allow-in-place-fine-tune for "
            "a deliberate in-place run."
        )
    resume_bundle = (
        torch.load(resume_model_path, map_location="cpu", weights_only=False)
        if resume_model_path is not None
        else None
    )
    learning_rate = resolve_learning_rate(
        args.learning_rate, fine_tuning=resume_bundle is not None
    )
    validate_fine_tune_learning_rate(
        learning_rate,
        resume_bundle is not None,
        args.allow_high_fine_tune_learning_rate,
    )

    if resume_bundle is not None and resume_bundle.get("label") != "top3_mask":
        raise ValueError(
            "Resume model was not trained with top3_mask and cannot be safely resumed"
        )
    source_model_kwargs = dict(resume_bundle.get("model_kwargs", {})) if resume_bundle else {}
    source_race_mode = source_model_kwargs.get("race_context_mode", "none")
    source_context_prototype_branch = bool(
        source_model_kwargs.get("context_prototype_branch", False)
    )
    training_scope = resolve_training_scope(
        args.fine_tune_scope,
        args.fine_tune_attention_head_only,
        resume_bundle is not None,
        source_race_mode,
    )
    if training_scope != "full_model" and resume_bundle is None:
        raise ValueError(
            f"--fine-tune-scope={training_scope} requires an existing output or "
            "--resume-model checkpoint; a frozen random backbone is invalid"
        )
    race_context_mode = args.race_context_mode or source_race_mode
    race_context_dim = (
        source_model_kwargs.get("race_context_dim", 32)
        if args.race_context_dim is None else args.race_context_dim
    )
    race_context_layers = (
        source_model_kwargs.get("race_context_layers", 1)
        if args.race_context_layers is None else args.race_context_layers
    )
    race_context_heads = (
        source_model_kwargs.get("race_context_heads", 2)
        if args.race_context_heads is None else args.race_context_heads
    )
    race_context_ff_dim = (
        source_model_kwargs.get("race_context_ff_dim", 64)
        if args.race_context_ff_dim is None else args.race_context_ff_dim
    )
    if race_context_dim < 1 or race_context_ff_dim < 1:
        raise SystemExit("Race context dimensions must be positive")
    race_context_residual = (
        source_model_kwargs.get("race_context_residual", True)
        if args.race_context_residual is None else args.race_context_residual
    )
    if race_context_layers < 1 or race_context_heads < 1:
        raise SystemExit("Race context layers and heads must be positive")
    if race_context_dim % race_context_heads:
        raise SystemExit("--race-context-dim must be divisible by --race-context-heads")
    if source_race_mode == "self_attention" and race_context_mode == "none":
        raise ValueError(
            "A race-aware checkpoint cannot be loaded into a non-race-aware architecture"
        )
    context_prototype_branch = (
        source_context_prototype_branch
        if args.context_prototype_branch is None
        else bool(args.context_prototype_branch)
    )
    if source_context_prototype_branch and not context_prototype_branch:
        raise ValueError(
            "A context-prototype checkpoint cannot be loaded with the branch disabled"
        )
    context_prototype_dim = (
        source_model_kwargs.get("context_prototype_dim", 16)
        if args.context_prototype_dim is None else args.context_prototype_dim
    )
    context_prototype_max_correction = (
        source_model_kwargs.get("context_prototype_max_correction", 0.5)
        if args.context_prototype_max_correction is None
        else args.context_prototype_max_correction
    )
    args.context_prototype_loss_weight = (
        (0.25 if context_prototype_branch else 0.0)
        if args.context_prototype_loss_weight is None
        else args.context_prototype_loss_weight
    )
    if training_scope != "full_model" and race_context_mode != "self_attention":
        raise ValueError(
            f"--fine-tune-scope={training_scope} requires race_context_mode=self_attention"
        )
    if args.attention_delta_pairwise_loss_weight > 0 and race_context_mode != "self_attention":
        raise ValueError(
            "--attention-delta-pairwise-loss-weight requires race_context_mode=self_attention"
        )
    if args.context_prototype_loss_weight > 0 and not context_prototype_branch:
        raise ValueError(
            "--context-prototype-loss-weight requires --context-prototype-branch; "
            "set its weight to 0 when the branch is disabled"
        )
    model_kwargs = dict(source_model_kwargs or {"max_classes": 2, "is_classifier": True})
    model_kwargs.update(
        race_context_mode=race_context_mode,
        race_context_dim=race_context_dim,
        race_context_layers=race_context_layers,
        race_context_heads=race_context_heads,
        race_context_ff_dim=race_context_ff_dim,
        race_context_residual=race_context_residual,
        encode_races_before_icl=(
            bool(source_model_kwargs.get("encode_races_before_icl", False))
            if args.encode_races_before_icl is None else bool(args.encode_races_before_icl)
        ),
        context_prototype_branch=context_prototype_branch,
        context_prototype_dim=context_prototype_dim,
        context_prototype_max_correction=context_prototype_max_correction,
    )
    feature_manifest_path = args.features_json or DEFAULT_FEATURES
    if resume_bundle is not None:
        feature_columns = list(resume_bundle["feature_columns"])
        zero_features = list(resume_bundle.get("zeroed_features", []))
        if args.features_json is not None:
            requested_feature_columns, zero_features = load_feature_manifest(
                args.features_json
            )
            if requested_feature_columns != feature_columns:
                raise ValueError(
                    "Fine-tune feature manifest differs from source model: "
                    f"requested_count={len(requested_feature_columns)} "
                    f"checkpoint_count={len(feature_columns)}; omit --features-json "
                    "to inherit the checkpoint manifest, or use --overwrite-existing "
                    "to train from scratch"
                )
        print(
            "fine_tune_feature_manifest "
            f"source_checkpoint count={len(feature_columns)} "
            f"explicit_override={args.features_json is not None}",
            flush=True,
        )
    else:
        feature_columns, zero_features = load_feature_manifest(feature_manifest_path)
    if context_prototype_branch:
        if (
            resume_bundle is not None
            and source_context_prototype_branch
            and "context_prototype_input_dim" not in source_model_kwargs
        ):
            raise ValueError(
                "This checkpoint uses the obsolete hidden-representation prototype "
                "branch. Start a fresh run with --overwrite-existing or use a new "
                "--output path for input-feature prototypes."
            )
        model_kwargs["context_prototype_input_dim"] = len(feature_columns)
    split_manifest = None
    if args.split_manifest is not None:
        split_manifest = load_and_validate_runtime_manifest(args.split_manifest, args.db)
    validation_context_race_ids = load_context_race_ids(args.context_json)
    requested_context_races_per_step = (
        len(validation_context_race_ids)
        if args.context_races_per_step is None
        else args.context_races_per_step
    )
    if requested_context_races_per_step > len(validation_context_race_ids):
        raise ValueError(
            "Requested context window cannot exceed the context manifest size: "
            f"requested={requested_context_races_per_step} "
            f"available={len(validation_context_race_ids)}"
        )
    print(
        "context_regime "
        f"requested_context_races={requested_context_races_per_step} "
        "training_context=most_recent_strictly_earlier_same_competition_races_per_query "
        "validation_context=most_recent_strictly_earlier_same_competition_races_per_query "
        "validation_context_pool=all_complete_partition_rows "
        "manifest_role=default_window_size_only "
        f"manifest_race_count={len(validation_context_race_ids)}",
        flush=True,
    )
    validate_feature_columns(args.db, feature_columns)
    print_race_selection_logic(args.db, args.min_race_number)
    if args.training_csv.resolve() == args.validation_csv.resolve():
        raise ValueError("--training-csv and --validation-csv must be different paths")
    export_rows_to_csv(
        args.db, feature_columns, TRAINING_ROWS_VIEW, args.training_csv
    )
    export_rows_to_csv(
        args.db, feature_columns, VALIDATION_ROWS_VIEW, args.validation_csv
    )
    (
        train_x,
        train_y_all,
        train_race_ids,
        train_times,
        train_competition_ids,
        train_validation_flags,
        train_market_fluc2,
    ) = load_rows_from_csv(args.training_csv, feature_columns)
    (
        valid_x_all,
        valid_y_all,
        valid_race_ids_all,
        valid_times,
        valid_competition_ids_all,
        valid_flags,
        valid_market_fluc2,
    ) = load_rows_from_csv(args.validation_csv, feature_columns)
    print(
        "training_record_source "
        f"training_csv={args.training_csv.resolve()} "
        f"validation_csv={args.validation_csv.resolve()}",
        flush=True,
    )
    if args.classroom_overfit_all_races:
        x = train_x
        y = train_y_all
        race_ids = train_race_ids
        times = train_times
        competition_ids = train_competition_ids
        validation_flags = train_validation_flags
        train_mask = np.ones(len(race_ids), dtype=bool)
        valid_mask = train_mask.copy()
    else:
        overlapping_races = sorted(
            set(map(int, train_race_ids)).intersection(map(int, valid_race_ids_all))
        )
        if overlapping_races:
            raise ValueError(
                f"{TRAINING_ROWS_VIEW} and {VALIDATION_ROWS_VIEW} overlap on "
                f"{len(overlapping_races)} race IDs: {overlapping_races[:10]}"
            )
        x = np.concatenate((train_x, valid_x_all), axis=0)
        y = np.concatenate((train_y_all, valid_y_all), axis=0)
        race_ids = np.concatenate((train_race_ids, valid_race_ids_all), axis=0)
        times = np.concatenate((train_times, valid_times), axis=0)
        competition_ids = np.concatenate(
            (train_competition_ids, valid_competition_ids_all), axis=0
        )
        validation_flags = np.concatenate(
            (train_validation_flags, valid_flags), axis=0
        )
        train_mask = np.arange(len(race_ids)) < len(train_race_ids)
        valid_mask = ~train_mask
    if args.fine_tune_race_id is not None:
        target_id = int(args.fine_tune_race_id)
        target_mask = race_ids == target_id
        if not np.any(target_mask):
            raise ValueError(f"fine-tune race {target_id} is not present in the database")
        # This is deliberately an experiment-only label leak: the requested
        # race is removed from validation and admitted as the sole query pool.
        train_mask[target_mask] = True
        valid_mask[target_mask] = False
        print(
            f"WARNING EXPERIMENT fine_tune_race_id={target_id}: target labels "
            "are intentionally used for adaptation; metrics are not OOS evidence.",
            flush=True,
        )
    partition_source = "separate_training_validation_views"
    if args.classroom_overfit_all_races:
        partition_source = "classroom_all_complete_view_races_overlap"
        print(
            "WARNING CLASSROOM OVERFIT MODE: every complete view race is used for "
            "both training and validation; validation/context labels leak into "
            "training and the resulting metrics are not generalization evidence.",
            flush=True,
        )
    if resume_bundle is not None:
        
        median = np.asarray(resume_bundle["median"], dtype=np.float32)
        scale = np.asarray(resume_bundle["scale"], dtype=np.float32)
        if len(feature_columns) != len(median) or len(feature_columns) != len(scale):
            raise ValueError("Resume bundle feature and preprocessing dimensions do not match")
    else:
        median, scale = fit_preprocessor(x[train_mask])

    valid_mask, skipped_validation_race_ids = exclude_invalid_races(
        y, race_ids, valid_mask, "Validation set"
    )
    context_validation_overlap: list[int] = []
    ordered_valid_races = list(dict.fromkeys(race_ids[valid_mask].tolist()))
    if (
        args.max_valid_races is not None
        and args.max_valid_races < len(ordered_valid_races)
    ):
        raise ValueError(
            "--max-valid-races cannot truncate cohort-aware validation: "
            f"requested={args.max_valid_races} available={len(ordered_valid_races)}"
        )
    selected_valid_races = ordered_valid_races
    if not selected_valid_races:
        raise ValueError("No complete validation races are available")
    valid_mask &= np.isin(race_ids, selected_valid_races)

    market_fluc2 = (
        train_market_fluc2
        if args.classroom_overfit_all_races
        else np.concatenate((train_market_fluc2, valid_market_fluc2), axis=0)
    )
    x = transform(x, median, scale)
    x = zero_feature_columns(x, feature_columns, zero_features)
    race_time_by_id: dict[int, object] = {}
    competition_by_race_id: dict[int, int] = {}
    for race_id_value, race_time, competition_id_value in zip(
        race_ids, times, competition_ids
    ):
        race_id = int(race_id_value)
        competition_id = int(competition_id_value)
        previous = race_time_by_id.setdefault(race_id, race_time)
        if previous != race_time:
            raise ValueError(f"Race {race_id} has inconsistent start times")
        previous_competition = competition_by_race_id.setdefault(
            race_id, competition_id
        )
        if previous_competition != competition_id:
            raise ValueError(f"Race {race_id} has inconsistent competition IDs")
    eligible_race_ids = load_race_number_eligible_ids(
        args.db, args.min_race_number
    )
    optimizer_eligible_mask = (
        np.ones(len(race_ids), dtype=bool)
        if args.min_race_number is None
        else np.isin(race_ids, list(eligible_race_ids))
    )
    training_pool_mask = train_mask & optimizer_eligible_mask
    training_pool_mask, skipped_training_race_ids = exclude_invalid_races(
        y, race_ids, training_pool_mask, "Training pool"
    )
    training_race_indices = build_race_indices(race_ids, training_pool_mask)
    schedule_race_numbers = load_race_numbers(
        args.db,
        set(training_race_indices),
    )
    effective_steps_per_epoch, scheduled_query_races_per_step = (
        resolve_training_schedule(
            len(training_race_indices),
            args.steps_per_epoch,
            args.query_races_per_step,
            args.auto_race_schedule,
        )
    )
    context_size_matching_validation = requested_context_races_per_step
    effective_context_races_per_step, effective_query_races_per_step = (
        resolve_races_per_step(
            len(training_race_indices),
            context_size_matching_validation,
            scheduled_query_races_per_step,
        )
    )
    if effective_context_races_per_step != requested_context_races_per_step:
        print(
            "training_context_size_adjusted "
            f"requested={requested_context_races_per_step} "
            f"effective={effective_context_races_per_step} "
            f"validation_context_races={len(validation_context_race_ids)} "
            f"query_races={effective_query_races_per_step} "
            f"available_training_races={len(training_race_indices)}",
            flush=True,
        )
    if not args.classroom_overfit_all_races:
        candidate_validation_races = list(
            map(int, dict.fromkeys(race_ids[valid_mask].tolist()))
        )
        validation_context_mask = training_pool_mask | valid_mask
        validation_context_race_indices = build_race_indices(
            race_ids, validation_context_mask
        )
        eligible_validation_races = eligible_query_race_ids_from_context(
            candidate_validation_races,
            list(validation_context_race_indices),
            race_time_by_id,
            effective_context_races_per_step,
            competition_by_race_id,
        )
        skipped_early_validation_races = sorted(
            set(candidate_validation_races) - set(eligible_validation_races)
        )
        if skipped_early_validation_races:
            print(
                "WARNING skipped_validation_races_without_earlier_same_competition_context "
                f"count={len(skipped_early_validation_races)} "
                f"required_context_races={effective_context_races_per_step} "
                f"preview={skipped_early_validation_races[:10]}",
                flush=True,
            )
        valid_mask &= np.isin(race_ids, eligible_validation_races)
        selected_valid_races = eligible_validation_races
        if not np.any(valid_mask):
            raise ValueError(
                "No validation races have enough strictly earlier "
                "same-competition context races"
            )
    else:
        validation_context_race_indices = build_race_indices(
            race_ids, training_pool_mask | valid_mask
        )
    # Slice after all validation eligibility rules, including causal context.
    valid_fluc2 = market_fluc2[valid_mask].copy()
    train_y = y[training_pool_mask]
    valid_x, valid_y = x[valid_mask], y[valid_mask]
    valid_race_ids = race_ids[valid_mask]
    validation_query_race_indices = build_race_indices(
        valid_race_ids, np.ones(len(valid_race_ids), dtype=bool)
    )
    fixed_probe_race_ids = select_fixed_probe_race_ids(
        valid_y, valid_race_ids, args.probe_races
    )
    fixed_probe_mask = np.isin(valid_race_ids, fixed_probe_race_ids)
    fixed_probe_x = valid_x[fixed_probe_mask]
    fixed_probe_y = valid_y[fixed_probe_mask]
    fixed_probe_row_race_ids = valid_race_ids[fixed_probe_mask]
    fixed_probe_query_race_indices = build_race_indices(
        fixed_probe_row_race_ids,
        np.ones(len(fixed_probe_row_race_ids), dtype=bool),
    )
    print(
        "FIXED_PROBE_SELECTION "
        f"races={len(fixed_probe_race_ids)} "
        f"race_ids={fixed_probe_race_ids.tolist()}",
        flush=True,
    )
    schedule_race_numbers.update(
        load_race_numbers(args.db, set(map(int, valid_race_ids)))
    )
    valid_cohorts, validation_cohort_source = load_validation_cohorts(
        args.db, valid_race_ids
    )
    validation_cohort_race_counts = {
        cohort: len(set(map(int, valid_race_ids[valid_cohorts == cohort])))
        for cohort in sorted(set(map(str, valid_cohorts)))
    }
    representative_races = validation_cohort_race_counts.get(
        "chronological_representative", 0
    )
    if representative_races < 1:
        raise ValueError(
            "Checkpoint selection requires at least one complete "
            "chronological_representative race; combined legacy validation "
            "cannot be used for selection"
        )
    if representative_races < MIN_CHECKPOINT_SELECTION_RACES:
        print(
            "WARNING checkpoint_selection_chronological_cohort_small "
            f"chronological_representative_races={representative_races} "
            f"reference_minimum={MIN_CHECKPOINT_SELECTION_RACES}; "
            "selection_remains_chronological_and_will_not_use_combined",
            flush=True,
        )
    early_stopping_enabled = (
        early_stopping_is_enabled(representative_races)
        or args.allow_small_cohort_early_stopping
    )
    if not early_stopping_enabled:
        print(
            "early_stopping_disabled reason=small_chronological_selection_cohort "
            f"chronological_races={representative_races} "
            f"required_races={MIN_CHECKPOINT_SELECTION_RACES}",
            flush=True,
        )
    elif representative_races < MIN_CHECKPOINT_SELECTION_RACES:
        print(
            "early_stopping_enabled override=allow_small_cohort_early_stopping "
            f"chronological_races={representative_races} "
            f"required_races={MIN_CHECKPOINT_SELECTION_RACES}",
            flush=True,
        )
    if "legacy_combined" in validation_cohort_race_counts:
        legacy_races = validation_cohort_race_counts["legacy_combined"]
        print(
            "WARNING validation_cohort_fallback "
            f"legacy_combined_races={legacy_races} "
            f"source={validation_cohort_source}; explicitly assigned cohorts are "
            "preserved and uncovered validation races remain in combined metrics",
            flush=True,
        )
    if len(np.unique(valid_y)) != 2:
        raise ValueError(
            "Selected validation races must contain both top-three and "
            "non-top-three runners"
        )

    progress_race_id = (
        args.progress_race_id
        if args.progress_race_id is not None
        else int(valid_race_ids[0])
    )
    progress_x_raw, progress_rows = load_progress_race(
        args.db, progress_race_id, feature_columns
    )
    progress_x = transform(progress_x_raw, median, scale)
    progress_x = zero_feature_columns(
        progress_x, feature_columns, zero_features
    )
    class_counts = np.bincount(train_y, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            "Training races must contain both top-three and non-top-three runners"
        )
    class_weights = len(train_y) / (2.0 * class_counts)
    training_race_ids = set(training_race_indices)
    chronological_query_race_ids = eligible_query_race_ids(
        list(training_race_indices),
        race_time_by_id,
        effective_context_races_per_step,
        competition_by_race_id,
    )
    excluded_early_query_races = len(training_race_ids) - len(
        chronological_query_race_ids
    )
    print(
        f"training_context_pool_races={len(training_race_ids)} "
        f"chronological_query_eligible_races={len(chronological_query_race_ids)} "
        f"excluded_early_query_races={excluded_early_query_races}",
        flush=True,
    )
    if not chronological_query_race_ids:
        raise ValueError(
            "No training query races have enough strictly earlier "
            "same-competition context races"
        )
    if effective_query_races_per_step > len(chronological_query_race_ids):
        print(
            "query_races_per_step_adjusted_for_chronology "
            f"requested={effective_query_races_per_step} "
            f"effective={len(chronological_query_race_ids)}",
            flush=True,
        )
        effective_query_races_per_step = len(chronological_query_race_ids)
    if args.auto_race_schedule:
        effective_steps_per_epoch, effective_query_races_per_step = (
            resolve_training_schedule(
                len(chronological_query_race_ids),
                args.steps_per_epoch,
                effective_query_races_per_step,
                True,
            )
        )
        print(
            "automatic_race_schedule "
            f"chronological_query_eligible_races={len(chronological_query_race_ids)} "
            f"steps_per_epoch={effective_steps_per_epoch} "
            f"query_races_per_step={effective_query_races_per_step} "
            f"query_slots={effective_steps_per_epoch * effective_query_races_per_step}",
            flush=True,
        )
    validation_race_id_set = set(int(race_id) for race_id in selected_valid_races)
    if progress_race_id in training_race_ids and not args.classroom_overfit_all_races:
        raise RuntimeError(
            "Progress race entered training; choose an is_validation = 1 race"
        )

    print("RACE COUNTS", flush=True)
    print(f"  Training races:   {len(training_race_indices):,}", flush=True)
    print(f"  Validation races: {len(selected_valid_races):,}", flush=True)
    print(
        f"db={args.db.resolve()} features={len(feature_columns)} "
        f"training_csv={args.training_csv.resolve()} "
        f"validation_csv={args.validation_csv.resolve()} "
        f"train_rows={len(train_y):,} train_races={len(training_race_indices):,} "
        f"valid_rows={len(valid_y):,} valid_races={len(selected_valid_races):,} "
        f"partition_source={partition_source} device={device}",
        flush=True,
    )
    print(
        f"train_top3_rate={train_y.mean():.4f} valid_top3_rate={valid_y.mean():.4f} "
        f"all_non_top3_valid_accuracy={1.0 - valid_y.mean():.4f} "
        f"class_weights=[{class_weights[0]:.3f}, {class_weights[1]:.3f}] "
        "training_context=most_recent_strictly_earlier_same_competition_races_per_query "
        "validation_context=most_recent_strictly_earlier_same_competition_races_per_query "
        f"context_races_per_query={effective_context_races_per_step} "
        f"independent_query_sequences_per_step={effective_query_races_per_step} "
        f"auto_race_schedule={args.auto_race_schedule} "
        f"min_race_number={args.min_race_number} "
        f"race_context_mode={race_context_mode} "
        f"context_prototype_branch={context_prototype_branch} "
        f"pairwise_loss_weight={args.pairwise_loss_weight} "
        f"attention_delta_pairwise_loss_weight="
        f"{args.attention_delta_pairwise_loss_weight} "
        f"context_prototype_loss_weight={args.context_prototype_loss_weight} "
        f"cardinality_loss_weight={args.cardinality_loss_weight} "
        f"validation_cohort_source={validation_cohort_source} "
        f"validation_cohort_races={validation_cohort_race_counts} "
        f"zeroed_features={zero_features} progress_race_id={progress_race_id}",
        flush=True,
    )
    if resume_bundle is not None:
        model = TabFM(**model_kwargs).to(device)
        architecture_unchanged = (
            source_race_mode == race_context_mode
            and source_context_prototype_branch == context_prototype_branch
        )
        incompatible = model.load_state_dict(
            resume_bundle["model_state_dict"],
            strict=architecture_unchanged,
        )
        if not architecture_unchanged:
            unexpected = list(incompatible.unexpected_keys)
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not (
                    key.startswith("race_set_head.")
                    or key.startswith("context_prototype_head.")
                )
            ]
            if unexpected or invalid_missing:
                raise ValueError(
                    "Base checkpoint is incompatible with race-aware upgrade: "
                    f"missing={invalid_missing} unexpected={unexpected}"
                )
        print(
            f"training_mode="
            f"{'fine_tune_existing_output' if auto_resumed_output else 'fine_tune_explicit_resume'} "
            f"resume_model={resume_model_path.resolve()} "
            f"source_best_epoch={resume_bundle.get('best_epoch', '-')}",
            flush=True,
        )
    else:
        model = TabFM(**model_kwargs).to(device)
        initialize_fourier_frequencies(model, args.seed)
        frequency_buffer = model.cell_embedder.fourier_frequencies
        print(
            f"fourier_frequencies=initialized "
            f"abs_mean={frequency_buffer.abs().mean().item():.4f} "
            f"abs_max={frequency_buffer.abs().max().item():.4f}",
            flush=True,
        )
    optimizer_parameters, trainable_parameter_count, total_parameter_count = (
        configure_trainable_parameters(model, training_scope)
    )
    trainable_names = trainable_parameter_names(model)
    if training_scope == "decoder_and_race_head":
        trainable_prefixes = ["icl_predictor.decoder", "race_set_head"]
    elif training_scope == "attention_head_only":
        trainable_prefixes = ["race_set_head"]
    elif training_scope == "icl_and_race_head":
        trainable_prefixes = ["icl_predictor", "race_set_head"]
    else:
        trainable_prefixes = sorted({name.split(".", 1)[0] for name in trainable_names})
    if (
        model.context_prototype_head is not None
        and training_scope != "full_model"
    ):
        trainable_prefixes.append("context_prototype_head")
    print(
        "trainable_module_prefixes=" + ",".join(trainable_prefixes),
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    run_training_mode = (
        "Fine-tune existing checkpoint"
        if resume_model_path is not None
        else "Train from scratch"
    )
    parameter_rows = [
        ("Training mode", run_training_mode),
        (
            "Resume model",
            str(resume_model_path.resolve()) if resume_model_path is not None else "None",
        ),
        (
            "Source best epoch",
            str(resume_bundle.get("best_epoch", "-"))
            if resume_bundle is not None else "-",
        ),
        ("Epochs", str(args.epochs)),
        ("Steps per epoch", str(effective_steps_per_epoch)),
        ("Probe races", str(len(fixed_probe_race_ids))),
        ("Probe cadence", f"every {args.probe_every_steps} optimizer steps"),
        (
            "Context stop warning",
            f"after step {max(CONTEXT_STOP_WARNING_MIN_STEP, args.probe_every_steps * CONTEXT_STOP_WARNING_PROBES)} "
            f"and {CONTEXT_STOP_WARNING_PROBES} ineffective permutation probes",
        ),
        ("Step loss window", str(args.step_loss_window)),
        ("Maximum optimizer steps", f"{args.epochs * effective_steps_per_epoch:,}"),
        ("Learning rate", f"{learning_rate:.8g}"),
        ("Optimizer", "AdamW"),
        ("Weight decay", f"{args.weight_decay:g}"),
        ("Gradient clipping", f"Maximum norm {args.max_grad_norm:g}"),
        ("Training scope", training_scope.replace("_", " ").title()),
        ("Trainable parameters", f"{trainable_parameter_count:,}"),
        ("Total parameters", f"{total_parameter_count:,}"),
        ("Device", str(device).upper()),
        ("Random seed", str(args.seed)),
        ("Context races per query", str(effective_context_races_per_step)),
        ("Query sequences per step", str(effective_query_races_per_step)),
        ("Training context strategy", "Most recent strictly earlier per query"),
        ("Automatic race schedule", str(args.auto_race_schedule)),
        ("Print race schedule", str(args.print_race_schedule)),
        ("Eligible training races", f"{len(training_race_indices):,}"),
        ("Batch layout", "One independently contextualised sequence per query"),
        ("Batch rows", "Padded variable sequences; --batch-rows is deprecated"),
        ("Classification loss weight", f"{args.classification_loss_weight:g}"),
        ("Pairwise loss weight", f"{args.pairwise_loss_weight:g}"),
        (
            "Attention-delta pairwise loss",
            f"{args.attention_delta_pairwise_loss_weight:g}",
        ),
        (
            "Context prototype direct loss",
            f"{args.context_prototype_loss_weight:g}",
        ),
        ("Cardinality loss weight", f"{args.cardinality_loss_weight:g}"),
        (
            "Context dependence weight",
            f"{args.context_dependence_loss_weight:g}",
        ),
        ("Context dependence margin", f"{args.context_dependence_margin:g}"),
        (
            "Context dependence formulation",
            "Detached correct-loss reference; maximize permuted loss to margin",
        ),
        ("Stress recall maximum drop", f"{args.stress_top3_recall_max_drop:g}"),
        ("Early-stopping patience", f"{args.early_stopping_patience} epochs"),
        ("Race context mode", race_context_mode),
        ("Context prototype branch", str(context_prototype_branch)),
        (
            "Context prototype source",
            "Normalized input features" if context_prototype_branch else "Disabled",
        ),
        ("Context prototype dimension", str(context_prototype_dim)),
        (
            "Context prototype max correction",
            f"{context_prototype_max_correction:g}",
        ),
        (
            "Minimum race number",
            str(args.min_race_number) if args.min_race_number is not None else "None",
        ),
        ("Zeroed features", ", ".join(zero_features) if zero_features else "None"),
        ("Number of features", str(len(feature_columns))),
    ]
    parameter_name_width = max(len(name) for name, _ in parameter_rows)
    parameter_value_width = max(len(value) for _, value in parameter_rows)
    table_width = parameter_name_width + parameter_value_width + 7
    print("TRAINING PARAMETERS", flush=True)
    print(
        f"  {'Parameter':<{parameter_name_width}} | {'Value':<{parameter_value_width}}",
        flush=True,
    )
    print(f"  {'-' * parameter_name_width}-+-{'-' * parameter_value_width}", flush=True)
    for name, value in parameter_rows:
        print(f"  {name:<{parameter_name_width}} | {value:<{parameter_value_width}}", flush=True)
    print(f"  {'-' * table_width}", flush=True)
    print(
        f"optimizer_learning_rate={learning_rate:.8g} "
        f"weight_decay={args.weight_decay:.8g} "
        f"training_scope={training_scope} "
        f"trainable_parameters={trainable_parameter_count:,} "
        f"total_parameters={total_parameter_count:,}",
        flush=True,
    )
    loss_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    market_scores = market_rank_scores(valid_fluc2)
    market_race_metrics = race_top3_metrics(valid_y, market_scores, valid_race_ids)
    market_metrics_by_cohort = {"combined": market_race_metrics}
    for cohort in sorted(set(map(str, valid_cohorts))):
        cohort_mask = valid_cohorts == cohort
        market_metrics_by_cohort[cohort] = race_top3_metrics(
            valid_y[cohort_mask], market_scores[cohort_mask], valid_race_ids[cohort_mask]
        )
    market_price_coverage = float(np.isfinite(market_scores).mean())
    print(f"MARKET BASELINE fluc2 coverage={market_price_coverage:.4f}", flush=True)
    for cohort, metrics in market_metrics_by_cohort.items():
        print(
            f"  {cohort} races={metrics['complete_races']} "
            f"top3_recall={metrics['top3_recall']:.4f} "
            f"exact_top3_set={metrics['exact_top3_set_rate']:.4f} "
            f"contained_top4={metrics['contained_top4_rate']:.4f} "
            f"contained_top5={metrics['contained_top5_rate']:.4f} "
            f"contained_top6={metrics['contained_top6_rate']:.4f}",
            flush=True,
        )

    best_fixed_probe_top3_recall = float("-inf")
    best_fixed_probe_auc = float("-inf")
    fixed_probe_regression_streak = 0
    ineffective_context_probe_streak = 0
    context_warning_start_step = max(
        CONTEXT_STOP_WARNING_MIN_STEP,
        args.probe_every_steps * CONTEXT_STOP_WARNING_PROBES,
    )

    def predict_with_causal_context(
        query_x: np.ndarray, query_race_indices: dict[int, np.ndarray],
        context_label_mode: str = "correct",
    ) -> np.ndarray:
        return predict_with_chronological_context(
            model,
            x,
            y,
            validation_context_race_indices,
            race_time_by_id,
            query_x,
            query_race_indices,
            effective_context_races_per_step,
            device,
            context_label_mode=context_label_mode,
            context_label_seed=args.seed,
            competition_by_race_id=competition_by_race_id,
        )

    def evaluate_fixed_probe(
        label: str, global_step: int
    ) -> tuple[dict[str, float | int], np.ndarray]:
        nonlocal best_fixed_probe_top3_recall
        nonlocal best_fixed_probe_auc
        nonlocal fixed_probe_regression_streak
        was_training = model.training
        model.eval()
        probe_probability = predict_with_causal_context(
            fixed_probe_x, fixed_probe_query_race_indices
        )
        probe_metrics = probability_metrics(
            fixed_probe_y, probe_probability, fixed_probe_row_race_ids
        )
        print(
            f"FIXED_PROBE label={label} step={global_step} "
            f"races={probe_metrics['complete_races']} "
            f"top3_recall={probe_metrics['top3_recall']:.4f} "
            f"exact_top3_set={probe_metrics['exact_top3_set_rate']:.4f} "
            f"contained_top4={probe_metrics['contained_top4_rate']:.4f} "
            f"contained_top5={probe_metrics['contained_top5_rate']:.4f} "
            f"contained_top6={probe_metrics['contained_top6_rate']:.4f} "
            f"auc={probe_metrics['roc_auc']:.4f} "
            f"logloss={probe_metrics['logloss']:.5f}",
            flush=True,
        )
        current_recall = float(probe_metrics["top3_recall"])
        current_auc = float(probe_metrics["roc_auc"])
        if label == "after_update" and global_step >= FIXED_PROBE_REGRESSION_WARNING_MIN_STEP:
            regressed = fixed_probe_has_material_regression(
                best_fixed_probe_top3_recall,
                best_fixed_probe_auc,
                current_recall,
                current_auc,
            )
            fixed_probe_regression_streak = (
                fixed_probe_regression_streak + 1 if regressed else 0
            )
            if regressed:
                print(
                    "TRAINING_WARNING type=fixed_probe_regression "
                    f"step={global_step} streak={fixed_probe_regression_streak} "
                    f"top3_recall_drop="
                    f"{best_fixed_probe_top3_recall - current_recall:.4f} "
                    f"auc_drop={best_fixed_probe_auc - current_auc:.4f} "
                    "action=observe_next_probe",
                    flush=True,
                )
            if fixed_probe_regression_streak >= FIXED_PROBE_STOP_WARNING_PROBES:
                print(
                    "TRAINING_STOP_WARNING severity=stop_recommended "
                    "reason=repeated_fixed_probe_regression "
                    f"step={global_step} consecutive_probes="
                    f"{fixed_probe_regression_streak} "
                    "action=stop_run_and_reduce_learning_rate",
                    flush=True,
                )
        best_fixed_probe_top3_recall = max(
            best_fixed_probe_top3_recall, current_recall
        )
        best_fixed_probe_auc = max(best_fixed_probe_auc, current_auc)
        if was_training:
            model.train()
        return probe_metrics, probe_probability

    def evaluate_context_ablation_probe(
        label: str, global_step: int, baseline_probability: np.ndarray
    ) -> None:
        """Compare fixed-probe predictions under deterministic label ablations."""
        nonlocal ineffective_context_probe_streak
        was_training = model.training
        model.eval()
        baseline_metrics = probability_metrics(
            fixed_probe_y, baseline_probability, fixed_probe_row_race_ids
        )
        print(
            f"CONTEXT_ABLATION_PROBE label={label} step={global_step} "
            f"variant=correct races={baseline_metrics['complete_races']} "
            "max_probability_delta=0.00000000 "
            "mean_probability_delta=0.00000000 "
            "ranking_changed_races=0 top3_changed_races=0 "
            "mean_rank_displacement=0.0000 "
            f"top3_recall={baseline_metrics['top3_recall']:.4f} "
            "top3_recall_delta=+0.0000 "
            f"auc={baseline_metrics['roc_auc']:.4f} auc_delta=+0.0000 "
            f"logloss={baseline_metrics['logloss']:.5f} "
            "logloss_delta=+0.00000",
            flush=True,
        )
        for variant in ("permuted", "zeroed", "flipped"):
            ablated_probability = predict_with_causal_context(
                fixed_probe_x,
                fixed_probe_query_race_indices,
                context_label_mode=variant,
            )
            ablated_metrics = probability_metrics(
                fixed_probe_y, ablated_probability, fixed_probe_row_race_ids
            )
            change = prediction_change_metrics(
                baseline_probability,
                ablated_probability,
                fixed_probe_row_race_ids,
            )
            auc_delta = (
                float(ablated_metrics["roc_auc"])
                - float(baseline_metrics["roc_auc"])
            )
            logloss_delta = (
                float(ablated_metrics["logloss"])
                - float(baseline_metrics["logloss"])
            )
            print(
                f"CONTEXT_ABLATION_PROBE label={label} step={global_step} "
                f"variant={variant} races={change['compared_races']} "
                f"max_probability_delta={change['max_probability_delta']:.8f} "
                f"mean_probability_delta={change['mean_probability_delta']:.8f} "
                f"ranking_changed_races={change['ranking_changed_races']} "
                f"top3_changed_races={change['top3_changed_races']} "
                f"mean_rank_displacement="
                f"{change['mean_absolute_rank_displacement']:.4f} "
                f"top3_recall={ablated_metrics['top3_recall']:.4f} "
                f"top3_recall_delta="
                f"{float(ablated_metrics['top3_recall']) - float(baseline_metrics['top3_recall']):+.4f} "
                f"auc={ablated_metrics['roc_auc']:.4f} "
                f"auc_delta="
                f"{auc_delta:+.4f} "
                f"logloss={ablated_metrics['logloss']:.5f} "
                f"logloss_delta="
                f"{logloss_delta:+.5f}",
                flush=True,
            )
            if variant == "permuted" and label == "after_update":
                ineffective = context_permutation_is_ineffective(
                    float(change["mean_probability_delta"]),
                    auc_delta,
                    logloss_delta,
                )
                ineffective_context_probe_streak = (
                    ineffective_context_probe_streak + 1 if ineffective else 0
                )
                if (
                    global_step >= context_warning_start_step
                    and ineffective_context_probe_streak
                    >= CONTEXT_STOP_WARNING_PROBES
                    and ineffective_context_probe_streak
                    % CONTEXT_STOP_WARNING_PROBES == 0
                ):
                    print(
                        "TRAINING_STOP_WARNING severity=stop_recommended "
                        "reason=context_permutation_ineffective "
                        f"step={global_step} consecutive_probes="
                        f"{ineffective_context_probe_streak} "
                        f"mean_probability_delta="
                        f"{float(change['mean_probability_delta']):.8f} "
                        f"auc_delta={auc_delta:+.4f} "
                        f"logloss_delta={logloss_delta:+.5f} "
                        "action=stop_run_and_review_context_objective",
                        flush=True,
                    )
        if was_training:
            model.train()

    _, before_training_probability = evaluate_fixed_probe("before_training", 0)
    evaluate_context_ablation_probe(
        "before_training", 0, before_training_probability
    )
    best_selection: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metrics: dict[str, float | int | None] = {}
    best_metrics_by_cohort: dict[str, dict[str, float | int]] = {}
    best_observed_stress_recall: float | None = None
    epochs_without_improvement = 0
    step_loss_history: deque[float] = deque(maxlen=args.step_loss_window)
    if resume_bundle is not None:
        model.eval()
        baseline_probability = predict_with_causal_context(
            valid_x, validation_query_race_indices
        )
        baseline_metrics_by_cohort = validation_metrics_by_cohort(
            valid_y, baseline_probability, valid_race_ids, valid_cohorts
        )
        baseline_race_metrics = baseline_metrics_by_cohort["combined"]
        best_selection = cohort_checkpoint_selection(baseline_metrics_by_cohort)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_metrics = {
            **baseline_race_metrics,
            "train_loss": None,
        }
        best_metrics_by_cohort = baseline_metrics_by_cohort
        stress_metrics = baseline_metrics_by_cohort.get("market_miss_stress")
        if stress_metrics is not None:
            best_observed_stress_recall = float(stress_metrics["top3_recall"])
        print("FINE-TUNE BASELINE epoch=0", flush=True)
        for cohort, metrics in baseline_metrics_by_cohort.items():
            print("  " + format_metric_line(cohort, metrics), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        classification_losses = []
        pairwise_losses = []
        attention_delta_pairwise_losses = []
        context_prototype_pairwise_losses = []
        context_prototype_abs_means = []
        context_prototype_permutation_deltas = []
        cardinality_losses = []
        context_dependence_losses = []
        permuted_context_prediction_losses = []
        context_margin_satisfied_steps = 0
        context_row_counts = []
        query_row_counts = []
        batch_row_counts = []
        gradient_norms = []
        clipped_gradient_steps = 0
        epoch_context_races: set[int] = set()
        epoch_query_races: set[int] = set()
        query_schedule = build_query_race_schedule(
            chronological_query_race_ids,
            effective_steps_per_epoch,
            effective_query_races_per_step,
            rng,
        )
        if args.print_race_schedule and epoch == 1:
            training_display = ", ".join(
                f"{race_id}:{schedule_race_numbers.get(int(race_id), '?')}"
                for race_id in sorted(training_race_indices)
            )
            validation_display = ", ".join(
                f"{race_id}:{schedule_race_numbers.get(int(race_id), '?')}"
                for race_id in dict.fromkeys(map(int, valid_race_ids))
            )
            print(
                "TRAINING RACES "
                f"races={len(training_race_indices)} "
                f"race_id:race_number=[{training_display}]",
                flush=True,
            )
            print(
                "VALIDATION RACES "
                f"races={len(set(map(int, valid_race_ids)))} "
                f"race_id:race_number=[{validation_display}]",
                flush=True,
            )
            print(
                "TRAINING AND VALIDATION CONTEXT "
                "strategy=most_recent_strictly_earlier_same_competition_races "
                "layout=independent_sequence_per_query "
                f"races_per_query={effective_context_races_per_step}",
                flush=True,
            )
        for step_index in range(effective_steps_per_epoch):
            global_step = (epoch - 1) * effective_steps_per_epoch + step_index + 1
            (
                batch_x,
                batch_y,
                batch_train_sizes,
                batch_context_race_ids,
                batch_query_race_ids,
                batch_race_group_ids,
                batch_valid_row_mask,
            ) = sample_independent_race_batch(
                x,
                y,
                training_race_indices,
                effective_context_races_per_step,
                query_schedule[step_index],
                race_time_by_id,
                competition_by_race_id,
                group_context_races=model.encode_races_before_icl,
            )
            permuted_batch_y = None
            if args.context_dependence_loss_weight > 0:
                permuted_batch_y = permute_context_labels(
                    batch_y,
                    batch_train_sizes,
                    seed=args.seed + global_step * 1_000_003,
                )
            if args.print_race_schedule:
                context_display = ", ".join(
                    f"{race_id}:{schedule_race_numbers.get(int(race_id), '?')}"
                    for race_id in batch_context_race_ids
                )
                query_display = ", ".join(
                    f"{race_id}:{schedule_race_numbers.get(int(race_id), '?')}"
                    for race_id in batch_query_race_ids
                )
                print(
                    f"RACE SCHEDULE epoch={epoch} step={step_index + 1}/"
                    f"{effective_steps_per_epoch} "
                    f"context_race_id:race_number=[{context_display}] "
                    f"query_race_id:race_number=[{query_display}]",
                    flush=True,
                )
            context_race_set = set(map(int, batch_context_race_ids))
            query_race_set = set(map(int, batch_query_race_ids))
            if not context_race_set <= training_race_ids:
                raise RuntimeError("A sampled context race is not a training race")
            if not query_race_set <= training_race_ids:
                raise RuntimeError("A sampled query race is not a training race")
            if (
                not args.classroom_overfit_all_races
                and (context_race_set | query_race_set) & validation_race_id_set
            ):
                raise RuntimeError("A validation race appeared in a training batch")
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if permuted_batch_y is not None:
                permuted_batch_y = permuted_batch_y.to(device)
            batch_train_sizes = batch_train_sizes.to(device)
            batch_race_group_ids = batch_race_group_ids.to(device)
            batch_valid_row_mask = batch_valid_row_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            return_auxiliary_deltas = (
                args.attention_delta_pairwise_loss_weight > 0
                or args.context_prototype_loss_weight > 0
            )
            model_output = model(
                batch_x,
                batch_y,
                batch_train_sizes,
                race_group_ids=batch_race_group_ids,
                valid_row_mask=batch_valid_row_mask,
                return_auxiliary_deltas=return_auxiliary_deltas,
            )
            if return_auxiliary_deltas:
                logits, auxiliary_deltas = model_output
                race_delta = auxiliary_deltas["race_delta"]
                context_prototype_delta = auxiliary_deltas[
                    "context_prototype_delta"
                ]
            else:
                logits = model_output
                race_delta = None
                context_prototype_delta = None
            positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
            query_mask = positions >= batch_train_sizes.unsqueeze(1)
            query_mask &= batch_valid_row_mask
            query_mask &= batch_y != -100
            query_mask &= batch_race_group_ids >= 0
            if not torch.any(query_mask):
                raise ValueError("Batch contains no valid query rows for loss calculation.")
            query_logits = logits[query_mask][:, :2]
            query_targets = batch_y[query_mask]
            query_row_race_ids_tensor = batch_race_group_ids[query_mask]
            prediction_loss, classification_loss, pairwise_loss, cardinality_loss = grouped_race_losses(
                query_logits,
                query_targets,
                query_row_race_ids_tensor,
                loss_weights,
                args.pairwise_loss_weight,
                args.cardinality_loss_weight,
                classification_loss_weight=args.classification_loss_weight,
            )
            loss = prediction_loss
            if args.attention_delta_pairwise_loss_weight > 0:
                if race_delta is None:
                    raise RuntimeError(
                        "Attention-delta loss was enabled but the model returned "
                        "no race correction"
                    )
                attention_delta_pairwise_loss = grouped_pairwise_loss(
                    race_delta[query_mask][:, :2],
                    query_targets,
                    query_row_race_ids_tensor,
                )
                loss = (
                    loss
                    + args.attention_delta_pairwise_loss_weight
                    * attention_delta_pairwise_loss
                )
            else:
                attention_delta_pairwise_loss = loss.new_zeros(())
            if args.context_prototype_loss_weight > 0:
                if context_prototype_delta is None:
                    raise RuntimeError(
                        "Context prototype direct loss was enabled but the model "
                        "returned no prototype correction"
                    )
                context_prototype_pairwise_loss = grouped_pairwise_loss(
                    context_prototype_delta[query_mask][:, :2],
                    query_targets,
                    query_row_race_ids_tensor,
                )
                loss = (
                    loss
                    + args.context_prototype_loss_weight
                    * context_prototype_pairwise_loss
                )
            else:
                context_prototype_pairwise_loss = loss.new_zeros(())
            permuted_context_prototype_delta = None
            if permuted_batch_y is not None:
                permuted_model_output = model(
                    batch_x,
                    permuted_batch_y,
                    batch_train_sizes,
                    race_group_ids=batch_race_group_ids,
                    valid_row_mask=batch_valid_row_mask,
                    return_auxiliary_deltas=(
                        args.context_prototype_loss_weight > 0
                    ),
                )
                if args.context_prototype_loss_weight > 0:
                    permuted_logits, permuted_auxiliary_deltas = (
                        permuted_model_output
                    )
                    permuted_context_prototype_delta = (
                        permuted_auxiliary_deltas["context_prototype_delta"]
                    )
                else:
                    permuted_logits = permuted_model_output
                permuted_query_logits = permuted_logits[query_mask][:, :2]
                (
                    permuted_context_prediction_loss,
                    _,
                    _,
                    _,
                ) = grouped_race_losses(
                    permuted_query_logits,
                    query_targets,
                    query_row_race_ids_tensor,
                    loss_weights,
                    args.pairwise_loss_weight,
                    args.cardinality_loss_weight,
                    classification_loss_weight=args.classification_loss_weight,
                )
                context_dependence_loss = context_dependence_margin_loss(
                    prediction_loss,
                    permuted_context_prediction_loss,
                    args.context_dependence_margin,
                )
                context_margin_satisfied_steps += int(
                    float(context_dependence_loss.detach()) == 0.0
                )
                loss = (
                    loss
                    + args.context_dependence_loss_weight
                    * context_dependence_loss
                )
            else:
                permuted_context_prediction_loss = loss.new_zeros(())
                context_dependence_loss = loss.new_zeros(())
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss encountered before backward pass"
                )
            step_loss = float(loss.detach())
            context_loss_value = float(context_dependence_loss.detach())
            context_loss_gap = float(
                (
                    permuted_context_prediction_loss.detach()
                    - prediction_loss.detach()
                )
            )
            if context_prototype_delta is not None:
                query_context_prototype_delta = context_prototype_delta[query_mask]
                context_prototype_abs_mean = float(
                    query_context_prototype_delta.detach().abs().mean()
                )
            else:
                context_prototype_abs_mean = 0.0
            if permuted_context_prototype_delta is not None:
                context_prototype_permutation_delta = float(
                    (
                        context_prototype_delta[query_mask].detach()
                        - permuted_context_prototype_delta[query_mask].detach()
                    ).abs().mean()
                )
            else:
                context_prototype_permutation_delta = 0.0
            step_race_metrics = pre_update_training_batch_metrics(
                query_targets.detach().cpu().numpy(),
                query_logits,
                query_row_race_ids_tensor.detach().cpu().numpy(),
            )
            step_loss_history.append(step_loss)
            print(
                f"pre_update_training_batch epoch={epoch:02d} "
                f"step={step_index + 1}/{effective_steps_per_epoch} "
                f"global_step={global_step} "
                f"complete_races={step_race_metrics['complete_races']} "
                f"loss={step_loss:.5f} "
                f"rolling_loss={np.mean(step_loss_history):.5f} "
                f"context_dependence_loss={context_loss_value:.5f} "
                f"permuted_minus_correct_loss={context_loss_gap:+.5f} "
                f"prototype_pairwise_loss="
                f"{float(context_prototype_pairwise_loss.detach()):.5f} "
                f"prototype_abs_mean={context_prototype_abs_mean:.6f} "
                f"prototype_permutation_delta="
                f"{context_prototype_permutation_delta:.6f} "
                "{"
                f"top3_recall={step_race_metrics['top3_recall']:.4f} "
                f"exact_top3_set={step_race_metrics['exact_top3_set_rate']:.4f} "
                f"contained_top4={step_race_metrics['contained_top4_rate']:.4f} "
                f"contained_top5={step_race_metrics['contained_top5_rate']:.4f} "
                f"contained_top6={step_race_metrics['contained_top6_rate']:.4f}"
                "}",
                flush=True,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            if not torch.isfinite(gradient_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite gradient norm encountered after backward pass"
                )
            gradient_norm_value = float(gradient_norm.detach())
            gradient_norms.append(gradient_norm_value)
            clipped_gradient_steps += int(gradient_norm_value > args.max_grad_norm)
            optimizer.step()
            base_model = model.module if hasattr(model, "module") else model
            if hasattr(base_model, "mark_weights_updated"):
                base_model.mark_weights_updated()
            if global_step % args.probe_every_steps == 0:
                _, after_update_probability = evaluate_fixed_probe(
                    "after_update", global_step
                )
                evaluate_context_ablation_probe(
                    "after_update", global_step, after_update_probability
                )
            losses.append(float(loss.detach()))
            classification_losses.append(float(classification_loss.detach()))
            pairwise_losses.append(float(pairwise_loss.detach()))
            attention_delta_pairwise_losses.append(
                float(attention_delta_pairwise_loss.detach())
            )
            context_prototype_pairwise_losses.append(
                float(context_prototype_pairwise_loss.detach())
            )
            context_prototype_abs_means.append(context_prototype_abs_mean)
            context_prototype_permutation_deltas.append(
                context_prototype_permutation_delta
            )
            cardinality_losses.append(float(cardinality_loss.detach()))
            context_dependence_losses.append(
                float(context_dependence_loss.detach())
            )
            permuted_context_prediction_losses.append(
                float(permuted_context_prediction_loss.detach())
            )
            context_row_counts.extend(
                map(float, batch_train_sizes.detach().cpu().numpy())
            )
            query_row_counts.extend(
                map(
                    float,
                    query_mask.sum(dim=1).detach().cpu().numpy(),
                )
            )
            batch_row_counts.append(int(batch_valid_row_mask.sum().item()))
            epoch_context_races.update(context_race_set)
            epoch_query_races.update(query_race_set)

        model.eval()
        valid_prob = predict_with_causal_context(
            valid_x, validation_query_race_indices
        )
        metrics_by_cohort = validation_metrics_by_cohort(
            valid_y, valid_prob, valid_race_ids, valid_cohorts
        )
        race_metrics = metrics_by_cohort["combined"]
        print(
            f"epoch={epoch:02d} weighted_train_loss={np.mean(losses):.5f} "
            f"classification_loss={np.mean(classification_losses):.5f} "
            f"pairwise_loss={np.mean(pairwise_losses):.5f} "
            f"attention_delta_pairwise_loss="
            f"{np.mean(attention_delta_pairwise_losses):.5f} "
            f"context_prototype_pairwise_loss="
            f"{np.mean(context_prototype_pairwise_losses):.5f} "
            f"context_prototype_abs_mean="
            f"{np.mean(context_prototype_abs_means):.6f} "
            f"context_prototype_permutation_delta="
            f"{np.mean(context_prototype_permutation_deltas):.6f} "
            f"cardinality_loss={np.mean(cardinality_losses):.5f} "
            f"context_dependence_loss={np.mean(context_dependence_losses):.5f} "
            f"permuted_context_prediction_loss="
            f"{np.mean(permuted_context_prediction_losses):.5f} "
            f"context_margin_satisfied_steps="
            f"{context_margin_satisfied_steps}/{len(losses)} "
            f"avg_context_rows_per_query={np.mean(context_row_counts):.1f} "
            f"avg_query_rows_per_race={np.mean(query_row_counts):.1f} "
            f"valid_rows_per_step_min={min(batch_row_counts)} "
            f"valid_rows_per_step_max={max(batch_row_counts)} "
            f"avg_gradient_norm={np.mean(gradient_norms):.5f} "
            f"max_gradient_norm={max(gradient_norms):.5f} "
            f"gradient_clipped_steps={clipped_gradient_steps}/{len(gradient_norms)} "
            f"unique_context_races={len(epoch_context_races)} "
            f"unique_query_races={len(epoch_query_races)}",
            flush=True,
        )
        for cohort, metrics in metrics_by_cohort.items():
            print("  " + format_metric_line(cohort, metrics), flush=True)
        progress_probability = predict_with_causal_context(
            progress_x,
            {progress_race_id: np.arange(len(progress_x), dtype=np.int64)},
        )
        print_progress_race(progress_race_id, progress_rows, progress_probability)
        selection = cohort_checkpoint_selection(metrics_by_cohort)
        guardrail_passed = stress_guardrail_passes(
            metrics_by_cohort,
            best_observed_stress_recall,
            args.stress_top3_recall_max_drop,
        )
        stress_metrics = metrics_by_cohort.get("market_miss_stress")
        if stress_metrics is not None:
            stress_recall = float(stress_metrics["top3_recall"])
            best_observed_stress_recall = (
                stress_recall
                if best_observed_stress_recall is None
                else max(best_observed_stress_recall, stress_recall)
            )
        selection_improved = (
            best_selection is None
            or checkpoint_selection_improves(
                metrics_by_cohort, best_metrics_by_cohort
            )
        )
        if selection_improved and guardrail_passed:
            best_selection = selection
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_metrics = {
                **race_metrics,
                "train_loss": float(np.mean(losses)),
            }
            best_metrics_by_cohort = metrics_by_cohort
            epochs_without_improvement = 0
            best_checkpoint_path = save_best_epoch_checkpoint(
                output=args.output,
                epoch=best_epoch,
                state_dict=best_state,
                model_kwargs=model_kwargs,
                feature_columns=feature_columns,
                median=median,
                scale=scale,
                context_races_per_step=effective_context_races_per_step,
                zeroed_features=zero_features,
                metrics=best_metrics,
                metrics_by_cohort=best_metrics_by_cohort,
            )
            print(
                f"NEW BEST epoch={best_epoch} "
                f"top3_recall={race_metrics['top3_recall']:.4f} "
                f"contained_top5={race_metrics['contained_top5_rate']:.4f} "
                f"contained_top4={race_metrics['contained_top4_rate']:.4f} "
                f"exact_top3_set={race_metrics['exact_top3_set_rate']:.4f} "
                f"auc={race_metrics['roc_auc']:.4f} "
                f"logloss={race_metrics['logloss']:.5f} "
                f"checkpoint={best_checkpoint_path}",
                flush=True,
            )
        else:
            epochs_without_improvement += 1
            if not guardrail_passed:
                print(
                    "STRESS GUARDRAIL REJECTED "
                    f"max_drop={args.stress_top3_recall_max_drop:.4f} "
                    f"best_observed_stress_recall={best_observed_stress_recall:.4f}",
                    flush=True,
                )
            print(
                f"no_improvement={epochs_without_improvement}/"
                f"{args.early_stopping_patience} best_epoch={best_epoch}",
                flush=True,
            )
            if (
                early_stopping_enabled
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(
                    f"EARLY STOP best_epoch={best_epoch} "
                    f"best_combined_top3_recall={best_metrics['top3_recall']:.4f} "
                    f"best_contained_top5={best_metrics['contained_top5_rate']:.4f} "
                    f"best_contained_top4={best_metrics['contained_top4_rate']:.4f} "
                    f"best_exact_top3_set={best_metrics['exact_top3_set_rate']:.4f} "
                    f"best_auc={best_metrics['roc_auc']:.4f}",
                    flush=True,
                )
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_mode = (
        "fine_tune_existing_output"
        if auto_resumed_output
        else "fine_tune_explicit_resume"
        if resume_model_path is not None
        else "from_scratch"
    )
    if args.classroom_overfit_all_races:
        training_filter = f"{TRAINING_ROWS_VIEW} (all complete races; overlaps validation)"
        validation_filter = f"{TRAINING_ROWS_VIEW} (all complete races; overlaps training)"
    else:
        training_filter = (
            TRAINING_ROWS_VIEW
            + (
                f" AND race_number >= {args.min_race_number}"
                if args.min_race_number is not None else ""
            )
        )
        validation_filter = VALIDATION_ROWS_VIEW
    checkpoint = {
            "model_state_dict": best_state,
            "model_kwargs": model_kwargs,
            "feature_columns": feature_columns,
            "median": median,
            "scale": scale,
            "partition_source": partition_source,
            "row_source": str(args.training_csv.resolve()),
            "validation_row_source": str(args.validation_csv.resolve()),
            "source_training_view": TRAINING_ROWS_VIEW,
            "source_validation_view": VALIDATION_ROWS_VIEW,
            "training_filter": training_filter,
            "optimizer_min_race_number": args.min_race_number,
            "validation_filter": validation_filter,
            "experiment_only": args.classroom_overfit_all_races,
            "production_eligible": not args.classroom_overfit_all_races,
            "classroom_overfit_all_races": args.classroom_overfit_all_races,
            "context_rows": None,
            "context_race_ids": [],
            "validation_context_race_ids": [],
            "context_sampling": (
                "most_recent_strictly_earlier_same_competition_races_per_query"
            ),
            "training_query_layout": "independent_padded_sequence_per_query_race",
            "validation_context_strategy": (
                "most_recent_strictly_earlier_same_competition_races"
            ),
            "validation_context_pool": "all_complete_partition_rows",
            "validation_context_races_per_prediction": effective_context_races_per_step,
            "context_manifest_race_ids": validation_context_race_ids,
            "context_manifest_role": "default_context_window_size_only",
            "context_races_per_step": effective_context_races_per_step,
            "requested_context_races_per_step": requested_context_races_per_step,
            "query_races_per_step": effective_query_races_per_step,
            "steps_per_epoch": effective_steps_per_epoch,
            "probe_race_ids": fixed_probe_race_ids.tolist(),
            "probe_every_steps": args.probe_every_steps,
            "training_warning_policy": {
                "context_warning_start_step": context_warning_start_step,
                "ineffective_context_probe_streak": CONTEXT_STOP_WARNING_PROBES,
                "permutation_minimum_probability_delta": 1e-4,
                "permutation_minimum_auc_change": 0.01,
                "permutation_minimum_logloss_change": 0.001,
                "fixed_probe_regression_warning_min_step": (
                    FIXED_PROBE_REGRESSION_WARNING_MIN_STEP
                ),
                "fixed_probe_stop_warning_streak": (
                    FIXED_PROBE_STOP_WARNING_PROBES
                ),
                "fixed_probe_minimum_top3_recall_drop": 0.10,
                "fixed_probe_minimum_auc_drop": 0.10,
            },
            "step_loss_window": args.step_loss_window,
            "requested_steps_per_epoch": args.steps_per_epoch,
            "auto_race_schedule": args.auto_race_schedule,
            "eligible_training_race_count": len(training_race_indices),
            "max_grad_norm": args.max_grad_norm,
            "learning_rate": learning_rate,
            "weight_decay": args.weight_decay,
            "training_scope": training_scope,
            "trainable_parameter_count": trainable_parameter_count,
            "loss_aggregation": "equal_mean_per_query_race",
            "skipped_invalid_training_race_ids": skipped_training_race_ids,
            "skipped_invalid_validation_race_ids": skipped_validation_race_ids,
            "excluded_context_validation_race_ids": context_validation_overlap,
            "zeroed_features": zero_features,
            "class_weights": class_weights.astype(np.float32),
            "fourier_frequency_init": "seeded_normal_logspace_0.1_to_10",
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "best_metrics_by_validation_cohort": best_metrics_by_cohort,
            "market_baseline_metrics": market_race_metrics,
            "market_baseline_metrics_by_validation_cohort": market_metrics_by_cohort,
            "market_price_coverage": market_price_coverage,
            "validation_cohort_source": validation_cohort_source,
            "validation_cohort_race_counts": validation_cohort_race_counts,
            "stress_top3_recall_max_drop": args.stress_top3_recall_max_drop,
            "allow_small_cohort_early_stopping": (
                args.allow_small_cohort_early_stopping
            ),
            "selection_metric": (
                "chronological_top3_recall_then_contained_top5_then_logloss_"
                "with_market_miss_stress_guardrail"
            ),
            "classification_loss_weight": args.classification_loss_weight,
            "label": "top3_mask",
            "race_context_mode": race_context_mode,
            "race_context_dim": race_context_dim,
            "race_context_layers": race_context_layers,
            "race_context_heads": race_context_heads,
            "race_context_ff_dim": race_context_ff_dim,
            "race_context_residual": race_context_residual,
            "context_prototype_branch": context_prototype_branch,
            "context_prototype_source": (
                "normalized_input_features" if context_prototype_branch else None
            ),
            "context_prototype_dim": context_prototype_dim,
            "context_prototype_input_dim": (
                len(feature_columns) if context_prototype_branch else None
            ),
            "context_prototype_max_correction": (
                context_prototype_max_correction
            ),
            "pairwise_loss_weight": args.pairwise_loss_weight,
            "attention_delta_pairwise_loss_weight": (
                args.attention_delta_pairwise_loss_weight
            ),
            "context_prototype_loss_weight": (
                args.context_prototype_loss_weight
            ),
            "cardinality_loss_weight": args.cardinality_loss_weight,
            "context_dependence_loss_weight": (
                args.context_dependence_loss_weight
            ),
            "context_dependence_margin": args.context_dependence_margin,
            "context_dependence_formulation": (
                "detached_correct_reference_margin_against_permuted_context"
            ),
            "output_semantics": "race_conditioned_uncalibrated_binary_top3_probability",
            "source_db": str(args.db.resolve()),
            "feature_manifest": str(feature_manifest_path.resolve()),
            "context_manifest": str(args.context_json.resolve()),
            "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
            "split_manifest_sha256": split_manifest.get("manifest_sha256") if split_manifest else None,
            "base_model": (
                str(resume_model_path.resolve())
                if resume_model_path is not None else None
            ),
            "training_mode": training_mode,
            "source_best_epoch": (
                resume_bundle.get("best_epoch") if resume_bundle is not None else None
            ),
        }
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary_output)
    temporary_output.replace(args.output)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model": "model.TabFM",
                "features": feature_columns,
                "label": "top3_mask",
                "race_context_mode": race_context_mode,
                "race_context_dim": race_context_dim,
                "race_context_layers": race_context_layers,
                "race_context_heads": race_context_heads,
                "race_context_ff_dim": race_context_ff_dim,
                "race_context_residual": race_context_residual,
                "context_prototype_branch": context_prototype_branch,
                "context_prototype_source": (
                    "normalized_input_features" if context_prototype_branch else None
                ),
                "context_prototype_dim": context_prototype_dim,
                "context_prototype_input_dim": (
                    len(feature_columns) if context_prototype_branch else None
                ),
                "context_prototype_max_correction": (
                    context_prototype_max_correction
                ),
                "pairwise_loss_weight": args.pairwise_loss_weight,
                "attention_delta_pairwise_loss_weight": (
                    args.attention_delta_pairwise_loss_weight
                ),
                "context_prototype_loss_weight": (
                    args.context_prototype_loss_weight
                ),
                "cardinality_loss_weight": args.cardinality_loss_weight,
                "context_dependence_loss_weight": (
                    args.context_dependence_loss_weight
                ),
                "context_dependence_margin": args.context_dependence_margin,
                "context_dependence_formulation": (
                    "detached_correct_reference_margin_against_permuted_context"
                ),
                "output_semantics": "race_conditioned_uncalibrated_binary_top3_probability",
                "partition_source": partition_source,
                "row_source": str(args.training_csv.resolve()),
                "validation_row_source": str(args.validation_csv.resolve()),
                "source_training_view": TRAINING_ROWS_VIEW,
                "source_validation_view": VALIDATION_ROWS_VIEW,
                "training_filter": training_filter,
                "optimizer_min_race_number": args.min_race_number,
                "validation_filter": validation_filter,
                "experiment_only": args.classroom_overfit_all_races,
                "production_eligible": not args.classroom_overfit_all_races,
                "classroom_overfit_all_races": args.classroom_overfit_all_races,
                "context_rows": None,
                "context_race_ids": [],
                "validation_context_race_ids": [],
                "validation_context_strategy": (
                    "most_recent_strictly_earlier_same_competition_races"
                ),
                "validation_context_pool": "all_complete_partition_rows",
                "validation_context_races_per_prediction": effective_context_races_per_step,
                "context_manifest_race_ids": validation_context_race_ids,
                "context_sampling": (
                    "most_recent_strictly_earlier_same_competition_races_per_query"
                ),
                "training_query_layout": "independent_padded_sequence_per_query_race",
                "context_races_per_step": effective_context_races_per_step,
                "requested_context_races_per_step": requested_context_races_per_step,
                "context_manifest_role": "default_context_window_size_only",
                "query_races_per_step": effective_query_races_per_step,
                "steps_per_epoch": effective_steps_per_epoch,
                "probe_race_ids": fixed_probe_race_ids.tolist(),
                "probe_every_steps": args.probe_every_steps,
                "training_warning_policy": {
                    "context_warning_start_step": context_warning_start_step,
                    "ineffective_context_probe_streak": CONTEXT_STOP_WARNING_PROBES,
                    "permutation_minimum_probability_delta": 1e-4,
                    "permutation_minimum_auc_change": 0.01,
                    "permutation_minimum_logloss_change": 0.001,
                    "fixed_probe_regression_warning_min_step": (
                        FIXED_PROBE_REGRESSION_WARNING_MIN_STEP
                    ),
                    "fixed_probe_stop_warning_streak": (
                        FIXED_PROBE_STOP_WARNING_PROBES
                    ),
                    "fixed_probe_minimum_top3_recall_drop": 0.10,
                    "fixed_probe_minimum_auc_drop": 0.10,
                },
                "step_loss_window": args.step_loss_window,
                "requested_steps_per_epoch": args.steps_per_epoch,
                "auto_race_schedule": args.auto_race_schedule,
                "eligible_training_race_count": len(training_race_indices),
                "max_grad_norm": args.max_grad_norm,
                "learning_rate": learning_rate,
                "weight_decay": args.weight_decay,
                "training_scope": training_scope,
                "trainable_parameter_count": trainable_parameter_count,
                "loss_aggregation": "equal_mean_per_query_race",
                "skipped_invalid_training_race_ids": skipped_training_race_ids,
                "skipped_invalid_validation_race_ids": skipped_validation_race_ids,
                "excluded_context_validation_race_ids": context_validation_overlap,
                "zeroed_features": zero_features,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "best_metrics_by_validation_cohort": best_metrics_by_cohort,
                "market_baseline_metrics": market_race_metrics,
                "market_baseline_metrics_by_validation_cohort": market_metrics_by_cohort,
                "market_price_coverage": market_price_coverage,
                "validation_cohort_source": validation_cohort_source,
                "validation_cohort_race_counts": validation_cohort_race_counts,
                "stress_top3_recall_max_drop": args.stress_top3_recall_max_drop,
                "allow_small_cohort_early_stopping": (
                    args.allow_small_cohort_early_stopping
                ),
                "selection_metric": (
                    "chronological_top3_recall_then_contained_top5_then_logloss_"
                    "with_market_miss_stress_guardrail"
                ),
                "classification_loss_weight": args.classification_loss_weight,
                "source_db": str(args.db.resolve()),
                "feature_manifest": str(feature_manifest_path.resolve()),
            "context_manifest": str(args.context_json.resolve()),
            "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
            "split_manifest_sha256": split_manifest.get("manifest_sha256") if split_manifest else None,
                "base_model": (
                    str(resume_model_path.resolve())
                    if resume_model_path is not None
                    else None
                ),
                "training_mode": training_mode,
                "source_best_epoch": (
                    resume_bundle.get("best_epoch")
                    if resume_bundle is not None else None
                ),
                "output": str(args.output.resolve()),
            },
            indent=2,
        ) + "\n"
    )
    print(
        f"selected_epoch={best_epoch} "
        f"combined_"
        f"contained_top5={best_metrics['contained_top5_rate']:.4f} "
        f"contained_top4={best_metrics['contained_top4_rate']:.4f} "
        f"exact_top3_set={best_metrics['exact_top3_set_rate']:.4f} "
        f"top3_recall={best_metrics['top3_recall']:.4f} "
        f"auc={best_metrics['roc_auc']:.4f} "
        f"logloss={best_metrics['logloss']:.5f}",
        flush=True,
    )
    for cohort, metrics in best_metrics_by_cohort.items():
        print("  selected_" + format_metric_line(cohort, metrics), flush=True)
    print(f"saved_model={args.output.resolve()}", flush=True)
    print(f"saved_metadata={metadata_path.resolve()}", flush=True)
    return 0
