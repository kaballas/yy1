"""TabFM optimiser configuration and behaviour-preserving training loop."""

from __future__ import annotations

import argparse
from collections import deque
import json
import random
import numpy as np
import torch
from src.checkpoint import resolve_resume_model_path
from src.constants import (
    MIN_CHECKPOINT_SELECTION_RACES,
    TRAINING_ROWS_VIEW,
    VALIDATION_ROWS_VIEW,
)
from src.context import context_race_id_changes, load_context_race_ids
from src.database import (load_context_rows, load_market_fluc2, load_race_number_eligible_ids, load_race_numbers, load_rows, load_validation_cohorts, print_race_selection_logic, validate_feature_columns)
from src.dataset import load_feature_columns
from src.losses import grouped_pairwise_loss, grouped_race_losses
from src.metrics import (checkpoint_selection_improves, cohort_checkpoint_selection, format_metric_line, pre_update_training_batch_metrics, probability_metrics, race_top3_metrics, select_fixed_probe_race_ids, stress_guardrail_passes, validation_metrics_by_cohort)
from src.model import TabFM
from src.prediction import market_rank_scores, predict
from src.preprocessing import fit_preprocessor, transform, zero_feature_columns
from src.progress import load_progress_race, print_progress_race
from src.sampling import build_query_race_schedule, sample_race_batch
from src.utilities import (
    initialize_fourier_frequencies,
    resolve_learning_rate,
    resolve_races_per_step,
    resolve_training_schedule,
)
from tabfm_split.runtime import load_and_validate_runtime_manifest
from src.validation import build_race_indices, exclude_invalid_races, validate_race_targets


def configure_trainable_parameters(
    model: TabFM, training_scope: str = "full_model", **legacy_kwargs: object
) -> tuple[list[torch.nn.Parameter], int, int]:
    """Configure and return optimizer parameters plus trainable/total counts."""
    if "attention_head_only" in legacy_kwargs:
        training_scope = (
            "attention_head_only" if legacy_kwargs["attention_head_only"] else "full_model"
        )
    if training_scope not in {"full_model", "attention_head_only", "icl_and_race_head"}:
        raise ValueError(f"Unknown training scope: {training_scope}")
    if training_scope != "full_model":
        if model.race_set_head is None:
            raise ValueError(
                "Partial race fine-tuning requires race_context_mode=self_attention"
            )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.race_set_head.parameters():
            parameter.requires_grad_(True)
        if training_scope == "icl_and_race_head":
            for parameter in model.icl_predictor.parameters():
                parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if not parameters:
        raise RuntimeError("No trainable model parameters are available")
    return parameters, trainable_count, total_count


def run_training(args: argparse.Namespace) -> int:
    """Execute the original training, evaluation, and checkpoint workflow."""
    if args.epochs < 1 or args.steps_per_epoch < 1:
        raise SystemExit("--epochs and --steps-per-epoch must be positive")
    if args.context_races_per_step < 1 or args.query_races_per_step < 1:
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
        args.pairwise_loss_weight < 0
        or args.attention_delta_pairwise_loss_weight < 0
        or args.cardinality_loss_weight < 0
    ):
        raise SystemExit("Race-level loss weights must be non-negative")
    if args.fine_tune_attention_head_only and args.fine_tune_scope not in (
        None, "attention_head_only"
    ):
        raise SystemExit(
            "--fine-tune-attention-head-only conflicts with --fine-tune-scope="
            f"{args.fine_tune_scope}"
        )
    training_scope = args.fine_tune_scope or (
        "attention_head_only" if args.fine_tune_attention_head_only else "full_model"
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
    resume_bundle = (
        torch.load(resume_model_path, map_location="cpu", weights_only=False)
        if resume_model_path is not None
        else None
    )
    if training_scope != "full_model" and resume_bundle is None:
        raise ValueError(
            f"--fine-tune-scope={training_scope} requires an existing output or "
            "--resume-model checkpoint; a frozen random backbone is invalid"
        )
    learning_rate = resolve_learning_rate(
        args.learning_rate, fine_tuning=resume_bundle is not None
    )
    
    if resume_bundle is not None and resume_bundle.get("label") != "top3_mask":
        raise ValueError(
            "Resume model was not trained with top3_mask and cannot be safely resumed"
        )
    source_model_kwargs = dict(resume_bundle.get("model_kwargs", {})) if resume_bundle else {}
    source_race_mode = source_model_kwargs.get("race_context_mode", "none")
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
    if training_scope != "full_model" and race_context_mode != "self_attention":
        raise ValueError(
            f"--fine-tune-scope={training_scope} requires race_context_mode=self_attention"
        )
    if args.attention_delta_pairwise_loss_weight > 0 and race_context_mode != "self_attention":
        raise ValueError(
            "--attention-delta-pairwise-loss-weight requires race_context_mode=self_attention"
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
    )
    requested_feature_columns = load_feature_columns(args.features_json)
    if resume_bundle is not None:
        feature_columns = list(resume_bundle["feature_columns"])
        if requested_feature_columns != feature_columns:
            raise ValueError(
                "Fine-tune feature manifest differs from source model: "
                f"requested_count={len(requested_feature_columns)} "
                f"checkpoint_count={len(feature_columns)}; restore the source "
                "feature manifest or use --overwrite-existing to train from scratch"
            )
    else:
        feature_columns = requested_feature_columns
    split_manifest = None
    if args.split_manifest is not None:
        split_manifest = load_and_validate_runtime_manifest(args.split_manifest, args.db)
        expected_context = {int(value) for value in split_manifest["fixed_context_race_ids"]}
        supplied_context = {int(value) for value in load_context_race_ids(args.context_json)}
        if supplied_context != expected_context:
            raise ValueError("context manifest does not match split-v2 fixed_context_race_ids")
    validation_context_race_ids = load_context_race_ids(args.context_json)
    if resume_bundle is not None:
        recorded_context_race_ids = resume_bundle.get(
            "validation_context_race_ids", resume_bundle.get("context_race_ids")
        )
        if recorded_context_race_ids is None:
            raise ValueError(
                "Resume model has no recorded validation context and cannot be "
                "safely resumed"
            )
        else:
            added, removed = context_race_id_changes(
                recorded_context_race_ids, validation_context_race_ids
            )
            if added or removed:
                print(
                    "fine_tune_context_transition "
                    f"added_count={len(added)} added_preview={added[:10]} "
                    f"removed_count={len(removed)} removed_preview={removed[:10]}"
                )
    zero_features = (
        list(resume_bundle.get("zeroed_features", []))
        if resume_bundle is not None and args.zero_features is None
        else list(args.zero_features or [])
    )
    if len(zero_features) != len(set(zero_features)):
        raise ValueError("--zero-features contains duplicates")
    validate_feature_columns(args.db, feature_columns)
    print_race_selection_logic(args.db, args.min_race_number)
    train_x, train_y_all, train_race_ids, train_times, train_validation_flags = (
        load_rows(args.db, feature_columns, TRAINING_ROWS_VIEW)
    )
    if args.classroom_overfit_all_races:
        x = train_x
        y = train_y_all
        race_ids = train_race_ids
        times = train_times
        validation_flags = train_validation_flags
        train_mask = np.ones(len(race_ids), dtype=bool)
        valid_mask = train_mask.copy()
    else:
        valid_x_all, valid_y_all, valid_race_ids_all, valid_times, valid_flags = (
            load_rows(args.db, feature_columns, VALIDATION_ROWS_VIEW)
        )
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
        validation_flags = np.concatenate(
            (train_validation_flags, valid_flags), axis=0
        )
        train_mask = np.arange(len(race_ids)) < len(train_race_ids)
        valid_mask = ~train_mask
    appended_context_fluc2 = np.empty(0, dtype=np.float32)
    if not args.classroom_overfit_all_races:
        (
            context_x_all,
            context_y_all,
            context_race_ids_all,
            context_times_all,
            context_flags_all,
            context_fluc2_all,
        ) = load_context_rows(
            args.db, feature_columns, validation_context_race_ids
        )
        context_append_mask = ~np.isin(context_race_ids_all, race_ids)
        if np.any(context_append_mask):
            append_count = int(np.sum(context_append_mask))
            appended_race_count = len(
                set(map(int, context_race_ids_all[context_append_mask]))
            )
            print(
                "validation_context_rows_loaded_outside_partition_views "
                f"races={appended_race_count} rows={append_count}",
                flush=True,
            )
            x = np.concatenate((x, context_x_all[context_append_mask]), axis=0)
            y = np.concatenate((y, context_y_all[context_append_mask]), axis=0)
            race_ids = np.concatenate(
                (race_ids, context_race_ids_all[context_append_mask]), axis=0
            )
            times = np.concatenate(
                (times, context_times_all[context_append_mask]), axis=0
            )
            validation_flags = np.concatenate(
                (validation_flags, context_flags_all[context_append_mask]), axis=0
            )
            train_mask = np.concatenate(
                (train_mask, np.zeros(append_count, dtype=bool)), axis=0
            )
            valid_mask = np.concatenate(
                (valid_mask, np.zeros(append_count, dtype=bool)), axis=0
            )
            appended_context_fluc2 = context_fluc2_all[context_append_mask]
    if args.fine_tune_race_id is not None:
        target_id = int(args.fine_tune_race_id)
        target_mask = race_ids == target_id
        if not np.any(target_mask):
            raise ValueError(f"fine-tune race {target_id} is not present in the database")
        if target_id in validation_context_race_ids:
            raise ValueError("fine-tune race cannot also be fixed context")
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
    context_validation_overlap = sorted(
        set(map(int, race_ids[valid_mask])).intersection(validation_context_race_ids)
    )
    if context_validation_overlap and not args.classroom_overfit_all_races:
        valid_mask &= ~np.isin(race_ids, context_validation_overlap)
        print(
            "validation_context_targets_excluded "
            f"count={len(context_validation_overlap)} "
            f"preview={context_validation_overlap[:10]}",
            flush=True,
        )
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

    train_market_fluc2 = load_market_fluc2(
        args.db, train_race_ids, TRAINING_ROWS_VIEW
    )
    market_fluc2 = (
        train_market_fluc2
        if args.classroom_overfit_all_races
        else np.concatenate(
            (
                train_market_fluc2,
                load_market_fluc2(
                    args.db, valid_race_ids_all, VALIDATION_ROWS_VIEW
                ),
                appended_context_fluc2,
            ),
            axis=0,
        )
    )
    valid_fluc2 = market_fluc2[valid_mask].copy()
    x = transform(x, median, scale)
    x = zero_feature_columns(x, feature_columns, zero_features)
    context_mask = np.isin(race_ids, validation_context_race_ids)
    resolved_context_races = set(int(race_id) for race_id in race_ids[context_mask])
    missing_context_races = sorted(
        set(validation_context_race_ids) - resolved_context_races
    )
    if missing_context_races:
        raise ValueError(
            f"Validation context races missing from {TRAINING_ROWS_VIEW}: "
            + ", ".join(map(str, missing_context_races))
        )
    if split_manifest is not None:
        if np.any(context_mask & (validation_flags == 1)):
            raise ValueError("split-v2 fixed context contains validation-flagged races")
        earliest_validation_time = min(times[valid_mask])
        if np.any(context_mask & (times >= earliest_validation_time)):
            raise ValueError("split-v2 fixed context is not before development races")
    #if np.any(context_mask & (validation_flags == 1)):
    #    raise ValueError(
    #        "Fixed context races must have is_validation = 0 so their labels are "
    #        "not included among validation targets"
    #    )
    #earliest_validation_time = min(times[valid_mask])
    #if np.any(context_mask & (times >= earliest_validation_time)):
    #    raise ValueError(
    #        "Every fixed training-context race must occur before the earliest "
    #        "selected validation race"
    #    )
    context_indices = np.flatnonzero(context_mask)
    context_order = sorted(
        context_indices,
        key=lambda index: (times[index], int(race_ids[index]), int(index)),
    )
    context_x = x[context_order]
    context_y = y[context_order]
    context_race_ids_for_predict = race_ids[context_order]
    if len(np.unique(context_y)) != 2:
        raise ValueError(
            "Fixed validation context must contain both top-three and "
            "non-top-three runners"
        )

    eligible_race_ids = load_race_number_eligible_ids(
        args.db, args.min_race_number
    )
    optimizer_eligible_mask = (
        np.ones(len(race_ids), dtype=bool)
        if args.min_race_number is None
        else np.isin(race_ids, list(eligible_race_ids))
    )
    training_pool_mask = train_mask & optimizer_eligible_mask
    if not args.classroom_overfit_all_races:
        training_pool_mask &= ~context_mask
    training_pool_mask, skipped_training_race_ids = exclude_invalid_races(
        y, race_ids, training_pool_mask, "Training pool"
    )
    validate_race_targets(y, race_ids, context_mask, "Validation context")
    training_race_indices = build_race_indices(race_ids, training_pool_mask)
    schedule_race_numbers = load_race_numbers(
        args.db,
        set(training_race_indices).union(validation_context_race_ids),
    )
    effective_steps_per_epoch, scheduled_query_races_per_step = (
        resolve_training_schedule(
            len(training_race_indices),
            args.steps_per_epoch,
            args.query_races_per_step,
            args.auto_race_schedule,
        )
    )
    context_size_matching_validation = min(
        args.context_races_per_step, len(validation_context_race_ids)
    )
    effective_context_races_per_step, effective_query_races_per_step = (
        resolve_races_per_step(
            len(training_race_indices),
            context_size_matching_validation,
            scheduled_query_races_per_step,
        )
    )
    if args.auto_race_schedule:
        print(
            "automatic_race_schedule "
            f"eligible_training_races={len(training_race_indices)} "
            f"steps_per_epoch={effective_steps_per_epoch} "
            f"query_races_per_step={effective_query_races_per_step} "
            f"query_slots={effective_steps_per_epoch * effective_query_races_per_step}",
            flush=True,
        )
    if effective_context_races_per_step != args.context_races_per_step:
        print(
            "training_context_size_adjusted "
            f"requested={args.context_races_per_step} "
            f"effective={effective_context_races_per_step} "
            f"validation_context_races={len(validation_context_race_ids)} "
            f"query_races={effective_query_races_per_step} "
            f"available_training_races={len(training_race_indices)}",
            flush=True,
        )
    train_y = y[training_pool_mask]
    valid_x, valid_y = x[valid_mask], y[valid_mask]
    valid_race_ids = race_ids[valid_mask]
    fixed_probe_race_ids = select_fixed_probe_race_ids(
        valid_y, valid_race_ids, args.probe_races
    )
    fixed_probe_mask = np.isin(valid_race_ids, fixed_probe_race_ids)
    fixed_probe_x = valid_x[fixed_probe_mask]
    fixed_probe_y = valid_y[fixed_probe_mask]
    fixed_probe_row_race_ids = valid_race_ids[fixed_probe_mask]
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
    if representative_races < MIN_CHECKPOINT_SELECTION_RACES:
        print(
            "WARNING checkpoint_selection_uses_combined "
            f"chronological_representative_races={representative_races} "
            f"minimum={MIN_CHECKPOINT_SELECTION_RACES}",
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
    context_rows = len(context_y)

    class_counts = np.bincount(train_y, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError(
            "Training races must contain both top-three and non-top-three runners"
        )
    class_weights = len(train_y) / (2.0 * class_counts)
    training_race_ids = set(training_race_indices)
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
        f"train_rows={len(train_y):,} train_races={len(training_race_indices):,} "
        f"valid_rows={len(valid_y):,} valid_races={len(selected_valid_races):,} "
        f"partition_source={partition_source} device={device}",
        flush=True,
    )
    print(
        f"train_top3_rate={train_y.mean():.4f} valid_top3_rate={valid_y.mean():.4f} "
        f"all_non_top3_valid_accuracy={1.0 - valid_y.mean():.4f} "
        f"class_weights=[{class_weights[0]:.3f}, {class_weights[1]:.3f}] "
        f"validation_context_races={len(validation_context_race_ids)} "
        f"validation_context_rows={context_rows} "
        f"context_races_per_step={effective_context_races_per_step} "
        f"query_races_per_step={effective_query_races_per_step} "
        f"auto_race_schedule={args.auto_race_schedule} "
        f"total_races_per_step="
        f"{effective_context_races_per_step + effective_query_races_per_step} "
        f"min_race_number={args.min_race_number} "
        f"race_context_mode={race_context_mode} "
        f"pairwise_loss_weight={args.pairwise_loss_weight} "
        f"attention_delta_pairwise_loss_weight="
        f"{args.attention_delta_pairwise_loss_weight} "
        f"cardinality_loss_weight={args.cardinality_loss_weight} "
        f"validation_cohort_source={validation_cohort_source} "
        f"validation_cohort_races={validation_cohort_race_counts} "
        f"zeroed_features={zero_features} progress_race_id={progress_race_id}",
        flush=True,
    )
    if resume_bundle is not None:
        model = TabFM(**model_kwargs).to(device)
        incompatible = model.load_state_dict(
            resume_bundle["model_state_dict"],
            strict=source_race_mode == race_context_mode,
        )
        if source_race_mode != race_context_mode:
            unexpected = list(incompatible.unexpected_keys)
            invalid_missing = [
                key for key in incompatible.missing_keys
                if not key.startswith("race_set_head.")
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
        ("Context races per step", str(effective_context_races_per_step)),
        ("Query races per step", str(effective_query_races_per_step)),
        ("Automatic race schedule", str(args.auto_race_schedule)),
        ("Print race schedule", str(args.print_race_schedule)),
        ("Eligible training races", f"{len(training_race_indices):,}"),
        (
            "Total races per step",
            str(effective_context_races_per_step + effective_query_races_per_step),
        ),
        ("Batch rows", "Variable; --batch-rows 256 is deprecated"),
        ("Pairwise loss weight", f"{args.pairwise_loss_weight:g}"),
        (
            "Attention-delta pairwise loss",
            f"{args.attention_delta_pairwise_loss_weight:g}",
        ),
        ("Cardinality loss weight", f"{args.cardinality_loss_weight:g}"),
        ("Stress recall maximum drop", f"{args.stress_top3_recall_max_drop:g}"),
        ("Early-stopping patience", f"{args.early_stopping_patience} epochs"),
        ("Race context mode", race_context_mode),
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

    def evaluate_fixed_probe(label: str, global_step: int) -> dict[str, float | int]:
        was_training = model.training
        model.eval()
        probe_probability = predict(
            model,
            context_x,
            context_y,
            fixed_probe_x,
            fixed_probe_row_race_ids,
            context_rows,
            device,
            context_race_ids=context_race_ids_for_predict,
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
        if was_training:
            model.train()
        return probe_metrics

    evaluate_fixed_probe("before_training", 0)
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
        baseline_probability = predict(
            model, context_x, context_y, valid_x, valid_race_ids, context_rows, device,
            context_race_ids=context_race_ids_for_predict,
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
        cardinality_losses = []
        context_row_counts = []
        query_row_counts = []
        batch_row_counts = []
        epoch_context_races: set[int] = set()
        epoch_query_races: set[int] = set()
        query_schedule = build_query_race_schedule(
            list(training_race_indices),
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
            fixed_context_display = ", ".join(
                f"{race_id}:{schedule_race_numbers.get(int(race_id), '?')}"
                for race_id in validation_context_race_ids
            )
            print(
                "FIXED VALIDATION CONTEXT "
                f"races={len(validation_context_race_ids)} "
                f"race_id:race_number=[{fixed_context_display}]",
                flush=True,
            )
        for step_index in range(effective_steps_per_epoch):
            (
                batch_x,
                batch_y,
                batch_context_rows,
                batch_context_race_ids,
                batch_query_race_ids,
                query_row_race_ids,
                batch_race_group_ids,
            ) = sample_race_batch(
                x,
                y,
                training_race_indices,
                effective_context_races_per_step,
                effective_query_races_per_step,
                rng,
                query_schedule[step_index],
                group_context_races=model.encode_races_before_icl,
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
            if context_race_set & query_race_set:
                raise RuntimeError("Context and query races overlap")
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
            batch_race_group_ids = batch_race_group_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            model_output = model(
                batch_x,
                batch_y,
                torch.tensor([batch_context_rows], device=device),
                race_group_ids=batch_race_group_ids,
                return_race_delta=args.attention_delta_pairwise_loss_weight > 0,
            )
            if args.attention_delta_pairwise_loss_weight > 0:
                logits, race_delta = model_output
            else:
                logits = model_output
                race_delta = None
            query_logits = logits[0, batch_context_rows:, :2]
            query_targets = batch_y[0, batch_context_rows:]
            query_row_race_ids_tensor = torch.from_numpy(query_row_race_ids).to(device)
            loss, classification_loss, pairwise_loss, cardinality_loss = grouped_race_losses(
                query_logits,
                query_targets,
                query_row_race_ids_tensor,
                loss_weights,
                args.pairwise_loss_weight,
                args.cardinality_loss_weight,
            )
            if race_delta is not None:
                attention_delta_pairwise_loss = grouped_pairwise_loss(
                    race_delta[0, batch_context_rows:, :2],
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
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss encountered before backward pass"
                )
            step_loss = float(loss.detach())
            step_race_metrics = pre_update_training_batch_metrics(
                query_targets.detach().cpu().numpy(),
                query_logits,
                query_row_race_ids,
            )
            step_loss_history.append(step_loss)
            global_step = (epoch - 1) * effective_steps_per_epoch + step_index + 1
            print(
                f"pre_update_training_batch epoch={epoch:02d} "
                f"step={step_index + 1}/{effective_steps_per_epoch} "
                f"global_step={global_step} "
                f"complete_races={step_race_metrics['complete_races']} "
                f"loss={step_loss:.5f} "
                f"rolling_loss={np.mean(step_loss_history):.5f} "
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
            optimizer.step()
            if global_step % args.probe_every_steps == 0:
                evaluate_fixed_probe("after_update", global_step)
            losses.append(float(loss.detach()))
            classification_losses.append(float(classification_loss.detach()))
            pairwise_losses.append(float(pairwise_loss.detach()))
            attention_delta_pairwise_losses.append(
                float(attention_delta_pairwise_loss.detach())
            )
            cardinality_losses.append(float(cardinality_loss.detach()))
            context_row_counts.append(batch_context_rows)
            query_row_counts.append(batch_y.shape[1] - batch_context_rows)
            batch_row_counts.append(batch_y.shape[1])
            epoch_context_races.update(context_race_set)
            epoch_query_races.update(query_race_set)

        model.eval()
        valid_prob = predict(
            model, context_x, context_y, valid_x, valid_race_ids, context_rows, device,
            context_race_ids=context_race_ids_for_predict,
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
            f"cardinality_loss={np.mean(cardinality_losses):.5f} "
            f"avg_context_rows={np.mean(context_row_counts):.1f} "
            f"avg_query_rows={np.mean(query_row_counts):.1f} "
            f"batch_rows_min={min(batch_row_counts)} "
            f"batch_rows_max={max(batch_row_counts)} "
            f"unique_context_races={len(epoch_context_races)} "
            f"unique_query_races={len(epoch_query_races)}",
            flush=True,
        )
        for cohort, metrics in metrics_by_cohort.items():
            print("  " + format_metric_line(cohort, metrics), flush=True)
        progress_probability = predict(
            model,
            context_x,
            context_y,
            progress_x,
            np.full(len(progress_x), progress_race_id, dtype=np.int64),
            context_rows,
            device,
            context_race_ids=context_race_ids_for_predict,
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
            print(
                f"NEW BEST epoch={best_epoch} "
                f"top3_recall={race_metrics['top3_recall']:.4f} "
                f"contained_top5={race_metrics['contained_top5_rate']:.4f} "
                f"contained_top4={race_metrics['contained_top4_rate']:.4f} "
                f"exact_top3_set={race_metrics['exact_top3_set_rate']:.4f} "
                f"auc={race_metrics['roc_auc']:.4f} "
                f"logloss={race_metrics['logloss']:.5f}",
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
            if epochs_without_improvement >= args.early_stopping_patience:
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
            "row_source": TRAINING_ROWS_VIEW,
            "validation_row_source": VALIDATION_ROWS_VIEW,
            "training_filter": training_filter,
            "optimizer_min_race_number": args.min_race_number,
            "validation_filter": validation_filter,
            "experiment_only": args.classroom_overfit_all_races,
            "production_eligible": not args.classroom_overfit_all_races,
            "classroom_overfit_all_races": args.classroom_overfit_all_races,
            "context_rows": context_rows,
            "context_race_ids": validation_context_race_ids,
            "validation_context_race_ids": validation_context_race_ids,
            "context_sampling": "random_complete_races_per_step",
            "context_races_per_step": effective_context_races_per_step,
            "requested_context_races_per_step": args.context_races_per_step,
            "query_races_per_step": effective_query_races_per_step,
            "steps_per_epoch": effective_steps_per_epoch,
            "probe_race_ids": fixed_probe_race_ids.tolist(),
            "probe_every_steps": args.probe_every_steps,
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
            "selection_metric": (
                "representative_when_at_least_20_races_else_combined_top3_recall_"
                "with_two_runner_tolerance_then_contained_top5_exact_top3_"
                "contained_top4_with_stress_guardrail"
            ),
            "label": "top3_mask",
            "race_context_mode": race_context_mode,
            "race_context_dim": race_context_dim,
            "race_context_layers": race_context_layers,
            "race_context_heads": race_context_heads,
            "race_context_ff_dim": race_context_ff_dim,
            "race_context_residual": race_context_residual,
            "pairwise_loss_weight": args.pairwise_loss_weight,
            "attention_delta_pairwise_loss_weight": (
                args.attention_delta_pairwise_loss_weight
            ),
            "cardinality_loss_weight": args.cardinality_loss_weight,
            "output_semantics": "race_conditioned_uncalibrated_top3_score",
            "source_db": str(args.db.resolve()),
            "feature_manifest": str(args.features_json.resolve()),
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
                "pairwise_loss_weight": args.pairwise_loss_weight,
                "attention_delta_pairwise_loss_weight": (
                    args.attention_delta_pairwise_loss_weight
                ),
                "cardinality_loss_weight": args.cardinality_loss_weight,
                "output_semantics": "race_conditioned_uncalibrated_top3_score",
                "partition_source": partition_source,
                "row_source": TRAINING_ROWS_VIEW,
                "validation_row_source": VALIDATION_ROWS_VIEW,
                "training_filter": training_filter,
                "optimizer_min_race_number": args.min_race_number,
                "validation_filter": validation_filter,
                "experiment_only": args.classroom_overfit_all_races,
                "production_eligible": not args.classroom_overfit_all_races,
                "classroom_overfit_all_races": args.classroom_overfit_all_races,
                "context_rows": context_rows,
                "context_race_ids": validation_context_race_ids,
                "validation_context_race_ids": validation_context_race_ids,
                "context_sampling": "random_complete_races_per_step",
                "context_races_per_step": effective_context_races_per_step,
                "requested_context_races_per_step": args.context_races_per_step,
                "query_races_per_step": effective_query_races_per_step,
                "steps_per_epoch": effective_steps_per_epoch,
                "probe_race_ids": fixed_probe_race_ids.tolist(),
                "probe_every_steps": args.probe_every_steps,
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
                "selection_metric": (
                    "representative_when_at_least_20_races_else_combined_top3_recall_"
                    "with_two_runner_tolerance_then_contained_top5_exact_top3_"
                    "contained_top4_with_stress_guardrail"
                ),
                "source_db": str(args.db.resolve()),
                "feature_manifest": str(args.features_json.resolve()),
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
