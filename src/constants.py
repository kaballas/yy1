"""Shared immutable training constants."""

TRAINING_ROWS_VIEW = "tabfm_trainable_validation_runners"
VALIDATION_ROWS_VIEW = "tabfm_validation_runners"
VALIDATION_COHORTS = {"chronological_representative", "market_miss_stress"}
MIN_CHECKPOINT_SELECTION_RACES = 20
CHECKPOINT_RECALL_TOLERANCE_RUNNERS = 2

# Native validation and prediction use the fixed context manifest (currently
# about 123 complete races).  Training samples a smaller, random subset per
# step to keep the sequence bounded, but it must remain in the same regime.
DEFAULT_CONTEXT_RACES_PER_STEP = 96
CONTEXT_RACES_PER_STEP_MIN = 64
CONTEXT_RACES_PER_STEP_MAX = 128
