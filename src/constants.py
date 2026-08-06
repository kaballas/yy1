"""Shared immutable training constants."""

TRAINING_ROWS_VIEW = "tabfm_trainable_validation_runners"
VALIDATION_ROWS_VIEW = "tabfm_validation_runners"
VALIDATION_COHORTS = {"chronological_representative", "market_miss_stress"}
MIN_CHECKPOINT_SELECTION_RACES = 20
CHECKPOINT_RECALL_TOLERANCE_RUNNERS = 2

# The effective context window defaults to the number of races in the supplied
# context manifest. Validation resolves actual context races chronologically at
# runtime; no historical/live race count is encoded here.
