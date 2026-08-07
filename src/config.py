"""Filesystem defaults for TabFM training."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "db/race_runners.sqlite"
DEFAULT_OUTPUT = ROOT / "outputs/tabfm_race_top3.pt"
DEFAULT_FEATURES = ROOT / "tabfm_features.json"
DEFAULT_CONTEXT = ROOT / "tabfm_context.json"
DEFAULT_TRAINING_CSV = ROOT / "outputs/training_records.csv"
DEFAULT_VALIDATION_CSV = ROOT / "outputs/validation_records.csv"
