"""Filesystem defaults for TabFM training."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = "/home/theo/yy1/db/race_runners.sqlite"
DEFAULT_OUTPUT = ROOT / "outputs/tabfm_race_top3.pt"
DEFAULT_FEATURES = ROOT / "tabfm_features.json"
DEFAULT_CONTEXT = ROOT / "tabfm_context.json"
