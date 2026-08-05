"""Application orchestration for TabFM training."""

from src.cli import parse_args
from src.training import run_training


def main() -> int:
    """Parse CLI arguments and execute the unchanged training workflow."""
    return run_training(parse_args())
