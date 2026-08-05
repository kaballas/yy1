#!/usr/bin/env python3
"""Compatibility facade for the modular TabFM training package."""

from src.checkpoint import *  # noqa: F401,F403
from src.cli import parse_args
from src.config import *  # noqa: F401,F403
from src.constants import *  # noqa: F401,F403
from src.context import *  # noqa: F401,F403
from src.database import *  # noqa: F401,F403
from src.dataset import *  # noqa: F401,F403
from src.losses import *  # noqa: F401,F403
from src.main import main
from src.metrics import *  # noqa: F401,F403
from src.prediction import *  # noqa: F401,F403
from src.preprocessing import *  # noqa: F401,F403
from src.progress import *  # noqa: F401,F403
from src.sampling import *  # noqa: F401,F403
from src.training import configure_trainable_parameters, run_training
from src.utilities import *  # noqa: F401,F403
from src.validation import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
