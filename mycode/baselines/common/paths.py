"""Path helpers for baselines package."""

from __future__ import annotations

import os
import sys

BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_ROOT = os.path.dirname(BASELINES_DIR)
MYCODE_ROOT = os.path.dirname(BASELINES_ROOT)
REPO_ROOT = os.path.dirname(MYCODE_ROOT)
DRRO_TRAIN_DIR = os.path.join(MYCODE_ROOT, "drro_train")

for path in (MYCODE_ROOT, DRRO_TRAIN_DIR, REPO_ROOT):
    if path and path not in sys.path:
        sys.path.insert(0, path)

from drro_train.drro_paths import get_path_config as _get_path_config  # noqa: E402
from drro_train.drro_paths import get_verl_config_dir as _get_verl_config_dir  # noqa: E402

def get_path_config() -> dict[str, str]:
    """Load path config values from DRRO_PATH_CONFIG and environment."""
    return _get_path_config()


def get_verl_config_dir() -> str | None:
    """Return the installed VERL trainer config directory."""
    return _get_verl_config_dir()
