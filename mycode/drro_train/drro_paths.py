"""Path helpers for locating the local VERL package."""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional


def _load_path_env() -> Dict[str, str]:
    mycode_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.environ.get("DRRO_PATH_CONFIG", os.path.join(mycode_root, "project_paths.env"))
    values: Dict[str, str] = {}
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip("'\"")
    for key in (
        "DRRO_VERL_ROOT",
        "DRRO_LOCAL_DATASET_DIR",
        "DRRO_RAY_TMPDIR",
        "DRRO_OUTPUT_ROOT",
        "DRRO_PROXY_PAIRS_DIR",
    ):
        if os.environ.get(key) is not None:
            values[key] = os.environ[key]
    return values


def get_path_config() -> Dict[str, str]:
    """Load path-related config from DRRO_PATH_CONFIG and environment."""
    return _load_path_env()


def ensure_verl_on_path() -> Optional[str]:
    candidates = []
    path_cfg = _load_path_env()
    env_root = os.environ.get("VERL_ROOT")
    if env_root:
        candidates.append(env_root)
    cfg_verl_root = path_cfg.get("DRRO_VERL_ROOT")
    if cfg_verl_root:
        candidates.append(cfg_verl_root)

    project_root = os.path.dirname(os.path.abspath(__file__))
    candidates.append(project_root)
    candidates.append(os.path.dirname(project_root))
    candidates.append(os.path.dirname(os.path.dirname(project_root)))

    for root in candidates:
        if not root:
            continue
        # Direct package layout: root/verl/__init__.py
        pkg_dir = os.path.join(root, "verl")
        if os.path.isfile(os.path.join(pkg_dir, "__init__.py")):
            if root not in sys.path:
                sys.path.insert(0, root)
            return root
        # Repo layout: root/verl/verl/__init__.py
        nested_root = os.path.join(root, "verl")
        if os.path.isfile(os.path.join(nested_root, "verl", "__init__.py")):
            if nested_root not in sys.path:
                sys.path.insert(0, nested_root)
            return nested_root
    return None
