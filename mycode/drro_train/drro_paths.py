"""Shared path/config helpers."""

from __future__ import annotations

import importlib.util
import os
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
    for key in ("DRRO_LOCAL_DATASET_DIR", "DRRO_RAY_TMPDIR", "DRRO_OUTPUT_ROOT", "DRRO_PROXY_PAIRS_DIR"):
        if os.environ.get(key) is not None:
            values[key] = os.environ[key]
    return values


def get_path_config() -> Dict[str, str]:
    """Load path-related config from DRRO_PATH_CONFIG and environment."""
    return _load_path_env()


def _installed_verl_paths() -> tuple[Optional[str], Optional[str]]:
    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        return None, None
    pkg_dir = os.path.dirname(os.path.abspath(spec.origin))
    import_root = os.path.dirname(pkg_dir)
    return import_root, pkg_dir


def get_verl_package_dir() -> Optional[str]:
    _, pkg_dir = _installed_verl_paths()
    return pkg_dir


def get_verl_config_dir() -> Optional[str]:
    pkg_dir = get_verl_package_dir()
    if pkg_dir is None:
        return None
    config_dir = os.path.join(pkg_dir, "trainer", "config")
    if os.path.isdir(config_dir):
        return config_dir
    return None
