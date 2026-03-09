"""Path helpers for locating the local VERL package."""

from __future__ import annotations

import os
import sys
from typing import Optional

DEFAULT_VERL_ROOT = "/home/ykwang/projects/DRRO"


def ensure_verl_on_path() -> Optional[str]:
    candidates = []
    env_root = os.environ.get("VERL_ROOT")
    if env_root:
        candidates.append(env_root)

    project_root = os.path.dirname(os.path.abspath(__file__))
    candidates.append(project_root)
    candidates.append(os.path.dirname(project_root))
    candidates.append(DEFAULT_VERL_ROOT)

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
