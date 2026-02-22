"""Compatibility wrapper for DRRO path helpers.

Canonical implementation lives in mycode/drro_train/drro_paths.py.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_MYCODE_ROOT = os.path.join(_PROJECT_ROOT, "mycode")
if _MYCODE_ROOT not in sys.path:
    sys.path.insert(0, _MYCODE_ROOT)

from drro_train.drro_paths import *  # noqa: F401,F403
