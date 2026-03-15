"""I/O helpers for baseline scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def parse_proxy_rm_list(text: str) -> List[str]:
    models = [item.strip() for item in text.split(",") if item.strip()]
    if not models:
        raise ValueError("proxy_rm_list is empty; provide comma-separated model names/paths.")
    return models


def load_proxy_manifest(path: str) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "models" in payload and isinstance(payload["models"], list):
            models = payload["models"]
        elif "members" in payload and isinstance(payload["members"], list):
            models = [m.get("path") for m in payload["members"] if isinstance(m, dict)]
        else:
            raise ValueError("manifest JSON must contain 'models' list or 'members' list.")
    elif isinstance(payload, list):
        models = payload
    else:
        raise ValueError("manifest JSON must be object or list.")

    out = [str(item).strip() for item in models if item]
    if not out:
        raise ValueError("No valid model paths found in manifest.")
    return out
