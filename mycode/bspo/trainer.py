"""BSPO-specific trainer class with extra validation logging."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from baselines.common.trainer_common import BaselineRayPPOTrainer


class BSPORayPPOTrainer(BaselineRayPPOTrainer):
    def _extra_row_fields(
        self,
        proxy_extra_accum: Dict[str, List[float]],
        proxy_scores: List[float],
        gold_scores: List[float],
    ) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for key in ("unsupported_fraction", "mean_behavior_prob", "epsilon_beta"):
            values = proxy_extra_accum.get(key, [])
            if values:
                row[key] = float(np.mean(values))
        return row
