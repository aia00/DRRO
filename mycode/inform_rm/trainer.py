"""InfoRM-specific trainer with extra validation diagnostics."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from baselines.common.trainer_common import BaselineRayPPOTrainer


class InfoRMBaselineTrainer(BaselineRayPPOTrainer):
    """Adds InfoRM latent diagnostics to validation logs."""

    def _extra_row_fields(
        self,
        proxy_extra_accum: Dict[str, List[float]],
        proxy_scores: List[float],
        gold_scores: List[float],
    ) -> Dict[str, float]:
        row: Dict[str, float] = {}
        latent_kl = proxy_extra_accum.get("latent_kl", [])
        latent_norm = proxy_extra_accum.get("latent_mu_norm", [])
        if latent_kl:
            row["latent_kl"] = float(np.mean(latent_kl))
        if latent_norm:
            row["latent_mu_norm"] = float(np.mean(latent_norm))
        return row
