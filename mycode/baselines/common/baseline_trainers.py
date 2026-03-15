"""Baseline-specific trainer subclasses with extra logging fields."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from baselines.common.trainer_common import BaselineRayPPOTrainer


class EnsembleBaselineTrainer(BaselineRayPPOTrainer):
    """Adds ensemble mean/variance diagnostics to validation logs."""

    def _extra_row_fields(
        self,
        proxy_extra_accum: Dict[str, List[float]],
        proxy_scores: List[float],
        gold_scores: List[float],
    ) -> Dict[str, float]:
        row: Dict[str, float] = {}
        ensemble_mean = proxy_extra_accum.get("ensemble_mean", [])
        ensemble_var = proxy_extra_accum.get("ensemble_var", [])
        if ensemble_mean:
            row["ensemble_mean"] = float(np.mean(ensemble_mean))
        if ensemble_var:
            row["ensemble_var"] = float(np.mean(ensemble_var))
        return row


class ConstraintBaselineTrainer(BaselineRayPPOTrainer):
    """Adds constraint component/dual diagnostics to validation logs."""

    def _extra_row_fields(
        self,
        proxy_extra_accum: Dict[str, List[float]],
        proxy_scores: List[float],
        gold_scores: List[float],
    ) -> Dict[str, float]:
        row: Dict[str, float] = {}
        for key, values in proxy_extra_accum.items():
            if key.startswith("component_") and values:
                row[key] = float(np.mean(values))

        state = {}
        if hasattr(self.proxy_reward_fn, "get_constraint_state"):
            state = self.proxy_reward_fn.get_constraint_state()  # type: ignore[assignment]

        lambdas = state.get("lambda", {}) if isinstance(state, dict) else {}
        violations = state.get("violation", {}) if isinstance(state, dict) else {}
        for name, value in lambdas.items():
            row[f"lambda_{name}"] = float(value)
        for name, value in violations.items():
            row[f"violation_{name}"] = float(value)

        row["constraint_step"] = float(state.get("step", 0) if isinstance(state, dict) else 0)
        return row
