"""Component reward manager with Lagrangian constraint shaping (mu/xi PPO)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from baselines.common.paths import ensure_verl_on_path
from baselines.common.reward_helpers import (
    build_reward_bundle,
    extract_prompt_response_pairs,
    make_device,
    resolve_dtype,
    scatter_sequence_scores_to_token_rewards,
    score_pairs_with_bundle,
)

VERL_ROOT = ensure_verl_on_path()
if VERL_ROOT is None:
    raise RuntimeError("Could not locate VERL package. Set VERL_ROOT or place verl/ next to this repo.")

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


@dataclass
class ComponentSpec:
    name: str
    model: str


def load_component_config(path: str) -> tuple[List[ComponentSpec], str]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    objective_name = str(payload.get("objective_name", "")).strip()
    components_raw = payload.get("components")
    if not isinstance(components_raw, list) or not components_raw:
        raise ValueError("component_config_json must contain non-empty 'components' list.")

    components: List[ComponentSpec] = []
    for item in components_raw:
        if not isinstance(item, dict):
            raise ValueError("Each component entry must be an object with name/model fields.")
        name = str(item.get("name", "")).strip()
        model = str(item.get("model", item.get("path", ""))).strip()
        if not name or not model:
            raise ValueError("Each component must provide non-empty name and model/path.")
        components.append(ComponentSpec(name=name, model=model))

    if not objective_name:
        objective_name = components[0].name
    names = [spec.name for spec in components]
    if objective_name not in names:
        raise ValueError(f"objective_name '{objective_name}' not found in component names: {names}")

    return components, objective_name


def load_theta_json(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("theta_json must be a JSON object mapping component names to threshold values.")
    theta: Dict[str, float] = {}
    for key, value in payload.items():
        theta[str(key)] = float(value)
    return theta


class ConstraintRewardManager(AbstractRewardManager):
    """Reward manager for multi-component constraints with dual updates."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        components: Optional[Sequence[ComponentSpec]] = None,
        objective_name: str = "",
        theta: Optional[Dict[str, float]] = None,
        constraint_mode: str = "mu",
        dual_lr: float = 0.05,
        dual_ema: float = 0.9,
        dual_clip: float = 10.0,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype | str] = None,
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.batch_size = batch_size
        self.max_length = max_length

        if not components:
            raise ValueError("components must include at least one component reward model.")

        self.components = list(components)
        self.objective_name = str(objective_name)
        self.theta = dict(theta or {})
        self.constraint_mode = str(constraint_mode).lower().strip()
        if self.constraint_mode not in {"mu", "xi"}:
            raise ValueError("constraint_mode must be one of: mu, xi")

        self.dual_lr = float(dual_lr)
        self.dual_ema = float(dual_ema)
        self.dual_clip = float(dual_clip)
        if self.dual_ema < 0 or self.dual_ema >= 1:
            raise ValueError("dual_ema must be in [0, 1).")

        self.device = make_device(device)
        model_dtype = resolve_dtype(dtype)
        if self.device.type != "cuda":
            model_dtype = torch.float32

        self.bundles = {
            spec.name: build_reward_bundle(spec.model, model_dtype, self.device) for spec in self.components
        }
        if self.objective_name not in self.bundles:
            raise ValueError(f"objective_name '{self.objective_name}' not found in components.")

        self.constrained_names = [name for name in self.bundles if name != self.objective_name]
        for name in self.constrained_names:
            if name not in self.theta:
                raise ValueError(
                    f"theta_json missing threshold for constrained component '{name}'. "
                    "Provide theta for all non-objective components."
                )

        self.lambdas = {name: 0.0 for name in self.constrained_names}
        self.ema_means = {name: None for name in self.constrained_names}
        self.last_violations = {name: 0.0 for name in self.constrained_names}
        self.dual_step = 0

    def get_constraint_state(self) -> Dict[str, Dict[str, float] | int | str]:
        return {
            "mode": self.constraint_mode,
            "step": self.dual_step,
            "lambda": {k: float(v) for k, v in self.lambdas.items()},
            "violation": {k: float(v) for k, v in self.last_violations.items()},
            "ema_mean": {k: (None if v is None else float(v)) for k, v in self.ema_means.items()},
            "theta": {k: float(v) for k, v in self.theta.items()},
        }

    def eval_violation_from_mean(self, component_name: str, mean_value: float) -> float:
        theta = float(self.theta[component_name])
        if self.constraint_mode == "mu":
            return theta - float(mean_value)
        return float(mean_value) - theta

    def _score_components(self, prompts: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        scores: Dict[str, torch.Tensor] = {}
        for name, bundle in self.bundles.items():
            scores[name] = score_pairs_with_bundle(
                bundle=bundle,
                prompts=prompts,
                responses=responses,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
            )
        return scores

    def _update_duals(self, component_scores: Dict[str, torch.Tensor]) -> None:
        if not self.constrained_names:
            return

        for name in self.constrained_names:
            batch_mean = float(component_scores[name].mean().item())
            ema_prev = self.ema_means[name]
            if ema_prev is None:
                ema_value = batch_mean
            else:
                ema_value = self.dual_ema * float(ema_prev) + (1.0 - self.dual_ema) * batch_mean
            self.ema_means[name] = ema_value

            if self.constraint_mode == "mu":
                violation = float(self.theta[name]) - ema_value
                new_lambda = max(0.0, float(self.lambdas[name]) + self.dual_lr * violation)
                if self.dual_clip > 0:
                    new_lambda = min(new_lambda, self.dual_clip)
            else:  # xi
                violation = ema_value - float(self.theta[name])
                new_lambda = float(self.lambdas[name]) + self.dual_lr * violation
                if self.dual_clip > 0:
                    new_lambda = max(-self.dual_clip, min(new_lambda, self.dual_clip))

            self.lambdas[name] = new_lambda
            self.last_violations[name] = violation

        self.dual_step += 1

    def _compute_shaped_reward(self, component_scores: Dict[str, torch.Tensor]) -> torch.Tensor:
        objective_scores = component_scores[self.objective_name]
        shaped = objective_scores.clone()
        for name in self.constrained_names:
            centered = component_scores[name] - float(self.theta[name])
            shaped = shaped + float(self.lambdas[name]) * centered
        return shaped

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        prompts, responses, response_ids, valid_response_lengths = extract_prompt_response_pairs(
            self.policy_tokenizer, data
        )
        component_scores = self._score_components(prompts=prompts, responses=responses)

        validate_mode = bool(data.meta_info.get("validate", False))
        if not validate_mode:
            self._update_duals(component_scores)

        objective_scores = component_scores[self.objective_name]
        shaped_scores = self._compute_shaped_reward(component_scores)

        reward_tensor = scatter_sequence_scores_to_token_rewards(
            sequence_scores=shaped_scores,
            response_ids=response_ids,
            valid_response_lengths=valid_response_lengths,
        )

        reward_extra_info: Dict[str, object] = {
            "proxy_score": objective_scores.tolist(),
            "objective_score": objective_scores.tolist(),
            "shaped_score": shaped_scores.tolist(),
            "constraint_mode": self.constraint_mode,
        }
        for name, values in component_scores.items():
            reward_extra_info[f"component_{name}"] = values.tolist()
        for name in self.constrained_names:
            reward_extra_info[f"lambda_{name}"] = float(self.lambdas[name])
            reward_extra_info[f"violation_{name}"] = float(self.last_violations[name])

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
