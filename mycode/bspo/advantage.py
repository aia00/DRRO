"""Behavior-supported advantage computation for PPO/GAE.

This adapts the paper's critic-side regularization to the current VERL-based repo.
The implementation is intentionally lightweight:
- behavior support is estimated token-wise from ScoreLM next-token probabilities
- unsupported actions are assigned a pessimistic target value `unsupported_value`
- GAE bootstrapping stops at unsupported actions
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import ray_trainer as verl_ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.utils.torch_functional import masked_whiten


GLOBAL_UNSUPPORTED_VALUE = -15.0
GLOBAL_EPSILON_BETA = 1e-4
ORIG_COMPUTE_ADV = verl_ray_trainer.compute_advantage


def set_bspo_config(unsupported_value: float, epsilon_beta: float) -> None:
    global GLOBAL_UNSUPPORTED_VALUE, GLOBAL_EPSILON_BETA
    GLOBAL_UNSUPPORTED_VALUE = float(unsupported_value)
    GLOBAL_EPSILON_BETA = float(epsilon_beta)


def get_bspo_state() -> Dict[str, float]:
    return {
        "unsupported_value": float(GLOBAL_UNSUPPORTED_VALUE),
        "epsilon_beta": float(GLOBAL_EPSILON_BETA),
    }


def _load_behavior_supported_mask(data: DataProto, response_mask: torch.Tensor) -> Optional[torch.Tensor]:
    raw = data.non_tensor_batch.get("behavior_supported_mask")
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        tensor = torch.as_tensor(raw, dtype=torch.float32, device=response_mask.device)
    else:
        tensor = torch.tensor(raw, dtype=torch.float32, device=response_mask.device)
    if tensor.shape != response_mask.shape:
        return None
    return tensor * response_mask.float()


def compute_bspo_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    behavior_supported_mask: torch.Tensor,
    gamma: float,
    lam: float,
    unsupported_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        nextvalues = torch.zeros(token_level_rewards.size(0), device=token_level_rewards.device)
        lastgaelam = torch.zeros(token_level_rewards.size(0), device=token_level_rewards.device)
        advantages_reversed = []
        seq_len = token_level_rewards.shape[-1]
        unsupported_fill = torch.full_like(nextvalues, float(unsupported_value))

        for t in reversed(range(seq_len)):
            valid_t = response_mask[:, t] > 0
            supported_t = valid_t & (behavior_supported_mask[:, t] > 0.5)
            unsupported_t = valid_t & ~supported_t

            standard_delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            unsupported_delta = unsupported_fill - values[:, t]
            delta = torch.where(unsupported_t, unsupported_delta, standard_delta)

            supported_adv = delta + gamma * lam * lastgaelam
            current_adv = torch.where(unsupported_t, unsupported_delta, supported_adv)
            current_adv = torch.where(valid_t, current_adv, lastgaelam)
            advantages_reversed.append(current_adv)

            standard_next = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            nextvalues = torch.where(unsupported_t, unsupported_fill, standard_next)
            nextvalues = torch.where(valid_t, nextvalues, standard_next)
            lastgaelam = current_adv

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_advantage_bspo(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = verl_ray_trainer.compute_response_mask(data)

    if adv_estimator == AdvantageEstimator.GAE:
        supported_mask = _load_behavior_supported_mask(data, data.batch["response_mask"])
        if supported_mask is not None:
            advantages, returns = compute_bspo_gae_advantage_return(
                token_level_rewards=data.batch["token_level_rewards"],
                values=data.batch["values"],
                response_mask=data.batch["response_mask"],
                behavior_supported_mask=supported_mask,
                gamma=gamma,
                lam=lam,
                unsupported_value=GLOBAL_UNSUPPORTED_VALUE,
            )
            data.batch["advantages"] = advantages
            data.batch["returns"] = returns
            return data

    return ORIG_COMPUTE_ADV(
        data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )


def enable_bspo(unsupported_value: float = -15.0, epsilon_beta: float = 1e-4) -> None:
    set_bspo_config(unsupported_value=unsupported_value, epsilon_beta=epsilon_beta)
    verl_ray_trainer.compute_advantage = compute_advantage_bspo
