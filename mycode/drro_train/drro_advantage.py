"""DRRO/DRO bonus logic and GRPO advantage patching.

Assignment modes:
- DRRO hard: compute score_i = r_i - delta * p_i, pick argmax_i(score_i), add +delta only to that winner.
- DRRO soft: use the same score_i = r_i - delta * p_i, then distribute bonus with a softmax/SNIS weighting.
- DRO soft: use delta * p_i for the SNIS target and subtract the resulting add-on from the reward.
"""

from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo import ray_trainer as verl_ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator


GLOBAL_FIXED_DELTA = 0.0
GLOBAL_DYNAMIC_DELTA_COEFF = 0.0
GLOBAL_DYNAMIC_KL_WINDOW = 20
GLOBAL_DYNAMIC_KL_ESTIMATOR = "k3"
GLOBAL_DYNAMIC_DELTA_MIN = 0.0
GLOBAL_DYNAMIC_DELTA_MAX = 0.0
GLOBAL_SOFT_ASSIGN_TAU = 2.0
GLOBAL_ASSIGN_MODE = "soft"
GLOBAL_ROBUST_OBJECTIVE = "drro"
GLOBAL_KL_HISTORY = deque(maxlen=20)
GLOBAL_LAST_DYNAMIC_KL = 0.0
GLOBAL_LAST_DYNAMIC_KL_MEAN = 0.0
GLOBAL_LAST_EFFECTIVE_DELTA = 0.0
ORIG_COMPUTE_ADV = verl_ray_trainer.compute_advantage


def set_fixed_delta(fixed_delta: float) -> None:
    global GLOBAL_FIXED_DELTA
    GLOBAL_FIXED_DELTA = fixed_delta


def set_soft_assign_tau(soft_assign_tau: float) -> None:
    global GLOBAL_SOFT_ASSIGN_TAU
    GLOBAL_SOFT_ASSIGN_TAU = max(float(soft_assign_tau), 1e-6)


def set_assign_mode(assign_mode: str) -> None:
    global GLOBAL_ASSIGN_MODE
    if assign_mode not in {"soft", "hard"}:
        raise ValueError(f"Unsupported assign_mode: {assign_mode}")
    GLOBAL_ASSIGN_MODE = assign_mode


def set_robust_objective(robust_objective: str) -> None:
    global GLOBAL_ROBUST_OBJECTIVE
    if robust_objective not in {"drro", "dro"}:
        raise ValueError(f"Unsupported robust_objective: {robust_objective}")
    GLOBAL_ROBUST_OBJECTIVE = robust_objective


def set_dynamic_delta_config(
    dynamic_delta_coeff: float,
    dynamic_kl_window: int,
    dynamic_kl_estimator: str = "k3",
    dynamic_delta_min: float = 0.0,
    dynamic_delta_max: float = 0.0,
) -> None:
    global GLOBAL_DYNAMIC_DELTA_COEFF, GLOBAL_DYNAMIC_KL_WINDOW, GLOBAL_DYNAMIC_KL_ESTIMATOR
    global GLOBAL_DYNAMIC_DELTA_MIN, GLOBAL_DYNAMIC_DELTA_MAX, GLOBAL_KL_HISTORY
    global GLOBAL_LAST_DYNAMIC_KL, GLOBAL_LAST_DYNAMIC_KL_MEAN, GLOBAL_LAST_EFFECTIVE_DELTA
    GLOBAL_DYNAMIC_DELTA_COEFF = float(dynamic_delta_coeff)
    GLOBAL_DYNAMIC_KL_WINDOW = max(int(dynamic_kl_window), 1)
    GLOBAL_DYNAMIC_KL_ESTIMATOR = dynamic_kl_estimator
    GLOBAL_DYNAMIC_DELTA_MIN = max(float(dynamic_delta_min), 0.0)
    GLOBAL_DYNAMIC_DELTA_MAX = max(float(dynamic_delta_max), 0.0)
    GLOBAL_KL_HISTORY = deque(maxlen=GLOBAL_DYNAMIC_KL_WINDOW)
    GLOBAL_LAST_DYNAMIC_KL = 0.0
    GLOBAL_LAST_DYNAMIC_KL_MEAN = 0.0
    GLOBAL_LAST_EFFECTIVE_DELTA = GLOBAL_FIXED_DELTA


def get_drro_delta_state() -> Dict[str, float]:
    return {
        "fixed_delta": float(GLOBAL_FIXED_DELTA),
        "dynamic_delta_coeff": float(GLOBAL_DYNAMIC_DELTA_COEFF),
        "effective_delta": float(GLOBAL_LAST_EFFECTIVE_DELTA),
        "dynamic_kl_last": float(GLOBAL_LAST_DYNAMIC_KL),
        "dynamic_kl_mean": float(GLOBAL_LAST_DYNAMIC_KL_MEAN),
        "dynamic_kl_window": float(GLOBAL_DYNAMIC_KL_WINDOW),
        "soft_assign_tau": float(GLOBAL_SOFT_ASSIGN_TAU),
        "assign_mode": GLOBAL_ASSIGN_MODE,
        "robust_objective": GLOBAL_ROBUST_OBJECTIVE,
        # Legacy aliases for existing log consumers.
        "delta_base": float(GLOBAL_FIXED_DELTA),
        "delta_alpha": float(GLOBAL_DYNAMIC_DELTA_COEFF),
        "delta_runtime": float(GLOBAL_LAST_EFFECTIVE_DELTA),
        "kl_est_last": float(GLOBAL_LAST_DYNAMIC_KL),
        "kl_est_window": float(GLOBAL_LAST_DYNAMIC_KL_MEAN),
        "delta_tau": float(GLOBAL_DYNAMIC_KL_WINDOW),
        "delta_softmax_tau": float(GLOBAL_SOFT_ASSIGN_TAU),
    }


def _estimate_kl_from_batch(data: DataProto, response_mask: torch.Tensor, old_log_probs: torch.Tensor) -> float | None:
    ref_log_prob = data.batch.get("ref_log_prob")
    if ref_log_prob is None:
        return None

    valid = response_mask > 0
    if not torch.any(valid):
        return None

    diff = (old_log_probs - ref_log_prob) * response_mask
    if GLOBAL_DYNAMIC_KL_ESTIMATOR == "k1":
        kl = diff[valid].mean()
    elif GLOBAL_DYNAMIC_KL_ESTIMATOR == "k2":
        kl = 0.5 * diff[valid].pow(2).mean()
    else:
        # Schulman k3 estimator for KL(q||p): (p/q - 1) - log(p/q),
        # here q=policy, p=reference, and diff=log(q/p).
        k3 = torch.exp(torch.clamp(-diff[valid], min=-20.0, max=20.0)) - 1.0 + diff[valid]
        kl = k3.mean()
    return float(kl.item())


def _resolve_effective_delta(
    data: DataProto,
    fixed_delta: float,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> float:
    global GLOBAL_LAST_DYNAMIC_KL, GLOBAL_LAST_DYNAMIC_KL_MEAN, GLOBAL_LAST_EFFECTIVE_DELTA
    if GLOBAL_DYNAMIC_DELTA_COEFF <= 0:
        GLOBAL_LAST_EFFECTIVE_DELTA = fixed_delta
        return fixed_delta

    kl_est = _estimate_kl_from_batch(data, response_mask, old_log_probs)
    if kl_est is None:
        GLOBAL_LAST_EFFECTIVE_DELTA = fixed_delta
        return fixed_delta

    GLOBAL_LAST_DYNAMIC_KL = kl_est
    GLOBAL_KL_HISTORY.append(kl_est)
    dynamic_kl_mean = float(np.mean(GLOBAL_KL_HISTORY))
    GLOBAL_LAST_DYNAMIC_KL_MEAN = dynamic_kl_mean

    effective_delta = GLOBAL_DYNAMIC_DELTA_COEFF * dynamic_kl_mean
    if GLOBAL_DYNAMIC_DELTA_MIN > 0:
        effective_delta = max(effective_delta, GLOBAL_DYNAMIC_DELTA_MIN)
    if GLOBAL_DYNAMIC_DELTA_MAX > 0:
        effective_delta = min(effective_delta, GLOBAL_DYNAMIC_DELTA_MAX)
    GLOBAL_LAST_EFFECTIVE_DELTA = float(effective_delta)
    return float(effective_delta)


def apply_drro_bonus(data: DataProto, fixed_delta: float) -> DataProto:
    if fixed_delta <= 0 and GLOBAL_DYNAMIC_DELTA_COEFF <= 0:
        return data
    if "old_log_probs" not in data.batch:
        raise ValueError("old_log_probs missing for DRRO delta computation.")
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = verl_ray_trainer.compute_response_mask(data)

    response_mask = data.batch["response_mask"].float()
    old_log_probs = data.batch["old_log_probs"]
    token_scores = data.batch["token_level_scores"]
    token_rewards = data.batch.get("token_level_rewards")

    lengths = response_mask.sum(dim=-1).clamp(min=1)
    effective_delta = _resolve_effective_delta(data, fixed_delta, response_mask, old_log_probs)
    if effective_delta <= 0:
        return data
    avg_logp = ((old_log_probs * response_mask).sum(dim=-1) / lengths).detach()
    scores = token_scores.sum(dim=-1).detach()

    uids = data.non_tensor_batch["uid"]
    if isinstance(uids, np.ndarray):
        uids = uids.tolist()

    delta_scores = torch.zeros_like(scores)
    groups: Dict[object, list[int]] = {}
    for idx, uid in enumerate(uids):
        groups.setdefault(uid, []).append(idx)

    temperature = max(float(GLOBAL_SOFT_ASSIGN_TAU), 1e-6)
    for idxs in groups.values():
        idx_tensor = torch.tensor(idxs, device=scores.device, dtype=torch.long)
        group_scores = scores[idx_tensor]
        group_logp = avg_logp[idx_tensor]
        proposal_q = torch.softmax(group_logp, dim=0).clamp(min=1e-12).detach()

        if GLOBAL_ROBUST_OBJECTIVE == "drro":
            target_scores = group_scores - effective_delta * proposal_q
            bonus_sign = 1.0
        else:
            # DRO smooths max_i delta * pi_i; its policy-gradient add-on has
            # the opposite sign from DRRO and does not include r_i in the target.
            target_scores = effective_delta * proposal_q
            bonus_sign = -1.0

        if GLOBAL_ASSIGN_MODE == "hard":
            winner = torch.argmax(target_scores).item()
            delta_scores[idx_tensor[winner]] += bonus_sign * effective_delta
        else:
            log_u = (target_scores / temperature) - torch.log(proposal_q)
            weights = torch.softmax(log_u, dim=0).detach()

            group_size = float(len(idxs))
            bonus = (bonus_sign * group_size * effective_delta) * (weights * proposal_q)
            bonus = torch.nan_to_num(bonus.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            delta_scores[idx_tensor] += bonus

    penalty = None
    if token_rewards is not None:
        penalty = token_rewards - token_scores

    token_scores = token_scores.clone()
    last_pos = (lengths.long().clamp(min=1) - 1).to(token_scores.device)
    token_scores[torch.arange(token_scores.size(0), device=token_scores.device), last_pos] += delta_scores
    data.batch["token_level_scores"] = token_scores

    if penalty is None:
        data.batch["token_level_rewards"] = token_scores
    else:
        data.batch["token_level_rewards"] = token_scores + penalty
    return data


def compute_advantage_drro(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
):
    if (GLOBAL_FIXED_DELTA > 0 or GLOBAL_DYNAMIC_DELTA_COEFF > 0) and adv_estimator == AdvantageEstimator.GRPO:
        data = apply_drro_bonus(data, GLOBAL_FIXED_DELTA)
    return ORIG_COMPUTE_ADV(
        data,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )


def enable_drro_grpo(
    fixed_delta: float,
    dynamic_delta_coeff: float = 0.0,
    dynamic_kl_window: int = 20,
    soft_assign_tau: float = 2.0,
    assign_mode: str = "soft",
    robust_objective: str = "drro",
    dynamic_kl_estimator: str = "k3",
    dynamic_delta_min: float = 0.0,
    dynamic_delta_max: float = 0.0,
) -> None:
    set_fixed_delta(fixed_delta)
    set_soft_assign_tau(soft_assign_tau)
    set_assign_mode(assign_mode)
    set_robust_objective(robust_objective)
    set_dynamic_delta_config(
        dynamic_delta_coeff=dynamic_delta_coeff,
        dynamic_kl_window=dynamic_kl_window,
        dynamic_kl_estimator=dynamic_kl_estimator,
        dynamic_delta_min=dynamic_delta_min,
        dynamic_delta_max=dynamic_delta_max,
    )
    verl_ray_trainer.compute_advantage = compute_advantage_drro
