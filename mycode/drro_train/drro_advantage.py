"""DRRO delta logic and GRPO advantage patching."""

from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np
import torch

from drro_paths import ensure_verl_on_path

VERL_ROOT = ensure_verl_on_path()
if VERL_ROOT is None:
    raise RuntimeError("Could not locate the VERL package. Set VERL_ROOT or place verl/ next to this script.")

from verl import DataProto
from verl.trainer.ppo import ray_trainer as verl_ray_trainer
from verl.trainer.ppo.core_algos import AdvantageEstimator


GLOBAL_DELTA = 0.0
GLOBAL_DELTA_ALPHA = 0.0
GLOBAL_DELTA_TAU = 20
GLOBAL_DELTA_KL_ESTIMATOR = "k3"
GLOBAL_DELTA_MIN = 0.0
GLOBAL_DELTA_MAX = 0.0
GLOBAL_SOFTMAX_TAU = 2.0
GLOBAL_KL_WINDOW = deque(maxlen=20)
GLOBAL_LAST_KL_EST = 0.0
GLOBAL_LAST_KL_WINDOW = 0.0
GLOBAL_LAST_DYNAMIC_DELTA = 0.0
ORIG_COMPUTE_ADV = verl_ray_trainer.compute_advantage


def set_drro_delta(delta: float) -> None:
    global GLOBAL_DELTA
    GLOBAL_DELTA = delta


def set_drro_softmax_tau(softmax_tau: float) -> None:
    global GLOBAL_SOFTMAX_TAU
    GLOBAL_SOFTMAX_TAU = max(float(softmax_tau), 1e-6)


def set_drro_dynamic_delta(
    alpha: float,
    tau: int,
    kl_estimator: str = "k3",
    delta_min: float = 0.0,
    delta_max: float = 0.0,
) -> None:
    global GLOBAL_DELTA_ALPHA, GLOBAL_DELTA_TAU, GLOBAL_DELTA_KL_ESTIMATOR
    global GLOBAL_DELTA_MIN, GLOBAL_DELTA_MAX, GLOBAL_KL_WINDOW
    global GLOBAL_LAST_KL_EST, GLOBAL_LAST_KL_WINDOW, GLOBAL_LAST_DYNAMIC_DELTA
    GLOBAL_DELTA_ALPHA = float(alpha)
    GLOBAL_DELTA_TAU = max(int(tau), 1)
    GLOBAL_DELTA_KL_ESTIMATOR = kl_estimator
    GLOBAL_DELTA_MIN = max(float(delta_min), 0.0)
    GLOBAL_DELTA_MAX = max(float(delta_max), 0.0)
    GLOBAL_KL_WINDOW = deque(maxlen=GLOBAL_DELTA_TAU)
    GLOBAL_LAST_KL_EST = 0.0
    GLOBAL_LAST_KL_WINDOW = 0.0
    GLOBAL_LAST_DYNAMIC_DELTA = GLOBAL_DELTA


def get_drro_delta_state() -> Dict[str, float]:
    return {
        "delta_base": float(GLOBAL_DELTA),
        "delta_alpha": float(GLOBAL_DELTA_ALPHA),
        "delta_runtime": float(GLOBAL_LAST_DYNAMIC_DELTA),
        "kl_est_last": float(GLOBAL_LAST_KL_EST),
        "kl_est_window": float(GLOBAL_LAST_KL_WINDOW),
        "delta_tau": float(GLOBAL_DELTA_TAU),
        "delta_softmax_tau": float(GLOBAL_SOFTMAX_TAU),
    }


def _estimate_kl_from_batch(data: DataProto, response_mask: torch.Tensor, old_log_probs: torch.Tensor) -> float | None:
    ref_log_prob = data.batch.get("ref_log_prob")
    if ref_log_prob is None:
        return None

    valid = response_mask > 0
    if not torch.any(valid):
        return None

    diff = (old_log_probs - ref_log_prob) * response_mask
    if GLOBAL_DELTA_KL_ESTIMATOR == "k1":
        kl = diff[valid].mean()
    elif GLOBAL_DELTA_KL_ESTIMATOR == "k2":
        kl = 0.5 * diff[valid].pow(2).mean()
    else:
        # Schulman k3 estimator for KL(q||p): (p/q - 1) - log(p/q),
        # here q=policy, p=reference, and diff=log(q/p).
        k3 = torch.exp(torch.clamp(-diff[valid], min=-20.0, max=20.0)) - 1.0 + diff[valid]
        kl = k3.mean()
    return float(kl.item())


def _resolve_dynamic_delta(data: DataProto, base_delta: float, response_mask: torch.Tensor, old_log_probs: torch.Tensor) -> float:
    global GLOBAL_LAST_KL_EST, GLOBAL_LAST_KL_WINDOW, GLOBAL_LAST_DYNAMIC_DELTA
    if GLOBAL_DELTA_ALPHA <= 0:
        GLOBAL_LAST_DYNAMIC_DELTA = base_delta
        return base_delta

    kl_est = _estimate_kl_from_batch(data, response_mask, old_log_probs)
    if kl_est is None:
        GLOBAL_LAST_DYNAMIC_DELTA = base_delta
        return base_delta

    GLOBAL_LAST_KL_EST = kl_est
    GLOBAL_KL_WINDOW.append(kl_est)
    kl_window = float(np.mean(GLOBAL_KL_WINDOW))
    GLOBAL_LAST_KL_WINDOW = kl_window

    delta = GLOBAL_DELTA_ALPHA * kl_window
    if GLOBAL_DELTA_MIN > 0:
        delta = max(delta, GLOBAL_DELTA_MIN)
    if GLOBAL_DELTA_MAX > 0:
        delta = min(delta, GLOBAL_DELTA_MAX)
    GLOBAL_LAST_DYNAMIC_DELTA = float(delta)
    return float(delta)


def apply_drro_delta(data: DataProto, delta: float) -> DataProto:
    if delta <= 0 and GLOBAL_DELTA_ALPHA <= 0:
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
    effective_delta = _resolve_dynamic_delta(data, delta, response_mask, old_log_probs)
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

    temperature = max(float(GLOBAL_SOFTMAX_TAU), 1e-6)
    for idxs in groups.values():
        idx_tensor = torch.tensor(idxs, device=scores.device, dtype=torch.long)
        group_scores = scores[idx_tensor]
        group_logp = avg_logp[idx_tensor]
        proposal_q = torch.softmax(group_logp, dim=0).clamp(min=1e-12).detach()

        # Softmax surrogate of max(r_i - delta * p_i), estimated via SNIS weights.
        soft_scores = group_scores - effective_delta * proposal_q
        log_u = (soft_scores / temperature) - torch.log(proposal_q)
        weights = torch.softmax(log_u, dim=0).detach()

        group_size = float(len(idxs))
        bonus = (group_size * effective_delta) * (weights * proposal_q)
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
    if (GLOBAL_DELTA > 0 or GLOBAL_DELTA_ALPHA > 0) and adv_estimator == AdvantageEstimator.GRPO:
        data = apply_drro_delta(data, GLOBAL_DELTA)
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
    delta: float,
    delta_alpha: float = 0.0,
    delta_tau: int = 20,
    delta_softmax_tau: float = 2.0,
    delta_kl_estimator: str = "k3",
    delta_min: float = 0.0,
    delta_max: float = 0.0,
) -> None:
    set_drro_delta(delta)
    set_drro_softmax_tau(delta_softmax_tau)
    set_drro_dynamic_delta(
        alpha=delta_alpha,
        tau=delta_tau,
        kl_estimator=delta_kl_estimator,
        delta_min=delta_min,
        delta_max=delta_max,
    )
    verl_ray_trainer.compute_advantage = compute_advantage_drro
