"""Ensemble reward manager for proxy-RM conservative aggregation."""

from __future__ import annotations

import os
import sys
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MYCODE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if MYCODE_ROOT not in sys.path:
    sys.path.insert(0, MYCODE_ROOT)

from baselines.common.reward_helpers import (
    build_reward_bundle,
    extract_prompt_response_pairs,
    make_device,
    resolve_dtype,
    scatter_sequence_scores_to_token_rewards,
    score_pairs_with_bundle,
)

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.abstract import AbstractRewardManager


class _RunningMemberNormalizer:
    """Online per-member z-score normalizer for ensemble reward scales."""

    def __init__(self, num_models: int, eps: float = 1e-6) -> None:
        self.num_models = int(num_models)
        self.eps = float(eps)
        self.count = 0
        self.mean = torch.zeros(self.num_models, dtype=torch.float32)
        self.m2 = torch.zeros(self.num_models, dtype=torch.float32)

    def update(self, scores: torch.Tensor) -> None:
        if scores.numel() == 0:
            return
        scores = scores.detach().float().cpu().reshape(-1, self.num_models)
        for row in scores:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / float(self.count)
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def active(self) -> bool:
        return self.count >= 2

    @property
    def std(self) -> torch.Tensor:
        if self.count < 2:
            return torch.ones_like(self.mean)
        var = self.m2 / float(self.count)
        return torch.sqrt(var.clamp_min(self.eps * self.eps))

    def transform(self, scores: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not self.active:
            return scores.float(), False
        mean = self.mean.to(scores.device)
        std = self.std.to(scores.device)
        return (scores.float() - mean) / std, True


class EnsembleRewardManager(AbstractRewardManager):
    """Reward manager that aggregates multiple proxy RMs (mean, WCO, UWO)."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        model_names: Optional[Sequence[str]] = None,
        aggregation: str = "uwo",
        uwo_lambda: float = 1.0,
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype | str] = None,
        calibrate_scores: bool = True,
        calibration_eps: float = 1e-6,
        allow_single_model: bool = False,
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.batch_size = batch_size
        self.max_length = max_length
        self.aggregation = aggregation.lower()
        self.uwo_lambda = float(uwo_lambda)
        self.calibrate_scores = bool(calibrate_scores)
        self.calibration_eps = float(calibration_eps)

        if self.aggregation not in {"mean", "wco", "uwo"}:
            raise ValueError("aggregation must be one of: mean, wco, uwo")
        if not model_names:
            raise ValueError("model_names must include at least one reward model.")

        self.device = make_device(device)
        model_dtype = resolve_dtype(dtype)
        if self.device.type != "cuda":
            model_dtype = torch.float32

        self.model_names = [str(name) for name in model_names]
        if len(self.model_names) < 2 and not allow_single_model:
            raise ValueError(
                "Ensemble baseline requires at least two proxy RMs. "
                "Pass multiple models with --proxy_rm_list/--proxy_rm_manifest, "
                "or set --allow_single_ensemble for a debugging-only single-model run."
            )
        self.bundles = [build_reward_bundle(name, model_dtype, self.device) for name in self.model_names]
        self.normalizer = _RunningMemberNormalizer(self.num_models, eps=self.calibration_eps)

    @property
    def num_models(self) -> int:
        return len(self.bundles)

    def _aggregate_scores(self, member_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # member_scores shape: [batch, num_models]
        ensemble_mean = member_scores.mean(dim=1)
        ensemble_var = member_scores.var(dim=1, unbiased=False)
        if self.aggregation == "mean":
            agg = ensemble_mean
        elif self.aggregation == "wco":
            agg = member_scores.min(dim=1).values
        else:  # uwo
            agg = ensemble_mean - self.uwo_lambda * ensemble_var
        return agg, ensemble_mean, ensemble_var

    def _score_pairs(self, prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
        per_model_scores: List[torch.Tensor] = []
        for bundle in self.bundles:
            scores = score_pairs_with_bundle(
                bundle=bundle,
                prompts=prompts,
                responses=responses,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
            )
            per_model_scores.append(scores)
        return torch.stack(per_model_scores, dim=1) if per_model_scores else torch.empty((0, 0))

    def _maybe_calibrate(self, raw_member_scores: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not self.calibrate_scores:
            return raw_member_scores.float(), False
        self.normalizer.update(raw_member_scores)
        return self.normalizer.transform(raw_member_scores)

    def _build_reward_output(
        self,
        data: DataProto,
        member_scores_raw: torch.Tensor,
        return_dict: bool,
    ):
        response_ids = data.batch["responses"]
        prompt_len = data.batch["prompts"].shape[-1]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)

        member_scores, calibrated = self._maybe_calibrate(member_scores_raw)
        agg_scores, ensemble_mean, ensemble_var = self._aggregate_scores(member_scores)
        raw_mean = member_scores_raw.mean(dim=1)
        raw_var = member_scores_raw.var(dim=1, unbiased=False)

        reward_tensor = scatter_sequence_scores_to_token_rewards(
            sequence_scores=agg_scores,
            response_ids=response_ids,
            valid_response_lengths=valid_response_lengths,
        )

        reward_extra_info: Dict[str, object] = {
            "proxy_score": agg_scores.tolist(),
            "ensemble_mean": ensemble_mean.tolist(),
            "ensemble_var": ensemble_var.tolist(),
            "ensemble_agg": agg_scores.tolist(),
            "ensemble_raw_mean": raw_mean.tolist(),
            "ensemble_raw_var": raw_var.tolist(),
            "ensemble_calibrated": [1.0 if calibrated else 0.0] * int(member_scores.shape[0]),
        }
        for idx in range(member_scores_raw.shape[1]):
            reward_extra_info[f"ensemble_member_{idx}"] = member_scores_raw[:, idx].tolist()
            reward_extra_info[f"ensemble_member_z_{idx}"] = member_scores[:, idx].tolist()

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    def __call__(self, data: DataProto, return_dict: bool = False):
        prompts, responses, response_ids, valid_response_lengths = extract_prompt_response_pairs(
            self.policy_tokenizer, data
        )
        del response_ids, valid_response_lengths

        member_scores_raw = self._score_pairs(prompts, responses)
        return self._build_reward_output(data, member_scores_raw, return_dict)


class LoopEnsembleRewardManager(RewardManagerBase):
    """Reward-loop compatible ensemble manager used during async rollouts."""

    _tokenizer_cache: ClassVar[Dict[str, object]] = {}
    _model_cache: ClassVar[Dict[Tuple[str, str, Optional[torch.dtype]], torch.nn.Module]] = {}

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        model_names: Optional[Sequence[str]] = None,
        aggregation: str = "uwo",
        uwo_lambda: float = 1.0,
        reward_batch_size: int = 8,
        reward_max_length: int = 512,
        dtype: str = "float32",
        device: Optional[str] = None,
        calibrate_scores: bool = True,
        calibration_eps: float = 1e-6,
        allow_single_model: bool = False,
        **_: object,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("reward_kwargs", {})
        model_names = model_names or reward_cfg.get("model_names", [])
        aggregation = str(aggregation or reward_cfg.get("aggregation", "uwo")).lower()
        uwo_lambda = float(uwo_lambda if uwo_lambda is not None else reward_cfg.get("uwo_lambda", 1.0))
        reward_batch_size = int(reward_batch_size or reward_cfg.get("reward_batch_size", 8))
        reward_max_length = int(reward_max_length or reward_cfg.get("reward_max_length", 512))
        dtype = dtype or reward_cfg.get("dtype", "float32")
        calibrate_scores = bool(reward_cfg.get("calibrate_scores", calibrate_scores))
        calibration_eps = float(reward_cfg.get("calibration_eps", calibration_eps))
        allow_single_model = bool(reward_cfg.get("allow_single_model", allow_single_model))

        self.delegate = EnsembleRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            model_names=list(model_names),
            aggregation=aggregation,
            uwo_lambda=uwo_lambda,
            batch_size=reward_batch_size,
            max_length=reward_max_length,
            device=device,
            dtype=dtype,
            calibrate_scores=calibrate_scores,
            calibration_eps=calibration_eps,
            allow_single_model=allow_single_model,
        )

    @staticmethod
    def _unwrap_single(value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "LoopEnsembleRewardManager only supports a single sample"
        out = self.delegate(data, return_dict=True)
        reward_score = float(out["reward_tensor"].sum(dim=-1)[0].item())
        reward_extra_info = {
            key: self._unwrap_single(value) for key, value in (out.get("reward_extra_info", {}) or {}).items()
        }
        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }
