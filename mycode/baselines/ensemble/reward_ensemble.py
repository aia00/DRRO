"""Ensemble reward manager for proxy-RM conservative aggregation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from baselines.common.reward_helpers import (
    build_reward_bundle,
    extract_prompt_response_pairs,
    make_device,
    resolve_dtype,
    scatter_sequence_scores_to_token_rewards,
    score_pairs_with_bundle,
)

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


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
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.batch_size = batch_size
        self.max_length = max_length
        self.aggregation = aggregation.lower()
        self.uwo_lambda = float(uwo_lambda)

        if self.aggregation not in {"mean", "wco", "uwo"}:
            raise ValueError("aggregation must be one of: mean, wco, uwo")
        if not model_names:
            raise ValueError("model_names must include at least one reward model.")

        self.device = make_device(device)
        model_dtype = resolve_dtype(dtype)
        if self.device.type != "cuda":
            model_dtype = torch.float32

        self.model_names = [str(name) for name in model_names]
        self.bundles = [build_reward_bundle(name, model_dtype, self.device) for name in self.model_names]

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

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        prompts, responses, response_ids, valid_response_lengths = extract_prompt_response_pairs(
            self.policy_tokenizer, data
        )

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

        member_scores = torch.stack(per_model_scores, dim=1) if per_model_scores else torch.empty((0, 0))
        agg_scores, ensemble_mean, ensemble_var = self._aggregate_scores(member_scores)

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
        }
        for idx in range(member_scores.shape[1]):
            reward_extra_info[f"ensemble_member_{idx}"] = member_scores[:, idx].tolist()

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
