"""InfoRM reward manager for downstream policy training."""

from __future__ import annotations

import os
import sys
from typing import ClassVar, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.abstract import AbstractRewardManager

from baselines.common.reward_helpers import (
    extract_prompt_response_pairs,
    scatter_sequence_scores_to_token_rewards,
)
from drro_train.drro_reward import HFRewardManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from .modeling_inform_rm import InfoRMModel
except ImportError:
    from modeling_inform_rm import InfoRMModel


class InfoRMRewardManager(AbstractRewardManager):
    """Scores prompt/response pairs with a trained InfoRM checkpoint."""

    _cache: ClassVar[Dict[Tuple[str, str, Optional[torch.dtype]], Tuple[InfoRMModel, AutoTokenizer]]] = {}

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        inform_rm_path: str = "",
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.inform_rm_path = inform_rm_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            dtype = torch.float32

        cache_key = (inform_rm_path, str(self.device), dtype)
        if cache_key not in self._cache:
            model, inform_tokenizer = InfoRMModel.from_pretrained(inform_rm_path, device=self.device, dtype=dtype)
            if inform_tokenizer.pad_token is None:
                inform_tokenizer.pad_token = inform_tokenizer.eos_token or inform_tokenizer.unk_token
            self._cache[cache_key] = (model, inform_tokenizer)
        self.inform_rm, self.inform_tokenizer = self._cache[cache_key]

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        prompts, responses, response_ids, valid_response_lengths = extract_prompt_response_pairs(
            self.policy_tokenizer,
            data,
        )

        seq_scores: List[torch.Tensor] = []
        latent_kls: List[torch.Tensor] = []
        latent_norms: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                end = min(start + self.batch_size, len(prompts))
                enc = self.inform_tokenizer(
                    prompts[start:end],
                    responses[start:end],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {key: value.to(self.device) for key, value in enc.items()}
                out = self.inform_rm.score_batch(enc, sample_latent=False)
                seq_scores.append(out["reward"].float().cpu())
                latent_kls.append(self.inform_rm._kl_to_standard_normal(out["mu"], out["std"]).float().cpu())
                latent_norms.append(out["mu"].float().norm(dim=-1).cpu())

        sequence_scores = torch.cat(seq_scores, dim=0) if seq_scores else torch.empty((0,), dtype=torch.float32)
        latent_kl = torch.cat(latent_kls, dim=0) if latent_kls else torch.empty((0,), dtype=torch.float32)
        latent_norm = torch.cat(latent_norms, dim=0) if latent_norms else torch.empty((0,), dtype=torch.float32)

        reward_tensor = scatter_sequence_scores_to_token_rewards(
            sequence_scores=sequence_scores,
            response_ids=response_ids,
            valid_response_lengths=valid_response_lengths,
        )

        reward_extra_info = {
            "proxy_score": sequence_scores.tolist(),
            "latent_kl": latent_kl.tolist(),
            "latent_mu_norm": latent_norm.tolist(),
        }
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


class ProxyInfoRMPenaltyRewardManager(AbstractRewardManager):
    """Use the shared proxy RM as reward and InfoRM as an uncertainty penalty.

    The optimized reward is:

        shaped_reward = proxy_rm_score - inform_penalty_coef * latent_kl

    Validation logs keep ``proxy_score`` as the raw proxy RM score for fair
    plotting against PPO/GRPO/DRRO, while ``shaped_score`` is the actual
    training signal.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        proxy_model_name: str = "",
        inform_rm_path: str = "",
        reward_batch_size: int = 8,
        reward_max_length: int = 512,
        inform_max_length: int = 512,
        inform_penalty_coef: float = 0.01,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.inform_penalty_coef = float(inform_penalty_coef)
        self.proxy_reward = HFRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            model_name=proxy_model_name,
            batch_size=reward_batch_size,
            max_length=reward_max_length,
            device=device,
            dtype=dtype,
            data_parallel=False,
        )
        self.inform_reward = InfoRMRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            inform_rm_path=inform_rm_path,
            batch_size=reward_batch_size,
            max_length=inform_max_length,
            device=device,
            dtype=dtype,
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        proxy_out = self.proxy_reward(data, return_dict=True)
        inform_out = self.inform_reward(data, return_dict=True)

        proxy_scores = proxy_out["reward_tensor"].sum(dim=-1).detach().cpu()
        inform_extra = dict(inform_out.get("reward_extra_info", {}) or {})
        latent_kl = torch.as_tensor(
            inform_extra.get("latent_kl", [0.0] * int(proxy_scores.numel())),
            dtype=torch.float32,
        )
        inform_scores = inform_out["reward_tensor"].sum(dim=-1).detach().cpu()
        shaped_scores = proxy_scores - self.inform_penalty_coef * latent_kl

        response_ids = data.batch["responses"]
        prompt_len = data.batch["prompts"].shape[-1]
        valid_response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
        reward_tensor = scatter_sequence_scores_to_token_rewards(
            sequence_scores=shaped_scores,
            response_ids=response_ids,
            valid_response_lengths=valid_response_lengths,
        )

        reward_extra_info = {
            "proxy_score": proxy_scores.tolist(),
            "proxy_rm_score": proxy_scores.tolist(),
            "inform_score": inform_scores.tolist(),
            "shaped_score": shaped_scores.tolist(),
            "latent_kl": latent_kl.tolist(),
            "latent_mu_norm": inform_extra.get("latent_mu_norm", []),
            "inform_penalty": (self.inform_penalty_coef * latent_kl).tolist(),
            "inform_penalty_coef": [self.inform_penalty_coef] * int(proxy_scores.numel()),
        }
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


class LoopInfoRMRewardManager(RewardManagerBase):
    """Reward-loop compatible wrapper around InfoRMRewardManager."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        inform_rm_path: str = "",
        reward_batch_size: int = 8,
        inform_max_length: int = 512,
        dtype: str = "float32",
        device: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("reward_kwargs", {})
        inform_rm_path = inform_rm_path or reward_cfg.get("inform_rm_path", "")
        reward_batch_size = int(reward_batch_size or reward_cfg.get("reward_batch_size", 8))
        inform_max_length = int(inform_max_length or reward_cfg.get("inform_max_length", 512))
        dtype_name = dtype or reward_cfg.get("dtype", "float32")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(dtype_name, torch.float32)

        self.delegate = InfoRMRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            inform_rm_path=inform_rm_path,
            batch_size=reward_batch_size,
            max_length=inform_max_length,
            device=device,
            dtype=reward_dtype,
        )

    @staticmethod
    def _unwrap_single(value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "LoopInfoRMRewardManager only supports a single sample"
        out = self.delegate(data, return_dict=True)
        reward_tensor = out["reward_tensor"]
        reward_score = float(reward_tensor.sum(dim=-1)[0].item())
        reward_extra_info = {
            key: self._unwrap_single(value) for key, value in (out.get("reward_extra_info", {}) or {}).items()
        }
        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }


class LoopProxyInfoRMPenaltyRewardManager(RewardManagerBase):
    """Reward-loop wrapper for proxy RM reward plus InfoRM latent-KL penalty."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        proxy_model_name: str = "",
        inform_rm_path: str = "",
        reward_batch_size: int = 8,
        reward_max_length: int = 512,
        inform_max_length: int = 512,
        inform_penalty_coef: float = 0.01,
        dtype: str = "float32",
        device: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("reward_kwargs", {})
        proxy_model_name = proxy_model_name or reward_cfg.get("proxy_model", reward_cfg.get("model_name", ""))
        inform_rm_path = inform_rm_path or reward_cfg.get("inform_rm_path", "")
        reward_batch_size = int(reward_batch_size or reward_cfg.get("reward_batch_size", 8))
        reward_max_length = int(reward_max_length or reward_cfg.get("reward_max_length", 512))
        inform_max_length = int(inform_max_length or reward_cfg.get("inform_max_length", 512))
        inform_penalty_coef = float(inform_penalty_coef or reward_cfg.get("inform_penalty_coef", 0.01))
        dtype_name = dtype or reward_cfg.get("dtype", "float32")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(dtype_name, torch.float32)

        self.delegate = ProxyInfoRMPenaltyRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            proxy_model_name=proxy_model_name,
            inform_rm_path=inform_rm_path,
            reward_batch_size=reward_batch_size,
            reward_max_length=reward_max_length,
            inform_max_length=inform_max_length,
            inform_penalty_coef=inform_penalty_coef,
            device=device,
            dtype=reward_dtype,
        )

    @staticmethod
    def _unwrap_single(value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "LoopProxyInfoRMPenaltyRewardManager only supports a single sample"
        out = self.delegate(data, return_dict=True)
        reward_tensor = out["reward_tensor"]
        reward_score = float(reward_tensor.sum(dim=-1)[0].item())
        reward_extra_info = {
            key: self._unwrap_single(value) for key, value in (out.get("reward_extra_info", {}) or {}).items()
        }
        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }
