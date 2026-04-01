"""Reward model helpers for DRRO-GRPO."""

from __future__ import annotations

from typing import ClassVar, Dict, List, Optional, Tuple

import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.abstract import AbstractRewardManager


class RewardStats:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std


class HFRewardManager(AbstractRewardManager):
    """Reward manager that scores prompt+completion with a HF reward model."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        model_name: str = "",
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        data_parallel: Optional[bool] = None,
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if self.device.type != "cuda":
            dtype = torch.float32

        self.reward_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.reward_tokenizer.pad_token is None:
            self.reward_tokenizer.pad_token = self.reward_tokenizer.eos_token or self.reward_tokenizer.unk_token

        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, torch_dtype=dtype
        )
        if self.reward_model.config.pad_token_id is None and self.reward_tokenizer.pad_token_id is not None:
            self.reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id
        self.reward_model.to(self.device)
        if self.device.type == "cuda":
            if data_parallel is None:
                data_parallel = os.environ.get("DRRO_REWARD_DP", "1") == "1"
            if data_parallel and torch.cuda.device_count() > 1:
                self.reward_model = torch.nn.DataParallel(self.reward_model)
                self.reward_model.to(self.device)
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad_(False)

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        prompts: List[str] = []
        responses: List[str] = []
        for i in range(len(data)):
            p_len = int(valid_prompt_lengths[i].item())
            r_len = int(valid_response_lengths[i].item())
            prompt = self.policy_tokenizer.decode(prompt_ids[i][-p_len:], skip_special_tokens=True)
            response = self.policy_tokenizer.decode(response_ids[i][:r_len], skip_special_tokens=True)
            prompts.append(prompt)
            responses.append(response)

        scores: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                end = start + self.batch_size
                enc = self.reward_tokenizer(
                    prompts[start:end],
                    responses[start:end],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.reward_model(**enc).logits.squeeze(-1)
                scores.append(logits.float().cpu())
        scores_tensor = torch.cat(scores, dim=0)

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        idx = torch.arange(reward_tensor.size(0), device=reward_tensor.device)
        last_pos = (valid_response_lengths.long().clamp(min=1) - 1).to(reward_tensor.device)
        reward_tensor[idx, last_pos] = scores_tensor.to(reward_tensor.device)

        reward_extra_info = {"reward": scores_tensor.numpy().tolist()}
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


class LoopHFRewardManager(RewardManagerBase):
    """Reward-loop compatible HF reward manager for public VERL."""

    _tokenizer_cache: ClassVar[Dict[str, AutoTokenizer]] = {}
    _model_cache: ClassVar[Dict[Tuple[str, str, Optional[torch.dtype]], torch.nn.Module]] = {}

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        model_name: str = "",
        reward_batch_size: int = 8,
        reward_max_length: int = 512,
        dtype: str = "float32",
        device: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("reward_kwargs", {})
        model_name = model_name or reward_cfg.get("model_name", "")
        reward_batch_size = int(reward_batch_size or reward_cfg.get("reward_batch_size", 8))
        reward_max_length = int(reward_max_length or reward_cfg.get("reward_max_length", 512))
        dtype = dtype or reward_cfg.get("dtype", "float32")
        self.policy_tokenizer = tokenizer
        self.model_name = model_name
        self.reward_batch_size = reward_batch_size
        self.reward_max_length = reward_max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.float32)
        if self.device.type != "cuda":
            self.dtype = torch.float32

        if model_name not in self._tokenizer_cache:
            reward_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if reward_tokenizer.pad_token is None:
                reward_tokenizer.pad_token = reward_tokenizer.eos_token or reward_tokenizer.unk_token
            self._tokenizer_cache[model_name] = reward_tokenizer
        self.reward_tokenizer = self._tokenizer_cache[model_name]

        model_key = (model_name, str(self.device), self.dtype)
        if model_key not in self._model_cache:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=1, torch_dtype=self.dtype
            )
            if reward_model.config.pad_token_id is None and self.reward_tokenizer.pad_token_id is not None:
                reward_model.config.pad_token_id = self.reward_tokenizer.pad_token_id
            reward_model.to(self.device)
            reward_model.eval()
            for param in reward_model.parameters():
                param.requires_grad_(False)
            self._model_cache[model_key] = reward_model
        self.reward_model = self._model_cache[model_key]

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "LoopHFRewardManager only supports a single sample"
        reward = self._score_single(data)
        return {
            "reward_score": reward,
            "reward_extra_info": {"reward": reward},
        }

    def _score_single(self, data: DataProto) -> float:
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        p_len = int(valid_prompt_lengths[0].item())
        r_len = int(valid_response_lengths[0].item())
        prompt = self.policy_tokenizer.decode(prompt_ids[0][-p_len:], skip_special_tokens=True)
        response = self.policy_tokenizer.decode(response_ids[0][:r_len], skip_special_tokens=True)

        with torch.no_grad():
            enc = self.reward_tokenizer(
                [prompt],
                [response],
                padding=True,
                truncation=True,
                max_length=self.reward_max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.reward_model(**enc).logits.squeeze(-1)
        return float(logits[0].float().cpu().item())
