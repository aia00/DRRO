"""ScoreLM reward manager for BSPO."""

from __future__ import annotations

import os
import sys
from typing import ClassVar, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.abstract import AbstractRewardManager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from .data import tokenize_prompt_response_batch
    from .modeling_scorelm import ScoreLMModel
except ImportError:
    from data import tokenize_prompt_response_batch
    from modeling_scorelm import ScoreLMModel

try:
    from drro_train.drro_reward import HFRewardManager
except ImportError:
    HFRewardManager = None


class ScoreLMRewardManager(AbstractRewardManager):
    """Reward manager that emits both scalar proxy rewards and behavior support stats."""

    _cache: ClassVar[Dict[Tuple[str, str, Optional[torch.dtype]], Tuple[ScoreLMModel, AutoTokenizer]]] = {}

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        scorelm_path: str = "",
        batch_size: int = 8,
        max_length: int = 1024,
        epsilon_beta: float = 1e-4,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.scorelm_path = scorelm_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.epsilon_beta = float(epsilon_beta)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            dtype = torch.float32

        cache_key = (scorelm_path, str(self.device), dtype)
        if cache_key not in self._cache:
            model, score_tokenizer = ScoreLMModel.from_pretrained(scorelm_path, device=self.device, dtype=dtype)
            if score_tokenizer.pad_token is None:
                score_tokenizer.pad_token = score_tokenizer.eos_token or score_tokenizer.unk_token
            self._cache[cache_key] = (model, score_tokenizer)
        self.scorelm, self.score_tokenizer = self._cache[cache_key]
        self.use_actor_token_ids = self._tokenizers_compatible()

    def _tokenizers_compatible(self) -> bool:
        if self.score_tokenizer.vocab_size != self.policy_tokenizer.vocab_size:
            return False
        if self.score_tokenizer.bos_token_id != self.policy_tokenizer.bos_token_id:
            return False
        if self.score_tokenizer.eos_token_id != self.policy_tokenizer.eos_token_id:
            return False

        probe_texts = [
            "hello",
            " hello",
            "\n\nHuman: test\n\nAssistant:",
            "1 2 3",
            "The quick brown fox.",
        ]
        for text in probe_texts:
            if self.score_tokenizer.encode(text, add_special_tokens=False) != self.policy_tokenizer.encode(
                text,
                add_special_tokens=False,
            ):
                return False
        return True

    def _extract_batch_tensors(
        self,
        data: DataProto,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        return prompt_ids, response_ids, valid_prompt_lengths, valid_response_lengths

    def _build_score_batch_from_actor_tokens(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        valid_prompt_lengths: torch.Tensor,
        valid_response_lengths: torch.Tensor,
        start: int,
        end: int,
    ) -> Dict[str, torch.Tensor]:
        pad_token_id = self.score_tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("ScoreLM tokenizer must define pad_token_id.")

        input_rows: List[List[int]] = []
        attention_rows: List[List[int]] = []
        response_mask_rows: List[List[int]] = []

        for idx in range(start, end):
            p_len = int(valid_prompt_lengths[idx].item())
            r_len = int(valid_response_lengths[idx].item())
            prompt_tokens = prompt_ids[idx, -p_len:].tolist() if p_len > 0 else []
            response_tokens = response_ids[idx, :r_len].tolist() if r_len > 0 else []

            if len(prompt_tokens) + len(response_tokens) > self.max_length:
                if len(response_tokens) >= self.max_length:
                    prompt_tokens = []
                    response_tokens = response_tokens[: self.max_length]
                else:
                    keep_prompt = self.max_length - len(response_tokens)
                    prompt_tokens = prompt_tokens[-keep_prompt:]

            seq = prompt_tokens + response_tokens
            if not seq:
                seq = [pad_token_id]
            prompt_span = len(prompt_tokens)
            response_mask = [0] * prompt_span + [1] * len(response_tokens)

            input_rows.append(seq)
            attention_rows.append([1] * len(seq))
            response_mask_rows.append(response_mask)

        max_seq_len = max(len(row) for row in input_rows)
        padded_ids: List[List[int]] = []
        padded_attn: List[List[int]] = []
        padded_resp_mask: List[List[int]] = []

        for ids, attn, resp_mask in zip(input_rows, attention_rows, response_mask_rows):
            pad = max_seq_len - len(ids)
            padded_ids.append(ids + [pad_token_id] * pad)
            padded_attn.append(attn + [0] * pad)
            padded_resp_mask.append(resp_mask + [0] * pad)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
            "response_token_mask": torch.tensor(padded_resp_mask, dtype=torch.long),
        }

    def _decode_prompt_response(self, data: DataProto) -> tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
        prompt_ids, response_ids, valid_prompt_lengths, valid_response_lengths = self._extract_batch_tensors(data)

        prompts: List[str] = []
        responses: List[str] = []
        for i in range(len(data)):
            p_len = int(valid_prompt_lengths[i].item())
            r_len = int(valid_response_lengths[i].item())
            prompt = self.policy_tokenizer.decode(
                prompt_ids[i][-p_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = self.policy_tokenizer.decode(
                response_ids[i][:r_len], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            prompts.append(prompt)
            responses.append(response)
        return prompts, responses, response_ids, valid_response_lengths

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        prompt_ids, response_ids, valid_prompt_lengths, valid_response_lengths = self._extract_batch_tensors(data)
        prompts, responses, response_ids, valid_response_lengths = self._decode_prompt_response(data)
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        behavior_probs = torch.zeros_like(response_ids, dtype=torch.float32)
        behavior_supported = torch.zeros_like(response_ids, dtype=torch.float32)
        seq_scores: List[torch.Tensor] = []

        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                end = min(start + self.batch_size, len(prompts))
                if self.use_actor_token_ids:
                    batch_enc = self._build_score_batch_from_actor_tokens(
                        prompt_ids=prompt_ids,
                        response_ids=response_ids,
                        valid_prompt_lengths=valid_prompt_lengths,
                        valid_response_lengths=valid_response_lengths,
                        start=start,
                        end=end,
                    )
                else:
                    batch_enc = tokenize_prompt_response_batch(
                        tokenizer=self.score_tokenizer,
                        prompts=prompts[start:end],
                        responses=responses[start:end],
                        max_length=self.max_length,
                        return_labels=False,
                    )
                batch_enc = {k: v.to(self.device) for k, v in batch_enc.items()}
                out = self.scorelm.forward_batch(
                    input_ids=batch_enc["input_ids"],
                    attention_mask=batch_enc["attention_mask"],
                    labels=None,
                )
                log_probs, resp_mask = self.scorelm.behavior_log_probs(
                    input_ids=batch_enc["input_ids"],
                    attention_mask=batch_enc["attention_mask"],
                    response_token_mask=batch_enc["response_token_mask"],
                )
                probs = torch.exp(log_probs) * resp_mask
                seq_scores.append(out["score"].detach().cpu())

                for local_idx in range(end - start):
                    global_idx = start + local_idx
                    score_resp_mask = batch_enc["response_token_mask"][local_idx].bool()
                    score_probs = probs[local_idx][score_resp_mask].detach().cpu()
                    actor_len = int(valid_response_lengths[global_idx].item())
                    if score_probs.numel() != actor_len:
                        raise ValueError(
                            "ScoreLM tokenizer is incompatible with the actor tokenizer for BSPO. "
                            f"Expected response length {actor_len}, got {score_probs.numel()}. "
                            "Use a tokenizer-compatible ScoreLM backbone or enable actor-token compatible scoring."
                        )
                    behavior_probs[global_idx, :actor_len] = score_probs
                    behavior_supported[global_idx, :actor_len] = (score_probs > self.epsilon_beta).float()

        seq_score_tensor = torch.cat(seq_scores, dim=0) if seq_scores else torch.empty((0,), dtype=torch.float32)
        idx = torch.arange(reward_tensor.size(0), device=reward_tensor.device)
        last_pos = (valid_response_lengths.long().clamp(min=1) - 1).to(reward_tensor.device)
        reward_tensor[idx, last_pos] = seq_score_tensor.to(reward_tensor.device)

        mean_behavior_prob = []
        unsupported_fraction = []
        for i in range(reward_tensor.size(0)):
            r_len = int(valid_response_lengths[i].item())
            if r_len <= 0:
                mean_behavior_prob.append(0.0)
                unsupported_fraction.append(0.0)
                continue
            probs_i = behavior_probs[i, :r_len]
            supp_i = behavior_supported[i, :r_len]
            mean_behavior_prob.append(float(probs_i.mean().item()))
            unsupported_fraction.append(float((1.0 - supp_i).mean().item()))

        reward_extra_info = {
            "proxy_score": seq_score_tensor.tolist(),
            "behavior_token_prob": behavior_probs.tolist(),
            "behavior_supported_mask": behavior_supported.tolist(),
            "unsupported_fraction": unsupported_fraction,
            "mean_behavior_prob": mean_behavior_prob,
            "epsilon_beta": [self.epsilon_beta] * len(prompts),
        }
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


class ProxyScoreLMSupportRewardManager(AbstractRewardManager):
    """Use the normal proxy RM as reward and ScoreLM only for support masks.

    This is the apples-to-apples BSPO-like variant: the optimized scalar reward
    is the same proxy RM used by PPO/GRPO/DRRO, while ScoreLM only supplies the
    behavior-supported token mask consumed by the BSPO advantage patch.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        proxy_model_name: str = "",
        scorelm_path: str = "",
        reward_batch_size: int = 8,
        reward_max_length: int = 512,
        scorelm_max_length: int = 1024,
        epsilon_beta: float = 1e-4,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if HFRewardManager is None:
            raise ImportError("Could not import drro_train.drro_reward.HFRewardManager")
        self.policy_tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
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
        self.support_reward = ScoreLMRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            scorelm_path=scorelm_path,
            batch_size=reward_batch_size,
            max_length=scorelm_max_length,
            epsilon_beta=epsilon_beta,
            device=device,
            dtype=dtype,
        )

    def __call__(self, data: DataProto, return_dict: bool = False):
        proxy_out = self.proxy_reward(data, return_dict=True)
        support_out = self.support_reward(data, return_dict=True)

        reward_tensor = proxy_out["reward_tensor"]
        proxy_scores = reward_tensor.sum(dim=-1).detach().cpu().tolist()
        reward_extra_info = dict(support_out.get("reward_extra_info", {}) or {})
        scorelm_scores = reward_extra_info.pop("proxy_score", None)
        reward_extra_info["proxy_score"] = proxy_scores
        reward_extra_info["proxy_rm_score"] = proxy_scores
        if scorelm_scores is not None:
            reward_extra_info["scorelm_score"] = scorelm_scores

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor


class LoopScoreLMRewardManager(RewardManagerBase):
    """Reward-loop compatible wrapper around ScoreLMRewardManager."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        scorelm_path: str = "",
        reward_batch_size: int = 8,
        scorelm_max_length: int = 1024,
        epsilon_beta: float = 1e-4,
        dtype: str = "float32",
        device: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("reward_kwargs", {})
        scorelm_path = scorelm_path or reward_cfg.get("scorelm_path", "")
        reward_batch_size = int(reward_batch_size or reward_cfg.get("reward_batch_size", 8))
        scorelm_max_length = int(
            scorelm_max_length
            or reward_cfg.get("scorelm_max_length", reward_cfg.get("reward_max_length", 1024))
        )
        epsilon_beta = float(epsilon_beta or reward_cfg.get("epsilon_beta", 1e-4))
        dtype_name = dtype or reward_cfg.get("dtype", "float32")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(dtype_name, torch.float32)

        self.delegate = ScoreLMRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            scorelm_path=scorelm_path,
            batch_size=reward_batch_size,
            max_length=scorelm_max_length,
            epsilon_beta=epsilon_beta,
            device=device,
            dtype=reward_dtype,
        )

    @staticmethod
    def _unwrap_single(value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "LoopScoreLMRewardManager only supports a single sample"
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


class LoopProxyScoreLMSupportRewardManager(RewardManagerBase):
    """Reward-loop wrapper for proxy-RM reward + ScoreLM support masks."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        proxy_model_name: str = "",
        scorelm_path: str = "",
        reward_batch_size: int = 8,
        reward_max_length: int = 512,
        scorelm_max_length: int = 1024,
        epsilon_beta: float = 1e-4,
        dtype: str = "float32",
        device: Optional[str] = None,
        **_: object,
    ) -> None:
        super().__init__(config, tokenizer, compute_score)
        reward_cfg = config.reward.get("reward_kwargs", {})
        proxy_model_name = proxy_model_name or reward_cfg.get("proxy_model", reward_cfg.get("model_name", ""))
        scorelm_path = scorelm_path or reward_cfg.get("scorelm_path", "")
        reward_batch_size = int(reward_batch_size or reward_cfg.get("reward_batch_size", 8))
        reward_max_length = int(reward_max_length or reward_cfg.get("reward_max_length", 512))
        scorelm_max_length = int(
            scorelm_max_length
            or reward_cfg.get("scorelm_max_length", reward_cfg.get("reward_max_length", 1024))
        )
        epsilon_beta = float(epsilon_beta or reward_cfg.get("epsilon_beta", 1e-4))
        dtype_name = dtype or reward_cfg.get("dtype", "float32")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(dtype_name, torch.float32)

        self.delegate = ProxyScoreLMSupportRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            proxy_model_name=proxy_model_name,
            scorelm_path=scorelm_path,
            reward_batch_size=reward_batch_size,
            reward_max_length=reward_max_length,
            scorelm_max_length=scorelm_max_length,
            epsilon_beta=epsilon_beta,
            device=device,
            dtype=reward_dtype,
        )

    @staticmethod
    def _unwrap_single(value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "LoopProxyScoreLMSupportRewardManager only supports a single sample"
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
