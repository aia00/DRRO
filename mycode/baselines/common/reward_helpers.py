"""Shared helpers for custom reward managers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baselines.common.paths import ensure_verl_on_path

VERL_ROOT = ensure_verl_on_path()
if VERL_ROOT is None:
    raise RuntimeError("Could not locate VERL package. Set VERL_ROOT or place verl/ next to this repo.")

from verl import DataProto


@dataclass
class RewardModelBundle:
    name: str
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def resolve_dtype(dtype: Optional[torch.dtype | str]) -> Optional[torch.dtype]:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    return DTYPE_MAP.get(str(dtype), None)


def make_device(device: Optional[str]) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_reward_bundle(model_name: str, dtype: Optional[torch.dtype], device: torch.device) -> RewardModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=dtype)
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return RewardModelBundle(name=model_name, tokenizer=tokenizer, model=model)


def extract_prompt_response_pairs(policy_tokenizer, data: DataProto) -> tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
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
        prompt = policy_tokenizer.decode(prompt_ids[i][-p_len:], skip_special_tokens=True)
        response = policy_tokenizer.decode(response_ids[i][:r_len], skip_special_tokens=True)
        prompts.append(prompt)
        responses.append(response)

    return prompts, responses, response_ids, valid_response_lengths


def score_pairs_with_bundle(
    bundle: RewardModelBundle,
    prompts: Sequence[str],
    responses: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            enc = bundle.tokenizer(
                list(prompts[start:end]),
                list(responses[start:end]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = bundle.model(**enc).logits.squeeze(-1)
            scores.append(logits.float().cpu())
    if not scores:
        return torch.empty((0,), dtype=torch.float32)
    return torch.cat(scores, dim=0)


def scatter_sequence_scores_to_token_rewards(
    sequence_scores: torch.Tensor,
    response_ids: torch.Tensor,
    valid_response_lengths: torch.Tensor,
) -> torch.Tensor:
    reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
    idx = torch.arange(reward_tensor.size(0), device=reward_tensor.device)
    last_pos = (valid_response_lengths.long().clamp(min=1) - 1).to(reward_tensor.device)
    reward_tensor[idx, last_pos] = sequence_scores.to(reward_tensor.device)
    return reward_tensor


def to_float_list(values: Iterable[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        out.append(float(value))
    return out


def merge_numeric_lists(storage: Dict[str, List[float]], new_data: Dict[str, object]) -> None:
    for key, value in new_data.items():
        if isinstance(value, np.ndarray):
            storage.setdefault(key, []).extend([float(v) for v in value.tolist()])
            continue
        if torch.is_tensor(value):
            storage.setdefault(key, []).extend([float(v) for v in value.detach().cpu().tolist()])
            continue
        if isinstance(value, (list, tuple)):
            storage.setdefault(key, []).extend([float(v) for v in value])
            continue
        if value is None:
            continue
        try:
            storage.setdefault(key, []).append(float(value))
        except (TypeError, ValueError):
            continue
