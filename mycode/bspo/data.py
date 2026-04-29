"""Data helpers for ScoreLM and BSPO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PreferencePairDataset(Dataset):
    """Loads prompt/chosen/rejected triples from JSONL."""

    def __init__(self, path: Path) -> None:
        self.rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.rows.append(
                    {
                        "prompt": str(row["prompt"]),
                        "chosen": str(row["chosen"]),
                        "rejected": str(row["rejected"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.rows[idx]


def _truncate_prompt_response(
    prompt_ids: List[int],
    response_ids: List[int],
    max_length: int,
    bos_token_id: int | None,
) -> tuple[List[int], List[int], int]:
    prefix_len = 1 if bos_token_id is not None else 0
    available = max(max_length - prefix_len, 1)
    if len(prompt_ids) + len(response_ids) <= available:
        return prompt_ids, response_ids, prefix_len

    if len(response_ids) >= available:
        return [], response_ids[:available], prefix_len

    prompt_keep = available - len(response_ids)
    return prompt_ids[-prompt_keep:], response_ids, prefix_len


def tokenize_prompt_response_batch(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    return_labels: bool,
) -> Dict[str, torch.Tensor]:
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id for batching.")

    input_ids_rows: List[List[int]] = []
    attention_rows: List[List[int]] = []
    response_mask_rows: List[List[int]] = []
    labels_rows: List[List[int]] = []

    for prompt, response in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        prompt_ids, response_ids, prefix_len = _truncate_prompt_response(
            prompt_ids, response_ids, max_length=max_length, bos_token_id=bos_token_id
        )

        seq: List[int] = []
        if bos_token_id is not None:
            seq.append(bos_token_id)
        seq.extend(prompt_ids)
        seq.extend(response_ids)
        if not seq:
            seq = [pad_token_id]

        prompt_span = prefix_len + len(prompt_ids)
        response_mask = [0] * prompt_span + [1] * len(response_ids)
        labels = [-100] * prompt_span + list(response_ids)

        input_ids_rows.append(seq)
        attention_rows.append([1] * len(seq))
        response_mask_rows.append(response_mask)
        labels_rows.append(labels)

    max_seq_len = max(len(row) for row in input_ids_rows)
    padded_ids: List[List[int]] = []
    padded_attn: List[List[int]] = []
    padded_resp_mask: List[List[int]] = []
    padded_labels: List[List[int]] = []

    for ids, attn, resp_mask, labels in zip(input_ids_rows, attention_rows, response_mask_rows, labels_rows):
        pad = max_seq_len - len(ids)
        padded_ids.append(ids + [pad_token_id] * pad)
        padded_attn.append(attn + [0] * pad)
        padded_resp_mask.append(resp_mask + [0] * pad)
        padded_labels.append(labels + [-100] * pad)

    batch: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attn, dtype=torch.long),
        "response_token_mask": torch.tensor(padded_resp_mask, dtype=torch.long),
    }
    if return_labels:
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return batch


def build_pair_collate(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def _collate(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        prompts = [row["prompt"] for row in batch]
        chosen = [row["chosen"] for row in batch]
        rejected = [row["rejected"] for row in batch]
        return {
            "chosen": tokenize_prompt_response_batch(
                tokenizer=tokenizer,
                prompts=prompts,
                responses=chosen,
                max_length=max_length,
                return_labels=True,
            ),
            "rejected": tokenize_prompt_response_batch(
                tokenizer=tokenizer,
                prompts=prompts,
                responses=rejected,
                max_length=max_length,
                return_labels=True,
            ),
        }

    return _collate
