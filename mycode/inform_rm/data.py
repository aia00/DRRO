"""Dataset helpers for InfoRM training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PreferencePairDataset(Dataset):
    """Loads prompt/chosen/rejected triples from a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        return {
            "prompt": str(row["prompt"]),
            "chosen": str(row["chosen"]),
            "rejected": str(row["rejected"]),
        }


class PromptResponseDataset(Dataset):
    """Loads prompt/response pairs from a JSONL file for latent extraction."""

    def __init__(self, path: Path, prompt_key: str = "prompt", response_key: str = "response") -> None:
        self.rows: List[Dict[str, object]] = []
        self.prompt_key = prompt_key
        self.response_key = response_key
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.rows[idx]
        return {
            "prompt": str(row[self.prompt_key]),
            "response": str(row[self.response_key]),
        }


def build_pair_collate(tokenizer: AutoTokenizer, max_length: int) -> Callable[[List[Dict[str, str]]], Dict[str, Dict[str, torch.Tensor]]]:
    def _collate(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        chosen_enc = tokenizer(
            prompts,
            chosen,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        rejected_enc = tokenizer(
            prompts,
            rejected,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {"chosen": chosen_enc, "rejected": rejected_enc}

    return _collate



def build_response_collate(tokenizer: AutoTokenizer, max_length: int) -> Callable[[List[Dict[str, str]]], Dict[str, torch.Tensor]]:
    def _collate(batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        return tokenizer(
            prompts,
            responses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    return _collate
