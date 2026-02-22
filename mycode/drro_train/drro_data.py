"""Dataset utilities and prompt templating for DRRO-GRPO."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple


def resolve_local_dataset_path(dataset_dir: str) -> Optional[str]:
    if not dataset_dir:
        return None
    candidates = [
        os.path.join(dataset_dir, "train.jsonl.gz"),
        os.path.join(dataset_dir, "train.jsonl"),
        os.path.join(dataset_dir, "helpful-base", "train.jsonl.gz"),
        os.path.join(dataset_dir, "helpful-base", "train.jsonl"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def extract_human_turn0(text: str) -> str:
    if not text:
        return ""
    marker = "Human:"
    start = text.find(marker)
    if start == -1:
        return text.strip()
    start += len(marker)
    end = text.find("\n\nAssistant:", start)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def load_hh_prompts(dataset: str, local_dir: str) -> List[str]:
    from datasets import load_dataset

    local_path = None
    if os.path.isdir(dataset):
        local_path = resolve_local_dataset_path(dataset)
    if local_path is None:
        local_path = resolve_local_dataset_path(local_dir)

    if local_path:
        ds = load_dataset("json", data_files=local_path, split="train")
    else:
        ds = load_dataset(dataset, split="train")

    if "human_turn0" in ds.column_names:
        raw_prompts = ds["human_turn0"]
    elif "prompt" in ds.column_names:
        raw_prompts = ds["prompt"]
    elif "chosen" in ds.column_names:
        raw_prompts = [extract_human_turn0(text) for text in ds["chosen"]]
    else:
        raise ValueError("Dataset missing human_turn0/prompt/chosen columns.")

    prompts = [text for text in raw_prompts if text and str(text).strip()]
    if not prompts:
        raise ValueError("No prompts found after preprocessing.")
    return prompts


def build_prompt_record(text: str, index: int) -> Dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": text}],
        "data_source": "hh",
        "reward_model": {"ground_truth": "", "style": "model"},
        "extra_info": {"index": index},
    }


def prepare_dataset_files(
    prompts: Sequence[str],
    eval_prompts: int,
    seed: int,
    output_dir: str,
) -> Tuple[str, str]:
    rng = random.Random(seed)
    indices = list(range(len(prompts)))
    rng.shuffle(indices)
    eval_indices = indices[:eval_prompts]
    train_indices = indices[eval_prompts:]
    if not train_indices:
        raise ValueError("Training split is empty; reduce --eval_prompts.")

    train_path = os.path.join(output_dir, "train_prompts.json")
    val_path = os.path.join(output_dir, "val_prompts.json")

    with open(train_path, "w", encoding="utf-8") as handle:
        for i, idx in enumerate(train_indices):
            record = build_prompt_record(prompts[idx], i)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    with open(val_path, "w", encoding="utf-8") as handle:
        for i, idx in enumerate(eval_indices):
            record = build_prompt_record(prompts[idx], i)
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return train_path, val_path


def get_custom_chat_template() -> str:
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "\n\nHuman: {{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "\n\nAssistant: {{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}\n\nAssistant:{% endif %}"
    )
