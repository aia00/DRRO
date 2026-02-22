#!/usr/bin/env python3
"""Build preference pairs using a gold reward model."""

from __future__ import annotations

import argparse
import json
import random
import time
import math
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

PROMPT_TEMPLATE = "\n\nHuman: {human_turn0}\n\nAssistant:"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_prompts(dataset_name: str, dataset_path: str | None, split: str, seed: int) -> List[str]:
    if dataset_path:
        ds = load_dataset(dataset_path, split=split)
    else:
        ds = load_dataset(dataset_name, split=split)
    prompts = [PROMPT_TEMPLATE.format(human_turn0=row["human_turn0"]) for row in ds]
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def generate_grouped_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_responses: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[List[str]]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_lens = enc["attention_mask"].sum(dim=1)

    outputs = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_responses,
        pad_token_id=tokenizer.eos_token_id,
    )

    grouped: List[List[str]] = [[] for _ in prompts]
    for i, output_ids in enumerate(outputs):
        prompt_idx = i // num_responses
        prompt_len = int(input_lens[prompt_idx].item())
        resp_ids = output_ids[prompt_len:]
        text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
        grouped[prompt_idx].append(text)
    return grouped


def score_pairs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> List[float]:
    scores: List[float] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            p_batch = prompts[i : i + batch_size]
            r_batch = responses[i : i + batch_size]
            enc = tokenizer(
                p_batch,
                r_batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(logits.detach().cpu().tolist())
    return scores


def build_pairs(
    prompts: List[str],
    policy_model: AutoModelForCausalLM,
    policy_tokenizer: AutoTokenizer,
    gold_model: AutoModelForSequenceClassification,
    gold_tokenizer: AutoTokenizer,
    num_pairs: int,
    num_responses: int,
    batch_size_prompts: int,
    rm_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    rm_max_length: int,
    device: torch.device,
    seed: int,
) -> List[Dict[str, object]]:
    pairs: List[Dict[str, object]] = []
    idx = 0
    rng = random.Random(seed)
    pbar = tqdm(total=num_pairs, desc="pairs")

    while len(pairs) < num_pairs:
        before = len(pairs)
        batch_prompts: List[str] = []
        for _ in range(batch_size_prompts):
            if idx >= len(prompts):
                rng.shuffle(prompts)
                idx = 0
            batch_prompts.append(prompts[idx])
            idx += 1

        grouped = generate_grouped_responses(
            policy_model,
            policy_tokenizer,
            batch_prompts,
            num_responses,
            max_new_tokens,
            temperature,
            top_p,
            device,
        )

        flat_prompts: List[str] = []
        flat_responses: List[str] = []
        for prompt, responses in zip(batch_prompts, grouped):
            for resp in responses:
                flat_prompts.append(prompt)
                flat_responses.append(resp)

        scores = score_pairs(
            gold_model,
            gold_tokenizer,
            flat_prompts,
            flat_responses,
            rm_batch_size,
            rm_max_length,
            device,
        )

        offset = 0
        for prompt, responses in zip(batch_prompts, grouped):
            resp_scores = scores[offset : offset + len(responses)]
            offset += len(responses)
            if not resp_scores:
                continue
            best_idx = int(torch.tensor(resp_scores).argmax().item())
            worst_idx = int(torch.tensor(resp_scores).argmin().item())
            if best_idx == worst_idx:
                continue
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": responses[best_idx],
                    "rejected": responses[worst_idx],
                    "chosen_score": float(resp_scores[best_idx]),
                    "rejected_score": float(resp_scores[worst_idx]),
                }
            )
            if len(pairs) >= num_pairs:
                break
        pbar.update(len(pairs) - before)

    pbar.close()
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build preference pairs with a gold RM.")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/hh-rlhf")
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--policy_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gold_rm", default="sileod/deberta-v3-large-tasksource-rlhf-reward-model")
    parser.add_argument("--num_pairs", type=int, default=50000)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_responses", type=int, default=4)
    parser.add_argument("--batch_size_prompts", type=int, default=8)
    parser.add_argument("--rm_batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rm_max_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.dataset_name, args.dataset_path, args.split, args.seed)
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            raise ValueError(f"shard_id must be in [0, {args.num_shards-1}]")
        prompts = [p for i, p in enumerate(prompts) if i % args.num_shards == args.shard_id]
        num_pairs = int(math.ceil(args.num_pairs / args.num_shards))
    else:
        num_pairs = args.num_pairs

    policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_model, use_fast=True)
    if policy_tokenizer.pad_token_id is None:
        policy_tokenizer.pad_token_id = policy_tokenizer.eos_token_id

    policy_model = AutoModelForCausalLM.from_pretrained(args.policy_model)
    policy_model.to(device)
    policy_model.eval()

    gold_tokenizer = AutoTokenizer.from_pretrained(args.gold_rm, use_fast=True)
    if gold_tokenizer.pad_token_id is None:
        gold_tokenizer.pad_token_id = gold_tokenizer.eos_token_id
    gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_rm, num_labels=1)
    gold_model.to(device)
    gold_model.eval()

    start = time.time()
    pairs = []
    with torch.no_grad():
        pairs = build_pairs(
            prompts,
            policy_model,
            policy_tokenizer,
            gold_model,
            gold_tokenizer,
            num_pairs,
            args.num_responses,
            args.batch_size_prompts,
            args.rm_batch_size,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.rm_max_length,
            device,
            args.seed,
        )

    random.Random(args.seed).shuffle(pairs)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    suffix = f".shard{args.shard_id}" if args.num_shards > 1 else ""
    train_path = output_dir / f"train{suffix}.jsonl"
    val_path = output_dir / f"val{suffix}.jsonl"
    all_path = output_dir / f"pairs_all{suffix}.jsonl"

    for path, data in [(train_path, train_pairs), (val_path, val_pairs), (all_path, pairs)]:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "policy_model": args.policy_model,
        "gold_rm": args.gold_rm,
        "num_pairs": len(pairs),
        "num_pairs_target": args.num_pairs,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_responses": args.num_responses,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rm_max_length": args.rm_max_length,
        "seed": args.seed,
        "elapsed_s": time.time() - start,
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {len(train_pairs)} train pairs and {len(val_pairs)} val pairs to {output_dir}")


if __name__ == "__main__":
    main()
