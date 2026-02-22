#!/usr/bin/env python3
"""Evaluate proxy RM against gold RM on preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_pairs(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def score_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    responses: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> List[float]:
    model.eval()
    scores: List[float] = []
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate proxy RM vs gold RM.")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--proxy_rm", required=True)
    parser.add_argument("--gold_rm", default="sileod/deberta-v3-large-tasksource-rlhf-reward-model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_pairs(Path(args.data_jsonl))
    if args.max_samples and len(data) > args.max_samples:
        rng = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(data), generator=rng)[: args.max_samples].tolist()
        data = [data[i] for i in indices]
    prompts = [str(row["prompt"]) for row in data]
    chosen = [str(row["chosen"]) for row in data]
    rejected = [str(row["rejected"]) for row in data]

    proxy_tokenizer = AutoTokenizer.from_pretrained(args.proxy_rm, use_fast=True)
    if proxy_tokenizer.pad_token_id is None:
        proxy_tokenizer.pad_token_id = proxy_tokenizer.eos_token_id
    proxy_model = AutoModelForSequenceClassification.from_pretrained(args.proxy_rm, num_labels=1).to(device)

    gold_tokenizer = AutoTokenizer.from_pretrained(args.gold_rm, use_fast=True)
    if gold_tokenizer.pad_token_id is None:
        gold_tokenizer.pad_token_id = gold_tokenizer.eos_token_id
    gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_rm, num_labels=1).to(device)

    proxy_chosen = score_model(proxy_model, proxy_tokenizer, prompts, chosen, args.batch_size, args.max_length, device)
    proxy_rejected = score_model(proxy_model, proxy_tokenizer, prompts, rejected, args.batch_size, args.max_length, device)
    gold_chosen = score_model(gold_model, gold_tokenizer, prompts, chosen, args.batch_size, args.max_length, device)
    gold_rejected = score_model(gold_model, gold_tokenizer, prompts, rejected, args.batch_size, args.max_length, device)

    proxy_pref = [pc > pr for pc, pr in zip(proxy_chosen, proxy_rejected)]
    gold_pref = [gc > gr for gc, gr in zip(gold_chosen, gold_rejected)]
    agreement = sum(p == g for p, g in zip(proxy_pref, gold_pref)) / max(len(proxy_pref), 1)

    abs_diffs = [abs(p - g) for p, g in zip(proxy_chosen + proxy_rejected, gold_chosen + gold_rejected)]
    diffs_tensor = torch.tensor(abs_diffs)
    mean_abs_diff = diffs_tensor.mean().item() if len(abs_diffs) else 0.0
    var_abs_diff = diffs_tensor.var(unbiased=False).item() if len(abs_diffs) else 0.0

    metrics = {
        "n_pairs": len(data),
        "agreement": agreement,
        "mean_abs_diff": mean_abs_diff,
        "var_abs_diff": var_abs_diff,
    }

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
