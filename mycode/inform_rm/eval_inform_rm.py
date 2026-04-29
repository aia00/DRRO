#!/usr/bin/env python3
"""Evaluate an InfoRM checkpoint on preference pairs and optionally compare to a gold RM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .data import PreferencePairDataset, build_pair_collate
    from .modeling_inform_rm import InfoRMModel
except ImportError:
    from data import PreferencePairDataset, build_pair_collate
    from modeling_inform_rm import InfoRMModel


@torch.no_grad()
def score_inform_rm(
    model: InfoRMModel,
    data_loader,
    device: torch.device,
) -> Dict[str, List[float]]:
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    model.eval()
    for batch in data_loader:
        chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
        rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
        chosen_out = model.score_batch(chosen, sample_latent=False)["reward"]
        rejected_out = model.score_batch(rejected, sample_latent=False)["reward"]
        chosen_scores.extend(chosen_out.detach().cpu().tolist())
        rejected_scores.extend(rejected_out.detach().cpu().tolist())
    return {"chosen": chosen_scores, "rejected": rejected_scores}


@torch.no_grad()
def score_hf_rm(
    model,
    tokenizer,
    rows: List[Dict[str, str]],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    model.eval()
    prompts = [row["prompt"] for row in rows]
    chosen = [row["chosen"] for row in rows]
    rejected = [row["rejected"] for row in rows]
    for i in range(0, len(rows), batch_size):
        p = prompts[i : i + batch_size]
        c = chosen[i : i + batch_size]
        r = rejected[i : i + batch_size]
        chosen_enc = tokenizer(p, c, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        rejected_enc = tokenizer(p, r, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        chosen_scores.extend(model(**chosen_enc).logits.squeeze(-1).detach().cpu().tolist())
        rejected_scores.extend(model(**rejected_enc).logits.squeeze(-1).detach().cpu().tolist())
    return {"chosen": chosen_scores, "rejected": rejected_scores}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate InfoRM checkpoint.")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--inform_rm", required=True)
    parser.add_argument("--gold_rm", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default="")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = PreferencePairDataset(Path(args.data_jsonl)).rows
    if args.max_samples and len(rows) > args.max_samples:
        gen = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(rows), generator=gen)[: args.max_samples].tolist()
        rows = [rows[i] for i in indices]

    normalized_rows = [
        {"prompt": str(row["prompt"]), "chosen": str(row["chosen"]), "rejected": str(row["rejected"])}
        for row in rows
    ]
    inform_model, tokenizer = InfoRMModel.from_pretrained(args.inform_rm, device=device)
    data_loader = torch.utils.data.DataLoader(
        normalized_rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_pair_collate(tokenizer, args.max_length),
    )
    inform_scores = score_inform_rm(inform_model, data_loader, device)
    accuracy = sum(c > r for c, r in zip(inform_scores["chosen"], inform_scores["rejected"])) / max(len(normalized_rows), 1)

    metrics = {
        "n_pairs": len(normalized_rows),
        "inform_accuracy": accuracy,
    }

    if args.gold_rm:
        gold_tokenizer = AutoTokenizer.from_pretrained(args.gold_rm, use_fast=True)
        if gold_tokenizer.pad_token is None:
            gold_tokenizer.pad_token = gold_tokenizer.eos_token or gold_tokenizer.unk_token
        gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_rm, num_labels=1).to(device)
        gold_scores = score_hf_rm(gold_model, gold_tokenizer, normalized_rows, args.batch_size, args.max_length, device)
        inform_pref = [c > r for c, r in zip(inform_scores["chosen"], inform_scores["rejected"])]
        gold_pref = [c > r for c, r in zip(gold_scores["chosen"], gold_scores["rejected"])]
        agreement = sum(i == g for i, g in zip(inform_pref, gold_pref)) / max(len(inform_pref), 1)
        diffs = [
            abs(i - g)
            for i, g in zip(
                inform_scores["chosen"] + inform_scores["rejected"],
                gold_scores["chosen"] + gold_scores["rejected"],
            )
        ]
        diff_tensor = torch.tensor(diffs) if diffs else torch.tensor([0.0])
        metrics["agreement_with_gold"] = float(agreement)
        metrics["mean_abs_diff_vs_gold"] = float(diff_tensor.mean().item())
        metrics["var_abs_diff_vs_gold"] = float(diff_tensor.var(unbiased=False).item())

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
