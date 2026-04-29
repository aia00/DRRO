#!/usr/bin/env python3
"""Evaluate ScoreLM on preference pairs and optionally compare to a gold RM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .data import PreferencePairDataset, build_pair_collate
    from .modeling_scorelm import ScoreLMModel
except ImportError:
    from data import PreferencePairDataset, build_pair_collate
    from modeling_scorelm import ScoreLMModel


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
    prompts = [row["prompt"] for row in rows]
    chosen = [row["chosen"] for row in rows]
    rejected = [row["rejected"] for row in rows]
    for start in range(0, len(rows), batch_size):
        end = start + batch_size
        chosen_enc = tokenizer(
            prompts[start:end],
            chosen[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        rejected_enc = tokenizer(
            prompts[start:end],
            rejected[start:end],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        chosen_scores.extend(model(**chosen_enc).logits.squeeze(-1).detach().cpu().tolist())
        rejected_scores.extend(model(**rejected_enc).logits.squeeze(-1).detach().cpu().tolist())
    return {"chosen": chosen_scores, "rejected": rejected_scores}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ScoreLM.")
    parser.add_argument("--data_jsonl", required=True)
    parser.add_argument("--scorelm", required=True)
    parser.add_argument("--gold_rm", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
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

    model, tokenizer = ScoreLMModel.from_pretrained(args.scorelm, device=device)
    loader = DataLoader(
        rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_pair_collate(tokenizer, args.max_length),
    )

    chosen_scores: List[float] = []
    rejected_scores: List[float] = []
    lm_losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            chosen_out = model.forward_batch(**chosen)
            rejected_out = model.forward_batch(**rejected)
            chosen_scores.extend(chosen_out["score"].detach().cpu().tolist())
            rejected_scores.extend(rejected_out["score"].detach().cpu().tolist())
            lm_losses.append(float((0.5 * (chosen_out["lm_loss"] + rejected_out["lm_loss"])).item()))

    metrics: Dict[str, float] = {
        "n_pairs": float(len(rows)),
        "scorelm_accuracy": float(sum(c > r for c, r in zip(chosen_scores, rejected_scores)) / max(len(rows), 1)),
        "scorelm_mean": float(torch.tensor(chosen_scores + rejected_scores).mean().item() if chosen_scores else 0.0),
        "scorelm_lm_loss": float(sum(lm_losses) / max(len(lm_losses), 1)),
    }

    if args.gold_rm:
        gold_tokenizer = AutoTokenizer.from_pretrained(args.gold_rm, use_fast=True)
        if gold_tokenizer.pad_token is None:
            gold_tokenizer.pad_token = gold_tokenizer.eos_token or gold_tokenizer.unk_token
        gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_rm, num_labels=1).to(device)
        gold_scores = score_hf_rm(gold_model, gold_tokenizer, rows, args.batch_size, args.max_length, device)
        scorelm_pref = [c > r for c, r in zip(chosen_scores, rejected_scores)]
        gold_pref = [c > r for c, r in zip(gold_scores["chosen"], gold_scores["rejected"])]
        diffs = [
            abs(s - g)
            for s, g in zip(
                chosen_scores + rejected_scores,
                gold_scores["chosen"] + gold_scores["rejected"],
            )
        ]
        diff_tensor = torch.tensor(diffs) if diffs else torch.tensor([0.0])
        metrics["agreement_with_gold"] = float(sum(a == b for a, b in zip(scorelm_pref, gold_pref)) / max(len(scorelm_pref), 1))
        metrics["mean_abs_diff_vs_gold"] = float(diff_tensor.mean().item())
        metrics["var_abs_diff_vs_gold"] = float(diff_tensor.var(unbiased=False).item())

    print(json.dumps(metrics, indent=2))
    if args.output_json:
        with Path(args.output_json).open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
