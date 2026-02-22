#!/usr/bin/env python3
"""Train a proxy reward model from preference pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


class PairDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(tokenizer: AutoTokenizer, max_length: int):
    def _collate(batch: List[Dict[str, str]]) -> Dict[str, Dict[str, torch.Tensor]]:
        prompts = [b["prompt"] for b in batch]
        chosen = [b["chosen"] for b in batch]
        rejected = [b["rejected"] for b in batch]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a proxy reward model.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--model_name", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many optimizer steps (0 = no limit).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            chosen_scores = model(**chosen).logits.squeeze(-1)
            rejected_scores = model(**rejected).logits.squeeze(-1)
            loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()
            total_loss += loss.item() * chosen_scores.size(0)
            total_correct += (chosen_scores > rejected_scores).sum().item()
            total += chosen_scores.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.bf16 or args.fp16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    train_ds = PairDataset(Path(args.train_jsonl))
    val_ds = PairDataset(Path(args.val_jsonl))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=1)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn(tokenizer, args.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    stop_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        optimizer.zero_grad()
        for batch in pbar:
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                chosen_scores = model(**chosen).logits.squeeze(-1)
                rejected_scores = model(**rejected).logits.squeeze(-1)
                loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores).mean()
                loss = loss / args.grad_accum
            loss.backward()
            global_step += 1
            if global_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if args.max_steps and (global_step // args.grad_accum) >= args.max_steps:
                    stop_early = True
                    break
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        if stop_early:
            break

        metrics = evaluate(model, val_loader, device)
        print(f"val loss={metrics['loss']:.4f} acc={metrics['accuracy']:.4f}")
        if args.save_best and metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    if not args.save_best:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
