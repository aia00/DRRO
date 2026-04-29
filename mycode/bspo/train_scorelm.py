#!/usr/bin/env python3
"""Train a ScoreLM proxy model for BSPO."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from .data import PreferencePairDataset, build_pair_collate
    from .modeling_scorelm import ScoreLMConfig, ScoreLMModel
except ImportError:
    from data import PreferencePairDataset, build_pair_collate
    from modeling_scorelm import ScoreLMConfig, ScoreLMModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ScoreLM for BSPO.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lm_coef", type=float, default=0.01)
    parser.add_argument("--score_head_dropout", type=float, default=0.0)
    parser.add_argument("--score_regularization", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: ScoreLMModel,
    data_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    lm_coef: float,
    score_regularization: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_reward = 0.0
    total_lm = 0.0
    total_correct = 0
    total = 0

    for batch in data_loader:
        chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
        rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            chosen_out = model.forward_batch(**chosen)
            rejected_out = model.forward_batch(**rejected)
            reward_loss = -F.logsigmoid(chosen_out["score"] - rejected_out["score"]).mean()
            lm_loss = 0.5 * (chosen_out["lm_loss"] + rejected_out["lm_loss"])
            reg_loss = 0.5 * (chosen_out["score"].pow(2).mean() + rejected_out["score"].pow(2).mean())
            loss = reward_loss + lm_coef * lm_loss + score_regularization * reg_loss

        batch_size = chosen["input_ids"].size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        total_reward += float(reward_loss.item()) * batch_size
        total_lm += float(lm_loss.item()) * batch_size
        total_correct += int((chosen_out["score"] > rejected_out["score"]).sum().item())

    denom = max(total, 1)
    return {
        "loss": total_loss / denom,
        "reward_loss": total_reward / denom,
        "lm_loss": total_lm / denom,
        "accuracy": total_correct / denom,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (args.bf16 or args.fp16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    train_ds = PreferencePairDataset(Path(args.train_jsonl))
    val_ds = PreferencePairDataset(Path(args.val_jsonl))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = ScoreLMModel(
        ScoreLMConfig(
            base_model_name=args.model_name,
            score_head_dropout=args.score_head_dropout,
        )
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_pair_collate(tokenizer, args.max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=build_pair_collate(tokenizer, args.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer_steps = 0
    best_acc = -1.0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        stop_early = False
        pbar = tqdm(train_loader, desc=f"scorelm epoch {epoch}")
        for batch in pbar:
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                chosen_out = model.forward_batch(**chosen)
                rejected_out = model.forward_batch(**rejected)
                reward_loss = -F.logsigmoid(chosen_out["score"] - rejected_out["score"]).mean()
                lm_loss = 0.5 * (chosen_out["lm_loss"] + rejected_out["lm_loss"])
                reg_loss = 0.5 * (chosen_out["score"].pow(2).mean() + rejected_out["score"].pow(2).mean())
                loss = reward_loss + args.lm_coef * lm_loss + args.score_regularization * reg_loss
                loss = loss / args.grad_accum
            loss.backward()
            global_step += 1
            if global_step % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                if args.max_steps and optimizer_steps >= args.max_steps:
                    stop_early = True
                    break
            pbar.set_postfix(
                {
                    "reward": f"{float(reward_loss.item()):.4f}",
                    "lm": f"{float(lm_loss.item()):.4f}",
                }
            )

        metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            lm_coef=args.lm_coef,
            score_regularization=args.score_regularization,
        )
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        if args.save_best and metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            model.save_pretrained(output_dir, tokenizer=tokenizer)
        if stop_early:
            break

    if not args.save_best or best_acc < 0.0:
        model.save_pretrained(output_dir, tokenizer=tokenizer)
        if history:
            best_acc = max(best_acc, max(item["accuracy"] for item in history))

    summary = {
        "model_name": args.model_name,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "lm_coef": args.lm_coef,
        "score_regularization": args.score_regularization,
        "seed": args.seed,
        "global_step": global_step,
        "optimizer_steps": optimizer_steps,
        "best_val_accuracy": float(best_acc if best_acc >= 0 else 0.0),
        "history": history,
    }
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
