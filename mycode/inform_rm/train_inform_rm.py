#!/usr/bin/env python3
"""Train an InfoRM reward model on prompt/chosen/rejected preference pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    from .data import PreferencePairDataset, build_pair_collate
    from .modeling_inform_rm import InfoRMConfig, InfoRMModel
except ImportError:
    from data import PreferencePairDataset, build_pair_collate
    from modeling_inform_rm import InfoRMConfig, InfoRMModel


@torch.no_grad()
def evaluate(
    model: InfoRMModel,
    data_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_bt = 0.0
    total_kl = 0.0
    total_correct = 0
    total = 0
    for batch in data_loader:
        chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
        rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model.forward_pair(chosen, rejected, sample_latent=False)
        total_loss += float(outputs["loss"].item()) * chosen["input_ids"].size(0)
        total_bt += float(outputs["bt_loss"].item()) * chosen["input_ids"].size(0)
        total_kl += float(outputs["kl_loss"].item()) * chosen["input_ids"].size(0)
        total_correct += (outputs["chosen_reward"] > outputs["rejected_reward"]).sum().item()
        total += chosen["input_ids"].size(0)
    return {
        "loss": total_loss / max(total, 1),
        "bt_loss": total_bt / max(total, 1),
        "kl_loss": total_kl / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train InfoRM on preference pairs.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--model_name", default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pooling", choices=["cls", "mean", "last_token"], default="cls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    return parser.parse_args()



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

    model = InfoRMModel(
        InfoRMConfig(
            base_model_name=args.model_name,
            latent_dim=args.latent_dim,
            beta=args.beta,
            dropout=args.dropout,
            pooling=args.pooling,
        )
    ).to(device)

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

    best_acc = -1.0
    global_step = 0
    optimizer_steps = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"inform epoch {epoch}")
        stop_early = False
        for batch in pbar:
            chosen = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected = {k: v.to(device) for k, v in batch["rejected"].items()}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model.forward_pair(chosen, rejected, sample_latent=True)
                loss = outputs["loss"] / args.grad_accum
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
                    "loss": f"{float(outputs['loss'].item()):.4f}",
                    "bt": f"{float(outputs['bt_loss'].item()):.4f}",
                    "kl": f"{float(outputs['kl_loss'].item()):.4f}",
                }
            )

        metrics = evaluate(model, val_loader, device, amp_dtype, use_amp)
        metrics["epoch"] = float(epoch)
        history.append(metrics)
        print(json.dumps(metrics, indent=2))

        if args.save_best and metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            model.save_pretrained(output_dir, tokenizer=tokenizer)
        if stop_early:
            break

    if not args.save_best:
        model.save_pretrained(output_dir, tokenizer=tokenizer)
    elif best_acc < 0.0:
        model.save_pretrained(output_dir, tokenizer=tokenizer)
        best_acc = history[-1]["accuracy"] if history else 0.0

    if best_acc < 0.0 and history:
        best_acc = max(item["accuracy"] for item in history)

    summary = {
        "model_name": args.model_name,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "seed": args.seed,
        "pooling": args.pooling,
        "global_step": global_step,
        "optimizer_steps": optimizer_steps,
        "best_val_accuracy": float(best_acc),
        "history": history,
    }
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
