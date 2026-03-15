#!/usr/bin/env python3
"""Train/export proxy RM ensemble members and write manifest JSON."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    default_train_script = str((Path(__file__).resolve().parents[2] / "proxy_rm" / "train_proxy_rm.py"))
    parser = argparse.ArgumentParser(description="Train multi-seed proxy RM ensemble.")
    parser.add_argument("--train_jsonl", type=str, default="")
    parser.add_argument("--val_jsonl", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument("--num_members", type=int, default=5)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_best", action="store_true")
    parser.add_argument(
        "--train_script",
        type=str,
        default=default_train_script,
        help="Path to existing proxy RM training script.",
    )
    parser.add_argument(
        "--pretrained_list",
        type=str,
        default="",
        help="Optional comma-separated existing proxy RM paths; if set, training is skipped and only manifest is written.",
    )
    parser.add_argument("--manifest_out", type=str, default="")
    return parser.parse_args()


def parse_seeds(seed_text: str, num_members: int) -> List[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        seeds = [42]
    while len(seeds) < num_members:
        seeds.append(seeds[-1] + 1)
    return seeds[:num_members]


def parse_pretrained_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def run_member_training(args: argparse.Namespace, seed: int, member_dir: Path) -> None:
    cmd = [
        sys.executable,
        args.train_script,
        "--train_jsonl",
        args.train_jsonl,
        "--val_jsonl",
        args.val_jsonl,
        "--model_name",
        args.model_name,
        "--output_dir",
        str(member_dir),
        "--batch_size",
        str(args.batch_size),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--max_length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--max_steps",
        str(args.max_steps),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_accum",
        str(args.grad_accum),
        "--seed",
        str(seed),
    ]
    if args.bf16:
        cmd.append("--bf16")
    if args.fp16:
        cmd.append("--fp16")
    if args.save_best:
        cmd.append("--save_best")

    print("[proxy-ensemble]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_dir / "proxy_ensemble_manifest.json"

    pretrained_models = parse_pretrained_list(args.pretrained_list)
    members = []

    if pretrained_models:
        for idx, path in enumerate(pretrained_models):
            members.append({"id": idx, "seed": None, "path": path})
    else:
        if not args.train_jsonl or not args.val_jsonl:
            raise ValueError("--train_jsonl and --val_jsonl are required when --pretrained_list is not provided.")

        seeds = parse_seeds(args.seeds, args.num_members)
        for idx, seed in enumerate(seeds):
            member_dir = output_dir / f"member_{idx:02d}_seed{seed}"
            member_dir.mkdir(parents=True, exist_ok=True)
            run_member_training(args=args, seed=seed, member_dir=member_dir)
            members.append({"id": idx, "seed": seed, "path": str(member_dir)})

    manifest = {
        "model_name": args.model_name,
        "num_members": len(members),
        "members": members,
        "models": [item["path"] for item in members],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[proxy-ensemble] wrote manifest: {manifest_out}")


if __name__ == "__main__":
    main()
