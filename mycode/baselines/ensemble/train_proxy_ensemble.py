#!/usr/bin/env python3
"""Train/export proxy RM ensemble members and write manifest JSON."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import List


DEFAULT_SMALL_MODELS = [
    "microsoft/MiniLM-L12-H384-uncased",
    "prajjwal1/bert-small",
    "google/electra-small-discriminator",
    "distilbert-base-uncased",
    "distilroberta-base",
]


def parse_args() -> argparse.Namespace:
    default_train_script = str((Path(__file__).resolve().parents[2] / "proxy_rm" / "train_proxy_rm.py"))
    parser = argparse.ArgumentParser(description="Train multi-seed proxy RM ensemble.")
    parser.add_argument("--train_jsonl", type=str, default="")
    parser.add_argument("--val_jsonl", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/MiniLM-L12-H384-uncased")
    parser.add_argument(
        "--member_model_list",
        type=str,
        default="",
        help="Comma-separated base models for ensemble members. Overrides --model_name/--num_members when set.",
    )
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
        "--parallel_workers",
        type=int,
        default=1,
        help="How many member trainings to run in parallel.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help="Comma-separated CUDA device ids for member training (e.g. 0,1,2,3).",
    )
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


def parse_gpu_ids(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_member_models(args: argparse.Namespace) -> List[str]:
    if args.member_model_list.strip():
        models = [item.strip() for item in args.member_model_list.split(",") if item.strip()]
        if not models:
            raise ValueError("--member_model_list is set but empty after parsing.")
        return models
    if args.num_members <= 0:
        raise ValueError("--num_members must be > 0.")
    return [args.model_name] * args.num_members


def run_member_training(
    args: argparse.Namespace,
    seed: int,
    member_dir: Path,
    model_name: str,
    gpu_id: str | None,
) -> None:
    cmd = [
        sys.executable,
        args.train_script,
        "--train_jsonl",
        args.train_jsonl,
        "--val_jsonl",
        args.val_jsonl,
        "--model_name",
        model_name,
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

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tag = f"gpu={gpu_id}" if gpu_id is not None else "gpu=default"
    print("[proxy-ensemble]", tag, " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def summarize_member_metrics(members: List[dict], output_dir: Path) -> dict:
    rows = []
    accuracies = []
    for member in members:
        metrics_path = Path(member["path"]) / "train_metrics.json"
        row = {
            "id": member.get("id"),
            "seed": member.get("seed"),
            "model_name": member.get("model_name"),
            "path": member.get("path"),
            "metrics_path": str(metrics_path),
            "final_val_accuracy": None,
            "best_val_accuracy": None,
        }
        if metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            final_acc = payload.get("final_val_accuracy")
            best_acc = payload.get("best_val_accuracy")
            row["final_val_accuracy"] = final_acc
            row["best_val_accuracy"] = best_acc
            if best_acc is not None:
                accuracies.append(float(best_acc))
        rows.append(row)

    summary = {
        "num_members": len(rows),
        "num_with_metrics": len(accuracies),
        "members": rows,
    }
    if accuracies:
        summary.update(
            {
                "best_val_accuracy_mean": float(statistics.fmean(accuracies)),
                "best_val_accuracy_std": float(statistics.pstdev(accuracies)) if len(accuracies) > 1 else 0.0,
                "best_val_accuracy_min": float(min(accuracies)),
                "best_val_accuracy_max": float(max(accuracies)),
            }
        )

    out_path = output_dir / "proxy_ensemble_training_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[proxy-ensemble] wrote summary: {out_path}", flush=True)
    if accuracies:
        print(
            "[proxy-ensemble] agreement(best val acc): "
            f"mean={summary['best_val_accuracy_mean']:.4f}, "
            f"std={summary['best_val_accuracy_std']:.4f}, "
            f"min={summary['best_val_accuracy_min']:.4f}, "
            f"max={summary['best_val_accuracy_max']:.4f}",
            flush=True,
        )
    return summary


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_dir / "proxy_ensemble_manifest.json"

    pretrained_models = parse_pretrained_list(args.pretrained_list)
    members = []
    member_models = parse_member_models(args)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if args.parallel_workers <= 0:
        raise ValueError("--parallel_workers must be > 0.")
    if args.parallel_workers > 1 and not gpu_ids:
        print(
            "[proxy-ensemble] parallel_workers > 1 but no --gpu_ids set; jobs may contend on one GPU.",
            flush=True,
        )
    if gpu_ids and args.parallel_workers > len(gpu_ids):
        print(
            f"[proxy-ensemble] parallel_workers={args.parallel_workers} > gpu_ids={len(gpu_ids)}; "
            "multiple jobs will share some GPUs.",
            flush=True,
        )

    if pretrained_models:
        for idx, path in enumerate(pretrained_models):
            members.append({"id": idx, "seed": None, "path": path})
    else:
        if not args.train_jsonl or not args.val_jsonl:
            raise ValueError("--train_jsonl and --val_jsonl are required when --pretrained_list is not provided.")

        seeds = parse_seeds(args.seeds, len(member_models))
        train_specs: list[tuple[int, Path, str, str | None]] = []
        for idx, (seed, model_name) in enumerate(zip(seeds, member_models)):
            member_dir = output_dir / f"member_{idx:02d}_seed{seed}"
            member_dir.mkdir(parents=True, exist_ok=True)
            gpu_id = gpu_ids[idx % len(gpu_ids)] if gpu_ids else None
            train_specs.append((seed, member_dir, model_name, gpu_id))
            members.append({"id": idx, "seed": seed, "model_name": model_name, "path": str(member_dir)})

        if args.parallel_workers == 1:
            for seed, member_dir, model_name, gpu_id in train_specs:
                run_member_training(
                    args=args,
                    seed=seed,
                    member_dir=member_dir,
                    model_name=model_name,
                    gpu_id=gpu_id,
                )
        else:
            with ThreadPoolExecutor(max_workers=args.parallel_workers) as pool:
                futures = [
                    pool.submit(
                        run_member_training,
                        args,
                        seed,
                        member_dir,
                        model_name,
                        gpu_id,
                    )
                    for seed, member_dir, model_name, gpu_id in train_specs
                ]
                for future in as_completed(futures):
                    future.result()

    manifest = {
        "default_model_name": args.model_name,
        "num_members": len(members),
        "members": members,
        "models": [item["path"] for item in members],
        "default_small_model_mix": DEFAULT_SMALL_MODELS,
    }
    summary = summarize_member_metrics(members, output_dir)
    if summary.get("num_with_metrics", 0) > 0:
        manifest["member_best_val_acc_mean"] = summary.get("best_val_accuracy_mean")
        manifest["member_best_val_acc_std"] = summary.get("best_val_accuracy_std")
    manifest["training_summary"] = str(output_dir / "proxy_ensemble_training_summary.json")
    manifest_out.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[proxy-ensemble] wrote manifest: {manifest_out}")


if __name__ == "__main__":
    main()
