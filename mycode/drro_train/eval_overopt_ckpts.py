#!/usr/bin/env python3
"""Evaluate saved checkpoints to produce KL_seq-based over-optimization logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, Iterable, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DRRO checkpoints for KL_seq plots.")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory with config.json and val_prompts.json.")
    parser.add_argument("--output_csv", type=str, default="", help="Output CSV path (default: run_dir/eval_log.csv).")
    parser.add_argument("--max_eval_prompts", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_list_to_text(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"\n\nHuman: {content}")
        elif role == "assistant":
            parts.append(f"\n\nAssistant: {content}")
    parts.append("\n\nAssistant:")
    return "".join(parts)


def load_eval_prompts(path: str, max_items: int) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            prompts.append(prompt_list_to_text(record["prompt"]))
            if max_items > 0 and len(prompts) >= max_items:
                break
    return prompts


def init_policy(model_name: str, dtype: torch.dtype, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def init_reward_model(model_name: str, dtype: torch.dtype, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=dtype)
    if model.config.pad_token_id is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def generate(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[str]:
    enc = tok(prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
    completions: List[str] = []
    prompt_lens = attention_mask.sum(dim=1).tolist()
    for seq, p_len in zip(gen, prompt_lens):
        comp_ids = seq[int(p_len) :]
        completions.append(tok.decode(comp_ids, skip_special_tokens=True))
    return completions


def compute_logp(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    texts = [p + c for p, c in zip(prompts, completions)]
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_enc = tok(prompts, padding=True, truncation=True, return_tensors="pt")
    prompt_lens = prompt_enc["attention_mask"].sum(dim=1).to(device)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    positions = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    token_mask = positions >= prompt_lens.unsqueeze(1)
    logprob_mask = token_mask[:, 1:]
    token_logp = token_logp * logprob_mask
    sum_logp = token_logp.sum(dim=1)
    token_count = logprob_mask.sum(dim=1).clamp(min=1)
    return sum_logp, token_count


def score_rewards(
    model: AutoModelForSequenceClassification,
    tok: AutoTokenizer,
    prompts: List[str],
    completions: List[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    with torch.no_grad():
        for pb, cb in zip(batch_iter(prompts, batch_size), batch_iter(completions, batch_size)):
            enc = tok(pb, cb, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.squeeze(-1)
            scores.append(logits.float().cpu())
    return torch.cat(scores, dim=0)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    output_csv = args.output_csv or os.path.join(run_dir, "eval_log.csv")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_json(os.path.join(run_dir, "config.json"))
    model_name = cfg["actor_rollout_ref"]["model"]["path"]
    reward_kwargs = cfg["reward_model"]["reward_kwargs"]
    proxy_rm_name = reward_kwargs["proxy_model"]
    gold_rm_name = reward_kwargs["gold_model"]
    delta = cfg.get("trainer", {}).get(
        "fixed_delta",
        cfg.get("trainer", {}).get("drro_delta", cfg.get("trainer", {}).get("delta", 0.0)),
    )
    beta_kl = cfg.get("trainer", {}).get("drro_beta_kl", 0.0)

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    base_model, tok = init_policy(model_name, dtype, device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    proxy_rm, proxy_tok = init_reward_model(proxy_rm_name, dtype, device)
    gold_rm, gold_tok = init_reward_model(gold_rm_name, dtype, device)

    eval_prompts = load_eval_prompts(os.path.join(run_dir, "val_prompts.json"), args.max_eval_prompts)

    # Baseline (step 0) for normalization.
    base_completions = []
    for batch in batch_iter(eval_prompts, args.batch_size):
        base_completions.extend(
            generate(base_model, tok, batch, args.max_new_tokens, args.temperature, args.top_p, device)
        )
    proxy_mean0 = float(score_rewards(proxy_rm, proxy_tok, eval_prompts, base_completions, device, args.batch_size).mean())
    gold_scores0 = score_rewards(gold_rm, gold_tok, eval_prompts, base_completions, device, args.batch_size)
    gold_mean0 = float(gold_scores0.mean())
    gold_std0 = float(gold_scores0.std()) if gold_scores0.numel() > 1 else 1.0

    ckpt_dirs = [
        d
        for d in os.listdir(run_dir)
        if re.match(r"global_step_\d+", d) and os.path.isdir(os.path.join(run_dir, d))
    ]
    ckpt_dirs.sort(key=lambda x: int(x.split("_")[-1]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "kl_seq",
                "kl_per_token",
                "proxy_raw",
                "gold_raw",
                "proxy_norm",
                "gold_norm",
                "delta",
                "beta_kl",
            ]
        )

        policy = None
        total_ckpts = len(ckpt_dirs)
        start_time = time.time()
        for idx, ckpt in enumerate(ckpt_dirs, start=1):
            step = int(ckpt.split("_")[-1])
            adapter_path = os.path.join(run_dir, ckpt, "actor", "lora_adapter")
            if not os.path.isdir(adapter_path):
                continue
            ckpt_start = time.time()
            print(f"[eval] {idx}/{total_ckpts} step={step} adapter={adapter_path}", flush=True)
            if policy is not None:
                if hasattr(policy, "unload"):
                    base_model = policy.unload()
                policy = None
                if hasattr(base_model, "peft_config"):
                    delattr(base_model, "peft_config")
                if hasattr(base_model, "active_adapter"):
                    delattr(base_model, "active_adapter")
                torch.cuda.empty_cache()
            policy = PeftModel.from_pretrained(base_model, adapter_path)
            if hasattr(policy, "set_adapter"):
                try:
                    policy.set_adapter("default")
                except ValueError:
                    pass
            policy.to(device)
            policy.eval()

            completions = []
            for batch in batch_iter(eval_prompts, args.batch_size):
                completions.extend(
                    generate(policy, tok, batch, args.max_new_tokens, args.temperature, args.top_p, device)
                )

            logp_new_sum, token_count = compute_logp(policy, tok, eval_prompts, completions, device)
            logp_ref_sum, _ = compute_logp(ref_model, tok, eval_prompts, completions, device)
            kl_seq = (logp_new_sum - logp_ref_sum).mean().item()
            kl_per_token = ((logp_new_sum - logp_ref_sum) / token_count).mean().item()

            proxy_raw = float(
                score_rewards(proxy_rm, proxy_tok, eval_prompts, completions, device, args.batch_size).mean()
            )
            gold_raw = float(
                score_rewards(gold_rm, gold_tok, eval_prompts, completions, device, args.batch_size).mean()
            )
            proxy_norm = proxy_raw - proxy_mean0
            gold_norm = (gold_raw - gold_mean0) / (gold_std0 + 1e-8)

            writer.writerow(
                [step, kl_seq, kl_per_token, proxy_raw, gold_raw, proxy_norm, gold_norm, delta, beta_kl]
            )
            ckpt_time = time.time() - ckpt_start
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = avg_time * (total_ckpts - idx)
            print(
                f"[eval] step={step} kl_seq={kl_seq:.4f} kl_tok={kl_per_token:.4f} "
                f"done in {ckpt_time:.1f}s | "
                f"avg {avg_time:.1f}s | eta {remaining/60:.1f}m",
                flush=True,
            )


if __name__ == "__main__":
    main()
