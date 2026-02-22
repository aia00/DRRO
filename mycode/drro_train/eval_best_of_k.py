#!/usr/bin/env python3
"""Evaluate best-of-k sampling with proxy/gold reward models."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-of-k evaluation with proxy/gold reward models.")
    parser.add_argument("--run_dir", type=str, default="", help="Run dir with config.json + val_prompts.json.")
    parser.add_argument("--prompts_jsonl", type=str, default="", help="JSONL prompts (val_prompts.json).")
    parser.add_argument("--output_csv", type=str, default="", help="Output CSV path.")
    parser.add_argument("--policy_model", type=str, default="", help="Policy model name or path.")
    parser.add_argument("--proxy_rm", type=str, default="", help="Proxy reward model name or path.")
    parser.add_argument("--gold_rm", type=str, default="", help="Gold reward model name or path.")
    parser.add_argument("--n_list", type=str, default="1,2,4,8,16", help="Comma-separated list of k values.")
    parser.add_argument("--max_eval_prompts", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
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
            if isinstance(record.get("prompt"), list):
                prompts.append(prompt_list_to_text(record["prompt"]))
            else:
                prompts.append(record["prompt"])
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


def generate_n(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> List[List[str]]:
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
            num_return_sequences=n,
        )
    prompt_lens = attention_mask.sum(dim=1).tolist()
    completions: List[List[str]] = [[] for _ in prompts]
    for idx, seq in enumerate(gen):
        p_idx = idx // n
        p_len = prompt_lens[p_idx]
        comp_ids = seq[int(p_len) :]
        completions[p_idx].append(tok.decode(comp_ids, skip_special_tokens=True))
    return completions


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    run_dir = args.run_dir
    if run_dir:
        cfg = load_json(os.path.join(run_dir, "config.json"))
        policy_model = cfg["actor_rollout_ref"]["model"]["path"]
        reward_kwargs = cfg["reward_model"]["reward_kwargs"]
        proxy_rm = reward_kwargs["proxy_model"]
        gold_rm = reward_kwargs["gold_model"]
        prompts_jsonl = os.path.join(run_dir, "val_prompts.json")
        output_csv = args.output_csv or os.path.join(run_dir, "bon_eval.csv")
    else:
        policy_model = args.policy_model
        proxy_rm = args.proxy_rm
        gold_rm = args.gold_rm
        prompts_jsonl = args.prompts_jsonl
        output_csv = args.output_csv

    if not policy_model or not proxy_rm or not gold_rm or not prompts_jsonl or not output_csv:
        raise ValueError("Missing required inputs. Provide --run_dir or all of --policy_model/--proxy_rm/--gold_rm/--prompts_jsonl/--output_csv.")

    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]
    eval_prompts = load_eval_prompts(prompts_jsonl, args.max_eval_prompts)

    policy, tok = init_policy(policy_model, dtype, device)
    proxy_model, proxy_tok = init_reward_model(proxy_rm, dtype, device)
    gold_model, gold_tok = init_reward_model(gold_rm, dtype, device)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "proxy_mean", "gold_mean", "avg_logp", "avg_logp_per_token"])

        for n in n_list:
            selected_completions: List[str] = []
            selected_proxy_scores: List[float] = []
            for batch in batch_iter(eval_prompts, args.batch_size):
                comps = generate_n(
                    policy,
                    tok,
                    batch,
                    n,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    device,
                )
                flat_prompts: List[str] = []
                flat_comps: List[str] = []
                for p, clist in zip(batch, comps):
                    for c in clist:
                        flat_prompts.append(p)
                        flat_comps.append(c)
                proxy_scores = score_rewards(proxy_model, proxy_tok, flat_prompts, flat_comps, device, args.batch_size)
                proxy_scores = proxy_scores.view(len(batch), n)
                for row, clist in zip(proxy_scores, comps):
                    best_idx = int(torch.argmax(row).item())
                    selected_completions.append(clist[best_idx])
                    selected_proxy_scores.append(float(row[best_idx].item()))

            gold_scores = score_rewards(gold_model, gold_tok, eval_prompts, selected_completions, device, args.batch_size)
            logp_sum, token_count = compute_logp(policy, tok, eval_prompts, selected_completions, device)
            avg_logp = float(logp_sum.mean().item())
            avg_logp_per_token = float((logp_sum / token_count).mean().item())

            writer.writerow(
                [
                    n,
                    float(sum(selected_proxy_scores) / max(len(selected_proxy_scores), 1)),
                    float(gold_scores.mean().item()),
                    avg_logp,
                    avg_logp_per_token,
                ]
            )


if __name__ == "__main__":
    main()
