"""Shared CLI/config helpers for baseline training entrypoints."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from baselines.common.paths import get_path_config, get_verl_config_dir
PATH_CFG = get_path_config()

from drro_train.drro_config import build_config as build_drro_config


def normalize_vllm_load_format(load_format: str) -> str:
    # vLLM 0.11+ removed the old "hf" alias. Preserve backward compatibility.
    return "auto" if load_format == "hf" else load_format


def add_common_training_args(parser: argparse.ArgumentParser, adv_default: str = "gae") -> None:
    """Add baseline training args compatible with existing DRRO scripts."""
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/hh-rlhf")
    parser.add_argument(
        "--local_dataset_dir",
        type=str,
        default=PATH_CFG.get("DRRO_LOCAL_DATASET_DIR", ""),
    )
    parser.add_argument("--policy_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--proxy_rm", type=str, default="OpenAssistant/reward-model-deberta-v3-base")
    parser.add_argument(
        "--gold_rm",
        type=str,
        default="sileod/deberta-v3-large-tasksource-rlhf-reward-model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PATH_CFG.get("DRRO_OUTPUT_ROOT", "runs"), "baseline"),
    )
    parser.add_argument("--num_steps", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--eval_prompts", type=int, default=512)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=0,
        help="Validation batch size (0 = use --batch_size_prompts).",
    )
    parser.add_argument("--batch_size_prompts", type=int, default=12)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_prompt_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument(
        "--rollout_backend",
        type=str,
        choices=["hf", "vllm"],
        default="vllm",
        help="Rollout backend for generation.",
    )
    parser.add_argument(
        "--vllm_tensor_parallel",
        type=int,
        default=0,
        help="Tensor parallel size for vLLM. Set 0 for auto.",
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=16384)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=1536)
    parser.add_argument("--vllm_max_model_len", type=int, default=0)
    parser.add_argument("--vllm_logprobs_mode", type=str, default="processed_logprobs")
    parser.add_argument(
        "--vllm_load_format",
        type=str,
        choices=["auto", "hf", "safetensors", "dummy"],
        default="auto",
        help="Weight loading mode for vLLM rollout.",
    )
    parser.add_argument("--vllm_enforce_eager", action="store_true", default=False)
    parser.add_argument("--vllm_enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--vllm_disable_prefix_caching", action="store_false", dest="vllm_enable_prefix_caching")
    parser.add_argument("--vllm_enable_chunked_prefill", action="store_true", default=True)
    parser.add_argument("--vllm_disable_chunked_prefill", action="store_false", dest="vllm_enable_chunked_prefill")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true", default=False)
    parser.add_argument("--vllm_disable_sleep_mode", action="store_false", dest="vllm_enable_sleep_mode")
    parser.add_argument("--vllm_free_cache_engine", action="store_true", default=False)
    parser.add_argument("--vllm_disable_free_cache_engine", action="store_false", dest="vllm_free_cache_engine")
    parser.add_argument("--vllm_layered_summon", action="store_true", default=True)
    parser.add_argument("--vllm_disable_layered_summon", action="store_false", dest="vllm_layered_summon")

    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["sdpa", "eager", "flash_attention_2"],
        default="flash_attention_2",
        help="Attention backend for the policy model.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--beta_kl", type=float, default=0.0)
    parser.add_argument(
        "--adv_estimator",
        type=str,
        choices=["grpo", "gae"],
        default=adv_default,
        help="Advantage estimator: gae for PPO baseline, grpo for GRPO-style baseline.",
    )
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1.0)

    # Keep DRRO knobs for compatibility with shared config composition; unused here.
    parser.add_argument("--fixed_delta", "--delta", dest="fixed_delta", type=float, default=None)
    parser.add_argument("--dynamic_delta_coeff", "--delta_alpha", dest="dynamic_delta_coeff", type=float, default=0.0)
    parser.add_argument("--dynamic_kl_window", "--delta_tau", dest="dynamic_kl_window", type=int, default=20)
    parser.add_argument("--soft_assign_tau", "--delta_softmax_tau", dest="soft_assign_tau", type=float, default=2.0)
    parser.add_argument(
        "--assign_mode",
        type=str,
        choices=["soft", "hard"],
        default="soft",
        help=(
            "DRRO assignment rule. "
            "'hard' = argmax_i(r_i - delta * p_i), winner gets +delta. "
            "'soft' = softmax/SNIS bonus over the same score r_i - delta * p_i."
        ),
    )
    parser.add_argument(
        "--robust_objective",
        type=str,
        choices=["drro", "dro"],
        default="drro",
        help="Compatibility knob for DRRO/DRO delta add-on composition.",
    )
    parser.add_argument(
        "--dynamic_kl_estimator",
        "--delta_kl_estimator",
        dest="dynamic_kl_estimator",
        type=str,
        choices=["k1", "k2", "k3"],
        default="k3",
    )
    parser.add_argument("--dynamic_delta_min", "--delta_min", dest="dynamic_delta_min", type=float, default=0.0)
    parser.add_argument("--dynamic_delta_max", "--delta_max", dest="dynamic_delta_max", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=3)
    parser.add_argument(
        "--reward_gpus",
        type=int,
        default=None,
        help="GPUs reserved for reward model inference (default: auto; 1 if --num_gpus > 1).",
    )
    parser.add_argument("--agent_workers", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--reward_batch_size", type=int, default=16)
    parser.add_argument("--reward_max_length", type=int, default=512)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--optimizer_offload", action="store_true", default=False)
    parser.add_argument("--no_optimizer_offload", action="store_false", dest="optimizer_offload")
    parser.add_argument("--param_offload", action="store_true", default=False)
    parser.add_argument("--no_param_offload", action="store_false", dest="param_offload")
    parser.add_argument("--grad_offload", action="store_true", default=False)
    parser.add_argument("--no_grad_offload", action="store_false", dest="grad_offload")
    parser.add_argument("--torch_compile", action="store_true", default=False)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="drro-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--recalibrate_for_plot",
        action="store_true",
        default=False,
        help="Normalize rewards using step-0 baseline for plotting.",
    )


def finalize_common_args(args: argparse.Namespace) -> argparse.Namespace:
    args.vllm_load_format = normalize_vllm_load_format(args.vllm_load_format)
    if args.fixed_delta is None:
        args.fixed_delta = 2.5 * args.num_generations
    args.delta = args.fixed_delta
    args.delta_alpha = args.dynamic_delta_coeff
    args.delta_tau = args.dynamic_kl_window
    args.delta_softmax_tau = args.soft_assign_tau
    args.delta_kl_estimator = args.dynamic_kl_estimator
    args.delta_min = args.dynamic_delta_min
    args.delta_max = args.dynamic_delta_max
    if args.beta_kl != 0.0:
        print("beta_kl is forced to 0.0 (no KL penalty in loss).")
        args.beta_kl = 0.0
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compose_verl_config(
    args: argparse.Namespace,
    train_path: str,
    val_path: str,
    method_name: str,
    method_kwargs: Optional[dict[str, object]] = None,
):
    """Compose baseline VERL config using the existing DRRO config builder."""
    config_dir = get_verl_config_dir()
    if config_dir is None:
        raise RuntimeError("Could not locate installed `verl` trainer config directory.")
    cfg = build_drro_config(args, train_path, val_path, config_dir)

    with open_dict(cfg.trainer):
        cfg.trainer["log_csv_path"] = os.path.join(args.output_dir, "log.csv")
        cfg.trainer["baseline_method"] = method_name
        cfg.trainer["drro_beta_kl"] = 0.0

    if method_kwargs:
        with open_dict(cfg.trainer):
            for key, value in method_kwargs.items():
                cfg.trainer[key] = value

    return cfg


def save_config_json(cfg, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(OmegaConf.to_container(cfg, resolve=True), handle, indent=2, ensure_ascii=True)
    return config_path
