#!/usr/bin/env python3
"""Train BSPO on HH-RLHF using the current VERL-based repo."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MYCODE_ROOT = os.path.dirname(SCRIPT_DIR)
if MYCODE_ROOT not in sys.path:
    sys.path.insert(0, MYCODE_ROOT)

import ray
import torch
from omegaconf import open_dict
from verl.trainer import main_ppo

from baselines.common.shared_config import (
    add_common_training_args,
    compose_verl_config,
    finalize_common_args,
    save_config_json,
    set_seed,
    normalize_vllm_load_format,
)
from baselines.common.trainer_common import BaseBaselineTaskRunner
from bspo.advantage import enable_bspo
from bspo.reward_manager import ProxyScoreLMSupportRewardManager
from bspo.trainer import BSPORayPPOTrainer
from drro_train.drro_data import load_hh_prompts, prepare_dataset_files
from drro_train.drro_reward import HFRewardManager


def finalize_bspo_args(args: argparse.Namespace) -> argparse.Namespace:
    args = finalize_common_args(args)
    args.vllm_load_format = normalize_vllm_load_format(args.vllm_load_format)
    if args.reward_gpus is None:
        args.reward_gpus = 1 if args.num_gpus > 1 else 0
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BSPO baseline.")
    add_common_training_args(parser, adv_default="gae")
    parser.add_argument("--scorelm_path", required=True, help="Path to a trained ScoreLM checkpoint.")
    parser.add_argument(
        "--unsupported_value",
        type=float,
        default=-15.0,
        help="Pessimistic target value used for unsupported actions (paper default: -15).",
    )
    parser.add_argument(
        "--epsilon_beta",
        type=float,
        default=1e-4,
        help="Behavior-support threshold on next-token probability (paper default: 1e-4).",
    )
    parser.add_argument(
        "--scorelm_max_length",
        type=int,
        default=1024,
        help="Max length used by ScoreLM for reward/support evaluation.",
    )
    args = parser.parse_args()
    return finalize_bspo_args(args)


class BSPOTaskRunner(BaseBaselineTaskRunner):
    trainer_cls = BSPORayPPOTrainer

    def run(self, config):
        # Ray starts the trainer in a fresh process, so patch VERL's advantage
        # function inside that process as well as in the launcher process.
        enable_bspo(
            unsupported_value=float(config.trainer.get("unsupported_value", -15.0)),
            epsilon_beta=float(config.trainer.get("epsilon_beta", 1e-4)),
        )
        return super().run(config)

    def build_reward_functions(self, config, tokenizer):
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(reward_kwargs.get("dtype"), torch.float32)

        proxy_reward_fn = ProxyScoreLMSupportRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            proxy_model_name=reward_kwargs["proxy_model"],
            scorelm_path=str(config.trainer.get("scorelm_path")),
            reward_batch_size=int(reward_kwargs.get("reward_batch_size", 8)),
            reward_max_length=int(reward_kwargs.get("reward_max_length", 512)),
            scorelm_max_length=int(config.trainer.get("scorelm_max_length", reward_kwargs.get("reward_max_length", 1024))),
            epsilon_beta=float(config.trainer.get("epsilon_beta", 1e-4)),
            dtype=reward_dtype,
        )
        gold_reward_fn = HFRewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            model_name=reward_kwargs["gold_model"],
            batch_size=int(reward_kwargs.get("reward_batch_size", 8)),
            max_length=int(reward_kwargs.get("reward_max_length", 512)),
            dtype=reward_dtype,
        )
        return proxy_reward_fn, gold_reward_fn, {}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    enable_bspo(unsupported_value=args.unsupported_value, epsilon_beta=args.epsilon_beta)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    prompts = load_hh_prompts(args.dataset, args.local_dataset_dir)
    train_path, val_path = prepare_dataset_files(prompts, args.eval_prompts, args.seed, args.output_dir)

    cfg = compose_verl_config(
        args,
        train_path=train_path,
        val_path=val_path,
        method_name="bspo_like_proxy_ppo",
        method_kwargs={
            "proxy_eval_key": "proxy_score",
            "scorelm_path": args.scorelm_path,
            "unsupported_value": args.unsupported_value,
            "epsilon_beta": args.epsilon_beta,
            "scorelm_max_length": args.scorelm_max_length,
        },
    )

    method_fields: Dict[str, Any] = {
        "unsupported_value": args.unsupported_value,
        "epsilon_beta": args.epsilon_beta,
        "scorelm_max_length": float(args.scorelm_max_length),
        "scorelm_path": args.scorelm_path,
        "proxy_reward_model": args.proxy_rm,
    }
    with open_dict(cfg.trainer):
        cfg.trainer["baseline_method_fields"] = method_fields
        cfg.trainer["scorelm_path"] = args.scorelm_path
        cfg.trainer["unsupported_value"] = args.unsupported_value
        cfg.trainer["epsilon_beta"] = args.epsilon_beta
        cfg.trainer["scorelm_max_length"] = args.scorelm_max_length
    with open_dict(cfg.reward):
        cfg.reward["reward_kwargs"] = {
            "proxy_model": args.proxy_rm,
            "scorelm_path": args.scorelm_path,
            "reward_batch_size": args.reward_batch_size,
            "reward_max_length": args.reward_max_length,
            "scorelm_max_length": args.scorelm_max_length,
            "epsilon_beta": args.epsilon_beta,
            "dtype": "bfloat16" if args.bf16 else "float16" if args.fp16 else "float32",
        }
        cfg.reward.reward_manager["source"] = "importlib"
        cfg.reward.reward_manager["name"] = "LoopProxyScoreLMSupportRewardManager"
        cfg.reward.reward_manager.module["path"] = os.path.join(SCRIPT_DIR, "reward_manager.py")

    save_config_json(cfg, args.output_dir)

    remote_kwargs = {"num_cpus": 1}
    if args.reward_gpus:
        remote_kwargs["num_gpus"] = args.reward_gpus

    task_runner_class = ray.remote(**remote_kwargs)(BSPOTaskRunner)
    main_ppo.run_ppo(cfg, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
