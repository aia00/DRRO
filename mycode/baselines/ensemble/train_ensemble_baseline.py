#!/usr/bin/env python3
"""Train ensemble-based PPO baseline (mean/WCO/UWO) on HH-RLHF."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MYCODE_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if MYCODE_ROOT not in sys.path:
    sys.path.insert(0, MYCODE_ROOT)

import ray
from omegaconf import open_dict
from verl.trainer import main_ppo

from baselines.common.baseline_trainers import EnsembleBaselineTrainer
from baselines.common.io_utils import load_proxy_manifest, parse_proxy_rm_list
from baselines.ensemble.reward_ensemble import EnsembleRewardManager
from baselines.common.shared_config import (
    add_common_training_args,
    compose_verl_config,
    finalize_common_args,
    save_config_json,
    set_seed,
)
from baselines.common.trainer_common import BaseBaselineTaskRunner
from drro_train.drro_data import load_hh_prompts, prepare_dataset_files
from drro_train.drro_reward import HFRewardManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble proxy-RM baseline.")
    add_common_training_args(parser, adv_default="grpo")

    parser.add_argument("--proxy_rm_list", type=str, default="", help="Comma-separated proxy RM paths.")
    parser.add_argument("--proxy_rm_manifest", type=str, default="", help="JSON manifest with proxy RM member paths.")
    parser.add_argument(
        "--ensemble_agg",
        type=str,
        choices=["mean", "wco", "uwo"],
        default="uwo",
        help="Aggregation for ensemble reward.",
    )
    parser.add_argument("--uwo_lambda", type=float, default=1.0, help="Variance penalty for UWO aggregation.")

    args = parser.parse_args()
    return finalize_common_args(args)


def resolve_ensemble_models(args: argparse.Namespace) -> List[str]:
    if args.proxy_rm_manifest and args.proxy_rm_list:
        raise ValueError("Provide only one of --proxy_rm_manifest or --proxy_rm_list.")
    if args.proxy_rm_manifest:
        models = load_proxy_manifest(args.proxy_rm_manifest)
    elif args.proxy_rm_list:
        models = parse_proxy_rm_list(args.proxy_rm_list)
    else:
        models = [str(args.proxy_rm)]
    if not models:
        raise ValueError("Resolved empty proxy RM ensemble.")
    return models


class EnsembleTaskRunner(BaseBaselineTaskRunner):
    trainer_cls = EnsembleBaselineTrainer

    def build_reward_functions(self, config, tokenizer):
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        dtype_str = reward_kwargs.get("dtype")
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(dtype_str, torch.float32)

        ensemble_models = config.trainer.get("ensemble_proxy_models")
        if not ensemble_models:
            raise ValueError("Missing ensemble_proxy_models in config.trainer.")

        proxy_reward_fn = EnsembleRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            model_names=list(ensemble_models),
            aggregation=str(config.trainer.get("ensemble_agg", "uwo")),
            uwo_lambda=float(config.trainer.get("uwo_lambda", 1.0)),
            batch_size=int(reward_kwargs.get("reward_batch_size", 16)),
            max_length=int(reward_kwargs.get("reward_max_length", 512)),
            dtype=reward_dtype,
        )
        gold_reward_fn = HFRewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            model_name=reward_kwargs["gold_model"],
            batch_size=int(reward_kwargs.get("reward_batch_size", 16)),
            max_length=int(reward_kwargs.get("reward_max_length", 512)),
            dtype=reward_dtype,
        )

        return proxy_reward_fn, gold_reward_fn, {}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ensemble_models = resolve_ensemble_models(args)
    args.proxy_rm = ensemble_models[0]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    prompts = load_hh_prompts(args.dataset, args.local_dataset_dir)
    train_path, val_path = prepare_dataset_files(prompts, args.eval_prompts, args.seed, args.output_dir)

    cfg = compose_verl_config(
        args,
        train_path=train_path,
        val_path=val_path,
        method_name="ensemble_ppo",
        method_kwargs={
            "ensemble_agg": args.ensemble_agg,
            "uwo_lambda": args.uwo_lambda,
            "proxy_eval_key": "proxy_score",
        },
    )

    method_fields: Dict[str, Any] = {
        "ensemble_agg": args.ensemble_agg,
        "num_ensemble": float(len(ensemble_models)),
        "uwo_lambda": args.uwo_lambda,
    }
    with open_dict(cfg.trainer):
        cfg.trainer["baseline_method_fields"] = method_fields
        cfg.trainer["ensemble_proxy_models"] = list(ensemble_models)

    save_config_json(cfg, args.output_dir)

    remote_kwargs = {"num_cpus": 1}
    if args.reward_gpus:
        remote_kwargs["num_gpus"] = args.reward_gpus

    task_runner_class = ray.remote(**remote_kwargs)(EnsembleTaskRunner)
    main_ppo.run_ppo(cfg, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
