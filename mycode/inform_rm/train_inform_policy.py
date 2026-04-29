#!/usr/bin/env python3
"""Run downstream policy training using a trained InfoRM checkpoint as the proxy reward."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
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
)
from baselines.common.trainer_common import BaseBaselineTaskRunner
from drro_train.drro_data import load_hh_prompts, prepare_dataset_files
from drro_train.drro_reward import HFRewardManager
from inform_rm.reward_manager import ProxyInfoRMPenaltyRewardManager
from inform_rm.trainer import InfoRMBaselineTrainer


def finalize_inform_args(args: argparse.Namespace) -> argparse.Namespace:
    args = finalize_common_args(args)
    if args.reward_gpus is None:
        args.reward_gpus = 1 if args.num_gpus > 1 else 0
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a policy against a trained InfoRM proxy reward.")
    add_common_training_args(parser, adv_default="gae")
    parser.add_argument("--inform_rm_path", required=True, help="Path to a trained InfoRM checkpoint.")
    parser.add_argument(
        "--inform_max_length",
        type=int,
        default=512,
        help="Max prompt+response length fed into the InfoRM proxy.",
    )
    parser.add_argument(
        "--inform_penalty_coef",
        type=float,
        default=0.01,
        help="Penalty coefficient for InfoRM latent KL when shaping proxy RM reward.",
    )
    args = parser.parse_args()
    return finalize_inform_args(args)


def load_inform_metadata(model_dir: str) -> Dict[str, Any]:
    config_path = Path(model_dir) / "inform_config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): value for key, value in payload.items()}


class InfoRMTaskRunner(BaseBaselineTaskRunner):
    trainer_cls = InfoRMBaselineTrainer

    def build_reward_functions(self, config, tokenizer):
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(reward_kwargs.get("dtype"), torch.float32)

        proxy_reward_fn = ProxyInfoRMPenaltyRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            proxy_model_name=reward_kwargs["proxy_model"],
            inform_rm_path=str(config.trainer.get("inform_rm_path")),
            reward_batch_size=int(reward_kwargs.get("reward_batch_size", 16)),
            reward_max_length=int(reward_kwargs.get("reward_max_length", 512)),
            inform_max_length=int(config.trainer.get("inform_max_length", reward_kwargs.get("reward_max_length", 512))),
            inform_penalty_coef=float(config.trainer.get("inform_penalty_coef", 0.01)),
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

    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    prompts = load_hh_prompts(args.dataset, args.local_dataset_dir)
    train_path, val_path = prepare_dataset_files(prompts, args.eval_prompts, args.seed, args.output_dir)

    method_name = f"inform_proxy_penalty_{args.adv_estimator}"
    cfg = compose_verl_config(
        args,
        train_path=train_path,
        val_path=val_path,
        method_name=method_name,
        method_kwargs={
            "proxy_eval_key": "proxy_score",
            "inform_rm_path": args.inform_rm_path,
            "inform_max_length": args.inform_max_length,
            "inform_penalty_coef": args.inform_penalty_coef,
        },
    )

    inform_meta = load_inform_metadata(args.inform_rm_path)
    method_fields: Dict[str, Any] = {
        "inform_rm_path": args.inform_rm_path,
        "inform_max_length": args.inform_max_length,
        "inform_penalty_coef": args.inform_penalty_coef,
        "proxy_reward_model": args.proxy_rm,
    }
    if "latent_dim" in inform_meta:
        method_fields["inform_latent_dim"] = inform_meta["latent_dim"]
    if "beta" in inform_meta:
        method_fields["inform_beta"] = inform_meta["beta"]
    if "pooling" in inform_meta:
        method_fields["inform_pooling"] = inform_meta["pooling"]
    if "base_model_name" in inform_meta:
        method_fields["inform_base_model"] = inform_meta["base_model_name"]

    with open_dict(cfg.trainer):
        cfg.trainer["baseline_method_fields"] = method_fields
        cfg.trainer["inform_rm_path"] = args.inform_rm_path
        cfg.trainer["inform_max_length"] = args.inform_max_length
        cfg.trainer["inform_penalty_coef"] = args.inform_penalty_coef
    with open_dict(cfg.reward):
        cfg.reward["reward_kwargs"] = {
            "proxy_model": args.proxy_rm,
            "inform_rm_path": args.inform_rm_path,
            "reward_batch_size": args.reward_batch_size,
            "reward_max_length": args.reward_max_length,
            "inform_max_length": args.inform_max_length,
            "inform_penalty_coef": args.inform_penalty_coef,
            "dtype": "bfloat16" if args.bf16 else "float16" if args.fp16 else "float32",
        }
        cfg.reward.reward_manager["source"] = "importlib"
        cfg.reward.reward_manager["name"] = "LoopProxyInfoRMPenaltyRewardManager"
        cfg.reward.reward_manager.module["path"] = os.path.join(SCRIPT_DIR, "reward_manager.py")

    save_config_json(cfg, args.output_dir)

    remote_kwargs = {"num_cpus": 1}
    if args.reward_gpus:
        remote_kwargs["num_gpus"] = args.reward_gpus

    task_runner_class = ray.remote(**remote_kwargs)(InfoRMTaskRunner)
    main_ppo.run_ppo(cfg, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
