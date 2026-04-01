#!/usr/bin/env python3
"""Train CMDP-style constraint PPO baselines (mu/xi) on HH-RLHF."""

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

from baselines.common.baseline_trainers import ConstraintBaselineTrainer
from baselines.constraint.reward_components import (
    ComponentSpec,
    ConstraintRewardManager,
    load_component_config,
    load_theta_json,
)
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
    parser = argparse.ArgumentParser(description="Train constraint RLHF baseline (mu/xi PPO).")
    add_common_training_args(parser, adv_default="gae")

    parser.add_argument("--component_config_json", type=str, required=True)
    parser.add_argument("--theta_json", type=str, required=True)
    parser.add_argument("--constraint_mode", type=str, choices=["mu", "xi"], default="mu")
    parser.add_argument("--dual_lr", type=float, default=0.05)
    parser.add_argument("--dual_ema", type=float, default=0.9)
    parser.add_argument("--dual_clip", type=float, default=10.0)

    args = parser.parse_args()
    return finalize_common_args(args)


class ConstraintTaskRunner(BaseBaselineTaskRunner):
    trainer_cls = ConstraintBaselineTrainer

    def build_reward_functions(self, config, tokenizer):
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        reward_dtype = dtype_map.get(reward_kwargs.get("dtype"), torch.float32)

        specs_raw = config.trainer.get("component_specs")
        if not specs_raw:
            raise ValueError("Missing component_specs in config.trainer.")
        components = [ComponentSpec(name=str(item["name"]), model=str(item["model"])) for item in specs_raw]

        objective_name = str(config.trainer.get("constraint_objective_name"))
        theta = dict(config.trainer.get("constraint_theta") or {})

        proxy_reward_fn = ConstraintRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            components=components,
            objective_name=objective_name,
            theta=theta,
            constraint_mode=str(config.trainer.get("constraint_mode", "mu")),
            dual_lr=float(config.trainer.get("dual_lr", 0.05)),
            dual_ema=float(config.trainer.get("dual_ema", 0.9)),
            dual_clip=float(config.trainer.get("dual_clip", 10.0)),
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

    components, objective_name = load_component_config(args.component_config_json)
    theta = load_theta_json(args.theta_json)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    prompts = load_hh_prompts(args.dataset, args.local_dataset_dir)
    train_path, val_path = prepare_dataset_files(prompts, args.eval_prompts, args.seed, args.output_dir)

    method_name = f"constraint_{args.constraint_mode}_ppo"
    cfg = compose_verl_config(
        args,
        train_path=train_path,
        val_path=val_path,
        method_name=method_name,
        method_kwargs={
            "proxy_eval_key": "objective_score",
            "constraint_mode": args.constraint_mode,
            "dual_lr": args.dual_lr,
            "dual_ema": args.dual_ema,
            "dual_clip": args.dual_clip,
        },
    )

    method_fields: Dict[str, Any] = {
        "constraint_mode": args.constraint_mode,
        "num_components": float(len(components)),
        "num_constraints": float(max(len(components) - 1, 0)),
        "dual_lr": args.dual_lr,
        "dual_ema": args.dual_ema,
        "dual_clip": args.dual_clip,
    }

    with open_dict(cfg.trainer):
        cfg.trainer["baseline_method_fields"] = method_fields
        cfg.trainer["component_specs"] = [{"name": spec.name, "model": spec.model} for spec in components]
        cfg.trainer["constraint_objective_name"] = objective_name
        cfg.trainer["constraint_theta"] = theta

    save_config_json(cfg, args.output_dir)

    remote_kwargs = {"num_cpus": 1}
    if args.reward_gpus:
        remote_kwargs["num_gpus"] = args.reward_gpus

    task_runner_class = ray.remote(**remote_kwargs)(ConstraintTaskRunner)
    main_ppo.run_ppo(cfg, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
