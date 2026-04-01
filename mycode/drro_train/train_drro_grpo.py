#!/usr/bin/env python3
"""Train DRRO-GRPO using VERL with proxy and gold reward models."""

from __future__ import annotations

import json
import os
from drro_paths import get_verl_config_dir

import ray
from omegaconf import OmegaConf
from verl.trainer import main_ppo

from drro_advantage import enable_drro_grpo
from drro_config import build_config, parse_args, set_seed
from drro_data import load_hh_prompts, prepare_dataset_files
from drro_trainer import DrroTaskRunner


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity

    prompts = load_hh_prompts(args.dataset, args.local_dataset_dir)
    train_path, val_path = prepare_dataset_files(
        prompts, args.eval_prompts, args.seed, args.output_dir
    )

    config_dir = get_verl_config_dir()
    if config_dir is None:
        raise RuntimeError("Could not locate installed `verl` trainer config directory.")
    config = build_config(args, train_path, val_path, config_dir)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(OmegaConf.to_container(config, resolve=True), handle, indent=2, ensure_ascii=True)

    enable_drro_grpo(
        args.fixed_delta,
        dynamic_delta_coeff=args.dynamic_delta_coeff,
        dynamic_kl_window=args.dynamic_kl_window,
        soft_assign_tau=args.soft_assign_tau,
        assign_mode=args.assign_mode,
        dynamic_kl_estimator=args.dynamic_kl_estimator,
        dynamic_delta_min=args.dynamic_delta_min,
        dynamic_delta_max=args.dynamic_delta_max,
    )

    remote_kwargs = {"num_cpus": 1}
    if args.reward_gpus:
        remote_kwargs["num_gpus"] = args.reward_gpus
    task_runner_class = ray.remote(**remote_kwargs)(DrroTaskRunner)
    main_ppo.run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
