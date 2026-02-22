#!/usr/bin/env python3
"""Train DRRO-GRPO using VERL with proxy and gold reward models."""

from __future__ import annotations

import json
import os
from drro_paths import ensure_verl_on_path

VERL_ROOT = ensure_verl_on_path()
if VERL_ROOT is None:
    raise RuntimeError("Could not locate the VERL package. Set VERL_ROOT or place verl/ next to this script.")

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

    config_dir = os.path.join(VERL_ROOT, "verl", "trainer", "config")
    config = build_config(args, train_path, val_path, config_dir)

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(OmegaConf.to_container(config, resolve=True), handle, indent=2, ensure_ascii=True)

    enable_drro_grpo(
        args.delta,
        delta_alpha=args.delta_alpha,
        delta_tau=args.delta_tau,
        delta_softmax_tau=args.delta_softmax_tau,
        delta_kl_estimator=args.delta_kl_estimator,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
    )

    remote_kwargs = {"num_cpus": 1}
    if args.reward_gpus:
        remote_kwargs["num_gpus"] = args.reward_gpus
    task_runner_class = ray.remote(**remote_kwargs)(DrroTaskRunner)
    main_ppo.run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
