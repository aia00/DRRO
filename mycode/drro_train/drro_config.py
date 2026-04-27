"""Argument parsing and VERL config composition."""

from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from drro_data import get_custom_chat_template
from drro_paths import get_path_config

PATH_CFG = get_path_config()


def normalize_vllm_load_format(load_format: str) -> str:
    # vLLM 0.11+ removed the old "hf" alias. Preserve backward compatibility.
    return "auto" if load_format == "hf" else load_format


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DRRO-GRPO with VERL.")
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/hh-rlhf")
    parser.add_argument(
        "--local_dataset_dir",
        type=str,
        default=PATH_CFG.get("DRRO_LOCAL_DATASET_DIR", ""),
    )
    parser.add_argument(
        "--policy_model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--proxy_rm",
        type=str,
        default="OpenAssistant/reward-model-deberta-v3-base",
    )
    parser.add_argument(
        "--gold_rm",
        type=str,
        default="sileod/deberta-v3-large-tasksource-rlhf-reward-model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PATH_CFG.get("DRRO_OUTPUT_ROOT", "runs"), "exp1"),
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
        help="Tensor parallel size for vLLM. Set 0 for auto (default).",
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
    parser.add_argument(
        "--vllm_disable_layered_summon",
        action="store_false",
        dest="vllm_layered_summon",
    )
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
        default="grpo",
        help="Advantage estimator (grpo for DRRO-GRPO, gae for PPO).",
    )
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--fixed_delta",
        "--delta",
        dest="fixed_delta",
        type=float,
        default=None,
        help="Fixed DRRO bonus. If unset, defaults to 2.5 * num_generations.",
    )
    parser.add_argument(
        "--dynamic_delta_coeff",
        "--delta_alpha",
        dest="dynamic_delta_coeff",
        type=float,
        default=0.0,
        help="Dynamic-delta coefficient: effective_delta = coeff * smoothed_KL.",
    )
    parser.add_argument(
        "--dynamic_kl_window",
        "--delta_tau",
        dest="dynamic_kl_window",
        type=int,
        default=20,
        help="Sliding-window size for KL smoothing in dynamic-delta mode.",
    )
    parser.add_argument(
        "--soft_assign_tau",
        "--delta_softmax_tau",
        dest="soft_assign_tau",
        type=float,
        default=2.0,
        help="Soft-assignment temperature (smaller -> closer to hard max).",
    )
    parser.add_argument(
        "--assign_mode",
        type=str,
        choices=["soft", "hard"],
        default="soft",
        help=(
            "DRRO bonus assignment rule. "
            "'hard' = argmax_i(r_i - delta * p_i), then only the winner gets +delta. "
            "'soft' = use the same score r_i - delta * p_i, then distribute bonus with softmax/SNIS."
        ),
    )
    parser.add_argument(
        "--robust_objective",
        type=str,
        choices=["drro", "dro"],
        default="drro",
        help=(
            "Robust objective used by the delta add-on. "
            "'drro' keeps the existing r_i - delta * pi_i target with positive add-on; "
            "'dro' uses delta * pi_i for SNIS and subtracts the add-on."
        ),
    )
    parser.add_argument(
        "--dynamic_kl_estimator",
        "--delta_kl_estimator",
        dest="dynamic_kl_estimator",
        type=str,
        choices=["k1", "k2", "k3"],
        default="k3",
        help="KL estimator used by dynamic delta.",
    )
    parser.add_argument(
        "--dynamic_delta_min",
        "--delta_min",
        dest="dynamic_delta_min",
        type=float,
        default=0.0,
        help="Optional lower bound for effective_delta (0 disables clamp).",
    )
    parser.add_argument(
        "--dynamic_delta_max",
        "--delta_max",
        dest="dynamic_delta_max",
        type=float,
        default=0.0,
        help="Optional upper bound for effective_delta (0 disables clamp).",
    )
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
    args = parser.parse_args()
    if args.fixed_delta is None:
        args.fixed_delta = 2.5 * args.num_generations
    # Backward-compatible aliases for older code paths and saved configs.
    args.delta = args.fixed_delta
    args.delta_alpha = args.dynamic_delta_coeff
    args.delta_tau = args.dynamic_kl_window
    args.delta_softmax_tau = args.soft_assign_tau
    args.delta_kl_estimator = args.dynamic_kl_estimator
    args.delta_min = args.dynamic_delta_min
    args.delta_max = args.dynamic_delta_max
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_precision(args: argparse.Namespace) -> Optional[str]:
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")
    if args.bf16:
        return "bfloat16"
    if args.fp16:
        return "float16"
    return None


def resolve_vllm_tp(num_gpus: int, requested_tp: int) -> int:
    if requested_tp > 0:
        if num_gpus % requested_tp != 0:
            raise ValueError(
                f"--vllm_tensor_parallel {requested_tp} must divide --num_gpus {num_gpus}."
            )
        return requested_tp
    if num_gpus <= 1:
        return 1
    if num_gpus % 2 == 0:
        return 2
    return 1


def build_config(
    args: argparse.Namespace,
    train_path: str,
    val_path: str,
    config_dir: str,
):
    args.vllm_load_format = normalize_vllm_load_format(args.vllm_load_format)
    if args.beta_kl != 0.0:
        print("beta_kl is forced to 0.0 (no KL penalty in loss).")
        args.beta_kl = 0.0
    if args.proxy_rm == args.gold_rm:
        raise ValueError(
            "--proxy_rm and --gold_rm are identical. Use two different reward models "
            "for proxy (training) and gold (eval)."
        )
    reward_gpus = args.reward_gpus
    if reward_gpus is None:
        reward_gpus = 1 if args.num_gpus > 1 else 0
    if reward_gpus < 0:
        raise ValueError("--reward_gpus must be >= 0.")
    args.reward_gpus = reward_gpus
    train_gpus = args.num_gpus - reward_gpus
    if train_gpus < 1:
        raise ValueError("--reward_gpus must be less than --num_gpus.")
    real_train_batch = args.batch_size_prompts * args.num_generations
    if real_train_batch % train_gpus != 0:
        adjusted = args.batch_size_prompts
        max_adjust = max(train_gpus, args.num_generations, 8)
        for _ in range(max_adjust):
            adjusted += 1
            if (adjusted * args.num_generations) % train_gpus == 0:
                print(
                    "Adjusted --batch_size_prompts from "
                    f"{args.batch_size_prompts} to {adjusted} so "
                    f"(batch_size_prompts * num_generations) is divisible by "
                    f"train_gpus={train_gpus}."
                )
                args.batch_size_prompts = adjusted
                real_train_batch = adjusted * args.num_generations
                break
        if real_train_batch % train_gpus != 0:
            raise ValueError(
                "batch_size_prompts * num_generations must be divisible by "
                f"train_gpus={train_gpus}. "
                f"Got {args.batch_size_prompts} * {args.num_generations} = "
                f"{real_train_batch}."
            )
    val_batch = args.val_batch_size if args.val_batch_size and args.val_batch_size > 0 else args.batch_size_prompts
    if val_batch % train_gpus != 0:
        adjusted_val = ((val_batch + train_gpus - 1) // train_gpus) * train_gpus
        print(
            "Adjusted --val_batch_size from "
            f"{val_batch} to {adjusted_val} so it is divisible by "
            f"train_gpus={train_gpus}."
        )
        args.val_batch_size = adjusted_val
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.output_dir)) or "drro-grpo"
        project_name = args.wandb_project if args.wandb else "drro-grpo"
        logger_list = "[console,wandb]" if args.wandb else "[console]"
        rollout_top_k = -1 if args.rollout_backend == "vllm" else 0
        vllm_tp = resolve_vllm_tp(train_gpus, args.vllm_tensor_parallel)
        overrides = [
            f"data.train_files={train_path}",
            f"data.val_files={val_path}",
            f"data.prompt_key=prompt",
            f"data.max_prompt_length={args.max_prompt_tokens}",
            f"data.max_response_length={args.max_new_tokens}",
            f"data.train_batch_size={args.batch_size_prompts}",
            f"data.val_batch_size={args.val_batch_size or args.batch_size_prompts}",
            f"data.dataloader_num_workers={args.dataloader_num_workers}",
            f"trainer.total_training_steps={args.num_steps}",
            f"trainer.test_freq={args.eval_every}",
            f"trainer.save_freq={args.save_every}",
            f"trainer.n_gpus_per_node={train_gpus}",
            f"trainer.project_name={project_name}",
            f"trainer.experiment_name={run_name}",
            f"actor_rollout_ref.model.path={args.policy_model}",
            f"+actor_rollout_ref.model.override_config.attn_implementation={args.attn_implementation}",
            "actor_rollout_ref.model.use_remove_padding=false",
            "actor_rollout_ref.model.use_fused_kernels=false",
            f"actor_rollout_ref.rollout.name={args.rollout_backend}",
            f"actor_rollout_ref.rollout.agent.num_workers={args.agent_workers}",
            f"actor_rollout_ref.rollout.n={args.num_generations}",
            f"actor_rollout_ref.rollout.temperature={args.temperature}",
            f"actor_rollout_ref.rollout.top_k={rollout_top_k}",
            f"actor_rollout_ref.rollout.top_p={args.top_p}",
            f"actor_rollout_ref.rollout.response_length={args.max_new_tokens}",
            f"actor_rollout_ref.rollout.prompt_length={args.max_prompt_tokens}",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={args.batch_size_prompts}",
            f"actor_rollout_ref.actor.clip_ratio={args.clip_eps}",
            f"actor_rollout_ref.actor.optim.lr={args.lr}",
            "actor_rollout_ref.actor.use_kl_loss=true",
            "actor_rollout_ref.actor.kl_loss_coef=0.0",
            f"actor_rollout_ref.actor.use_torch_compile={'true' if args.torch_compile else 'false'}",
            f"actor_rollout_ref.ref.use_torch_compile={'true' if args.torch_compile else 'false'}",
            f"actor_rollout_ref.actor.fsdp_config.use_torch_compile={'true' if args.torch_compile else 'false'}",
            f"actor_rollout_ref.ref.fsdp_config.use_torch_compile={'true' if args.torch_compile else 'false'}",
            f"algorithm.adv_estimator={args.adv_estimator}",
            "algorithm.use_kl_in_reward=false",
            "reward_model.enable=false",
            "reward_model.enable_resource_pool=false",
            f"trainer.logger={logger_list}",
            f"trainer.default_local_dir={args.output_dir}",
        ]
        if args.adv_estimator == "gae":
            overrides.extend(
                [
                    "critic.enable=true",
                    f"algorithm.lam={args.gae_lambda}",
                    f"algorithm.gamma={args.gamma}",
                    f"critic.model.path={args.policy_model}",
                    f"critic.model.tokenizer_path={args.policy_model}",
                    f"+critic.model.override_config.attn_implementation={args.attn_implementation}",
                    "critic.ppo_micro_batch_size_per_gpu=1",
                ]
            )
        else:
            overrides.append("critic.enable=false")
        if args.rollout_backend == "vllm":
            overrides.extend(
                [
                    f"actor_rollout_ref.rollout.tensor_model_parallel_size={vllm_tp}",
                    f"actor_rollout_ref.rollout.gpu_memory_utilization={args.vllm_gpu_memory_utilization}",
                    f"actor_rollout_ref.rollout.max_num_batched_tokens={args.vllm_max_num_batched_tokens}",
                    f"actor_rollout_ref.rollout.max_num_seqs={args.vllm_max_num_seqs}",
                    f"actor_rollout_ref.rollout.load_format={args.vllm_load_format}",
                    f"actor_rollout_ref.rollout.layered_summon={'true' if args.vllm_layered_summon else 'false'}",
                    f"actor_rollout_ref.rollout.enable_prefix_caching={'true' if args.vllm_enable_prefix_caching else 'false'}",
                    f"actor_rollout_ref.rollout.enable_chunked_prefill={'true' if args.vllm_enable_chunked_prefill else 'false'}",
                    f"actor_rollout_ref.rollout.free_cache_engine={'true' if args.vllm_free_cache_engine else 'false'}",
                    f"actor_rollout_ref.rollout.enforce_eager={'true' if args.vllm_enforce_eager else 'false'}",
                    f"actor_rollout_ref.rollout.logprobs_mode={args.vllm_logprobs_mode}",
                ]
            )
            if args.vllm_max_model_len and args.vllm_max_model_len > 0:
                overrides.append(f"actor_rollout_ref.rollout.max_model_len={args.vllm_max_model_len}")
        if args.use_lora:
            overrides.extend(
                [
                    f"actor_rollout_ref.model.lora_rank={args.lora_r}",
                    f"actor_rollout_ref.model.lora_alpha={args.lora_alpha}",
                ]
            )
        else:
            overrides.append("actor_rollout_ref.model.lora_rank=0")

        precision = resolve_precision(args)
        if precision:
            overrides.extend(
                [
                    f"actor_rollout_ref.actor.fsdp_config.dtype={precision}",
                    f"actor_rollout_ref.ref.fsdp_config.dtype={precision}",
                    f"actor_rollout_ref.rollout.dtype={precision}",
                ]
            )

        cfg = hydra.compose(config_name="ppo_trainer", overrides=overrides)

    project_root = os.path.dirname(os.path.abspath(__file__))
    cfg.actor_rollout_ref.model.custom_chat_template = get_custom_chat_template()
    dtype_str = "bfloat16" if args.bf16 else "float16" if args.fp16 else "float32"
    reward_kwargs = {
        "proxy_model": args.proxy_rm,
        "gold_model": args.gold_rm,
        "reward_batch_size": args.reward_batch_size,
        "reward_max_length": args.reward_max_length,
        "dtype": dtype_str,
    }
    with open_dict(cfg.reward_model):
        cfg.reward_model["reward_kwargs"] = reward_kwargs
    with open_dict(cfg.reward):
        cfg.reward["reward_kwargs"] = {
            "model_name": args.proxy_rm,
            "reward_batch_size": args.reward_batch_size,
            "reward_max_length": args.reward_max_length,
            "dtype": dtype_str,
        }
        cfg.reward.reward_manager["source"] = "importlib"
        cfg.reward.reward_manager["name"] = "LoopHFRewardManager"
        cfg.reward.reward_manager.module["path"] = os.path.join(project_root, "drro_reward.py")
    with open_dict(cfg.trainer):
        cfg.trainer["log_csv_path"] = os.path.join(args.output_dir, "log.csv")
        cfg.trainer["fixed_delta"] = args.fixed_delta
        cfg.trainer["dynamic_delta_coeff"] = args.dynamic_delta_coeff
        cfg.trainer["dynamic_kl_window"] = args.dynamic_kl_window
        cfg.trainer["soft_assign_tau"] = args.soft_assign_tau
        cfg.trainer["assign_mode"] = args.assign_mode
        cfg.trainer["robust_objective"] = args.robust_objective
        cfg.trainer["dynamic_kl_estimator"] = args.dynamic_kl_estimator
        cfg.trainer["dynamic_delta_min"] = args.dynamic_delta_min
        cfg.trainer["dynamic_delta_max"] = args.dynamic_delta_max
        # Legacy keys kept for backward compatibility with old logs/eval scripts.
        cfg.trainer["drro_delta"] = args.delta
        cfg.trainer["drro_delta_alpha"] = args.delta_alpha
        cfg.trainer["drro_delta_tau"] = args.delta_tau
        cfg.trainer["drro_delta_softmax_tau"] = args.delta_softmax_tau
        cfg.trainer["drro_delta_kl_estimator"] = args.delta_kl_estimator
        cfg.trainer["drro_delta_min"] = args.delta_min
        cfg.trainer["drro_delta_max"] = args.delta_max
        cfg.trainer["drro_beta_kl"] = args.beta_kl
        cfg.trainer["recalibrate_for_plot"] = bool(args.recalibrate_for_plot)
    with open_dict(cfg.actor_rollout_ref.actor.fsdp_config):
        cfg.actor_rollout_ref.actor.fsdp_config["optimizer_offload"] = bool(args.optimizer_offload)
        cfg.actor_rollout_ref.actor.fsdp_config["param_offload"] = bool(args.param_offload)
        cfg.actor_rollout_ref.actor.fsdp_config["grad_offload"] = bool(args.grad_offload)
    if args.rollout_backend == "vllm":
        with open_dict(cfg.actor_rollout_ref.rollout):
            cfg.actor_rollout_ref.rollout["load_format"] = args.vllm_load_format
            cfg.actor_rollout_ref.rollout["enable_sleep_mode"] = bool(args.vllm_enable_sleep_mode)

    mycode_root = os.path.dirname(project_root)
    python_paths = [project_root, mycode_root]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    pythonpath_value = ":".join([p for p in python_paths if p])

    with open_dict(cfg.ray_kwargs):
        if cfg.ray_kwargs.get("ray_init") is None:
            cfg.ray_kwargs["ray_init"] = {}
    with open_dict(cfg.ray_kwargs.ray_init):
        default_ray_tmp = PATH_CFG.get("DRRO_RAY_TMPDIR", "")
        ray_temp_dir = (
            os.environ.get("RAY_TMPDIR")
            or os.environ.get("RAY_TEMP_DIR")
            or default_ray_tmp
        )
        if not ray_temp_dir:
            raise ValueError(
                "Ray temp dir is not configured. Set DRRO_RAY_TMPDIR in project_paths.env "
                "or export RAY_TMPDIR/RAY_TEMP_DIR."
            )
        os.makedirs(ray_temp_dir, exist_ok=True)
        runtime_env = cfg.ray_kwargs.ray_init.get("runtime_env") or {}
        env_vars = runtime_env.get("env_vars") or {}
        env_vars["PYTHONPATH"] = pythonpath_value
        env_vars["RAY_TMPDIR"] = ray_temp_dir
        env_vars["RAY_TEMP_DIR"] = ray_temp_dir
        runtime_env["env_vars"] = env_vars
        cfg.ray_kwargs.ray_init["_temp_dir"] = ray_temp_dir
        cfg.ray_kwargs.ray_init["runtime_env"] = runtime_env
    return cfg
