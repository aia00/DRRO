"""Shared VERL trainer/task-runner classes for baseline experiments."""

from __future__ import annotations

import csv
import os
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from omegaconf import OmegaConf

from baselines.common.reward_helpers import merge_numeric_lists

import torch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer import main_ppo
from verl.trainer.ppo import ray_trainer as verl_ray_trainer
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.config import validate_config
from verl.utils.fs import copy_to_local
from verl.workers.reward_manager.abstract import AbstractRewardManager


class BaselineRayPPOTrainer(RayPPOTrainer):
    """RayPPOTrainer variant with KL/proxy/gold logging for baselines."""

    def __init__(
        self,
        *args,
        reward_fn: AbstractRewardManager,
        val_reward_fn: AbstractRewardManager,
        log_csv_path: str,
        method_name: str,
        beta_kl: float,
        method_fields: Optional[Dict[str, Any]] = None,
        proxy_eval_key: str = "proxy_score",
        **kwargs,
    ) -> None:
        super().__init__(*args, reward_fn=reward_fn, val_reward_fn=val_reward_fn, **kwargs)
        self.proxy_reward_fn = reward_fn
        self.gold_reward_fn = val_reward_fn
        self.log_csv_path = log_csv_path
        self.method_name = method_name
        self.beta_kl = beta_kl
        self.method_fields = dict(method_fields or {})
        self.proxy_eval_key = proxy_eval_key
        self.reward_norm_proxy: Optional[tuple[float, float]] = None
        self.reward_norm_gold: Optional[tuple[float, float]] = None

    def _write_log_row(self, row: Dict[str, Any]) -> None:
        file_exists = os.path.isfile(self.log_csv_path)
        with open(self.log_csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def _coerce_scores(value: object, batch_size: int) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value.tolist()
            return [float(x) for x in arr]
        if torch.is_tensor(value):
            arr = value.detach().cpu().tolist()
            if isinstance(arr, list):
                return [float(x) for x in arr]
            return [float(arr)] * batch_size
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value]
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            return None
        return [scalar] * batch_size

    def _extract_proxy_scores(self, proxy_result: Dict[str, Any]) -> List[float]:
        reward_tensor = proxy_result["reward_tensor"]
        batch_size = int(reward_tensor.shape[0])
        extra = proxy_result.get("reward_extra_info", {}) or {}
        if self.proxy_eval_key:
            scores = self._coerce_scores(extra.get(self.proxy_eval_key), batch_size=batch_size)
            if scores is not None:
                return scores
        return reward_tensor.sum(dim=-1).detach().cpu().tolist()

    def _extra_row_fields(
        self,
        proxy_extra_accum: Dict[str, List[float]],
        proxy_scores: List[float],
        gold_scores: List[float],
    ) -> Dict[str, Any]:
        return {}

    def _validate(self, merged: bool = False):
        proxy_scores: List[float] = []
        gold_scores: List[float] = []
        proxy_extra_accum: Dict[str, List[float]] = defaultdict(list)

        kl_sum = 0.0
        kl_count = 0.0
        kl_seq_sum = 0.0
        kl_seq_count = 0.0

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            gen_batch = self._get_gen_batch(test_batch)
            gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            gen_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, gen_divisor)
            if not self.async_rollout_mode:
                gen_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
            else:
                gen_output_padded = self.async_rollout_manager.generate_sequences(gen_batch_padded)
            gen_output = unpad_dataproto(gen_output_padded, pad_size=pad_size)

            test_batch = test_batch.union(gen_output)
            if "response_mask" not in test_batch.batch:
                test_batch.batch["response_mask"] = verl_ray_trainer.compute_response_mask(test_batch)

            logprob_divisor = self.actor_rollout_wg.world_size
            if len(test_batch) % logprob_divisor != 0:
                padded_batch, pad_size_lp = pad_dataproto_to_divisor(test_batch, logprob_divisor)
                old_log_prob, _ = self._compute_old_log_prob(padded_batch)
                old_log_prob = unpad_dataproto(old_log_prob, pad_size=pad_size_lp)
                test_batch = test_batch.union(old_log_prob)
                ref_log_prob = self._compute_ref_log_prob(padded_batch)
                ref_log_prob = unpad_dataproto(ref_log_prob, pad_size=pad_size_lp)
                test_batch = test_batch.union(ref_log_prob)
            else:
                old_log_prob, _ = self._compute_old_log_prob(test_batch)
                test_batch = test_batch.union(old_log_prob)
                ref_log_prob = self._compute_ref_log_prob(test_batch)
                test_batch = test_batch.union(ref_log_prob)

            response_mask = test_batch.batch["response_mask"]
            logp = test_batch.batch["old_log_probs"]
            ref_logp = test_batch.batch.get("ref_log_prob")
            if ref_logp is not None:
                diff = (logp - ref_logp) * response_mask
                kl_sum += diff.sum().item()
                kl_count += response_mask.sum().item()
                kl_seq = diff.sum(dim=-1)
                kl_seq_sum += kl_seq.sum().item()
                kl_seq_count += kl_seq.numel()

            proxy_result = self.proxy_reward_fn(test_batch, return_dict=True)
            gold_result = self.gold_reward_fn(test_batch, return_dict=True)

            proxy_scores.extend(self._extract_proxy_scores(proxy_result))
            gold_reward = gold_result["reward_tensor"].sum(dim=-1).detach().cpu().tolist()
            gold_scores.extend(gold_reward)

            proxy_extra = proxy_result.get("reward_extra_info", {}) or {}
            merge_numeric_lists(proxy_extra_accum, proxy_extra)

        proxy_mean = float(np.mean(proxy_scores)) if proxy_scores else 0.0
        proxy_std = float(np.std(proxy_scores)) if proxy_scores else 0.0
        gold_mean = float(np.mean(gold_scores)) if gold_scores else 0.0
        gold_std = float(np.std(gold_scores)) if gold_scores else 0.0
        kl_mean = kl_sum / max(kl_count, 1.0)
        kl_seq_mean = kl_seq_sum / max(kl_seq_count, 1.0)

        if self.reward_norm_proxy is None:
            self.reward_norm_proxy = (proxy_mean, proxy_std)
            self.reward_norm_gold = (gold_mean, gold_std)

        recalibrate = bool(self.config.trainer.get("recalibrate_for_plot", True))
        if recalibrate:
            proxy_norm = (proxy_mean - self.reward_norm_proxy[0]) / (self.reward_norm_proxy[1] + 1e-8)
            gold_norm = (gold_mean - self.reward_norm_gold[0]) / (self.reward_norm_gold[1] + 1e-8)
        else:
            proxy_norm = proxy_mean
            gold_norm = gold_mean

        row: Dict[str, Any] = {
            "step": self.global_steps,
            "method": self.method_name,
            "beta_kl": self.beta_kl,
            "delta": 0.0,
            "kl": kl_mean,
            "kl_per_token": kl_mean,
            "kl_seq": kl_seq_mean,
            "proxy_score": proxy_mean,
            "gold_score": gold_mean,
            "proxy_score_norm": proxy_norm,
            "gold_score_norm": gold_norm,
        }
        for key, value in self.method_fields.items():
            if value is None:
                continue
            if isinstance(value, (bool, int, float, str)):
                row[key] = value
            else:
                row[key] = str(value)

        row.update(self._extra_row_fields(proxy_extra_accum, proxy_scores, gold_scores))
        self._write_log_row(row)

        return {
            "val/proxy_mean": proxy_mean,
            "val/gold_mean": gold_mean,
            "val/kl": kl_mean,
            "val/kl_seq": kl_seq_mean,
        }


class BaseBaselineTaskRunner(main_ppo.TaskRunner):
    """Shared task-runner that wires baseline reward managers into VERL."""

    trainer_cls = BaselineRayPPOTrainer

    def build_reward_functions(
        self,
        config,
        tokenizer,
    ) -> Tuple[AbstractRewardManager, AbstractRewardManager, Dict[str, Any]]:
        raise NotImplementedError

    def run(self, config):
        from pprint import pprint

        print("TaskRunner started")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=main_ppo.need_reference_policy(self.role_worker_mapping),
            use_critic=main_ppo.need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, trust_remote_code=config.data.get("trust_remote_code", False), use_fast=True)

        custom_template = config.actor_rollout_ref.model.get("custom_chat_template", None)
        if custom_template:
            if processor is not None:
                processor.chat_template = custom_template
            else:
                tokenizer.chat_template = custom_template
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

        reward_fn, gold_reward_fn, trainer_extra_kwargs = self.build_reward_functions(config=config, tokenizer=tokenizer)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = main_ppo.create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = main_ppo.create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = main_ppo.create_rl_sampler(config.data, train_dataset)

        method_fields = config.trainer.get("baseline_method_fields", {})
        if method_fields is None:
            method_fields = {}
        method_fields = dict(method_fields)

        trainer = self.trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=gold_reward_fn,
            log_csv_path=config.trainer.get("log_csv_path"),
            method_name=config.trainer.get("baseline_method", "baseline"),
            beta_kl=config.trainer.get("drro_beta_kl", 0.0),
            method_fields=method_fields,
            proxy_eval_key=config.trainer.get("proxy_eval_key", "proxy_score"),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            **trainer_extra_kwargs,
        )
        trainer.init_workers()
        trainer.fit()
