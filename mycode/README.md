# DRRO-GRPO Over-Optimization Curve (VERL)

This folder contains an installed-VERL-based DRRO-GRPO training stack with proxy and gold reward models, plus plotting/eval scripts for KL vs reward curves.

Code layout (consolidated):

- `drro_train/` canonical DRRO training implementation.
- `proxy_rm/` proxy RM data generation/training/eval pipeline.
- `baselines/` additional comparison baselines:
  - `baselines/ensemble/` ensemble PPO (`mean`, `wco`, `uwo`) + proxy-ensemble utilities
  - `baselines/constraint/` constraint PPO (`mu`, `xi`) + proxy-point estimation
  - `baselines/common/` shared baseline trainer/config utilities

## Quickstart

Install dependencies:

```
pip install -r requirements.txt
pip install "verl==0.7.1"
```

Configure paths once in `project_paths.env` (output root, ray tmp, local dataset, proxy pair dir).

Run GRPO (`fixed_delta=0`) and DRRO-GRPO (`fixed_delta>0`) with vLLM rollout (Ray is launched internally):

```bash
python drro_train/train_drro_grpo.py --fixed_delta 0.0 --output_dir runs/grpo
python drro_train/train_drro_grpo.py --fixed_delta 0.5 --output_dir runs/drro_delta0.5
```

DRRO assignment modes:

- `--assign_mode hard`: compute `r_i - delta * p_i`, pick the argmax, give `+delta` only to that winner.
- `--assign_mode soft`: use the same `r_i - delta * p_i`, then distribute bonus with the softmax/SNIS surrogate.

Run the DRO variant from `DRO_RLHF` by switching the robust objective:

```bash
python drro_train/train_drro_grpo.py --robust_objective dro --fixed_delta 0.5 --output_dir runs/dro_delta0.5
```

For DRO, the SNIS target uses `delta * p_i` instead of `r_i - delta * p_i`, and the add-on is subtracted from the original reward.

Run PPO (GAE) instead of GRPO:

```bash
python drro_train/train_drro_grpo.py --adv_estimator gae --output_dir runs/ppo
```

Use HF rollout instead:

```bash
python drro_train/train_drro_grpo.py --rollout_backend hf --fixed_delta 0.5 --output_dir runs/drro_hf
```

Enable Weights & Biases logging:

```bash
export WANDB_MODE=online
export WANDB_ENTITY=your_entity  # optional
python drro_train/train_drro_grpo.py --fixed_delta 0.5 --output_dir runs/drro_delta0.5 --wandb --wandb_project drro-grpo
```

Plot the KL curve:

```bash
python drro_train/plot_overopt_curve.py --inputs runs/grpo/log.csv runs/drro_delta0.5/log.csv --out runs/overopt_kl_curve.png
```

Best-of-k evaluation (BoN) on validation prompts:

```bash
python drro_train/eval_best_of_k.py --run_dir runs/grpo --n_list 1,2,4,8,16
```

## Baseline experiments

Train ensemble baseline (PPO + LoRA):

```bash
bash baselines/ensemble/run_ensemble_lora.sh
```

Train constraint baselines (CMDP style):

```bash
COMPONENT_CONFIG_JSON=/path/to/component_config.json \
THETA_JSON=/path/to/theta.json \
bash baselines/constraint/run_constraint_mu_lora.sh

COMPONENT_CONFIG_JSON=/path/to/component_config.json \
THETA_JSON=/path/to/theta.json \
bash baselines/constraint/run_constraint_xi_lora.sh
```

Train/export a 5-member proxy RM ensemble + manifest:

```bash
bash baselines/ensemble/run_train_proxy_ensemble.sh
```

Estimate proxy-point thresholds (gold-peak rule):

```bash
python baselines/constraint/estimate_proxy_points.py \
  --inputs /path/to/eval_log.csv \
  --output_theta_json /path/to/theta.json
```

## Dataset notes

By default, the script targets `HuggingFaceH4/hh-rlhf`.
For local datasets, set `DRRO_LOCAL_DATASET_DIR` in `project_paths.env` (or export it in shell), for example:

```
DRRO_LOCAL_DATASET_DIR=./datasets/Anthropic_hh-rlhf
```

You can also override at runtime:

```
--local_dataset_dir /path/to/dataset_dir
```

The script converts prompts into a JSONL file with a stable "Human/Assistant" chat template.

## Tips

- To change GPU count, set `--num_gpus` (default 3).
- If rewards are CPU-bound, set `--reward_gpus` (or omit it to auto-reserve 1 GPU when `--num_gpus > 1`) and reduce `--num_gpus` accordingly.
  - `batch_size_prompts * num_generations` must be divisible by `train_gpus = num_gpus - reward_gpus`; the script will auto-adjust upward if needed.
  - `val_batch_size` must also be divisible by `train_gpus`; it will auto-adjust if needed.
- LoRA is enabled by default (`--no_lora` to disable).
- Optimizer CPU offload is disabled by default (pass `--optimizer_offload` to enable).
- Set `--eval_every` and `--eval_prompts` to control evaluation cost.
- Use `--bf16` or `--fp16` to set mixed precision for rollout/actor/ref and reward models.
- The resolved VERL config is saved to `output_dir/config.json`.
- W&B uses `trainer.project_name` and `trainer.experiment_name` (defaults to `drro-grpo` and the `output_dir` basename).
- vLLM rollout is enabled by default; `--vllm_tensor_parallel 0` auto-selects TP (default), or set a fixed TP that divides `--num_gpus`.
- If vLLM stalls, try `--vllm_disable_sleep_mode --vllm_enforce_eager` first; disabling chunked prefill can crash some models.
