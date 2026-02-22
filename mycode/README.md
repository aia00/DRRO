# DRRO-GRPO Over-Optimization Curve (VERL)

This folder contains a VERL-based DRRO-GRPO training stack with proxy and gold reward models, plus plotting/eval scripts for KL vs reward curves.

Code layout (consolidated):

- `drro_train/` canonical DRRO training implementation.
- `proxy_rm/` proxy RM data generation/training/eval pipeline.

## Quickstart

Install dependencies:

```
pip install -r requirements.txt
```

Run GRPO (delta=0) and DRRO-GRPO (delta>0) with vLLM rollout (Ray is launched internally):

```bash
python drro_train/train_drro_grpo.py --delta 0.0 --output_dir runs/grpo
python drro_train/train_drro_grpo.py --delta 0.5 --output_dir runs/drro_delta0.5
```

Run PPO (GAE) instead of GRPO:

```bash
python drro_train/train_drro_grpo.py --adv_estimator gae --output_dir runs/ppo
```

Use HF rollout instead:

```bash
python drro_train/train_drro_grpo.py --rollout_backend hf --delta 0.5 --output_dir runs/drro_hf
```

Enable Weights & Biases logging:

```bash
export WANDB_MODE=online
export WANDB_ENTITY=your_entity  # optional
python drro_train/train_drro_grpo.py --delta 0.5 --output_dir runs/drro_delta0.5 --wandb --wandb_project drro-grpo
```

Plot the KL curve:

```bash
python drro_train/plot_overopt_curve.py --inputs runs/grpo/log.csv runs/drro_delta0.5/log.csv --out runs/overopt_kl_curve.png
```

Best-of-k evaluation (BoN) on validation prompts:

```bash
python drro_train/eval_best_of_k.py --run_dir runs/grpo --n_list 1,2,4,8,16
```

## Dataset notes

By default, the script targets `HuggingFaceH4/hh-rlhf` but will automatically use the local dataset if it exists at:

```
/home/ykwang/common_dataset_model/dataset/Anthropic_hh-rlhf
```

You can override with:

```
--dataset /path/to/dataset_dir
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
