# DRRO Research Project

This repository contains RLHF experiments for reward over-optimization:
- DRRO / GRPO training
- PPO and BoN evaluation utilities
- Proxy reward-model data/training pipeline
- Baseline methods (ensemble and constraint RLHF)

## Repository Layout

```text
DRRO/
├── mycode/
│   ├── drro_train/      # DRRO/GRPO/PPO training + eval + plotting
│   ├── proxy_rm/        # Build preference pairs, train/eval proxy RM
│   ├── baselines/
│   │   ├── ensemble/    # Ensemble PPO baselines (mean/WCO/UWO)
│   │   ├── constraint/  # Constraint RLHF baselines (mu/xi)
│   │   └── common/      # Shared baseline utilities
│   ├── project_paths.env.example
│   └── requirements.txt
```

## 1) Setup

```bash
cd mycode
pip install -r requirements.txt
pip install "verl==0.7.1"
```

## 2) Path Configuration (important)

Do **not** commit machine-specific paths.

```bash
cp mycode/project_paths.env.example mycode/project_paths.env
```

Edit `mycode/project_paths.env`:
- `DRRO_LOCAL_DATASET_DIR`
- `DRRO_RAY_TMPDIR`
- `DRRO_OUTPUT_ROOT`
- `DRRO_PROXY_PAIRS_DIR`

Install `verl` in the active environment before running training or baselines.

If you want a custom location:
```bash
export DRRO_PATH_CONFIG=/path/to/project_paths.env
```

## 3) Main Training Workflows

Run from `mycode/drro_train/`.

### DRRO (LoRA)
```bash
bash run_drro_delta_only_lora.sh
```

### Two-run compare script (e.g., GRPO vs DRRO)
```bash
bash run_drro_lora.sh
```

### No-LoRA run
```bash
bash run_drro_no_lora.sh
```

### Python entrypoint (direct)
```bash
python train_drro_grpo.py --help
```

## 4) Baseline Workflows

Run from `mycode/baselines/...`.

### Ensemble PPO baseline
```bash
bash mycode/baselines/ensemble/run_ensemble_lora.sh
```

### Constraint RLHF baselines
```bash
COMPONENT_CONFIG_JSON=/path/to/component_config.json \
THETA_JSON=/path/to/theta.json \
bash mycode/baselines/constraint/run_constraint_mu_lora.sh

COMPONENT_CONFIG_JSON=/path/to/component_config.json \
THETA_JSON=/path/to/theta.json \
bash mycode/baselines/constraint/run_constraint_xi_lora.sh
```

### Proxy ensemble training utility
```bash
bash mycode/baselines/ensemble/run_train_proxy_ensemble.sh
```

## 5) Proxy RM Pipeline

Run from `mycode/proxy_rm/`.

```bash
bash run_build_proxy_pairs_4gpu.sh
bash run_proxy_pipeline_50k.sh
python eval_proxy_rm.py --help
```

Debug / weak-data scripts:
- `run_proxy_pipeline_debug_40.sh`
- `run_proxy_pipeline_weak.sh`

## 6) Evaluation and Plotting

### Evaluate checkpoints
```bash
python mycode/drro_train/eval_overopt_ckpts.py --help
python mycode/drro_train/eval_best_of_k.py --help
```

### Plot KL-reward curves
```bash
python mycode/drro_train/plot_overopt_curve.py --help
```

Expected logs:
- `log.csv` for training-time metrics
- `eval_log.csv` for checkpoint evaluation

## 7) Reproducibility Notes

- KL penalty in your DRRO/PPO objectives is controlled by config/CLI (`beta_kl`).
- Output names in scripts include method/tau/rollout where configured.
- Checkpoint and eval frequency are controlled by `--save_every` and `--eval_every`.
- Most scripts support extra CLI args via `EXTRA_TRAIN_ARGS`.

## 8) Git / Privacy Best Practice

- `mycode/project_paths.env` is ignored by git.
- Commit `mycode/project_paths.env.example` only.
- Never commit local absolute paths in scripts/config.

If `project_paths.env` was committed before, untrack it once:
```bash
git rm --cached mycode/project_paths.env
git commit -m "stop tracking local path config"
```

## 9) Troubleshooting

- `ModuleNotFoundError: ray`: activate/install the intended env.
- Ray tmp full: set `DRRO_RAY_TMPDIR` to a large disk.
- Empty plots: verify CSV columns (`kl_seq`, `gold_score`/`gold_score_norm`, etc.).
- Slow eval: reduce eval prompts or eval frequency.
