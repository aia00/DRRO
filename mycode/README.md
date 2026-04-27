# DRRO-GRPO Training (VERL)

This directory contains the DRRO-GRPO training code built on top of VERL.

## Structure

- `drro_train/`: main training and evaluation entrypoints.

## Installation

```bash
pip install -r requirements.txt
pip install "verl==0.7.1"
```

## Configuration

Set the local paths in `project_paths.env` before running training:

```bash
DRRO_OUTPUT_ROOT=/path/to/outputs
DRRO_RAY_TMP=/path/to/ray_tmp
DRRO_LOCAL_DATASET_DIR=/path/to/dataset_dir
```

## Training

Run GRPO:

```bash
python drro_train/train_drro_grpo.py --fixed_delta 0.0 --output_dir runs/grpo
```

Run DRRO-GRPO:

```bash
python drro_train/train_drro_grpo.py --fixed_delta 0.5 --output_dir runs/drro_delta0.5
```

Ray is launched internally by the training script.

Optional: enable Weights & Biases logging:

```bash
export WANDB_MODE=online
export WANDB_ENTITY=your_entity
python drro_train/train_drro_grpo.py --fixed_delta 0.5 --output_dir runs/drro_delta0.5 --wandb --wandb_project drro-grpo
```

## Evaluation

Run best-of-k evaluation on a training run:

```bash
python drro_train/eval_best_of_k.py --run_dir runs/grpo --n_list 1,2,4,8,16
```

## Dataset

The default dataset is `HuggingFaceH4/hh-rlhf`.

To use a local dataset, set `DRRO_LOCAL_DATASET_DIR` in `project_paths.env`, or pass:

```bash
--local_dataset_dir /path/to/dataset_dir
```
