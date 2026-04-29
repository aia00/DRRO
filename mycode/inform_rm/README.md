# InfoRM in This Repo

This folder contains a lightweight implementation of the core method from:

- *InfoRM: Mitigating Reward Hacking in RLHF via Information-Theoretic Reward Modeling*
- arXiv: `2402.09345`

What is implemented here:

1. `train_inform_rm.py`
   - trains an information-bottleneck reward model on `prompt/chosen/rejected` JSONL pairs
   - objective: Bradley-Terry preference loss + `beta * KL(q(z|x) || N(0, I))`
2. `eval_inform_rm.py`
   - evaluates InfoRM pairwise accuracy and optionally agreement with a gold RM
3. `extract_ib_latents.py`
   - exports IB latent means for prompt/response JSONL files
4. `compute_csi.py`
   - computes the paper's Cluster Separation Index (CSI) from two latent sets
5. `run_train_inform_rm.sh`
   - convenience launcher using the repo's `project_paths.env`
6. `train_inform_policy.py`
   - uses a trained InfoRM checkpoint as the proxy reward in downstream PPO/GRPO-style policy training
7. `run_inform_lora.sh`
   - convenience launcher for LoRA policy training with InfoRM as proxy reward

## Notes

This is an adaptation to the current DRRO repo, not a full copy of the paper authors' OpenRLHF codebase.
The focus here is on the paper's two core components:

- the variational information-bottleneck reward model (InfoRM)
- the CSI latent-space overoptimization indicator

## Training

```bash
bash mycode/inform_rm/run_train_inform_rm.sh
```

Common overrides:

```bash
PAIR_DIR=/home/ykwang/mtdata2/DRRO/runs/proxy_pairs/proxy_pairs_50k \
LATENT_DIM=64 \
BETA=0.01 \
POOLING=cls \
BATCH_SIZE=16 \
EPOCHS=1 \
BF16=1 \
bash mycode/inform_rm/run_train_inform_rm.sh
```

Direct Python usage:

```bash
python mycode/inform_rm/train_inform_rm.py \
  --train_jsonl /path/to/train.jsonl \
  --val_jsonl /path/to/val.jsonl \
  --model_name microsoft/MiniLM-L12-H384-uncased \
  --output_dir /path/to/inform_rm_ckpt \
  --latent_dim 128 \
  --beta 0.01 \
  --pooling cls \
  --epochs 1 \
  --lr 5e-6
```

## Evaluation

```bash
python mycode/inform_rm/eval_inform_rm.py \
  --data_jsonl /path/to/val.jsonl \
  --inform_rm /path/to/inform_rm_ckpt \
  --gold_rm sileod/deberta-v3-large-tasksource-rlhf-reward-model
```

## Downstream Policy Training

Train a policy against a trained InfoRM checkpoint:

```bash
INFORM_RM_PATH=/path/to/inform_rm_ckpt \
bash mycode/inform_rm/run_inform_lora.sh
```

Example explicit Python command:

```bash
python mycode/inform_rm/train_inform_policy.py \
  --policy_model Qwen/Qwen2.5-0.5B-Instruct \
  --inform_rm_path /path/to/inform_rm_ckpt \
  --gold_rm sileod/deberta-v3-large-tasksource-rlhf-reward-model \
  --output_dir /path/to/inform_policy_run \
  --num_steps 250 \
  --batch_size_prompts 12 \
  --num_generations 16 \
  --inform_max_length 512 \
  --use_lora \
  --bf16
```

This downstream stage reuses the repo's VERL-based baseline trainer wiring:

- training reward = InfoRM proxy reward
- evaluation reward = fixed gold RM
- logs remain compatible with the repo's `log.csv` / `kl_seq` plotting workflow

## CSI Workflow

1. Export latent vectors for the reference/SFT responses.
2. Export latent vectors for the RLHF responses.
3. Compute CSI.

Reference/SFT latent export:

```bash
python mycode/inform_rm/extract_ib_latents.py \
  --model_dir /path/to/inform_rm_ckpt \
  --input_jsonl /path/to/sft_outputs.jsonl \
  --output_npy /path/to/sft_latents.npy \
  --prompt_key prompt \
  --response_key response
```

RLHF latent export:

```bash
python mycode/inform_rm/extract_ib_latents.py \
  --model_dir /path/to/inform_rm_ckpt \
  --input_jsonl /path/to/rlhf_outputs.jsonl \
  --output_npy /path/to/rlhf_latents.npy \
  --prompt_key prompt \
  --response_key response
```

CSI calculation:

```bash
python mycode/inform_rm/compute_csi.py \
  --blue_points /path/to/sft_latents.npy \
  --red_points /path/to/rlhf_latents.npy
```
