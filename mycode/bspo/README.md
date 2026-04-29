# BSPO in This Repo

This folder contains a lightweight implementation of the main idea from:

- *Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization*
- arXiv: `2503.18130`

What is implemented here:

1. `train_scorelm.py`
   - trains a ScoreLM-style proxy model
   - keeps the causal LM head for next-token behavior prediction
   - adds a scalar score head for preference reward modeling
   - objective: Bradley-Terry reward loss + `lm_coef * LM loss`
2. `train_bspo.py`
   - runs PPO-style RLHF using the current VERL-based repo
   - uses ScoreLM as the proxy reward model
   - applies behavior-supported GAE regularization with:
     - `unsupported_value` (`V_min` in the paper)
     - `epsilon_beta` support threshold on next-token probability
3. `eval_scorelm.py`
   - evaluates ScoreLM pairwise accuracy and optional agreement with a gold RM

## Important compatibility note

BSPO requires a causal-lm-style reward model with a next-token distribution over the same action space as the actor.
In practice, that means:

- the actor model and ScoreLM should use tokenizer-compatible model families
- the safest setup is training ScoreLM from the same base model family as the actor

This is different from the DeBERTa-style proxy RMs used elsewhere in this repo.
Those sequence-classification reward models do **not** provide the next-token behavior distribution needed by BSPO.

## Recommended workflow

### 1) Train ScoreLM

```bash
bash mycode/bspo/run_train_scorelm.sh
```

Example override:

```bash
PAIR_DIR=/home/ykwang/mtdata2/DRRO/proxy_pairs \
OUT_ROOT=/home/ykwang/mtdata2/DRRO/runs \
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct \
LM_COEF=0.01 \
EPOCHS=2 \
BF16=1 \
bash mycode/bspo/run_train_scorelm.sh
```

### 2) Evaluate ScoreLM

```bash
python mycode/bspo/eval_scorelm.py \
  --data_jsonl /path/to/val.jsonl \
  --scorelm /path/to/scorelm_ckpt \
  --gold_rm sileod/deberta-v3-large-tasksource-rlhf-reward-model
```

### 3) Run BSPO

```bash
SCORELM_PATH=/path/to/scorelm_ckpt \
bash mycode/bspo/run_bspo_lora.sh
```

Example explicit command:

```bash
python mycode/bspo/train_bspo.py \
  --policy_model Qwen/Qwen2.5-0.5B-Instruct \
  --scorelm_path /path/to/scorelm_ckpt \
  --gold_rm sileod/deberta-v3-large-tasksource-rlhf-reward-model \
  --output_dir /path/to/bspo_run \
  --num_steps 250 \
  --batch_size_prompts 12 \
  --num_generations 16 \
  --unsupported_value -15 \
  --epsilon_beta 1e-4 \
  --use_lora \
  --bf16
```

## Defaults chosen from the paper when possible

- `unsupported_value = -15`
- `epsilon_beta = 1e-4`
- ScoreLM `lm_coef = 0.01`

## Scope

This is a repo-adapted implementation, not a line-by-line reproduction of the authors' training stack.
The main algorithmic ingredients implemented here are:

- ScoreLM behavior distribution prediction
- behavior-supported support masking from next-token probabilities
- pessimistic value regularization in PPO/GAE for unsupported actions
