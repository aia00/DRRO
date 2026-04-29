#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYCODE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PATH_CFG="${DRRO_PATH_CONFIG:-${MYCODE_ROOT}/project_paths.env}"
if [[ -f "${PATH_CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${PATH_CFG}"
fi

ENV_NAME="${ENV_NAME:-${DRRO_CONDA_ENV:-verl_vllm_fa2}}"
PAIR_DIR="${PAIR_DIR:-${DRRO_PROXY_PAIRS_DIR:-}}"
OUT_ROOT="${OUT_ROOT:-${DRRO_OUTPUT_ROOT:-}}"

if [[ -z "${PAIR_DIR}" ]]; then
  echo "Set DRRO_PROXY_PAIRS_DIR in project_paths.env or export PAIR_DIR." >&2
  exit 1
fi
if [[ -z "${OUT_ROOT}" ]]; then
  echo "Set DRRO_OUTPUT_ROOT in project_paths.env or export OUT_ROOT." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_DIR}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LM_COEF="${LM_COEF:-0.01}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-scorelm_qwen0p5b_lmcoef${LM_COEF}}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN_NAME}}"

CMD=(
  python train_scorelm.py
  --train_jsonl "${PAIR_DIR}/train.jsonl"
  --val_jsonl "${PAIR_DIR}/val.jsonl"
  --model_name "${MODEL_NAME}"
  --output_dir "${RUN_DIR}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_accum "${GRAD_ACCUM}"
  --lm_coef "${LM_COEF}"
  --seed "${SEED}"
  --save_best
)

if [[ "${BF16:-1}" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "${FP16:-0}" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ "${MAX_STEPS:-}" != "" ]]; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi

"${CMD[@]}"

echo "Done. ScoreLM saved to ${RUN_DIR}"
