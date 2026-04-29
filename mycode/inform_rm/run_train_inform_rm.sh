#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYCODE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PATH_CFG="${DRRO_PATH_CONFIG:-${MYCODE_ROOT}/project_paths.env}"
if [[ -f "${PATH_CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${PATH_CFG}"
fi

ENV_NAME="${ENV_NAME:-${DRRO_CONDA_ENV:-verl_vllm}}"
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
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_DIR}"

MODEL_NAME="${MODEL_NAME:-microsoft/MiniLM-L12-H384-uncased}"
LATENT_DIM="${LATENT_DIM:-128}"
BETA="${BETA:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-5e-6}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
POOLING="${POOLING:-cls}"
SEED="${SEED:-42}"
SAVE_BEST="${SAVE_BEST:-1}"
RUN_NAME="${RUN_NAME:-inform_rm_lat${LATENT_DIM}_beta${BETA}}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN_NAME}}"

CMD=(
  python train_inform_rm.py
  --train_jsonl "${PAIR_DIR}/train.jsonl"
  --val_jsonl "${PAIR_DIR}/val.jsonl"
  --model_name "${MODEL_NAME}"
  --output_dir "${RUN_DIR}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --grad_accum "${GRAD_ACCUM}"
  --latent_dim "${LATENT_DIM}"
  --beta "${BETA}"
  --pooling "${POOLING}"
  --seed "${SEED}"
)

if [[ "${SAVE_BEST}" == "1" ]]; then
  CMD+=(--save_best)
fi
if [[ "${BF16:-0}" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "${FP16:-0}" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  CMD+=(--max_steps "${MAX_STEPS}")
fi

"${CMD[@]}"

echo "Done. InfoRM saved to ${RUN_DIR}"
