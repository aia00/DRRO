#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYCODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATH_CFG="${DRRO_PATH_CONFIG:-${MYCODE_ROOT}/project_paths.env}"
if [[ -f "${PATH_CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${PATH_CFG}"
fi
ENV_NAME="${ENV_NAME:-${DRRO_CONDA_ENV:-verl_vllm}}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u
  conda activate "${ENV_NAME}"
  set -u
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${DRRO_OUTPUT_ROOT:-}}"
if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "Set DRRO_OUTPUT_ROOT in project_paths.env or export OUTPUT_ROOT." >&2
  exit 1
fi
INPUT_PAIR_DIR="${INPUT_PAIR_DIR:-${OUTPUT_ROOT}/proxy_pairs_50k}"
WEAK_PAIR_DIR="${WEAK_PAIR_DIR:-${OUTPUT_ROOT}/proxy_pairs_2k_noisy}"
PROXY_OUT="${PROXY_OUT:-${OUTPUT_ROOT}/proxy_rm_distilbert_noisy2k}"

TRAIN_SIZE="${TRAIN_SIZE:-2000}"
VAL_SIZE="${VAL_SIZE:-200}"
FLIP_RATIO="${FLIP_RATIO:-0.2}"
SEED="${SEED:-42}"

MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-100}"

python "${SCRIPT_DIR}/build_weak_subset.py" \
  --train_jsonl "${INPUT_PAIR_DIR}/train.jsonl" \
  --val_jsonl "${INPUT_PAIR_DIR}/val.jsonl" \
  --out_dir "${WEAK_PAIR_DIR}" \
  --train_size "${TRAIN_SIZE}" \
  --val_size "${VAL_SIZE}" \
  --flip_ratio "${FLIP_RATIO}" \
  --seed "${SEED}"

python "${SCRIPT_DIR}/train_proxy_rm.py" \
  --train_jsonl "${WEAK_PAIR_DIR}/train.jsonl" \
  --val_jsonl "${WEAK_PAIR_DIR}/val.jsonl" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${PROXY_OUT}" \
  --batch_size "${BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS}" \
  --max_steps "${MAX_STEPS}"

python "${SCRIPT_DIR}/eval_proxy_rm.py" \
  --data_jsonl "${WEAK_PAIR_DIR}/val.jsonl" \
  --proxy_rm "${PROXY_OUT}"

echo "[done] weak proxy RM trained at ${PROXY_OUT}"
