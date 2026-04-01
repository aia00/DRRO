#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYCODE_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
PATH_CFG="${DRRO_PATH_CONFIG:-${MYCODE_ROOT}/project_paths.env}"
if [[ -f "${PATH_CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${PATH_CFG}"
fi
ENV_NAME="${ENV_NAME:-${DRRO_CONDA_ENV:-verl_vllm}}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${ENV_NAME}"
set -u

cd "${PROJECT_DIR}"

PAIR_DIR="${PAIR_DIR:-${DRRO_PROXY_PAIRS_DIR:-}}"
if [[ -z "${PAIR_DIR}" ]]; then
  echo "Set DRRO_PROXY_PAIRS_DIR in project_paths.env or export PAIR_DIR." >&2
  exit 1
fi
TRAIN_JSONL="${TRAIN_JSONL:-${PAIR_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PAIR_DIR}/val.jsonl}"
if [[ ! -f "${TRAIN_JSONL}" || ! -f "${VAL_JSONL}" ]]; then
  echo "Missing pair files." >&2
  echo "  TRAIN_JSONL=${TRAIN_JSONL}" >&2
  echo "  VAL_JSONL=${VAL_JSONL}" >&2
  echo "Set PAIR_DIR (or TRAIN_JSONL/VAL_JSONL) to an existing proxy-pair folder." >&2
  echo "Example: PAIR_DIR=\${DRRO_OUTPUT_ROOT}/runs/proxy_pairs/proxy_pairs_50k" >&2
  exit 1
fi
OUT_BASE="${DRRO_OUTPUT_ROOT:-}"
if [[ -z "${OUT_BASE}" && -z "${OUT_DIR:-}" ]]; then
  echo "Set DRRO_OUTPUT_ROOT in project_paths.env or export OUT_DIR." >&2
  exit 1
fi
OUT_DIR="${OUT_DIR:-${OUT_BASE}/proxy_ensemble}"

MODEL_NAME="${MODEL_NAME:-microsoft/MiniLM-L12-H384-uncased}"
MEMBER_MODEL_LIST="${MEMBER_MODEL_LIST:-microsoft/MiniLM-L12-H384-uncased,prajjwal1/bert-small,google/electra-small-discriminator,distilbert-base-uncased,distilroberta-base}"
NUM_MEMBERS="${NUM_MEMBERS:-5}"
SEEDS="${SEEDS:-42,43,44,45,46}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"

CMD=(
  python train_proxy_ensemble.py
  --train_jsonl "${TRAIN_JSONL}"
  --val_jsonl "${VAL_JSONL}"
  --output_dir "${OUT_DIR}"
  --model_name "${MODEL_NAME}"
  --member_model_list "${MEMBER_MODEL_LIST}"
  --num_members "${NUM_MEMBERS}"
  --seeds "${SEEDS}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --max_length "${MAX_LENGTH}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --grad_accum "${GRAD_ACCUM}"
  --max_steps "${MAX_STEPS}"
  --parallel_workers "${PARALLEL_WORKERS}"
  --gpu_ids "${GPU_IDS}"
)

if [[ "${BF16:-0}" == "1" ]]; then
  CMD+=(--bf16)
fi
if [[ "${FP16:-0}" == "1" ]]; then
  CMD+=(--fp16)
fi
if [[ "${SAVE_BEST:-0}" == "1" ]]; then
  CMD+=(--save_best)
fi
if [[ -n "${PRETRAINED_LIST:-}" ]]; then
  CMD+=(--pretrained_list "${PRETRAINED_LIST}")
fi

echo "[run] ${CMD[*]}"
"${CMD[@]}"

echo "Done. Manifest at ${OUT_DIR}/proxy_ensemble_manifest.json"
