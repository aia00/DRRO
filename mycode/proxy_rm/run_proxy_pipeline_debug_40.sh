#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${ENV_NAME:-verl_vllm}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ykwang/mtdata2/DRRO}"
PAIR_DIR="${PAIR_DIR:-${OUTPUT_ROOT}/proxy_pairs_debug40}"
PROXY_OUT="${PROXY_OUT:-${OUTPUT_ROOT}/proxy_rm_minilm_debug40}"
DATASET_PATH="${DATASET_PATH:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${PAIR_DIR}"

# 1) build pairs (4-GPU parallel by default)
OUTPUT_DIR="${PAIR_DIR}" \
NUM_PAIRS=40 \
NUM_SHARDS="${NUM_SHARDS}" \
NUM_RESPONSES=2 \
MAX_NEW_TOKENS=64 \
TEMPERATURE=1.0 \
TOP_P=0.95 \
RM_MAX_LENGTH=256 \
DATASET_PATH="${DATASET_PATH}" \
  bash "${SCRIPT_DIR}/run_build_proxy_pairs_4gpu.sh"

# 2) merge shards into train/val
cat "${PAIR_DIR}"/train.shard*.jsonl > "${PAIR_DIR}/train.jsonl"
cat "${PAIR_DIR}"/val.shard*.jsonl > "${PAIR_DIR}/val.jsonl"

python "${SCRIPT_DIR}/train_proxy_rm.py" \
  --train_jsonl "${PAIR_DIR}/train.jsonl" \
  --val_jsonl "${PAIR_DIR}/val.jsonl" \
  --output_dir "${PROXY_OUT}" \
  --epochs 1 \
  --batch_size 4 \
  --eval_batch_size 4

python "${SCRIPT_DIR}/eval_proxy_rm.py" \
  --data_jsonl "${PAIR_DIR}/val.jsonl" \
  --proxy_rm "${PROXY_OUT}"

echo "[done] debug proxy RM trained at ${PROXY_OUT}"
