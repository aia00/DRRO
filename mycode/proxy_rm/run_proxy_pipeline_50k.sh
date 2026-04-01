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
PAIR_DIR="${PAIR_DIR:-${OUTPUT_ROOT}/proxy_pairs_50k}"
PROXY_OUT="${PROXY_OUT:-${OUTPUT_ROOT}/proxy_rm_minilm_50k}"
DATASET_PATH="${DATASET_PATH:-${DRRO_LOCAL_DATASET_DIR:-}}"
NUM_PAIRS="${NUM_PAIRS:-50000}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_RESPONSES="${NUM_RESPONSES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
RM_MAX_LENGTH="${RM_MAX_LENGTH:-512}"

mkdir -p "${PAIR_DIR}"

# 1) build pairs (4-GPU parallel by default)
OUTPUT_DIR="${PAIR_DIR}" \
NUM_PAIRS="${NUM_PAIRS}" \
NUM_SHARDS="${NUM_SHARDS}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
NUM_RESPONSES="${NUM_RESPONSES}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
RM_MAX_LENGTH="${RM_MAX_LENGTH}" \
DATASET_PATH="${DATASET_PATH}" \
  bash "${SCRIPT_DIR}/run_build_proxy_pairs_4gpu.sh"

# 2) merge shards into train/val
cat "${PAIR_DIR}"/train.shard*.jsonl > "${PAIR_DIR}/train.jsonl"
cat "${PAIR_DIR}"/val.shard*.jsonl > "${PAIR_DIR}/val.jsonl"

# 3) train proxy RM
python "${SCRIPT_DIR}/train_proxy_rm.py" \
  --train_jsonl "${PAIR_DIR}/train.jsonl" \
  --val_jsonl "${PAIR_DIR}/val.jsonl" \
  --output_dir "${PROXY_OUT}"

# 4) eval proxy RM vs gold
python "${SCRIPT_DIR}/eval_proxy_rm.py" \
  --data_jsonl "${PAIR_DIR}/val.jsonl" \
  --proxy_rm "${PROXY_OUT}"

echo "[done] proxy RM trained at ${PROXY_OUT}"
