#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${ENV_NAME:-verl_vllm}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-/home/ykwang/mtdata2/DRRO/proxy_pairs}"
DATASET_PATH="${DATASET_PATH:-}"  # e.g. /home/ykwang/common_dataset_model/dataset/Anthropic_hh-rlhf
NUM_PAIRS="${NUM_PAIRS:-50000}"
NUM_RESPONSES="${NUM_RESPONSES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
RM_MAX_LENGTH="${RM_MAX_LENGTH:-512}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_IDS=(0 1 2 3)

EXTRA_ARGS=()
if [[ -n "${DATASET_PATH}" ]]; then
  EXTRA_ARGS+=("--dataset_path" "${DATASET_PATH}")
fi

mkdir -p "${OUTPUT_DIR}"

pids=()
for ((i=0; i<NUM_SHARDS; i++)); do
  GPU_ID=${GPU_IDS[$i]}
  echo "[run] shard ${i}/${NUM_SHARDS} on GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python "${SCRIPT_DIR}/build_proxy_pairs.py" \
      --output_dir "${OUTPUT_DIR}" \
      --num_pairs "${NUM_PAIRS}" \
      --num_responses "${NUM_RESPONSES}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --rm_max_length "${RM_MAX_LENGTH}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_id "${i}" \
      "${EXTRA_ARGS[@]}" \
      "$@" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "[done] all shards finished. output: ${OUTPUT_DIR}"
