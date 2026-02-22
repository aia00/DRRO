#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="verl_vllm"
RAY_TMPDIR="${RAY_TMPDIR:-/home/ykwang/mtdata2/ray_tmp}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

cd "${PROJECT_DIR}"

mkdir -p "${RAY_TMPDIR}"
export RAY_TMPDIR
export RAY_TEMP_DIR="${RAY_TMPDIR}"

OUT_BASE="${OUT_BASE:-/home/ykwang/mtdata2/DRRO}"
NUM_GPUS="${NUM_GPUS:-3}"
REWARD_GPUS="${REWARD_GPUS:-}"
if [[ -z "${REWARD_GPUS}" ]]; then
  if [[ "${NUM_GPUS}" -ge 2 ]]; then
    REWARD_GPUS=1
  else
    REWARD_GPUS=0
  fi
fi
DELTA1="${DELTA1:-0.0}"
DELTA2="${DELTA2:-}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
VLLM_TP="${VLLM_TP:-0}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.5}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-8192}"
VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-1024}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROXY_RM="${PROXY_RM:-OpenAssistant/reward-model-deberta-v3-base}"
GOLD_RM="${GOLD_RM:-sileod/deberta-v3-large-tasksource-rlhf-reward-model}"
NUM_STEPS="${NUM_STEPS:-300}"
if [[ -z "${DELTA2}" ]]; then
  DELTA2="$(awk "BEGIN{printf \"%.6f\", ${NUM_GENERATIONS}*2.5}")"
fi

RUN1_DIR="${OUT_BASE}/grpo"
RUN2_DIR="${OUT_BASE}/drro_delta${DELTA2}"

TRAIN_ARGS=(
  --num_gpus "${NUM_GPUS}"
  --rollout_backend "${ROLLOUT_BACKEND}"
  --num_generations "${NUM_GENERATIONS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --policy_model "${POLICY_MODEL}"
  --proxy_rm "${PROXY_RM}"
  --gold_rm "${GOLD_RM}"
  --num_steps "${NUM_STEPS}"
  --eval_every 10
  --save_every 20
  --no_lora
)
if [[ -n "${REWARD_GPUS}" ]]; then
  TRAIN_ARGS+=(--reward_gpus "${REWARD_GPUS}")
fi
if [[ "${ROLLOUT_BACKEND}" == "vllm" ]]; then
  TRAIN_ARGS+=(
    --vllm_tensor_parallel "${VLLM_TP}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEM}"
    --vllm_max_num_batched_tokens "${VLLM_MAX_BATCHED_TOKENS}"
    --vllm_max_num_seqs "${VLLM_MAX_SEQS}"
  )
  if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    TRAIN_ARGS+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
  fi
  if [[ "${VLLM_DISABLE_PREFIX_CACHING:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--vllm_disable_prefix_caching)
  else
    TRAIN_ARGS+=(--vllm_enable_prefix_caching)
  fi
  if [[ "${VLLM_DISABLE_CHUNKED_PREFILL:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--vllm_disable_chunked_prefill)
  else
    TRAIN_ARGS+=(--vllm_enable_chunked_prefill)
  fi
  if [[ "${VLLM_DISABLE_SLEEP_MODE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--vllm_disable_sleep_mode)
  fi
  if [[ "${VLLM_DISABLE_FREE_CACHE_ENGINE:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--vllm_disable_free_cache_engine)
  fi
  if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--vllm_enforce_eager)
  fi
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_TRAIN_ARGS}"
fi

WANDB_ARGS=()
if [[ "${ENABLE_WANDB:-1}" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "${WANDB_PROJECT:-drro-grpo}")
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

python train_drro_grpo.py \
  --delta "${DELTA1}" \
  --output_dir "${RUN1_DIR}" \
  "${TRAIN_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

python train_drro_grpo.py \
  --delta "${DELTA2}" \
  --output_dir "${RUN2_DIR}" \
  "${TRAIN_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

python plot_overopt_curve.py \
  --inputs "${RUN1_DIR}/log.csv" "${RUN2_DIR}/log.csv" \
  --out "${OUT_BASE}/overopt_kl_curve.png"

echo "Done. Plot saved to ${OUT_BASE}/overopt_kl_curve.png"
