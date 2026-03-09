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
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROXY_RM="${PROXY_RM:-OpenAssistant/reward-model-deberta-v3-base}"
GOLD_RM="${GOLD_RM:-sileod/deberta-v3-large-tasksource-rlhf-reward-model}"
NUM_STEPS="${NUM_STEPS:-300}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
if [[ -z "${DELTA2}" ]]; then
  DELTA2="$(awk "BEGIN{printf \"%.6f\", ${NUM_GENERATIONS}*2.5}")"
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_TRAIN_ARGS}"
fi

extract_cli_value() {
  local key="$1"
  shift
  local args=("$@")
  local idx token
  for ((idx = 0; idx < ${#args[@]}; idx++)); do
    token="${args[$idx]}"
    if [[ "${token}" == "${key}" ]]; then
      if ((idx + 1 < ${#args[@]})); then
        printf "%s" "${args[$((idx + 1))]}"
        return 0
      fi
    elif [[ "${token}" == "${key}="* ]]; then
      printf "%s" "${token#*=}"
      return 0
    fi
  done
  return 1
}

format_num() {
  local value="$1"
  awk "BEGIN{printf \"%g\", ${value}}"
}

DELTA_ALPHA_VAL="${DELTA_ALPHA:-}"
if [[ -z "${DELTA_ALPHA_VAL}" ]]; then
  DELTA_ALPHA_VAL="$(extract_cli_value --delta_alpha "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${DELTA_ALPHA_VAL}" ]]; then
  DELTA_ALPHA_VAL="0.0"
fi

SOFTMAX_TAU_VAL="${DELTA_SOFTMAX_TAU:-}"
if [[ -z "${SOFTMAX_TAU_VAL}" ]]; then
  SOFTMAX_TAU_VAL="$(extract_cli_value --delta_softmax_tau "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${SOFTMAX_TAU_VAL}" ]]; then
  SOFTMAX_TAU_VAL="2.0"
fi

ALPHA_TAG="$(format_num "${DELTA_ALPHA_VAL}")"
TAU_TAG="$(format_num "${SOFTMAX_TAU_VAL}")"

build_run_name() {
  local delta_value="$1"
  local delta_tag
  delta_tag="$(format_num "${delta_value}")"
  if awk "BEGIN{exit !(${DELTA_ALPHA_VAL} > 0)}"; then
    printf "drro_dynamic_a%s_tau%s_rollout%s" "${ALPHA_TAG}" "${TAU_TAG}" "${NUM_GENERATIONS}"
  else
    printf "drro_fix_delta%s_tau%s_rollout%s" "${delta_tag}" "${TAU_TAG}" "${NUM_GENERATIONS}"
  fi
}

RUN1_DIR="${RUN1_DIR:-${OUT_BASE}/$(build_run_name "${DELTA1}")}"
RUN2_DIR="${RUN2_DIR:-${OUT_BASE}/$(build_run_name "${DELTA2}")}"

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
  --use_lora
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
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
