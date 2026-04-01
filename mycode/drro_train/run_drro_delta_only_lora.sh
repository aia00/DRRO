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
RAY_TMPDIR="${RAY_TMPDIR:-${DRRO_RAY_TMPDIR:-}}"
if [[ -z "${RAY_TMPDIR}" ]]; then
  echo "Set DRRO_RAY_TMPDIR in project_paths.env or export RAY_TMPDIR." >&2
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

mkdir -p "${RAY_TMPDIR}"
export RAY_TMPDIR
export RAY_TEMP_DIR="${RAY_TMPDIR}"

OUT_BASE="${OUT_BASE:-${DRRO_OUTPUT_ROOT:-}}"
if [[ -z "${OUT_BASE}" ]]; then
  echo "Set DRRO_OUTPUT_ROOT in project_paths.env or export OUT_BASE." >&2
  exit 1
fi
NUM_GPUS="${NUM_GPUS:-3}"
REWARD_GPUS="${REWARD_GPUS:-}"
if [[ -z "${REWARD_GPUS}" ]]; then
  if [[ "${NUM_GPUS}" -ge 2 ]]; then
    REWARD_GPUS=1
  else
    REWARD_GPUS=0
  fi
fi
DELTA2="${DELTA2:-}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
VLLM_TP="${VLLM_TP:-0}"
VLLM_GPU_MEM="${VLLM_GPU_MEM:-0.7}"
VLLM_MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-16384}"
VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-1536}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROXY_RM="${PROXY_RM:-OpenAssistant/reward-model-deberta-v3-base}"
GOLD_RM="${GOLD_RM:-sileod/deberta-v3-large-tasksource-rlhf-reward-model}"
NUM_STEPS="${NUM_STEPS:-400}"
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

DYNAMIC_DELTA_COEFF_VAL="${DYNAMIC_DELTA_COEFF:-${DELTA_ALPHA:-}}"
if [[ -z "${DYNAMIC_DELTA_COEFF_VAL}" ]]; then
  DYNAMIC_DELTA_COEFF_VAL="$(extract_cli_value --dynamic_delta_coeff "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${DYNAMIC_DELTA_COEFF_VAL}" ]]; then
  DYNAMIC_DELTA_COEFF_VAL="$(extract_cli_value --delta_alpha "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${DYNAMIC_DELTA_COEFF_VAL}" ]]; then
  DYNAMIC_DELTA_COEFF_VAL="0.0"
fi

SOFT_ASSIGN_TAU_VAL="${SOFT_ASSIGN_TAU:-${DELTA_SOFTMAX_TAU:-}}"
if [[ -z "${SOFT_ASSIGN_TAU_VAL}" ]]; then
  SOFT_ASSIGN_TAU_VAL="$(extract_cli_value --soft_assign_tau "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${SOFT_ASSIGN_TAU_VAL}" ]]; then
  SOFT_ASSIGN_TAU_VAL="$(extract_cli_value --delta_softmax_tau "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${SOFT_ASSIGN_TAU_VAL}" ]]; then
  SOFT_ASSIGN_TAU_VAL="2.0"
fi

ASSIGN_MODE_VAL="${ASSIGN_MODE:-}"
if [[ -z "${ASSIGN_MODE_VAL}" ]]; then
  ASSIGN_MODE_VAL="$(extract_cli_value --assign_mode "${EXTRA_ARGS[@]}" || true)"
fi
if [[ -z "${ASSIGN_MODE_VAL}" ]]; then
  ASSIGN_MODE_VAL="soft"
fi

DELTA_TAG="$(format_num "${DELTA2}")"
COEFF_TAG="$(format_num "${DYNAMIC_DELTA_COEFF_VAL}")"
TAU_TAG="$(format_num "${SOFT_ASSIGN_TAU_VAL}")"
ASSIGN_TAG="${ASSIGN_MODE_VAL}"

if awk "BEGIN{exit !(${DYNAMIC_DELTA_COEFF_VAL} > 0)}"; then
  if [[ "${ASSIGN_MODE_VAL}" == "hard" ]]; then
    RUN_NAME="drro_dynamic_coeff${COEFF_TAG}_assign${ASSIGN_TAG}_rollout${NUM_GENERATIONS}"
  else
    RUN_NAME="drro_dynamic_coeff${COEFF_TAG}_assign${ASSIGN_TAG}_tau${TAU_TAG}_rollout${NUM_GENERATIONS}"
  fi
else
  if [[ "${ASSIGN_MODE_VAL}" == "hard" ]]; then
    RUN_NAME="drro_fixed_delta${DELTA_TAG}_assign${ASSIGN_TAG}_rollout${NUM_GENERATIONS}"
  else
    RUN_NAME="drro_fixed_delta${DELTA_TAG}_assign${ASSIGN_TAG}_tau${TAU_TAG}_rollout${NUM_GENERATIONS}"
  fi
fi

RUN_DIR="${RUN_DIR:-${OUT_BASE}/${RUN_NAME}}"

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
if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb --wandb_project "${WANDB_PROJECT:-drro-grpo}")
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    WANDB_ARGS+=(--wandb_run_name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

python train_drro_grpo.py \
  --fixed_delta "${DELTA2}" \
  --output_dir "${RUN_DIR}" \
  "${TRAIN_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

echo "Done. Output in ${RUN_DIR}"
