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

NUM_STEPS="${NUM_STEPS:-400}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
GOLD_RM="${GOLD_RM:-sileod/deberta-v3-large-tasksource-rlhf-reward-model}"
DUAL_LR="${DUAL_LR:-0.05}"
DUAL_EMA="${DUAL_EMA:-0.9}"
DUAL_CLIP="${DUAL_CLIP:-10.0}"
COMPONENT_CONFIG_JSON="${COMPONENT_CONFIG_JSON:-}"
THETA_JSON="${THETA_JSON:-}"

if [[ -z "${COMPONENT_CONFIG_JSON}" || -z "${THETA_JSON}" ]]; then
  echo "Please set COMPONENT_CONFIG_JSON and THETA_JSON." >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-constraint_xi_rollout${NUM_GENERATIONS}}"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_TRAIN_ARGS}"
fi

CMD=(
  python train_constraint_baseline.py
  --output_dir "${RUN_DIR}"
  --num_gpus "${NUM_GPUS}"
  --reward_gpus "${REWARD_GPUS}"
  --policy_model "${POLICY_MODEL}"
  --gold_rm "${GOLD_RM}"
  --component_config_json "${COMPONENT_CONFIG_JSON}"
  --theta_json "${THETA_JSON}"
  --constraint_mode xi
  --dual_lr "${DUAL_LR}"
  --dual_ema "${DUAL_EMA}"
  --dual_clip "${DUAL_CLIP}"
  --num_steps "${NUM_STEPS}"
  --num_generations "${NUM_GENERATIONS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --use_lora
)

if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  CMD+=(--wandb --wandb_project "${WANDB_PROJECT:-drro-grpo}")
  if [[ -n "${WANDB_RUN_NAME:-}" ]]; then
    CMD+=(--wandb_run_name "${WANDB_RUN_NAME}")
  fi
  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    CMD+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "[run] ${CMD[*]}"
"${CMD[@]}"

echo "Done. Output in ${RUN_DIR}"
