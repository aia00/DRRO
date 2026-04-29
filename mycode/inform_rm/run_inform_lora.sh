#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYCODE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PATH_CFG="${DRRO_PATH_CONFIG:-${MYCODE_ROOT}/project_paths.env}"
if [[ -f "${PATH_CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${PATH_CFG}"
fi

ENV_NAME="${ENV_NAME:-${DRRO_CONDA_ENV:-verl_vllm_fa2}}"
OUT_ROOT="${OUT_ROOT:-${DRRO_OUTPUT_ROOT:-}}"
RAY_TMPDIR="${RAY_TMPDIR:-${DRRO_RAY_TMPDIR:-}}"
INFORM_RM_PATH="${INFORM_RM_PATH:-}"

if [[ -z "${OUT_ROOT}" ]]; then
  echo "Set DRRO_OUTPUT_ROOT in project_paths.env or export OUT_ROOT." >&2
  exit 1
fi
if [[ -z "${INFORM_RM_PATH}" ]]; then
  echo "Export INFORM_RM_PATH=/path/to/trained_inform_rm before running downstream training." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${ENV_NAME}"
set -u

if [[ -n "${RAY_TMPDIR}" ]]; then
  export RAY_TMPDIR
fi

cd "${PROJECT_DIR}"

POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PROXY_RM="${PROXY_RM:-OpenAssistant/reward-model-deberta-v3-base}"
GOLD_RM="${GOLD_RM:-sileod/deberta-v3-large-tasksource-rlhf-reward-model}"
NUM_GPUS="${NUM_GPUS:-3}"
REWARD_GPUS="${REWARD_GPUS:-1}"
SHARE_REWARD_GPU="${SHARE_REWARD_GPU:-0}"
REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES:-}"
if [[ "${SHARE_REWARD_GPU}" == "1" ]]; then
  REWARD_GPUS=0
  FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES:-${FIRST_VISIBLE_GPU:-0}}"
fi
NUM_STEPS="${NUM_STEPS:-300}"
BATCH_SIZE_PROMPTS="${BATCH_SIZE_PROMPTS:-16}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INFORM_MAX_LENGTH="${INFORM_MAX_LENGTH:-512}"
INFORM_PENALTY_COEF="${INFORM_PENALTY_COEF:-0.01}"
LR="${LR:-1e-5}"
ACTOR_MICRO_BATCH="${ACTOR_MICRO_BATCH:-8}"
LOGPROB_MICRO_BATCH="${LOGPROB_MICRO_BATCH:-8}"
INFORM_TAG="${INFORM_TAG:-$(basename "${INFORM_RM_PATH}")}"
RUN_NAME="${RUN_NAME:-inform_${INFORM_TAG}_rollout${NUM_GENERATIONS}}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN_NAME}}"

python train_inform_policy.py \
  --policy_model "${POLICY_MODEL}" \
  --proxy_rm "${PROXY_RM}" \
  --gold_rm "${GOLD_RM}" \
  --inform_rm_path "${INFORM_RM_PATH}" \
  --output_dir "${RUN_DIR}" \
  --num_steps "${NUM_STEPS}" \
  --batch_size_prompts "${BATCH_SIZE_PROMPTS}" \
  --num_generations "${NUM_GENERATIONS}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --inform_max_length "${INFORM_MAX_LENGTH}" \
  --inform_penalty_coef "${INFORM_PENALTY_COEF}" \
  --lr "${LR}" \
  --actor_micro_batch_size_per_gpu "${ACTOR_MICRO_BATCH}" \
  --logprob_micro_batch_size_per_gpu "${LOGPROB_MICRO_BATCH}" \
  --num_gpus "${NUM_GPUS}" \
  --reward_gpus "${REWARD_GPUS}" \
  --reward_cuda_visible_devices "${REWARD_CUDA_VISIBLE_DEVICES}" \
  --use_lora \
  --bf16 \
  ${EXTRA_TRAIN_ARGS:-}
