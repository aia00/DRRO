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
SHARE_REWARD_GPU="${SHARE_REWARD_GPU:-0}"
REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES:-}"
if [[ "${SHARE_REWARD_GPU}" == "1" ]]; then
  REWARD_GPUS=0
  FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  REWARD_CUDA_VISIBLE_DEVICES="${REWARD_CUDA_VISIBLE_DEVICES:-${FIRST_VISIBLE_GPU:-0}}"
fi


NUM_STEPS="${NUM_STEPS:-300}"
NUM_GENERATIONS="${NUM_GENERATIONS:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
ACTOR_MICRO_BATCH="${ACTOR_MICRO_BATCH:-8}"
LOGPROB_MICRO_BATCH="${LOGPROB_MICRO_BATCH:-8}"
ENSEMBLE_AGG="${ENSEMBLE_AGG:-uwo}"
UWO_LAMBDA="${UWO_LAMBDA:-1.0}"
ENSEMBLE_CALIBRATION="${ENSEMBLE_CALIBRATION:-1}"
ALLOW_SINGLE_ENSEMBLE="${ALLOW_SINGLE_ENSEMBLE:-0}"
ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
GOLD_RM="${GOLD_RM:-sileod/deberta-v3-large-tasksource-rlhf-reward-model}"
PROXY_RM="${PROXY_RM:-OpenAssistant/reward-model-deberta-v3-base}"
PROXY_RM_LIST="${PROXY_RM_LIST:-}"
PROXY_RM_MANIFEST="${PROXY_RM_MANIFEST:-}"

if [[ -n "${PROXY_RM_LIST}" ]]; then
  IFS=',' read -r -a MODELS <<< "${PROXY_RM_LIST}"
  NUM_ENSEMBLE="${#MODELS[@]}"
elif [[ -n "${PROXY_RM_MANIFEST}" && -f "${PROXY_RM_MANIFEST}" ]]; then
  NUM_ENSEMBLE="$(
    python - "${PROXY_RM_MANIFEST}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
if isinstance(payload.get("models"), list):
    print(len(payload["models"]))
elif isinstance(payload.get("members"), list):
    print(len(payload["members"]))
else:
    print(0)
PY
  )"
else
  NUM_ENSEMBLE="${NUM_ENSEMBLE:-1}"
fi

RUN_NAME="${RUN_NAME:-ensemble_${ENSEMBLE_AGG}_n${NUM_ENSEMBLE}_rollout${NUM_GENERATIONS}}"
RUN_DIR="${RUN_DIR:-${OUT_BASE}/${RUN_NAME}}"

EXTRA_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_TRAIN_ARGS}"
fi

CMD=(
  python train_ensemble_baseline.py
  --output_dir "${RUN_DIR}"
  --num_gpus "${NUM_GPUS}"
  --reward_gpus "${REWARD_GPUS}"
  --policy_model "${POLICY_MODEL}"
  --gold_rm "${GOLD_RM}"
  --proxy_rm "${PROXY_RM}"
  --ensemble_agg "${ENSEMBLE_AGG}"
  --uwo_lambda "${UWO_LAMBDA}"
  --num_steps "${NUM_STEPS}"
  --num_generations "${NUM_GENERATIONS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --actor_micro_batch_size_per_gpu "${ACTOR_MICRO_BATCH}"
  --logprob_micro_batch_size_per_gpu "${LOGPROB_MICRO_BATCH}"
  --adv_estimator "${ADV_ESTIMATOR}"
  --use_lora
)

if [[ -n "${PROXY_RM_MANIFEST}" ]]; then
  CMD+=(--proxy_rm_manifest "${PROXY_RM_MANIFEST}")
elif [[ -n "${PROXY_RM_LIST}" ]]; then
  CMD+=(--proxy_rm_list "${PROXY_RM_LIST}")
fi

if [[ "${ENSEMBLE_CALIBRATION}" != "1" ]]; then
  CMD+=(--no_ensemble_calibration)
fi

if [[ "${ALLOW_SINGLE_ENSEMBLE}" == "1" ]]; then
  CMD+=(--allow_single_ensemble)
fi

if [[ -n "${REWARD_CUDA_VISIBLE_DEVICES}" ]]; then
  CMD+=(--reward_cuda_visible_devices "${REWARD_CUDA_VISIBLE_DEVICES}")
fi

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
