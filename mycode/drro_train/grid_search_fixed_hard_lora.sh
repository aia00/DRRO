#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MYCODE_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PATH_CFG="${DRRO_PATH_CONFIG:-${MYCODE_ROOT}/project_paths.env}"
if [[ -f "${PATH_CFG}" ]]; then
  # shellcheck disable=SC1090
  source "${PATH_CFG}"
fi

OUT_ROOT="${OUT_BASE:-${DRRO_OUTPUT_ROOT:-}}"
if [[ -z "${OUT_ROOT}" ]]; then
  echo "Set DRRO_OUTPUT_ROOT in project_paths.env or export OUT_BASE." >&2
  exit 1
fi

GRID_OUT_BASE="${GRID_OUT_BASE:-${OUT_ROOT}/grid_fixed_hard_lora}"
FIXED_DELTA_GRID="${FIXED_DELTA_GRID:-10 20 40 80}"
COMMON_EXTRA_TRAIN_ARGS="${COMMON_EXTRA_TRAIN_ARGS:-}"

mkdir -p "${GRID_OUT_BASE}"

for fixed_delta in ${FIXED_DELTA_GRID}; do
  echo "==> fixed+hard: fixed_delta=${fixed_delta}"
  EXTRA_ARGS="--assign_mode hard --dynamic_delta_coeff 0"
  if [[ -n "${COMMON_EXTRA_TRAIN_ARGS}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} ${COMMON_EXTRA_TRAIN_ARGS}"
  fi
  OUT_BASE="${GRID_OUT_BASE}" \
  DELTA2="${fixed_delta}" \
  EXTRA_TRAIN_ARGS="${EXTRA_ARGS}" \
  bash "${PROJECT_DIR}/run_drro_delta_only_lora.sh"
done

echo "Done. Grid outputs in ${GRID_OUT_BASE}"
