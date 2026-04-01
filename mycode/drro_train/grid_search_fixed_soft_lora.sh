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

GRID_OUT_BASE="${GRID_OUT_BASE:-${OUT_ROOT}/grid_fixed_soft_lora}"
FIXED_DELTA_GRID="${FIXED_DELTA_GRID:-10 20 40 80}"
SOFT_ASSIGN_TAU_GRID="${SOFT_ASSIGN_TAU_GRID:-1 2 5 10}"
COMMON_EXTRA_TRAIN_ARGS="${COMMON_EXTRA_TRAIN_ARGS:-}"

mkdir -p "${GRID_OUT_BASE}"

for fixed_delta in ${FIXED_DELTA_GRID}; do
  for tau in ${SOFT_ASSIGN_TAU_GRID}; do
    echo "==> fixed+soft: fixed_delta=${fixed_delta}, tau=${tau}"
    EXTRA_ARGS="--assign_mode soft --dynamic_delta_coeff 0 --soft_assign_tau ${tau}"
    if [[ -n "${COMMON_EXTRA_TRAIN_ARGS}" ]]; then
      EXTRA_ARGS="${EXTRA_ARGS} ${COMMON_EXTRA_TRAIN_ARGS}"
    fi
    OUT_BASE="${GRID_OUT_BASE}" \
    DELTA2="${fixed_delta}" \
    EXTRA_TRAIN_ARGS="${EXTRA_ARGS}" \
    bash "${PROJECT_DIR}/run_drro_delta_only_lora.sh"
  done
done

echo "Done. Grid outputs in ${GRID_OUT_BASE}"
