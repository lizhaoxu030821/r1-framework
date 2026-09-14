#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRAIN_SCRIPT="${ATOM_TRAIN_ROOT}/robolab/scripts/rsl_rl/train.py"

TASK="${TASK:-zky_speed_repair}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-zky}"
SOURCE_RUN="${SOURCE_RUN:-2026-08-04_02-57-30_high_speed_3p5_from_52000}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-model_60000.pt}"
RUN_NAME="${RUN_NAME:-speed_repair_40k_from_60000}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40001}"
LOGGER="${LOGGER:-tensorboard}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        "/home/a/miniconda3/envs/robo_lzx/bin/python" \
        "/home/a/miniconda3/envs/isaaclab/bin/python" \
        "/home/a/miniconda3/envs/isaaclab_zxd/bin/python" \
        "/home/a/miniconda3/envs/env_isaaclab/bin/python" \
        "python"; do
        if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -c 'import isaacsim' >/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
    PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "[INFO] task=${TASK}"
echo "[INFO] source=${SOURCE_EXPERIMENT}/${SOURCE_RUN}/${SOURCE_CHECKPOINT}"
echo "[INFO] run_name=${RUN_NAME}"
echo "[INFO] max_iterations=${MAX_ITERATIONS}, num_envs=${NUM_ENVS}"
echo "[INFO] python=${PYTHON_BIN}"

exec env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --task "${TASK}" --headless --resume --reset_optimizer \
    --resume_experiment_name "${SOURCE_EXPERIMENT}" \
    --load_run "^${SOURCE_RUN}$" \
    --checkpoint "^${SOURCE_CHECKPOINT//./\\.}$" \
    --logger "${LOGGER}" --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" \
    --run_name "${RUN_NAME}" "$@"
