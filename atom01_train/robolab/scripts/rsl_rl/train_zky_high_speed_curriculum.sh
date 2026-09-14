#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/k205-2/miniconda3/envs/isaaclab/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-2026-08-03_11-35-08_velocity_continuous_roll_repair}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-model_52000.pt}"
RUN_NAME="${RUN_NAME:-high_speed_3p5_from_52000}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-40001}"

exec env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "${PYTHON_BIN}" "${ATOM_TRAIN_ROOT}/robolab/scripts/rsl_rl/train.py" \
    --task zky --headless --resume --reset_optimizer \
    --load_run "^${SOURCE_RUN}$" \
    --checkpoint "^${SOURCE_CHECKPOINT//./\\.}$" \
    --logger tensorboard --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" \
    --run_name "${RUN_NAME}" "$@"
