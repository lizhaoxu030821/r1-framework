#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TASK="${TASK:-zky_vertical_obstacle}"
INITIAL_RUN="${INITIAL_RUN:-2026-08-03_11-35-08_velocity_continuous_roll_repair}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-model_52000.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-60001}"
LOGGER="${LOGGER:-tensorboard}"
PYTHON_BIN="${PYTHON_BIN:-/home/k205-2/miniconda3/envs/isaaclab/bin/python}"

cd "${ATOM_TRAIN_ROOT}"
exec env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
    "${PYTHON_BIN}" robolab/scripts/rsl_rl/train.py \
    --task "${TASK}" \
    --headless \
    --resume \
    --reset_optimizer \
    --resume_experiment_name zky \
    --load_run "^${INITIAL_RUN}$" \
    --checkpoint "^${INITIAL_CHECKPOINT//./\\.}$" \
    --logger "${LOGGER}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --run_name "vertical_step_v2_from_52000" \
    "$@"
