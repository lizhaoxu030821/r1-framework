#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG_FILE="${ATOM_TRAIN_ROOT}/robolab/robolab/tasks/direct/base/base_config.py"
TRAIN_SCRIPT="${ATOM_TRAIN_ROOT}/robolab/scripts/rsl_rl/train.py"
LOG_ROOT="${ATOM_TRAIN_ROOT}/logs/rsl_rl/zky"

TASK="${TASK:-zky}"
PHASE_ITERS="${PHASE_ITERS:-2000}"
INITIAL_RUN="${INITIAL_RUN:-2026-07-17_20-14-26}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-model_20000.pt}"
LOGGER="${LOGGER:-tensorboard}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    for candidate in \
        "/home/cq/miniconda3/envs/isaaclab/bin/python" \
        "/home/a/miniconda3/envs/isaaclab/bin/python" \
        "/home/a/miniconda3/envs/isaaclab_zxd/bin/python" \
        "/home/a/miniconda3/envs/robo_lzx/bin/python"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c 'import isaacsim' >/dev/null 2>&1; then
            PYTHON_BIN="${candidate}"
            break
        fi
    done
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

EXTRA_ARGS=("$@")

set_push_range() {
    local push="$1"
    "${PYTHON_BIN}" - "${CONFIG_FILE}" "${push}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
push = sys.argv[2]

block = f'''    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3, 6),
        params={{
            "velocity_range": {{
                "x": (-{push}, {push}),
                "y": (-{push}, {push}),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-{push}, {push}),
            }}
        }},
    )
'''

lines = path.read_text().splitlines(keepends=True)
start = None
for index, line in enumerate(lines):
    if line.startswith("    push_robot = EventTerm("):
        start = index
        break

if start is None:
    raise RuntimeError(f"Cannot find push_robot EventTerm in {path}")

depth = 0
end = None
for index in range(start, len(lines)):
    depth += lines[index].count("(") - lines[index].count(")")
    if index > start and depth == 0:
        end = index + 1
        break

if end is None:
    raise RuntimeError(f"Cannot find end of push_robot EventTerm in {path}")

lines[start:end] = [line + "\n" for line in block.splitlines()]
path.write_text("".join(lines))
PY
}

run_regex() {
    printf '^%s$' "$1"
}

checkpoint_regex() {
    local checkpoint="$1"
    printf '^%s$' "${checkpoint//./\\.}"
}

latest_run_for() {
    local run_name="$1"
    local latest
    latest="$(
        find "${LOG_ROOT}" -maxdepth 1 -type d -name "*_${run_name}" -printf '%f\n' \
            | sort \
            | tail -n 1
    )"
    if [[ -z "${latest}" ]]; then
        echo "No run directory found for run_name=${run_name} under ${LOG_ROOT}" >&2
        exit 1
    fi
    printf '%s' "${latest}"
}

latest_checkpoint_in_run() {
    local run_dir="$1"
    local latest
    latest="$(
        find "${LOG_ROOT}/${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' \
            | sort -V \
            | tail -n 1
    )"
    if [[ -z "${latest}" ]]; then
        echo "No model_*.pt checkpoint found in ${LOG_ROOT}/${run_dir}" >&2
        exit 1
    fi
    printf '%s' "${latest}"
}

train_phase() {
    local push="$1"
    local run_name="$2"

    echo
    echo "=== push range: -${push} ~ ${push}; train ${PHASE_ITERS} iterations ==="
    echo "resume: ${LOG_ROOT}/${CURRENT_RUN}/${CURRENT_CHECKPOINT}"
    set_push_range "${push}"

    (
        cd "${ATOM_TRAIN_ROOT}"
        "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
            --task "${TASK}" \
            --headless \
            --resume \
            --load_run "$(run_regex "${CURRENT_RUN}")" \
            --checkpoint "$(checkpoint_regex "${CURRENT_CHECKPOINT}")" \
            --max_iterations "${PHASE_ITERS}" \
            --logger "${LOGGER}" \
            --run_name "${run_name}" \
            "${EXTRA_ARGS[@]}"
    )

    CURRENT_RUN="$(latest_run_for "${run_name}")"
    CURRENT_CHECKPOINT="$(latest_checkpoint_in_run "${CURRENT_RUN}")"
    echo "finished: ${LOG_ROOT}/${CURRENT_RUN}/${CURRENT_CHECKPOINT}"
}

if [[ ! -f "${LOG_ROOT}/${INITIAL_RUN}/${INITIAL_CHECKPOINT}" ]]; then
    echo "Initial checkpoint not found: ${LOG_ROOT}/${INITIAL_RUN}/${INITIAL_CHECKPOINT}" >&2
    exit 1
fi

CURRENT_RUN="${INITIAL_RUN}"
CURRENT_CHECKPOINT="${INITIAL_CHECKPOINT}"

train_phase "0.15" "push_0p15"
train_phase "0.3" "push_0p30"
train_phase "0.5" "push_0p50"
train_phase "0.8" "push_0p80"

echo
echo "Final checkpoint: ${LOG_ROOT}/${CURRENT_RUN}/${CURRENT_CHECKPOINT}"
