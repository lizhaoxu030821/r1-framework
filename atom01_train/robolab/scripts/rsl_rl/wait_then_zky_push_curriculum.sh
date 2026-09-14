#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_TRAIN_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LOG_ROOT="${ATOM_TRAIN_ROOT}/logs/rsl_rl/zky"
CURRICULUM_SCRIPT="${SCRIPT_DIR}/train_zky_push_curriculum.sh"

TASK="${TASK:-zky}"
INITIAL_RUN="${INITIAL_RUN:-}"
INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT:-model_20000.pt}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_SECONDS="${STABLE_SECONDS:-30}"

EXTRA_ARGS=("$@")

latest_non_push_run() {
    local latest=""
    local run
    while IFS= read -r run; do
        if [[ "${run}" == *_push_* ]]; then
            continue
        fi
        if find "${LOG_ROOT}/${run}" -maxdepth 1 -type f -name 'model_*.pt' -print -quit | grep -q .; then
            latest="${run}"
        fi
    done < <(find "${LOG_ROOT}" -maxdepth 1 -mindepth 1 -type d -printf '%T@ %f\n' | sort -n | sed 's/^[^ ]* //')

    if [[ -z "${latest}" ]]; then
        echo "No non-push run with model_*.pt found under ${LOG_ROOT}" >&2
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
    printf '%s' "${latest}"
}

wait_for_stable_file() {
    local path="$1"
    local before
    local after

    while true; do
        before="$(stat -c '%s' "${path}")"
        sleep "${STABLE_SECONDS}"
        after="$(stat -c '%s' "${path}")"
        if [[ "${before}" == "${after}" ]]; then
            return
        fi
        echo "$(date '+%F %T') ${path} is still changing; waiting for a stable checkpoint file"
    done
}

if [[ -z "${INITIAL_RUN}" ]]; then
    INITIAL_RUN="$(latest_non_push_run)"
fi

CHECKPOINT_PATH="${LOG_ROOT}/${INITIAL_RUN}/${INITIAL_CHECKPOINT}"
MARKER_PATH="${LOG_ROOT}/${INITIAL_RUN}/.${INITIAL_CHECKPOINT}.push_curriculum_started"

echo "Watching checkpoint: ${CHECKPOINT_PATH}"
echo "Poll seconds: ${POLL_SECONDS}; stable seconds: ${STABLE_SECONDS}"

while [[ ! -f "${CHECKPOINT_PATH}" ]]; do
    latest_checkpoint="$(latest_checkpoint_in_run "${INITIAL_RUN}")"
    if [[ -n "${latest_checkpoint}" ]]; then
        echo "$(date '+%F %T') waiting for ${INITIAL_CHECKPOINT}; latest is ${latest_checkpoint}"
    else
        echo "$(date '+%F %T') waiting for ${INITIAL_CHECKPOINT}; no model_*.pt yet"
    fi
    sleep "${POLL_SECONDS}"
done

wait_for_stable_file "${CHECKPOINT_PATH}"

if [[ -f "${MARKER_PATH}" ]]; then
    echo "Push curriculum was already started for ${INITIAL_RUN}/${INITIAL_CHECKPOINT}: ${MARKER_PATH}" >&2
    exit 1
fi
touch "${MARKER_PATH}"

echo "$(date '+%F %T') starting push curriculum from ${INITIAL_RUN}/${INITIAL_CHECKPOINT}"
exec env \
    TASK="${TASK}" \
    INITIAL_RUN="${INITIAL_RUN}" \
    INITIAL_CHECKPOINT="${INITIAL_CHECKPOINT}" \
    "${CURRICULUM_SCRIPT}" \
    "${EXTRA_ARGS[@]}"
