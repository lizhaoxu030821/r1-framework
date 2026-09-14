#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/lzx/projects/robo_party_getup/r1框架/atom01_train}"
RUN_DIR="${RUN_DIR:?RUN_DIR must identify the formal v50 run directory}"
TRAIN_PID="${TRAIN_PID:?TRAIN_PID must identify the formal training process}"
OUTPUT_LOG="${OUTPUT_LOG:-/tmp/zky_v50_overlap_eval.log}"
PYTHON_BIN="${PYTHON_BIN:-/data/lzx/conda_envs/robo_lzx/bin/python}"
EVALUATOR="${PROJECT_DIR}/robolab/scripts/tools/beyondmimic/evaluate_zky_getup_checkpoint.py"
DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-64}"
INITIAL_ITERATION="${INITIAL_ITERATION:-38199}"
FINAL_ITERATION="${FINAL_ITERATION:-73198}"

maximum_speed_for_checkpoint() {
    local added=$(( $1 - INITIAL_ITERATION ))
    if ((added < 3000)); then echo "0.50"
    elif ((added < 6000)); then echo "0.525"
    elif ((added < 9000)); then echo "0.55"
    elif ((added < 12000)); then echo "0.575"
    elif ((added < 15000)); then echo "0.60"
    elif ((added < 19000)); then echo "0.65"
    elif ((added < 23000)); then echo "0.70"
    elif ((added < 27000)); then echo "0.80"
    elif ((added < 31000)); then echo "0.90"
    else echo "1.00"
    fi
}

checkpoint_is_stable() {
    local checkpoint="$1" first_size second_size
    first_size="$(stat -c %s "$checkpoint" 2>/dev/null)" || return 1
    sleep 5
    second_size="$(stat -c %s "$checkpoint" 2>/dev/null)" || return 1
    [[ "$first_size" = "$second_size" && "$second_size" -gt 0 ]]
}

evaluate_speed() {
    local iteration="$1" checkpoint="$2" speed="$3"
    local marker="checkpoint_complete=${iteration} speed=${speed}"
    grep -Fqx "$marker" "$OUTPUT_LOG" && return 0
    {
        echo "$(date '+%F %T') evaluation_start iteration=${iteration} speed=${speed} checkpoint=${checkpoint}"
        "$PYTHON_BIN" "$EVALUATOR" \
            --checkpoint "$checkpoint" --num_envs "$NUM_ENVS" \
            --device "$DEVICE" --headless --playback_speed "$speed"
        status=$?
        echo "evaluation_exit_status=${status}"
        ((status == 0)) && echo "$marker"
        echo "$(date '+%F %T') evaluation_end iteration=${iteration} speed=${speed}"
        return "$status"
    } >> "$OUTPUT_LOG" 2>&1
}

mkdir -p "$(dirname "$OUTPUT_LOG")"
touch "$OUTPUT_LOG"
echo "$(date '+%F %T') watcher_start run_dir=${RUN_DIR} device=${DEVICE}" >> "$OUTPUT_LOG"

iterations="$(seq 39000 1000 72000) ${FINAL_ITERATION}"
for iteration in $iterations; do
    checkpoint="${RUN_DIR}/model_${iteration}.pt"
    missing_process_checks=0
    while ! checkpoint_is_stable "$checkpoint"; do
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            missing_process_checks=$((missing_process_checks + 1))
            if ((missing_process_checks >= 3)); then
                echo "$(date '+%F %T') watcher_abort missing_checkpoint=${checkpoint}" >> "$OUTPUT_LOG"
                exit 1
            fi
        fi
        sleep 30
    done
    maximum_speed="$(maximum_speed_for_checkpoint "$iteration")"
    evaluate_speed "$iteration" "$checkpoint" "0.50" || true
    if [[ "$maximum_speed" != "0.50" ]]; then
        evaluate_speed "$iteration" "$checkpoint" "$maximum_speed" || true
    fi
done
echo "$(date '+%F %T') watcher_complete" >> "$OUTPUT_LOG"
