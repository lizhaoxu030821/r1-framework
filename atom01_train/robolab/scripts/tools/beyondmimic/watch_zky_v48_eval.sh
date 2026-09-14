#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/lzx/projects/robo_party_getup/r1框架/atom01_train}"
RUN_DIR="${RUN_DIR:?RUN_DIR must identify the formal v48 run directory}"
OUTPUT_LOG="${OUTPUT_LOG:-/tmp/zky_v48_long35k_eval.log}"
PYTHON_BIN="${PYTHON_BIN:-/data/lzx/conda_envs/robo_lzx/bin/python}"
DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-64}"
TRAIN_PID="${TRAIN_PID:?TRAIN_PID must identify the formal training process}"
EVALUATOR="${PROJECT_DIR}/robolab/scripts/tools/beyondmimic/evaluate_zky_getup_checkpoint.py"
INITIAL_ITERATION="${INITIAL_ITERATION:-30000}"
FIRST_PERIODIC_ITERATION="${FIRST_PERIODIC_ITERATION:-31000}"
LAST_PERIODIC_ITERATION="${LAST_PERIODIC_ITERATION:-64000}"
FINAL_ITERATION="${FINAL_ITERATION:-64999}"
BEST_LINK_NAME="${BEST_LINK_NAME:-model_best_v48.pt}"

playback_speed_for_checkpoint() {
    local added_iterations=$(( $1 - INITIAL_ITERATION ))
    if ((added_iterations < 10000)); then
        echo "0.50"
    elif ((added_iterations < 15000)); then
        echo "0.65"
    elif ((added_iterations < 20000)); then
        echo "0.80"
    elif ((added_iterations < 25000)); then
        echo "0.90"
    else
        echo "1.00"
    fi
}

checkpoint_is_stable() {
    local checkpoint="$1"
    local first_size second_size
    first_size="$(stat -c %s "$checkpoint" 2>/dev/null)" || return 1
    sleep 5
    second_size="$(stat -c %s "$checkpoint" 2>/dev/null)" || return 1
    [[ "$first_size" = "$second_size" && "$second_size" -gt 0 ]]
}

mkdir -p "$(dirname "$OUTPUT_LOG")"
touch "$OUTPUT_LOG"
echo "$(date '+%F %T') watcher_start run_dir=${RUN_DIR} device=${DEVICE}" >> "$OUTPUT_LOG"

for iteration in $(seq "$FIRST_PERIODIC_ITERATION" 1000 "$LAST_PERIODIC_ITERATION") "$FINAL_ITERATION"; do
    checkpoint="${RUN_DIR}/model_${iteration}.pt"
    marker="checkpoint_complete=${iteration}"
    grep -Fqx "$marker" "$OUTPUT_LOG" && continue

    missing_process_checks=0
    while ! checkpoint_is_stable "$checkpoint"; do
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            missing_process_checks=$((missing_process_checks + 1))
            echo "$(date '+%F %T') training_process_missing pid=${TRAIN_PID} waiting_for=${checkpoint}" >> "$OUTPUT_LOG"
            if ((missing_process_checks >= 3)); then
                echo "$(date '+%F %T') watcher_abort missing_checkpoint=${checkpoint}" >> "$OUTPUT_LOG"
                exit 1
            fi
        fi
        sleep 30
    done

    speed="$(playback_speed_for_checkpoint "$iteration")"
    {
        echo "$(date '+%F %T') evaluation_start iteration=${iteration} speed=${speed} checkpoint=${checkpoint}"
        "$PYTHON_BIN" "$EVALUATOR" \
            --checkpoint "$checkpoint" \
            --num_envs "$NUM_ENVS" \
            --device "$DEVICE" \
            --headless \
            --playback_speed "$speed"
        status=$?
        echo "evaluation_exit_status=${status}"
        ((status == 0)) && echo "$marker"
        echo "$(date '+%F %T') evaluation_end iteration=${iteration}"
    } >> "$OUTPUT_LOG" 2>&1
done

selection="$(awk '
    /evaluation_start iteration=/ {
        split($0, fields, "iteration="); split(fields[2], value, " ")
        iteration = value[1] + 0; physical = joint = body = orientation = -1
    }
    /^physical_success_rate=/ { split($0, value, "="); physical = value[2] + 0 }
    /^final_joint_mae_mean=/ { split($0, value, "="); joint = value[2] + 0 }
    /^final_body_rmse_mean=/ { split($0, value, "="); body = value[2] + 0 }
    /^final_anchor_ori_error_mean=/ { split($0, value, "="); orientation = value[2] + 0 }
    /evaluation_end iteration=/ && physical >= 0 && joint >= 0 && body >= 0 && orientation >= 0 {
        score = joint + body + orientation
        tier = physical >= 0.90 ? 2 : (physical > 0 ? 1 : 0)
        if (!found || tier > best_tier || (tier == best_tier && (physical > best_physical || (physical == best_physical && score < best_score)))) {
            found = 1; best_tier = tier; best_iteration = iteration
            best_physical = physical; best_score = score
        }
    }
    END { if (found) printf "%d", best_iteration }
' "$OUTPUT_LOG")"

if [[ -n "$selection" ]]; then
    ln -sfn "model_${selection}.pt" "${RUN_DIR}/${BEST_LINK_NAME}"
    echo "$(date '+%F %T') best_iteration=${selection}" >> "$OUTPUT_LOG"
fi
echo "$(date '+%F %T') watcher_complete" >> "$OUTPUT_LOG"
