#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/lzx/projects/robo_party_getup/r1框架/atom01_train}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/logs/rsl_rl/zky_getup_beyondmimic/2026-08-26_18-56-37_npz_mimic_native_curriculum_v47_fresh35k_20260826}"
OUTPUT_LOG="${OUTPUT_LOG:-/tmp/zky_v47_fresh_eval.log}"
PYTHON_BIN="${PYTHON_BIN:-/data/lzx/conda_envs/robo_lzx/bin/python}"
DEVICE="${DEVICE:-cuda:2}"
NUM_ENVS="${NUM_ENVS:-64}"
TRAIN_PID="${TRAIN_PID:-2001339}"
EVALUATOR="${PROJECT_DIR}/robolab/scripts/tools/beyondmimic/evaluate_zky_getup_checkpoint.py"

playback_speed_for_iteration() {
    local iteration="$1"
    if ((iteration < 5000)); then
        echo "0.50"
    elif ((iteration < 10000)); then
        echo "0.65"
    elif ((iteration < 17500)); then
        echo "0.80"
    elif ((iteration < 25000)); then
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

for iteration in $(seq 0 1000 34000) 34999; do
    checkpoint="${RUN_DIR}/model_${iteration}.pt"
    marker="checkpoint_complete=${iteration}"

    if grep -Fqx "$marker" "$OUTPUT_LOG"; then
        continue
    fi

    while ! checkpoint_is_stable "$checkpoint"; do
        if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
            echo "$(date '+%F %T') training_process_missing pid=${TRAIN_PID} waiting_for=${checkpoint}" >> "$OUTPUT_LOG"
        fi
        sleep 30
    done

    speed="$(playback_speed_for_iteration "$iteration")"
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
        if ((status == 0)); then
            echo "$marker"
        fi
        echo "$(date '+%F %T') evaluation_end iteration=${iteration}"
    } >> "$OUTPUT_LOG" 2>&1
done

selection="$({
    awk '
        /evaluation_start iteration=/ {
            split($0, fields, "iteration=")
            split(fields[2], value, " ")
            iteration = value[1] + 0
            physical = joint = body = orientation = -1
        }
        /^physical_success_rate=/ { split($0, value, "="); physical = value[2] + 0 }
        /^final_joint_mae_mean=/ { split($0, value, "="); joint = value[2] + 0 }
        /^final_body_rmse_mean=/ { split($0, value, "="); body = value[2] + 0 }
        /^final_anchor_ori_error_mean=/ { split($0, value, "="); orientation = value[2] + 0 }
        /evaluation_end iteration=/ && physical >= 0 && joint >= 0 && body >= 0 && orientation >= 0 {
            score = joint + body + orientation
            tier = physical >= 0.90 ? 2 : (physical > 0 ? 1 : 0)
            if (!found || tier > best_tier || (tier == best_tier && (physical > best_physical || (physical == best_physical && score < best_score)))) {
                found = 1
                best_tier = tier
                best_iteration = iteration
                best_physical = physical
                best_score = score
                best_joint = joint
                best_body = body
                best_orientation = orientation
            }
        }
        END {
            if (found) {
                printf "best_iteration=%d physical_success=%.6f imitation_score=%.6f joint_mae=%.6f body_rmse=%.6f root_ori_error=%.6f", best_iteration, best_physical, best_score, best_joint, best_body, best_orientation
            }
        }
    ' "$OUTPUT_LOG"
} || true)"

if [[ -n "$selection" ]]; then
    echo "$(date '+%F %T') ${selection}" >> "$OUTPUT_LOG"
    best_iteration="$(sed -n 's/.*best_iteration=\([0-9]\+\).*/\1/p' <<< "$selection")"
    ln -sfn "model_${best_iteration}.pt" "${RUN_DIR}/model_best_v47.pt"
fi

echo "$(date '+%F %T') watcher_complete" >> "$OUTPUT_LOG"
