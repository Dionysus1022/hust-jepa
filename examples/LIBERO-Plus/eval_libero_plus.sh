#!/bin/bash
set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=${MUJOCO_GL:-egl}

# ===== LIBERO-Plus paths =====
export LIBERO_HOME=/home/WangBizi/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH:-}
export PYTHONPATH=$(pwd):${PYTHONPATH}

sim_python=/home/WangBizi/miniconda3/envs/libero_plus/bin/python

# ===== Checkpoint =====
your_ckpt=/home/WangBizi/VLA-JEPA/checkpoints/robot_ft/final_model/pytorch_model.pt
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# ===== Evaluation =====
items=(
  "Background Textures"
  "Camera Viewpoints"
  "Language Instructions"
  "Light Conditions"
  "Objects Layout"
  "Robot Initial States"
  "Sensor Noise"
)

task_suite_name=libero_mix
host=127.0.0.1
base_port=${BASE_PORT:-14082}
num_trials_per_task=1
with_state=true

eval_mode=${1:-${EVAL_MODE:-full}}
case "${eval_mode}" in
    full|all)
        eval_mode=full
        max_eval_tasks=0
        output_mode_dir=full
        save_videos=${SAVE_VIDEOS:-true}
        max_success_videos=${MAX_SUCCESS_VIDEOS:-10}
        max_failure_videos=${MAX_FAILURE_VIDEOS:-10}
        ;;
    100|limit100|quick)
        eval_mode=limit100
        max_eval_tasks=100
        output_mode_dir=limit100
        save_videos=true
        max_success_videos=${MAX_SUCCESS_VIDEOS:-100}
        max_failure_videos=${MAX_FAILURE_VIDEOS:-100}
        ;;
    *)
        echo "Usage: $0 [full|all|100|limit100|quick]"
        echo "Or set EVAL_MODE=full|100."
        exit 1
        ;;
esac

# Usage:
#   bash examples/LIBERO-Plus/eval_libero_plus.sh full
#   bash examples/LIBERO-Plus/eval_libero_plus.sh 100
#   EVAL_MODE=100 bash examples/LIBERO-Plus/eval_libero_plus.sh
#   GPUS="0 1 2 3" bash examples/LIBERO-Plus/eval_libero_plus.sh
#   GPUS="0 1" JOBS_PER_GPU=2 bash examples/LIBERO-Plus/eval_libero_plus.sh
if [ -n "${GPUS:-}" ]; then
    read -ra gpus <<< "${GPUS}"
else
    gpus=(4 5)
fi

jobs_per_gpu=${JOBS_PER_GPU:-1}
slots=()
for gpu in "${gpus[@]}"; do
    for _ in $(seq 1 "$jobs_per_gpu"); do
        slots+=("$gpu")
    done
done

num_workers=${#slots[@]}
if [ "$num_workers" -eq 0 ]; then
    echo "Error: no evaluation workers configured."
    exit 1
fi

queue_dir=$(mktemp -d /tmp/libero_plus_eval.XXXXXX)
queue_file="${queue_dir}/next_job"
queue_lock="${queue_dir}/queue.lock"
echo 0 > "${queue_file}"
touch "${queue_lock}"

cleanup() {
    rm -rf "${queue_dir}"
}
trap cleanup EXIT

claim_job() {
    local job_idx
    exec 9>"${queue_lock}"
    flock 9
    job_idx=$(<"${queue_file}")
    if [ "$job_idx" -ge "${#items[@]}" ]; then
        flock -u 9
        exec 9>&-
        return 1
    fi
    echo $((job_idx + 1)) > "${queue_file}"
    flock -u 9
    exec 9>&-
    echo "${job_idx}"
}

wait_for_server() {
    local port=$1
    local pid=$2
    local timeout=${SERVER_TIMEOUT:-600}
    local elapsed=0

    while true; do
        if ! kill -0 "${pid}" 2>/dev/null; then
            return 1
        fi
        if nc -z "${host}" "${port}" >/dev/null 2>&1; then
            return 0
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
}

run_one_perturbation() {
    local perturbation_name=$1
    local gpu=$2
    local port=$3
    local worker_id=$4

    local perturbation_file_name=${perturbation_name// /_}
    local output_path="results/plus_${task_suite_name}/${perturbation_file_name}/${folder_name}/${output_mode_dir}"
    mkdir -p "${output_path}"

    local server_log="${output_path}/server_gpu${gpu}_worker${worker_id}.log"
    local eval_log="${output_path}/eval_gpu${gpu}_worker${worker_id}.log"

    echo "[worker ${worker_id}] Starting '${perturbation_name}' on GPU ${gpu}, port ${port}"

    if nc -z "${host}" "${port}" >/dev/null 2>&1; then
        echo "[worker ${worker_id}] Port ${port} is already in use before starting '${perturbation_name}'."
        return 1
    fi

    CUDA_VISIBLE_DEVICES=${gpu} python ./deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --cuda 0 \
        > "${server_log}" 2>&1 &
    local server_pid=$!

    if ! wait_for_server "${port}" "${server_pid}"; then
        echo "[worker ${worker_id}] Server failed for '${perturbation_name}'. See ${server_log}"
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
        return 1
    fi

    set +e
    CUDA_VISIBLE_DEVICES=${gpu} "${sim_python}" ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.max-eval-tasks "${max_eval_tasks}" \
        --args.video-out-path "${output_path}" \
        --args.category_value "${perturbation_name}" \
        --args.with_state "${with_state}" \
        --args.save_videos "${save_videos}" \
        --args.max-success-videos "${max_success_videos}" \
        --args.max-failure-videos "${max_failure_videos}" \
        > "${eval_log}" 2>&1
    local eval_status=$?
    set -e

    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true

    echo "[worker ${worker_id}] Finished '${perturbation_name}', status=${eval_status}"
    return "${eval_status}"
}

worker_loop() {
    local worker_id=$1
    local gpu=$2
    local port=$((base_port + worker_id + 1))
    local status=0
    local job_idx

    while job_idx=$(claim_job); do
        if ! run_one_perturbation "${items[$job_idx]}" "${gpu}" "${port}" "${worker_id}"; then
            status=1
        fi
    done
    return "${status}"
}

echo "GPUs: ${gpus[*]}"
echo "Workers: ${num_workers} (${slots[*]})"
echo "EVAL_MODE: ${eval_mode}"
echo "MAX_EVAL_TASKS: ${max_eval_tasks} (0 means all)"
echo "SAVE_VIDEOS: ${save_videos}"
echo "VIDEO_QUOTA: ${max_success_videos} success + ${max_failure_videos} failure per perturbation"

pids=()
for worker_id in "${!slots[@]}"; do
    worker_loop "${worker_id}" "${slots[$worker_id]}" &
    pids+=($!)
done

overall_status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || overall_status=1
done

if [ "$overall_status" -ne 0 ]; then
    echo "Some LIBERO-Plus evaluations failed. Check their logs."
    exit "$overall_status"
fi

echo "All LIBERO-Plus evaluations finished."
