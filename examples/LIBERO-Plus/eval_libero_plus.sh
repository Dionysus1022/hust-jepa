#!/bin/bash

set -u

export LIBERO_HOME=/data/LiYuhang/benchmarks/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=${PYTHONPATH:-}:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export sim_python=/data/LiYuhang/conda_envs/libero_plus/bin/python
export policy_python=${policy_python:-/data/LiYuhang/conda_envs/vlajepa/bin/python}
export LD_LIBRARY_PATH=/data/LiYuhang/conda_envs/libero_plus/lib:${LD_LIBRARY_PATH:-}

default_ckpt=/data/LiYuhang/models/VLA-JEPA/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
your_ckpt=${your_ckpt:-$default_ckpt}

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items=("Background Textures" "Camera Viewpoints" "Language Instructions" "Light Conditions" "Objects Layout" "Robot Initial States" "Sensor Noise")
task_suite_name=libero_mix

host="127.0.0.1"
base_port=${base_port:-14082}
with_state="true"
num_trials_per_task=${num_trials_per_task:-1} # must be 1 for perturbation evaluation
GPU_IDS_STR=${GPU_IDS_STR:-"0 1 2 3 4 5"}
read -r -a gpu_ids <<< "$GPU_IDS_STR"

if [ ${#gpu_ids[@]} -eq 0 ]; then
    echo "GPU_IDS_STR must contain at least one GPU id" >&2
    exit 1
fi

eval_pids=()
server_pids=()
running=0

cleanup_servers() {
    for server_pid in "${server_pids[@]:-}"; do
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    done
    server_pids=()
}

wait_batch() {
    local rc=0
    for eval_pid in "${eval_pids[@]:-}"; do
        wait "$eval_pid" || rc=1
    done
    eval_pids=()
    cleanup_servers
    running=0
    return "$rc"
}

trap cleanup_servers EXIT

for i in "${!items[@]}"
do
    perturbation_name=${items[$i]}
    perturbation_file_name=${perturbation_name// /_}
    gpu=${gpu_ids[$running]}
    port=$((base_port+i+1))

    video_out_path="/data/LiYuhang/outputs/VLA-JEPA/libero-plus/results/plus_${task_suite_name}/${perturbation_file_name}/${folder_name}"
    LOG_DIR="/data/LiYuhang/outputs/VLA-JEPA/libero-plus/logs/$(date +"%Y%m%d_%H%M%S")"
    mkdir -p "$LOG_DIR"
    mkdir -p "$video_out_path"

    "$policy_python" ./deployment/model_server/server_policy.py \
        --ckpt_path "$your_ckpt" \
        --port "$port" \
        --use_bf16 \
        --cuda "$gpu" > "${video_out_path}/server.log" 2>&1 &
    server_pids+=("$!")

    "$sim_python" ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path "$your_ckpt" \
        --args.host "$host" \
        --args.port "$port" \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" \
        --args.category_value "$perturbation_name" \
        --args.with_state "$with_state" > "${video_out_path}/eval.log" 2>&1 &
    eval_pids+=("$!")

    running=$((running+1))
    if [ "$running" -eq "${#gpu_ids[@]}" ]; then
        wait_batch || exit 1
    fi
done

if [ "$running" -gt 0 ]; then
    wait_batch || exit 1
fi

trap - EXIT
