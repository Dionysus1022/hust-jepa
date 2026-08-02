#!/bin/bash
set -e

export PYTHONDONTWRITEBYTECODE=1

# ===== LIBERO 路径配置 =====
export LIBERO_HOME=/home/WangBizi/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero

export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH:-}
export PYTHONPATH=$(pwd):${PYTHONPATH}

export sim_python=/home/WangBizi/miniconda3/envs/libero/bin/python

# 如果渲染慢，建议加这个
export MUJOCO_GL=egl

# ===== checkpoint 配置 =====
your_ckpt=/home/WangBizi/VLA-JEPA/checkpoints/robot_ft/final_model/pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

# ===== 评测配置 =====
items=("libero_10" "libero_goal" "libero_object" "libero_spatial")

host="127.0.0.1"
base_port=15083
num_trials_per_task=50
with_state="true"

# ===== GPU 配置 =====
# 用法：
# bash run_eval.sh                 # 自动用所有 GPU，每张卡 1 个 worker
# GPUS="1 2 4 5" bash run_eval.sh  # 指定 GPU
# JOBS_PER_GPU=2 bash run_eval.sh  # 每张 GPU 跑 2 个 worker

if [ -n "${GPUS:-}" ]; then
    read -ra gpus <<< "${GPUS}"
else
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

jobs_per_gpu=${JOBS_PER_GPU:-1}

num_gpus=${#gpus[@]}

if [ "$num_gpus" -eq 0 ]; then
    echo "Error: No GPU found."
    exit 1
fi

# 构造 worker slots
# 例如 gpus=(0 3), JOBS_PER_GPU=2
# slots 就是 0 0 3 3
slots=()
for gpu in "${gpus[@]}"; do
    for j in $(seq 1 "$jobs_per_gpu"); do
        slots+=("$gpu")
    done
done

num_slots=${#slots[@]}

echo "Detected GPUs: ${gpus[*]}"
echo "JOBS_PER_GPU: ${jobs_per_gpu}"
echo "Total parallel workers: ${num_slots}"
echo "Worker slots: ${slots[*]}"
echo

run_one_suite() {
    local task_suite_name=$1
    local gpu=$2
    local port=$3
    local worker_id=$4

    local video_out_path="results/${task_suite_name}/${folder_name}_worker${worker_id}"
    mkdir -p "${video_out_path}"

    local server_log="${video_out_path}/server_gpu${gpu}_port${port}.log"
    local eval_log="${video_out_path}/eval_gpu${gpu}_port${port}.log"

    echo "=================================================="
    echo "Start task suite: ${task_suite_name}"
    echo "Physical GPU: ${gpu}"
    echo "Worker ID: ${worker_id}"
    echo "Port: ${port}"
    echo "Result dir: ${video_out_path}"
    echo "Server log: ${server_log}"
    echo "Eval log: ${eval_log}"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=${gpu} python ./deployment/model_server/server_policy.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --cuda 0 \
        > "${server_log}" 2>&1 &

    local server_pid=$!

    cleanup() {
        echo "Killing server PID: ${server_pid} for ${task_suite_name}"
        kill "${server_pid}" 2>/dev/null || true
    }

    trap cleanup EXIT

    echo "Server PID for ${task_suite_name}: ${server_pid}"
    echo "Waiting for server to start..."
    sleep 15

    set +e

    CUDA_VISIBLE_DEVICES=${gpu} "${sim_python}" ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path "${your_ckpt}" \
        --args.host "${host}" \
        --args.port "${port}" \
        --args.task-suite-name "${task_suite_name}" \
        --args.num-trials-per-task "${num_trials_per_task}" \
        --args.video-out-path "${video_out_path}" \
        --args.with_state "${with_state}" \
        > "${eval_log}" 2>&1

    local eval_status=$?

    set -e

    echo "Eval finished for ${task_suite_name}, status=${eval_status}"

    cleanup
    trap - EXIT

    sleep 3

    echo "Finished task suite: ${task_suite_name}"
    echo

    return ${eval_status}
}

# ===== 主循环：按照 slot 数量并行 =====
num_items=${#items[@]}
i=0

while [ $i -lt $num_items ]; do
    pids=()

    for slot_idx in $(seq 0 $((num_slots - 1))); do
        task_idx=$((i + slot_idx))

        if [ $task_idx -lt $num_items ]; then
            task_suite_name=${items[$task_idx]}
            gpu=${slots[$slot_idx]}
            port=$((base_port + task_idx + 1))
            worker_id=${slot_idx}

            run_one_suite "${task_suite_name}" "${gpu}" "${port}" "${worker_id}" &
            pids+=($!)
        fi
    done

    batch_status=0

    for pid in "${pids[@]}"; do
        wait "$pid" || batch_status=$?
    done

    if [ "$batch_status" -ne 0 ]; then
        echo "Warning: Some task in this batch failed, status=${batch_status}"
    fi

    i=$((i + num_slots))
done

echo "All LIBERO evaluations finished."