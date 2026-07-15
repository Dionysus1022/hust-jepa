#!/bin/bash

set -u

export LIBERO_HOME=${LIBERO_HOME:-/home/WangBizi/LIBERO}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LD_LIBRARY_PATH=${LIBERO_CONDA_LIB:-/home/WangBizi/miniconda3/envs/libero/lib}:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg-cache}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-/tmp/mesa-cache}
export MUJOCO_GL=${MUJOCO_GL:-egl}

export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH:-}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=${sim_python:-/home/WangBizi/miniconda3/envs/libero/bin/python}
export starvla_python=${starvla_python:-/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/python}

your_ckpt=${your_ckpt:-/home/WangBizi/VLA-JEPA/checkpoints/robot_ft/final_model/pytorch_model.pt}
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items_str=${items_str:-"libero_10|libero_goal|libero_object|libero_spatial"}
IFS='|' read -r -a items <<< "$items_str"

host=${host:-127.0.0.1}
base_port=${base_port:-15082}
gpu_ids_str=${gpu_ids_str:-"0"}
read -r -a gpu_ids <<< "$gpu_ids_str"
num_servers=${#gpu_ids[@]}
shards_per_suite=${shards_per_suite:-2}
max_batch_size=${max_batch_size:-4}
batch_timeout_ms=${batch_timeout_ms:-20}
min_free_mem_mb=${min_free_mem_mb:-12000}
server_start_timeout_s=${server_start_timeout_s:-900}
num_trials_per_task=${num_trials_per_task:-50}
with_state=${with_state:-true}
save_video=${save_video:-false}
run_name=${run_name:-$(date +"%Y%m%d_%H%M%S")}
log_root=${log_root:-logs/libero_sharded_${run_name}}

if (( num_servers <= 0 )); then
    echo "gpu_ids_str must contain at least one GPU id" >&2
    exit 1
fi
if (( shards_per_suite <= 0 )); then
    echo "shards_per_suite must be positive" >&2
    exit 1
fi
if [[ ! -f "${your_ckpt}" ]]; then
    echo "Checkpoint does not exist: ${your_ckpt}" >&2
    exit 1
fi

check_gpu_memory() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for GPU memory preflight" >&2
        exit 1
    fi

    local bad=0
    for cuda_id in "${gpu_ids[@]}"; do
        free_mb=$(nvidia-smi --id="${cuda_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ -z "${free_mb}" ]]; then
            echo "Could not read free memory for GPU ${cuda_id}" >&2
            bad=1
            continue
        fi
        echo "GPU ${cuda_id}: ${free_mb} MiB free"
        if (( free_mb < min_free_mem_mb )); then
            echo "GPU ${cuda_id} has less than min_free_mem_mb=${min_free_mem_mb} MiB free; refusing to start." >&2
            bad=1
        fi
    done

    if (( bad != 0 )); then
        echo "Choose freer GPUs with gpu_ids_str, lower min_free_mem_mb if you know it is safe, or wait for other jobs to finish." >&2
        exit 1
    fi
}

wait_for_server() {
    local port="$1"
    local server_pid="$2"
    local server_log="$3"
    local deadline=$((SECONDS + server_start_timeout_s))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "Server process exited before becoming ready; see ${server_log}" >&2
            return 1
        fi
        if "${starvla_python}" - "${host}" "${port}" <<'PY' >/dev/null 2>&1
import sys
import websockets.sync.client
from deployment.model_server.tools import msgpack_numpy

host, port = sys.argv[1], int(sys.argv[2])
with websockets.sync.client.connect(
    f"ws://{host}:{port}",
    compression=None,
    max_size=None,
    open_timeout=5,
) as conn:
    msgpack_numpy.unpackb(conn.recv())
PY
        then
            echo "Server on ${host}:${port} is ready"
            return 0
        fi
        sleep 5
    done
    echo "Timed out waiting for server on ${host}:${port}" >&2
    return 1
}

server_pids=()
eval_pids=()

cleanup_servers() {
    for pid in "${server_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_servers EXIT

mkdir -p "${log_root}"
check_gpu_memory

if [[ "${save_video}" == "true" ]]; then
    save_video_args=(--args.save-video)
else
    save_video_args=(--args.no-save-video)
fi

for server_idx in "${!gpu_ids[@]}"
do
    port=$((base_port + server_idx + 1))
    cuda_id=${gpu_ids[$server_idx]}
    CUDA_VISIBLE_DEVICES="${cuda_id}" "${starvla_python}" ./deployment/model_server/server_policy_batched.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --cuda 0 \
        --max_batch_size "${max_batch_size}" \
        --batch_timeout_ms "${batch_timeout_ms}" \
        > "${log_root}/server_${server_idx}_gpu${cuda_id}_port${port}.log" 2>&1 &
    server_pids+=($!)
done

for server_idx in "${!gpu_ids[@]}"
do
    port=$((base_port + server_idx + 1))
    if ! kill -0 "${server_pids[$server_idx]}" 2>/dev/null; then
        echo "Server ${server_idx} exited before becoming ready; see ${log_root}/server_${server_idx}_gpu${gpu_ids[$server_idx]}_port${port}.log" >&2
        exit 1
    fi
    server_log="${log_root}/server_${server_idx}_gpu${gpu_ids[$server_idx]}_port${port}.log"
    wait_for_server "${port}" "${server_pids[$server_idx]}" "${server_log}" || {
        echo "Server ${server_idx} did not become ready; see ${log_root}/server_${server_idx}_gpu${gpu_ids[$server_idx]}_port${port}.log" >&2
        exit 1
    }
done

global_shard=0
for task_suite_name in "${items[@]}"
do
    for shard_index in $(seq 0 $((shards_per_suite - 1)))
    do
        server_idx=$((global_shard % num_servers))
        port=$((base_port + server_idx + 1))
        eval_cuda_id=${gpu_ids[$server_idx]}
        video_out_path="results/${task_suite_name}_sharded/${folder_name}/shard_${shard_index}_of_${shards_per_suite}"
        mkdir -p "${video_out_path}"

        MUJOCO_EGL_DEVICE_ID="${eval_cuda_id}" "${sim_python}" ./examples/LIBERO/eval_libero_sharded.py \
            --args.pretrained-path "${your_ckpt}" \
            --args.host "${host}" \
            --args.port "${port}" \
            --args.task-suite-name "${task_suite_name}" \
            --args.benchmark-mode libero \
            --args.num-trials-per-task "${num_trials_per_task}" \
            --args.with-state "${with_state}" \
            "${save_video_args[@]}" \
            --args.shard-index "${shard_index}" \
            --args.num-shards "${shards_per_suite}" \
            --args.video-out-path "${video_out_path}" \
            > "${video_out_path}/eval.log" 2>&1 &
        eval_pids+=($!)
        global_shard=$((global_shard + 1))
    done
done

status=0
for pid in "${eval_pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
