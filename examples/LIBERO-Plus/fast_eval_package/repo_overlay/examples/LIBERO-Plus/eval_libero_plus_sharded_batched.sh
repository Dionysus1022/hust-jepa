#!/bin/bash

set -u

export LIBERO_HOME=${LIBERO_HOME:-/home/WangBizi/LIBERO-plus}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LD_LIBRARY_PATH=${LIBERO_CONDA_LIB:-/home/WangBizi/miniconda3/envs/libero_plus/lib}:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg-cache}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-/tmp/mesa-cache}

export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH:-}
export PYTHONPATH=$(pwd):${PYTHONPATH}
export sim_python=${sim_python:-/home/WangBizi/miniconda3/envs/libero_plus/bin/python}
export starvla_python=${starvla_python:-/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/python}

your_ckpt=${your_ckpt:-/home/WangBizi/VLA-JEPA/checkpoints/robot_ft/final_model/pytorch_model.pt}
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

items_str=${items_str:-"Background Textures|Camera Viewpoints|Language Instructions|Light Conditions|Objects Layout|Robot Initial States|Sensor Noise"}
IFS='|' read -r -a items <<< "$items_str"

task_suite_name=${task_suite_name:-libero_mix}
host=${host:-127.0.0.1}
base_port=${base_port:-14082}
gpu_ids_str=${gpu_ids_str:-"0"}
read -r -a gpu_ids <<< "$gpu_ids_str"
num_servers=${#gpu_ids[@]}
shards_per_category=${shards_per_category:-4}
max_batch_size=${max_batch_size:-8}
batch_timeout_ms=${batch_timeout_ms:-20}
num_trials_per_task=${num_trials_per_task:-1}
with_state=${with_state:-true}
save_video=${save_video:-false}
run_name=${run_name:-$(date +"%Y%m%d_%H%M%S")}
log_root=${log_root:-logs/libero_plus_sharded_${run_name}}

if (( num_servers <= 0 )); then
    echo "gpu_ids_str must contain at least one GPU id" >&2
    exit 1
fi
if (( shards_per_category <= 0 )); then
    echo "shards_per_category must be positive" >&2
    exit 1
fi

server_pids=()
eval_pids=()

cleanup_servers() {
    for pid in "${server_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_servers EXIT

mkdir -p "${log_root}"

if [[ "${save_video}" == "true" ]]; then
    save_video_args=(--args.save-video)
else
    save_video_args=(--args.no-save-video)
fi

for server_idx in "${!gpu_ids[@]}"
do
    port=$((base_port + server_idx + 1))
    cuda_id=${gpu_ids[$server_idx]}
    CUDA_VISIBLE_DEVICES="${cuda_id}" ${starvla_python} ./deployment/model_server/server_policy_batched.py \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        --cuda 0 \
        --max_batch_size "${max_batch_size}" \
        --batch_timeout_ms "${batch_timeout_ms}" \
        > "${log_root}/server_${server_idx}_gpu${cuda_id}_port${port}.log" 2>&1 &
    server_pids+=($!)
done

global_shard=0
for perturbation_name in "${items[@]}"
do
    perturbation_file_name=${perturbation_name// /_}
    for shard_index in $(seq 0 $((shards_per_category - 1)))
    do
        server_idx=$((global_shard % num_servers))
        port=$((base_port + server_idx + 1))
        eval_cuda_id=${gpu_ids[$server_idx]}
        video_out_path="results/plus_${task_suite_name}_sharded/${perturbation_file_name}/${folder_name}/shard_${shard_index}_of_${shards_per_category}"
        mkdir -p "${video_out_path}"

        MUJOCO_EGL_DEVICE_ID="${eval_cuda_id}" ${sim_python} ./examples/LIBERO/eval_libero_sharded.py \
            --args.pretrained-path "${your_ckpt}" \
            --args.host "${host}" \
            --args.port "${port}" \
            --args.task-suite-name "${task_suite_name}" \
            --args.benchmark-mode libero_plus \
            --args.num-trials-per-task "${num_trials_per_task}" \
            --args.category-value "${perturbation_name}" \
            --args.with-state "${with_state}" \
            "${save_video_args[@]}" \
            --args.shard-index "${shard_index}" \
            --args.num-shards "${shards_per_category}" \
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
