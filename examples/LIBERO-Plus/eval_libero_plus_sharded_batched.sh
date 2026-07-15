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
max_batch_size=${max_batch_size:-1}
batch_timeout_ms=${batch_timeout_ms:-20}
min_free_mem_mb=${min_free_mem_mb:-12000}
max_servers_per_gpu=${max_servers_per_gpu:-7}
server_start_timeout_s=${server_start_timeout_s:-1500}
num_trials_per_task=${num_trials_per_task:-1}
max_steps=${max_steps:-}
replan_steps=${replan_steps:-3}
with_state=${with_state:-true}
save_video=${save_video:-false}
dry_run=${dry_run:-false}
clean_existing_results=${clean_existing_results:-true}
run_name=${run_name:-$(date +"%Y%m%d_%H%M%S")}
log_root=${log_root:-logs/libero_plus_sharded_${run_name}}
combined_results_path=${combined_results_path:-${log_root}/combined_results.json}

if (( num_servers <= 0 )); then
    echo "gpu_ids_str must contain at least one GPU id" >&2
    exit 1
fi
num_categories=${#items[@]}
if (( num_categories != 7 )); then
    echo "LIBERO-Plus one-server-one-eval mode expects exactly 7 categories, got ${num_categories}" >&2
    exit 1
fi
if (( num_servers % num_categories != 0 )); then
    echo "num_servers/eval workers must be divisible by ${num_categories}; got num_servers=${num_servers}" >&2
    exit 1
fi
shards_per_category=$((num_servers / num_categories))
echo "One-server-one-eval mode: num_servers=${num_servers}, categories=${num_categories}, shards_per_category=${shards_per_category}"

declare -A servers_per_gpu=()
for cuda_id in "${gpu_ids[@]}"; do
    servers_per_gpu["${cuda_id}"]=$(( ${servers_per_gpu["${cuda_id}"]:-0} + 1 ))
done
for cuda_id in "${!servers_per_gpu[@]}"; do
    if (( servers_per_gpu["${cuda_id}"] > max_servers_per_gpu )); then
        echo "GPU ${cuda_id} is assigned ${servers_per_gpu["${cuda_id}"]} VLA servers; max_servers_per_gpu=${max_servers_per_gpu}." >&2
        echo "This run is likely to OOM during inference/rendering. Use fewer workers, more GPUs, or explicitly raise max_servers_per_gpu if you accept the risk." >&2
        exit 1
    fi
done

check_gpu_memory() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for GPU memory preflight" >&2
        exit 1
    fi

    local bad=0
    for cuda_id in "${!servers_per_gpu[@]}"; do
        free_mb=$(nvidia-smi --id="${cuda_id}" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [[ -z "${free_mb}" ]]; then
            echo "Could not read free memory for GPU ${cuda_id}" >&2
            bad=1
            continue
        fi
        echo "GPU ${cuda_id}: ${free_mb} MiB free for ${servers_per_gpu["${cuda_id}"]} assigned server/eval workers"
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
    local deadline=$((SECONDS + server_start_timeout_s))
    while (( SECONDS < deadline )); do
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

write_combined_results() {
    "${starvla_python}" - "${combined_results_path}" "${task_suite_name}" "${folder_name}" <<'PY'
import collections
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
task_suite_name = sys.argv[2]
folder_name = sys.argv[3]
repo_root = pathlib.Path.cwd()
summaries = []
pattern = f"results/plus_{task_suite_name}_sharded/*/{folder_name}/shard_*_of_*/result_summary.json"
for summary_path in sorted(repo_root.glob(pattern)):
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    summary["summary_path"] = str(summary_path)
    summaries.append(summary)

categories = []
by_category = collections.defaultdict(list)
for summary in summaries:
    by_category[summary.get("category_value", "")].append(summary)

for category, category_summaries in sorted(by_category.items()):
    episodes = sum(int(s.get("total_episodes", 0)) for s in category_summaries)
    successes = sum(int(s.get("total_successes", 0)) for s in category_summaries)
    categories.append(
        {
            "category_value": category,
            "total_episodes": episodes,
            "total_successes": successes,
            "success_rate": successes / episodes if episodes else 0.0,
            "shards": category_summaries,
        }
    )

total_episodes = sum(c["total_episodes"] for c in categories)
total_successes = sum(c["total_successes"] for c in categories)
combined = {
    "total_episodes": total_episodes,
    "total_successes": total_successes,
    "success_rate": total_successes / total_episodes if total_episodes else 0.0,
    "categories": categories,
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Wrote combined results to {out_path}")
PY
}

server_pids=()
eval_pids=()

cleanup_children() {
    for pid in "${eval_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${server_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_children EXIT

mkdir -p "${log_root}"

if [[ "${save_video}" == "true" ]]; then
    save_video_args=(--args.save-video)
else
    save_video_args=(--args.no-save-video)
fi
max_steps_args=()
if [[ -n "${max_steps}" ]]; then
    max_steps_args=(--args.max-steps "${max_steps}")
fi

global_shard=0
for perturbation_name in "${items[@]}"
do
    perturbation_file_name=${perturbation_name// /_}
    for shard_index in $(seq 0 $((shards_per_category - 1)))
    do
        if (( global_shard >= num_servers )); then
            echo "Internal assignment error: global_shard=${global_shard} exceeds num_servers=${num_servers}" >&2
            exit 1
        fi
        server_idx=${global_shard}
        port=$((base_port + server_idx + 1))
        eval_cuda_id=${gpu_ids[$server_idx]}
        video_out_path="results/plus_${task_suite_name}_sharded/${perturbation_file_name}/${folder_name}/shard_${shard_index}_of_${shards_per_category}"
        echo "Assign eval worker ${global_shard}/${num_servers}: category='${perturbation_name}', shard=${shard_index}/${shards_per_category}, server=${server_idx}, gpu=${eval_cuda_id}, port=${port}"
        global_shard=$((global_shard + 1))
    done
done
if (( global_shard != num_servers )); then
    echo "Internal assignment error: planned ${global_shard} eval workers, expected num_servers=${num_servers}" >&2
    exit 1
fi
if [[ "${dry_run}" == "true" ]]; then
    echo "Dry run complete; no servers or eval workers were launched."
    exit 0
fi

check_gpu_memory
if [[ "${clean_existing_results}" == "true" && -d "results/plus_${task_suite_name}_sharded" ]]; then
    find "results/plus_${task_suite_name}_sharded" \
        -mindepth 2 \
        -maxdepth 2 \
        -type d \
        -name "${folder_name}" \
        -exec rm -rf {} +
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

for server_idx in "${!gpu_ids[@]}"
do
    port=$((base_port + server_idx + 1))
    if ! kill -0 "${server_pids[$server_idx]}" 2>/dev/null; then
        echo "Server ${server_idx} exited before becoming ready; see ${log_root}/server_${server_idx}_gpu${gpu_ids[$server_idx]}_port${port}.log" >&2
        exit 1
    fi
    wait_for_server "${port}" || {
        echo "Server ${server_idx} did not become ready; see ${log_root}/server_${server_idx}_gpu${gpu_ids[$server_idx]}_port${port}.log" >&2
        exit 1
    }
done

global_shard=0
for perturbation_name in "${items[@]}"
do
    perturbation_file_name=${perturbation_name// /_}
    for shard_index in $(seq 0 $((shards_per_category - 1)))
    do
        if (( global_shard >= num_servers )); then
            echo "Internal assignment error: global_shard=${global_shard} exceeds num_servers=${num_servers}" >&2
            exit 1
        fi
        server_idx=${global_shard}
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
            --args.replan-steps "${replan_steps}" \
            "${max_steps_args[@]}" \
            "${save_video_args[@]}" \
            --args.shard-index "${shard_index}" \
            --args.num-shards "${shards_per_category}" \
            --args.video-out-path "${video_out_path}" \
            > "${video_out_path}/eval.log" 2>&1 &
        eval_pids+=($!)
        global_shard=$((global_shard + 1))
    done
done
if (( global_shard != num_servers )); then
    echo "Internal assignment error: launched ${global_shard} eval workers, expected num_servers=${num_servers}" >&2
    exit 1
fi

status=0
for pid in "${eval_pids[@]}"; do
    wait "$pid" || status=1
    write_combined_results
done
write_combined_results
exit "$status"
