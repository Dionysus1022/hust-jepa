#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export SimplerEnv_PATH=${SimplerEnv_PATH:-/home/WangBizi/SimplerEnv}
export LD_LIBRARY_PATH=${SIMPLER_CONDA_LIB:-/home/WangBizi/miniconda3/envs/simpler_env/lib}:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/xdg-cache}
export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-/tmp/mesa-cache}
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}
export SVULKAN2_CPU_COPY=${SVULKAN2_CPU_COPY:-1}
export SVULKAN2_DISABLE_DENOISER=${SVULKAN2_DISABLE_DENOISER:-1}

export PYTHONPATH=${SimplerEnv_PATH}:${PYTHONPATH:-}
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}
export sim_python=${sim_python:-/home/WangBizi/miniconda3/envs/simpler_env/bin/python}
export starvla_python=${starvla_python:-/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/python}

your_ckpt=${your_ckpt:-/data/WangBizi/VLA_JEPA_models/pretrain_checkpoint/SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt}
folder_name=${folder_name:-$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')}

items_str=${items_str:-"bridge_put_on|drawer|move_near|pick_coke_can|long_horizon_apple_in_drawer"}
IFS='|' read -r -a items <<< "$items_str"

task_suite_name=${task_suite_name:-simpler_env}
host=${host:-127.0.0.1}
base_port=${base_port:-14082}
gpu_ids_str=${gpu_ids_str:-"0"}
read -r -a gpu_ids <<< "$gpu_ids_str"
num_servers=${#gpu_ids[@]}
max_batch_size=${max_batch_size:-1}
batch_timeout_ms=${batch_timeout_ms:-20}
max_servers_per_gpu=${max_servers_per_gpu:-7}
server_start_timeout_s=${server_start_timeout_s:-1500}
num_trials_per_task=${num_trials_per_task:-1}
max_steps=${max_steps:-}
unnorm_key=${unnorm_key:-auto}
gripper_encoding=${gripper_encoding:-zero_one}
with_state=${with_state:-true}
seed=${seed:-7}
reset_state_history_on_subtask_change=${reset_state_history_on_subtask_change:-true}
save_video=${save_video:-false}
dry_run=${dry_run:-false}
clean_existing_results=${clean_existing_results:-true}
run_name=${run_name:-$(date +"%Y%m%d_%H%M%S")}
log_root=${log_root:-logs/simpler_env_sharded_${run_name}}
combined_results_path=${combined_results_path:-${log_root}/combined_results.json}

if [[ -n "${replan_steps+x}" ]]; then
    echo "NOTE: replan_steps=${replan_steps} is ignored. SimplerEnv now uses official per-step replanning with adaptive action ensembling."
fi

if (( num_servers <= 0 )); then
    echo "gpu_ids_str must contain at least one GPU id" >&2
    exit 1
fi
if [[ "${gripper_encoding}" != "zero_one" && "${gripper_encoding}" != "minus_one_one" ]]; then
    echo "gripper_encoding must be zero_one or minus_one_one" >&2
    exit 1
fi
if [[ "${with_state}" != "true" && "${with_state}" != "false" && "${with_state}" != "zero" ]]; then
    echo "with_state must be true, false, or zero" >&2
    exit 1
fi
if [[ "${reset_state_history_on_subtask_change}" != "true" && "${reset_state_history_on_subtask_change}" != "false" ]]; then
    echo "reset_state_history_on_subtask_change must be true or false" >&2
    exit 1
fi
num_categories=${#items[@]}
if (( num_categories <= 0 )); then
    echo "items_str must contain at least one SimplerEnv category" >&2
    exit 1
fi
for category in "${items[@]}"; do
    case "${category}" in
        bridge_put_on|drawer|move_near|pick_coke_can|long_horizon_apple_in_drawer) ;;
        *)
            echo "Unknown category in items_str: ${category}" >&2
            exit 1
            ;;
    esac
done
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

validate_checkpoint_metadata() {
    "${starvla_python}" - \
        "${your_ckpt}" \
        "${expected_data_mix:-}" \
        "${expected_unnorm_key:-}" <<'PY'
import json
import pathlib
import sys

from omegaconf import OmegaConf

checkpoint = pathlib.Path(sys.argv[1])
expected_data_mix = sys.argv[2]
expected_unnorm_key = sys.argv[3]
run_dir = checkpoint.parents[1]
config_path = run_dir / "config.yaml"
statistics_path = run_dir / "dataset_statistics.json"

if not config_path.is_file():
    raise FileNotFoundError(f"Missing training config: {config_path}")
if not statistics_path.is_file():
    raise FileNotFoundError(f"Missing dataset statistics: {statistics_path}")

config = OmegaConf.load(config_path)
checks = {
    "framework.action_model.state_dim": (
        config.framework.action_model.state_dim,
        7,
    ),
    "framework.vlanext_conditioning.use_proprio_input_vlm": (
        config.framework.vlanext_conditioning.use_proprio_input_vlm,
        True,
    ),
    "datasets.vla_data.with_state": (config.datasets.vla_data.with_state, True),
    "datasets.vla_data.proprio_encoding": (
        config.datasets.vla_data.get("proprio_encoding"),
        "vla_jepa_7d",
    ),
}
if expected_data_mix:
    checks["datasets.vla_data.data_mix"] = (
        config.datasets.vla_data.data_mix,
        expected_data_mix,
    )

failures = [
    f"{key}: got {actual!r}, expected {expected!r}"
    for key, (actual, expected) in checks.items()
    if actual != expected
]
with statistics_path.open("r", encoding="utf-8") as file:
    statistics = json.load(file)
if expected_unnorm_key and expected_unnorm_key not in statistics:
    failures.append(
        f"dataset_statistics key: missing {expected_unnorm_key!r}; "
        f"available={sorted(statistics)}"
    )
if failures:
    raise ValueError("Checkpoint is incompatible with this evaluator:\n- " + "\n- ".join(failures))

print(
    f"Validated checkpoint: data_mix={config.datasets.vla_data.data_mix}, "
    "state_dim=7, proprio_encoding=vla_jepa_7d"
)
PY
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
pattern = f"results/{task_suite_name}_sharded/*/{folder_name}/shard_*_of_*/result_summary.json"
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
    final_successes = sum(
        int(s.get("total_final_successes", s.get("total_successes", 0)))
        for s in category_summaries
    )
    any_successes = sum(
        int(s.get("total_any_successes", s.get("total_successes", 0)))
        for s in category_summaries
    )
    categories.append(
        {
            "category_value": category,
            "total_episodes": episodes,
            # Backward-compatible generic fields use the official final-state metric.
            "total_successes": final_successes,
            "success_rate": final_successes / episodes if episodes else 0.0,
            "total_final_successes": final_successes,
            "final_success_rate": final_successes / episodes if episodes else 0.0,
            "total_any_successes": any_successes,
            "any_success_rate": any_successes / episodes if episodes else 0.0,
            "transient_only_successes": any_successes - final_successes,
            "shards": category_summaries,
        }
    )

total_episodes = sum(c["total_episodes"] for c in categories)
total_final_successes = sum(c["total_final_successes"] for c in categories)
total_any_successes = sum(c["total_any_successes"] for c in categories)
combined = {
    "total_episodes": total_episodes,
    # Backward-compatible generic fields use the official final-state metric.
    "total_successes": total_final_successes,
    "success_rate": total_final_successes / total_episodes if total_episodes else 0.0,
    "total_final_successes": total_final_successes,
    "final_success_rate": total_final_successes / total_episodes if total_episodes else 0.0,
    "total_any_successes": total_any_successes,
    "any_success_rate": total_any_successes / total_episodes if total_episodes else 0.0,
    "transient_only_successes": total_any_successes - total_final_successes,
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
if [[ "${reset_state_history_on_subtask_change}" == "true" ]]; then
    state_history_args=(--args.reset-state-history-on-subtask-change)
else
    state_history_args=(--args.keep-state-history-on-subtask-change)
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
        video_out_path="results/${task_suite_name}_sharded/${perturbation_file_name}/${folder_name}/shard_${shard_index}_of_${shards_per_category}"
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

if [[ ! -f "${your_ckpt}" ]]; then
    echo "Checkpoint does not exist: ${your_ckpt}" >&2
    echo "Pass a checkpoint with: your_ckpt=/path/to/model.pt $0" >&2
    exit 1
fi
if ! validate_checkpoint_metadata; then
    echo "Checkpoint metadata validation failed: ${your_ckpt}" >&2
    exit 1
fi

if [[ "${clean_existing_results}" == "true" && -d "results/${task_suite_name}_sharded" ]]; then
    find "results/${task_suite_name}_sharded" \
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
        --seed "${seed}" \
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
        video_out_path="results/${task_suite_name}_sharded/${perturbation_file_name}/${folder_name}/shard_${shard_index}_of_${shards_per_category}"
        mkdir -p "${video_out_path}"

        CUDA_VISIBLE_DEVICES="${eval_cuda_id}" ${sim_python} ./examples/SimplerEnv/eval_files/eval_simpler_sharded.py \
            --args.pretrained-path "${your_ckpt}" \
            --args.host "${host}" \
            --args.port "${port}" \
            --args.task-suite-name "${task_suite_name}" \
            --args.benchmark-mode simpler_env \
            --args.num-trials-per-task "${num_trials_per_task}" \
            --args.category-value "${perturbation_name}" \
            --args.with-state "${with_state}" \
            --args.seed "${seed}" \
            --args.unnorm-key "${unnorm_key}" \
            --args.gripper-encoding "${gripper_encoding}" \
            "${max_steps_args[@]}" \
            "${save_video_args[@]}" \
            "${state_history_args[@]}" \
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
