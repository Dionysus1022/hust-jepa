#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

ACCELERATE_BIN="${ACCELERATE_BIN:-/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/accelerate}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MASTER_PORT="${MASTER_PORT:-29522}"
CONFIG_YAML="${SCRIPT_DIR}/vlajepa_ft_widowx_state7d.yaml"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

COMMAND=(
  "${ACCELERATE_BIN}" launch
  --main_process_port "${MASTER_PORT}"
  --config_file "${REPO_ROOT}/starVLA/config/deepseeds/deepspeed_zero2.yaml"
  --num_processes "${NUM_PROCESSES}"
  "${REPO_ROOT}/starVLA/training/train_starvla.py"
  --config_yaml "${CONFIG_YAML}"
  "$@"
)

cd "${REPO_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
