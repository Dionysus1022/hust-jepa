#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"

# Centralized environment variables for all child eval scripts.
# You can override them via environment variables when running this script.
: "${sim_python:=/home/WangBizi/miniconda3/envs/simpler_env/bin/python}"
: "${SimplerEnv_PATH:=/home/WangBizi/SimplerEnv}"
: "${star_vla_python:=/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/python}"
: "${MODEL_PATH:=/data/WangBizi/VLA_JEPA_models/pretrain_checkpoint/SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt}"
: "${VK_ICD_FILENAMES:=/etc/vulkan/icd.d/nvidia_icd.json}"
export sim_python
export SimplerEnv_PATH
export star_vla_python
export MODEL_PATH
export VK_ICD_FILENAMES
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

[[ -x "${sim_python}" ]] || { echo "Missing SimplerEnv Python: ${sim_python}" >&2; exit 1; }
[[ -x "${star_vla_python}" ]] || { echo "Missing VLA-JEPA Python: ${star_vla_python}" >&2; exit 1; }
[[ -d "${SimplerEnv_PATH}/simpler_env" ]] || { echo "Invalid SimplerEnv_PATH: ${SimplerEnv_PATH}" >&2; exit 1; }
[[ -f "${MODEL_PATH}" ]] || { echo "Missing checkpoint: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${VK_ICD_FILENAMES}" ]] || { echo "Missing Vulkan ICD: ${VK_ICD_FILENAMES}" >&2; exit 1; }

cd "${REPO_ROOT}"

sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_bridge.sh" "${MODEL_PATH}"

sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_drawer_visual_matching.sh" "${MODEL_PATH}"
sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_move_near_visual_matching.sh" "${MODEL_PATH}"
sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_pick_coke_can_visual_matching.sh" "${MODEL_PATH}"
sim_python="${sim_python}" SimplerEnv_PATH="${SimplerEnv_PATH}" bash "${SCRIPT_DIR}/star_put_in_drawer_visual_matching.sh" "${MODEL_PATH}"
