#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_CKPT="${REPO_ROOT}/checkpoints/direct_ft/oxe/simpler_ft_widowx_state7d_Qwen3.5_2B/final_model/pytorch_model.pt"
if (( $# > 1 )); then
    echo "Usage: $0 [checkpoint.pt]" >&2
    exit 2
fi
if (( $# == 1 )); then
    export your_ckpt="$1"
else
    export your_ckpt="${your_ckpt:-${DEFAULT_CKPT}}"
fi

# Bridge/WidowX tasks only. One worker covers the complete Bridge task category.
export items_str="bridge_put_on"
export task_suite_name="${task_suite_name:-simpler_env_widowx_state7d}"
export unnorm_key="oxe_bridge"
export expected_unnorm_key="oxe_bridge"
export expected_data_mix="widowx"
export with_state="true"
export gpu_ids_str="${gpu_ids_str:-0}"
export base_port="${base_port:-14200}"
export run_name="${run_name:-widowx_state7d_$(date +%Y%m%d_%H%M%S)}"

cd "${REPO_ROOT}"
exec bash "${SCRIPT_DIR}/eval_simpler_env_sharded_batched.sh"
