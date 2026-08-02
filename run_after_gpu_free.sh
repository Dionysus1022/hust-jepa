#!/usr/bin/env bash
set -euo pipefail

need_mb=80000

wait_gpus_free() {
  echo "Waiting for GPU 0,1 to have at least ${need_mb} MiB free..."
  while true; do
    free0=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    free1=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')

    echo "$(date '+%F %T') GPU0=${free0}MiB GPU1=${free1}MiB"

    if (( free0 >= need_mb && free1 >= need_mb )); then
      echo "GPU 0,1 are free enough."
      break
    fi

    sleep 60
  done
}

wait_gpus_free

folder_name=my_eval_20000 \
replan_steps=7 \
max_servers_per_gpu=20 \
gpu_ids_str="$(python - <<'PY'
print(' '.join(str(i % 2) for i in range(35)))
PY
)" \
your_ckpt="/home/WangBizi/VLA-JEPA/checkpoints/robot_ft_future/checkpoints/steps_20000_pytorch_model.pt" \
num_trials_per_task=1 \
bash examples/LIBERO-Plus/eval_libero_plus_sharded_batched.sh

wait_gpus_free

bash /home/WangBizi/VLA-JEPA/scripts/vlajepa_cotrain_future.sh
