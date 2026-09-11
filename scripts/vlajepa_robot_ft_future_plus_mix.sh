export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29513

export TORCHINDUCTOR_COMPILE_THREADS=4
export MAX_JOBS=4

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

unset NCCL_BLOCKING_WAIT
unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
  --main_process_port "${MASTER_PORT}" \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 2 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_robot_ft_future_plus_mix.yaml
