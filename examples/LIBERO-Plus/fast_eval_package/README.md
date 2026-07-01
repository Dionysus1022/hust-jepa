# LIBERO-Plus Fast Evaluation Package

这个文件夹是一个可以单独发给别人的 LIBERO-Plus 加速评测包。对方的 repo 不需要提前包含我们的改动；本包自带 `repo_overlay/`，可以把需要的文件安装到目标 repo 中。

## 目录结构

```text
fast_eval_package/
├── README.md
├── install_into_repo.sh
├── run_fast_eval.sh
└── repo_overlay/
    ├── deployment/model_server/...
    ├── examples/LIBERO/...
    ├── examples/LIBERO-Plus/...
    └── tests/...
```

`repo_overlay/` 里的路径就是要复制到目标 repo 的相对路径。`install_into_repo.sh` 会按这些路径复制文件；如果目标文件已经存在，会先生成 `.bak.<timestamp>` 备份。

## 这个包包含哪些改动

核心加速文件：

- `examples/LIBERO-Plus/eval_libero_plus_sharded_batched.sh`
- `deployment/model_server/server_policy_batched.py`
- `deployment/model_server/tools/batched_websocket_policy_server.py`
- `deployment/model_server/tools/batch_inference_queue.py`
- `deployment/model_server/tools/image_tools.py`
- `deployment/model_server/tools/websocket_policy_server.py`
- `examples/LIBERO/eval_libero_sharded.py`
- `examples/LIBERO/eval_libero.py`
- `examples/LIBERO/model2libero_interface.py`
- `examples/LIBERO-Plus/libero_plus_init.py`

验证文件：

- `tests/test_batched_policy_server.py`
- `tests/test_server_policy_batched.py`
- `tests/test_libero_sharded_eval.py`
- `tests/test_libero_eval_env.py`

这些文件一起提供两类加速：

1. 按 perturbation category 和 task shard 并行启动多个 LIBERO-Plus eval worker。
2. 用 batched policy server 把多个 websocket inference 请求合成一个 batch 后调用一次 `policy.predict_action`。

## 给别人怎么用

假设你把整个 `fast_eval_package/` 发给了别人，对方本地有一个 StarVLA/VLA-JEPA repo：

```text
/path/to/their/repo
```

第一步，把本包里的 overlay 安装到对方 repo：

```bash
cd /path/to/fast_eval_package
TARGET_REPO=/path/to/their/repo bash install_into_repo.sh
```

也可以用位置参数：

```bash
bash install_into_repo.sh /path/to/their/repo
```

安装脚本会输出每个被复制的文件。如果目标 repo 里已有同名文件，会保留备份，例如：

```text
examples/LIBERO/eval_libero.py.bak.20260701_153000
```

第二步，配置 LIBERO-Plus 环境并运行评测：

```bash
TARGET_REPO=/path/to/their/repo \
LIBERO_HOME=/path/to/LIBERO-plus \
sim_python=/path/to/libero_plus/bin/python \
starvla_python=/path/to/VLA_JEPA/bin/python \
your_ckpt=/path/to/checkpoint.pt \
gpu_ids_str="0 1 2 3" \
shards_per_category=4 \
max_batch_size=8 \
batch_timeout_ms=20 \
num_trials_per_task=1 \
with_state=true \
run_name=my_libero_plus_eval \
bash run_fast_eval.sh
```

如果已经进入目标 repo，也可以直接运行安装后的脚本：

```bash
cd /path/to/their/repo

LIBERO_HOME=/path/to/LIBERO-plus \
sim_python=/path/to/libero_plus/bin/python \
starvla_python=/path/to/VLA_JEPA/bin/python \
your_ckpt=/path/to/checkpoint.pt \
gpu_ids_str="0 1 2 3" \
shards_per_category=4 \
max_batch_size=8 \
batch_timeout_ms=20 \
num_trials_per_task=1 \
with_state=true \
run_name=my_libero_plus_eval \
bash examples/LIBERO-Plus/eval_libero_plus_sharded_batched.sh
```

## 环境准备

需要两套 Python 环境：

- `starvla_python`：能加载 VLA-JEPA / StarVLA 模型的环境。
- `sim_python`：能运行 LIBERO-Plus / robosuite / mujoco 仿真的环境。

需要设置 LIBERO-Plus 路径：

```bash
export LIBERO_HOME=/path/to/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
```

LIBERO-Plus benchmark 需要启用 perturbation suite。如果目标 LIBERO-Plus 安装还没有启用，可以把安装到 repo 后的文件复制过去：

```bash
cp /path/to/their/repo/examples/LIBERO-Plus/libero_plus_init.py \
  ${LIBERO_HOME}/libero/libero/benchmark/__init__.py
```

## 常用参数

这些参数都通过环境变量传给 `eval_libero_plus_sharded_batched.sh`：

- `TARGET_REPO`：目标 repo 路径。使用外部 `run_fast_eval.sh` 时需要；在 repo 内直接运行共享脚本时不需要。
- `your_ckpt`：要评测的模型 checkpoint。
- `LIBERO_HOME`：LIBERO-Plus 仓库路径。
- `LIBERO_CONFIG_PATH`：通常是 `${LIBERO_HOME}/libero`。
- `sim_python`：LIBERO-Plus 仿真环境的 Python。
- `starvla_python`：模型服务端环境的 Python。
- `gpu_ids_str`：模型 server 和仿真 worker 使用的 GPU 列表，例如 `"0 1 2 3"`。
- `shards_per_category`：每个 perturbation category 切成多少个 shard。
- `max_batch_size`：batched policy server 一次最多合并多少个请求。
- `batch_timeout_ms`：server 等待凑 batch 的最长时间，单位毫秒。
- `num_trials_per_task`：每个 task rollout 次数。LIBERO-Plus perturbation evaluation 通常用 `1`。
- `with_state`：是否把 robot state 发给 action head，通常保持 `true`。
- `items_str`：要跑的 perturbation category，用 `|` 分隔。
- `base_port`：server 起始端口，默认 `14082`。实际端口是 `base_port + server_index + 1`。
- `run_name`：本次评测日志目录名。

## 本机默认配置

在 `/home/WangBizi/VLA-JEPA` 中，本包已经安装过 overlay，并把默认路径改成本机路径：

```bash
TARGET_REPO=/home/WangBizi/VLA-JEPA
LIBERO_HOME=/home/WangBizi/LIBERO-plus
sim_python=/home/WangBizi/miniconda3/envs/libero_plus/bin/python
starvla_python=/home/WangBizi/miniconda3/envs/VLA_JEPA/bin/python
your_ckpt=/home/WangBizi/VLA-JEPA/checkpoints/robot_ft/final_model/pytorch_model.pt
```

默认直接从 repo 根目录运行即可：

```bash
bash examples/LIBERO-Plus/fast_eval_package/run_fast_eval.sh
```

如需评测其他 checkpoint，只覆盖 `your_ckpt` 即可。

默认 `items_str` 会跑全部七类：

```bash
Background Textures|Camera Viewpoints|Language Instructions|Light Conditions|Objects Layout|Robot Initial States|Sensor Noise
```

只跑一部分 category：

```bash
TARGET_REPO=/path/to/their/repo \
items_str="Background Textures|Camera Viewpoints" \
gpu_ids_str="0 1" \
shards_per_category=2 \
bash run_fast_eval.sh
```

## 输出位置

server 日志：

```text
logs/libero_plus_sharded_${run_name}/server_<idx>_gpu<gpu>_port<port>.log
```

每个 shard 的评测日志：

```text
results/plus_libero_mix_sharded/<category>/<checkpoint_name>/shard_<i>_of_<n>/eval.log
```

如果 `save_video=true`，视频也会写在对应 shard 目录下。

## 推荐配置

单卡 smoke test：

```bash
TARGET_REPO=/path/to/their/repo \
gpu_ids_str="0" \
shards_per_category=1 \
items_str="Background Textures" \
num_trials_per_task=1 \
max_batch_size=4 \
bash run_fast_eval.sh
```

4 卡正式评测：

```bash
TARGET_REPO=/path/to/their/repo \
gpu_ids_str="0 1 2 3" \
shards_per_category=4 \
num_trials_per_task=1 \
max_batch_size=8 \
batch_timeout_ms=20 \
bash run_fast_eval.sh
```

如果显存紧张，优先降低 `max_batch_size`。如果 GPU 利用率低，可以增加 `shards_per_category` 或 `max_batch_size`。

## 可选验证

安装到目标 repo 后，如果目标环境有 pytest，可以跑：

```bash
cd /path/to/their/repo
python -m pytest \
  tests/test_batched_policy_server.py \
  tests/test_server_policy_batched.py \
  tests/test_libero_sharded_eval.py \
  tests/test_libero_eval_env.py \
  -q
```

这些测试是静态/轻量测试，不会启动模型或 mujoco 仿真。

## 常见问题

### 不想覆盖目标 repo 的文件

安装脚本会自动备份已有文件，备份后缀是 `.bak.<timestamp>`。如果需要人工检查，可以先查看本包：

```bash
find repo_overlay -type f | sort
```

### 端口冲突

改 `base_port`：

```bash
TARGET_REPO=/path/to/their/repo base_port=15082 bash run_fast_eval.sh
```

### 找不到 LIBERO-Plus benchmark

确认：

```bash
export LIBERO_HOME=/path/to/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${LIBERO_HOME}:${PYTHONPATH}
```

并确认 `${LIBERO_HOME}/libero/libero/benchmark/__init__.py` 已经包含 LIBERO-Plus perturbation benchmark。

### server 正常启动但 eval 连不上

检查：

- `base_port` 是否被占用。
- `logs/libero_plus_sharded_${run_name}/server_*.log` 是否有模型加载错误。
- `starvla_python` 是否能 import 目标 repo 代码。
- `sim_python` 是否能 import LIBERO-Plus。

### batch 没有提速

确认同时有多个 eval shard 在请求同一个 server。`gpu_ids_str` 里每张 GPU 会启动一个 server，所有 shard 会轮询分配到这些 server。`shards_per_category` 太小或只跑一个 category 时，batch 可能凑不起来。
