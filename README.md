# SonicStar

G1 版 VLA 代码仓库，分成两块：

- `starVLA/`: 训练、数据集、推理部署
- `wbc/`: 部署、遥操作采集、仿真

详细背景和通用流程可直接看：

- https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/
- https://starvla.github.io/docs/zh-cn/

## 采集数据

在 `wbc/` 下启动采集链路，核心入口是：

```bash
python gear_sonic/scripts/launch_data_collection.py --sim
python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cylinder and throw it into the trash bin"
```

采集后会在 `wbc/outputs/<dataset_name>/` 生成 `data/`、`videos/` 和 `meta/`。

## 训练 VLA

训练入口在 `starVLA/`：

```bash
bash examples/SonicLatent/train_files/run_sonic_latent_train.sh
```

默认配置见 `examples/SonicLatent/train_files/train_sonic_latent.yaml`。

## 部署推理

先起 policy server：

```bash
bash examples/SonicLatent/eval_files/run_policy_server.sh
```

再起在线推理：

```bash
PYTHONPATH=$PWD python examples/SonicLatent/eval_files/run_starvla_inference.py \
  --ckpt-path <ckpt> \
  --host 127.0.0.1 \
  --port 10093 \
  --prompt "pick up the cylinder and throw it into the trash bin"
```

## 目录说明

- `starVLA/examples/SonicLatent/`: G1 VLA 训练和部署
- `wbc/gear_sonic/scripts/`: 采集、推理、仿真入口
- `wbc/gear_sonic_deploy/`: G1 部署代码

