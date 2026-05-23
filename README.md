# SonicStar

Unitree G1 的 VLA 开源仓库，分成两块：

- `starVLA/`: 训练、数据集、推理部署
- `wbc/`: 部署、遥操作采集、仿真

详细背景和通用流程可直接看：

- https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/
- https://starvla.github.io/docs/zh-cn/

## 采集数据

在 `wbc/` 下启动采集链路,在不同终端下依次运行（建议把https://github.com/NVlabs/GR00T-WholeBodyControl.git 克隆下来，在那个仓库下跑，本仓库仅供参考示意）：

```bash
python gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port 5555
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
bash deploy.sh --input-type zmq_manager sim
python gear_sonic/scripts/run_data_exporter.py --task-prompt "pick up the cylinder and throw it into the trash bin"
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

采集完毕后根据starVLA教程https://starvla.github.io/docs/zh-cn/training/lerobot-dataset/ 修改数据集的meta/modality.json
也可以参考我自己采集的数据集https://huggingface.co/datasets/Tang-keke/merged_dataset_001
注意我的数据集中meta/的source_episodes、stats_gr00t、steps_data_index不是必要的文件，不需要参考

## 训练 VLA

```bash
bash examples/SonicLatent/train_files/run_sonic_latent_train.sh
```

默认配置：

```bash
examples/SonicLatent/train_files/train_sonic_latent.yaml
```

我的训练数据集从`GR00T-WholeBodyControl/` 也就是`wbc/` 采集后，放在starVLA/playground/Datasets/里面（建议把https://github.com/starVLA/starVLA.git 克隆下来，在那个仓库下跑，本仓库仅供参考示意）

## 部署推理

在 `starVLA/` 下先起 policy server,启动前更换run_policy_server.sh的模型路径，换成自己训练的模型：

```bash
bash examples/SonicLatent/eval_files/run_policy_server.sh
```

再起在线推理：

```bash
PYTHONPATH=$PWD python examples/SonicLatent/eval_files/run_starvla_inference.py \
  --ckpt-path /playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/<ckpt> \
  --host 127.0.0.1 \
  --port 10093 \
  --prompt "pick up the cylinder and throw it into the trash bin"
  --rate 1.0
```

然后在 `wbc/` 下依次启动

```bash
python gear_sonic/scripts/run_sim_loop.py --enable-image-publish --enable-offscreen --camera-port 5555
python gear_sonic/scripts/run_camera_viewer.py --camera-host localhost --camera-port 5555
bash deploy.sh --input-type zmq_manager sim
python gear_sonic/scripts/send_keyboard_cmd.py k 
```

send_keyboard_cmd作用与运行时机：

- `k`: 先启动 deploy.sh 完毕后，机器人完全进入init状态，发送键k可让机器人进入CONTROL模式，机器人会在空中挣扎（没有挣扎的话重启deploy.sh），然后在MuJoCo界面按9可将其放下
- `i`: 机器人放下后，发送键i可让机器人张开双手，准备执行任务（没有张开手的话再发送一次i）
- `p`: 机器人准备好之后发送键p启动/暂停 VLA policy 

比如要发送键`k`，直接在独立终端运行：

```bash
python gear_sonic/scripts/send_keyboard_cmd.py k 
```

send_keyboard_cmd这个脚本请在gear_sonic_sim环境里运行。

## 目录说明

- `starVLA/examples/SonicLatent/`: G1 VLA 训练和部署
- `wbc/gear_sonic/scripts/`: 采集、推理、仿真入口
- `wbc/gear_sonic_deploy/`: G1 部署代码
