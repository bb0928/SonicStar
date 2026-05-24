# whole body vla baseline 5.24

本文档总结当前本地/远端 Whole-body VLA baseline 的模块关系、启动脚本、模型结构、Sonic 控制接口和数据格式。

## 1. 模块概述和启动脚本

### 总体链路

当前 baseline 是一个四进程部署链路：

1. 远端 StarVLA policy server
   - 加载 StarVLA checkpoint。
   - 对外提供 WebSocket policy inference 服务。
   - 远端端口：`10093`。

2. 本地 SSH 隧道
   - 把本地 `127.0.0.1:10092` 转发到远端 `127.0.0.1:10093`。
   - 本地 VLA client 只连 `127.0.0.1:10092`。

3. 本地 MuJoCo Sonic simulator
   - 跑 G1 仿真、相机渲染和 Unitree lowstate/lowcmd。
   - 发布相机图像到 `5555`。
   - 通过 DDS 与 C++ Sonic 控制器交换低层状态/命令。

4. 本地 C++ Sonic deploy
   - 加载 TensorRT policy encoder/decoder 和 planner。
   - 从 VLA client 的 ZMQ `5556` 收 `command/planner/pose`。
   - 向 VLA client 的 ZMQ `5557` 发 `g1_debug/robot_config`。
   - 输出低层控制到 MuJoCo。

5. 本地 StarVLA inference client
   - 从 MuJoCo 相机 `5555` 取 `ego_view`。
   - 从 C++ deploy `5557` 取机器人状态。
   - 调远端 policy server 做 VLA 推理。
   - 把 `motion_token + hands` 打包成 `pose` 发到 C++ deploy 的 `5556`。

### 当前推荐一键启动

本地已经有 tmux 脚本：

```bash
/home/bob/SonicStar/start_vla_tmux.sh
```

它会创建 `sonic_vla` tmux session，并启动四个本地 pane：

```bash
bash /home/bob/SonicStar/start_vla_tmux.sh
```

### 远端 policy server

远端目录：

```bash
/cpfs/user/xingliangjun/SonicStar/starVLA
```

远端启动：

```bash
cd /cpfs/user/xingliangjun/SonicStar/starVLA
bash examples/SonicLatent/eval_files/run_policy_server.sh
```

远端脚本默认加载：

```bash
playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/steps_90000_pytorch_model.pt
```

远端监听：

```text
0.0.0.0:10093
```

### 本地端口约定

| 端口 | 方向 | 角色 |
|---:|---|---|
| `10092` | local -> remote | 本地 WebSocket 隧道，转发到远端 policy server `10093` |
| `5555` | MuJoCo -> VLA client | 相机图像服务，主要是 `ego_view` |
| `5556` | VLA client -> C++ deploy | `command/planner/pose` ZMQ 控制输入 |
| `5557` | C++ deploy -> VLA client/exporter | `g1_debug/robot_config` 状态输出 |
| `5580` | old keyboard helper | 旧键盘触发端口；当前 VLA client 已改成自动 start，不再依赖 |

## 2. 每一个启动脚本的含义和角色

### `/home/bob/SonicStar/start_vla_tmux.sh`

本地总启动脚本。它会先清理旧进程，再启动 tmux 四分屏。

Pane 0: SSH 隧道

```bash
ssh -p 983 -N -L 10092:127.0.0.1:10093 root@123.57.187.96
```

作用：把本地 `10092` 映射到远端 policy server 的 `10093`。

Pane 1: MuJoCo Sonic sim

```bash
cd /home/bob/GR00T-WholeBodyControl
PYTHONPATH=$PWD /home/bob/anaconda3/envs/sonic-mcp/bin/python \
  gear_sonic/scripts/run_sim_loop.py \
  --enable-image-publish \
  --enable-offscreen \
  --camera-port 5555
```

作用：

- 启动 G1 MuJoCo 仿真。
- 启动相机服务 `5555`。
- 通过 Unitree DDS 与 C++ deploy 通讯。

Pane 2: C++ Sonic deploy

```bash
cd /home/bob/GR00T-WholeBodyControl/gear_sonic_deploy
printf '\n' | CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 \
  bash deploy.sh --input-type zmq_manager sim
```

作用：

- 加载 Sonic C++ 控制器。
- 加载 TensorRT 模型：
  - `policy/release/model_encoder.onnx`
  - `policy/release/model_decoder.onnx`
  - 对应 `.trt` engine
  - `planner/target_vel/V2/planner_sonic.onnx`
- 使用 `zmq_manager` 输入模式。
- 仿真模式 `sim` 下使用 `lo` interface，并关闭 CRC。
- 在 `5556` 接收 VLA 控制，在 `5557` 发布状态。

Pane 3: StarVLA inference client

```bash
cd /home/bob/SonicStar/starVLA
PYTHONPATH=$PWD:/home/bob/GR00T-WholeBodyControl \
  /home/bob/anaconda3/envs/sonic-mcp/bin/python \
  examples/SonicLatent/eval_files/run_starvla_inference.py \
  --ckpt-path /home/bob/SonicStar/starVLA/playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/steps_90000_pytorch_model.pt \
  --host 127.0.0.1 \
  --port 10092 \
  --camera-host localhost \
  --camera-port 5555 \
  --state-zmq-host localhost \
  --state-zmq-port 5557 \
  --action-zmq-host localhost \
  --action-zmq-port 5556 \
  --prompt "pick up the cylinder and throw it into the trash bin" \
  --rate 1.0
```

作用：

- 读 `ego_view` 图像。
- 读 C++ 发出的 `g1_debug` 状态。
- 调远端 policy server。
- 把 VLA 输出转换成 Sonic C++ 能吃的 latent pose message。

### `/home/bob/SonicStar/starVLA/examples/SonicLatent/eval_files/run_policy_server.sh`

远端 policy server 启动脚本。

默认关键变量：

```bash
STARVLA_PYTHON=/home/user/miniconda3/envs/starVLA/bin/python
CKPT_PATH=playground/Checkpoints/sonic_latent_scratch_frozen_vlm/checkpoints/steps_90000_pytorch_model.pt
GPU_ID=0
PORT=10093
```

实际执行：

```bash
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" \
  deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16
```

### `/home/bob/GR00T-WholeBodyControl/gear_sonic_deploy/deploy.sh`

C++ Sonic deploy 启动器。

当前使用方式：

```bash
CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 \
  bash deploy.sh --input-type zmq_manager sim
```

关键点：

- `--input-type zmq_manager`：控制输入来自 ZMQ，而不是本地 motion file 或键盘 planner。
- `sim`：仿真部署，interface 解析为 `lo`，CRC disabled。
- 依赖 TensorRT。ONNX 文件是模型表达，TensorRT engine 是 C++ 实时推理实际加载的后端。

### `/home/bob/GR00T-WholeBodyControl/gear_sonic/scripts/run_sim_loop.py`

MuJoCo Sonic 仿真入口。

当前使用：

```bash
PYTHONPATH=$PWD python gear_sonic/scripts/run_sim_loop.py \
  --enable-image-publish \
  --enable-offscreen \
  --camera-port 5555
```

关键点：

- `--enable-image-publish`：打开相机图像服务。
- `--enable-offscreen`：允许 offscreen 渲染，给 VLA client 提供图像。
- `--camera-port 5555`：VLA client 从这里拿图。

当前场景配置：

```text
/home/bob/GR00T-WholeBodyControl/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml
```

其中 `ROBOT_SCENE` 指向：

```text
gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml
```

`ENABLE_ELASTIC_BAND` 当前设为 `False`，否则机器人会像被虚拟弹簧吊住。

## 3. backbone 和 DiT action expert 的结构

### StarVLA framework

训练配置：

```text
/home/bob/SonicStar/starVLA/examples/SonicLatent/train_files/train_sonic_latent.yaml
```

实际 checkpoint 配置：

```text
/home/bob/SonicStar/starVLA/playground/Checkpoints/sonic_latent_scratch_frozen_vlm/config.yaml
```

框架名：

```yaml
framework:
  name: QwenGR00T
```

### Backbone

Backbone 是 Qwen-VL 系列 VLM：

```yaml
qwenvl:
  base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
  attn_implementation: flash_attention_2
  vl_hidden_dim: 2048
```

训练时冻结 Qwen/VLM 分支：

```yaml
freeze_modules: qwen_vl_interface
learning_rate:
  qwen_vl_interface: 0.0
```

所以这个 baseline 的主要训练对象不是视觉语言 backbone，而是后面的 action expert。

### 输入

训练配置注释里写明：

```text
Inputs: ego_view + observation.state(43) + projected_gravity(3) + language
```

也就是：

- 第一视角图像：`observation.images.ego_view`
- 本体状态：`observation.state`，43 维
- 重力投影：`observation.projected_gravity`，3 维
- 语言 prompt

### 输出

训练配置注释里写明：

```text
Outputs: motion_token(64) + left_hand_joints(7) + right_hand_joints(7)
```

总维度：

```text
64 + 7 + 7 = 78
```

对应：

```yaml
action_dim: 78
```

### DiT action expert

Action expert 是 DiT-B：

```yaml
action_model:
  action_model_type: DiT-B
  hidden_size: 1024
  action_dim: 78
  state_dim: 46
  action_horizon: 40
  future_action_window_size: 39
  past_action_window_size: 0
  repeated_diffusion_steps: 2
  num_timestep_buckets: 1000
  num_inference_timesteps: 4
  num_target_vision_tokens: 32
```

DiT block 配置：

```yaml
diffusion_model_cfg:
  dropout: 0.2
  final_dropout: true
  interleave_self_attention: true
  norm_type: ada_norm
  num_layers: 16
  output_dim: 1024
```

注意：训练 yaml 里 `cross_attention_dim` 写的是 `2048`，但当前部署 checkpoint 的 config 里是：

```yaml
cross_attention_dim: 2560
```

写文档和排错时应以 checkpoint config 为准。

### 训练要点

当前 run：

```yaml
run_id: sonic_latent_scratch_frozen_vlm
max_train_steps: 90000
save_interval: 30000
pretrained_checkpoint: null
```

含义：

- 从 Qwen-VL base 初始化视觉语言分支。
- 不加载已有 GR00T action head checkpoint。
- VLM 冻结。
- 主要训练随机初始化的 DiT action head。

这也是它和“已经有人形本体预训练”的策略不同的地方：这个 baseline 更像是用冻结 VLM 提供视觉语言条件，再让 DiT 学 Sonic latent action 分布。

## 4. Sonic 的控制格式以及 C++ 接口

### C++ 入口

C++ 输入接口核心文件：

```text
/home/bob/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/input_interface/zmq_manager.hpp
```

`ZMQManager` 是当前部署用的 input interface。

### ZMQManager 订阅格式

`ZMQManager` 在同一个 host/port 上订阅三个 topic：

```text
tcp://localhost:5556
```

| topic | 作用 | 模式 |
|---|---|---|
| `command` | start/stop/mode switch | 全局控制 |
| `planner` | 速度、朝向、高度等 planner 命令 | PLANNER |
| `pose` | VLA/Sonic latent pose stream | STREAMED_MOTION |

### command topic

command message 语义：

```text
{ start: bool, stop: bool, planner: bool, delta_heading?: f32 }
```

其中：

- `planner=true`：切到 PLANNER mode。
- `planner=false`：切到 STREAMED_MOTION mode，也就是吃 `pose` topic。
- `start=true`：启动控制。
- `stop=true`：停止控制。

当前 `run_starvla_inference.py` 已经改成自动发启动命令：

1. 先 `start=True, planner=True`
2. 发布 initial pose
3. 再 `start=True, planner=False`

这样不再需要额外起 5580 键盘 publisher。

### planner topic

planner topic 用于普通 locomotion 命令，包括：

- mode
- movement direction
- facing direction
- speed
- height
- 可选 upper-body / hand / VR 三点信息

如果 1 秒没有收到 planner message，C++ 会自动把 locomotion reset 到 IDLE，并清掉上肢/手部控制 flag。

### pose topic

VLA client 发给 C++ 的核心是 `pose` topic。

当前 StarVLA client 里打包函数：

```text
/home/bob/SonicStar/starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py
```

相关函数：

- `pack_latent_action_message`
- `send_cpp_control_command`
- `publish_initial_pose`

VLA action 会被拆成：

```python
motion_token = actions[:, :64]
left_hand_joints = actions[:, 64:71]
right_hand_joints = actions[:, 71:78]
```

然后打包成 pose protocol v4：

```python
pose_data = {
    "token_state": motion_token,
    "frame_index": frame_index,
    "left_hand_joints": left_hand_joints,
    "right_hand_joints": right_hand_joints,
}
```

再通过 `pack_pose_message(..., topic="pose", version=4)` 发到 `5556`。

### C++ deploy 的状态输出

C++ deploy 会在 `5557` 发布：

- `g1_debug`：机器人状态、动作、base quat 等。
- `robot_config`：机器人配置，data exporter 和 VLA client 可读取。

VLA client 使用：

```python
ZMQStateSubscriber(host="localhost", port=5557)
```

### C++ policy 观测维度

C++ deploy 日志里当前 policy observation：

```text
token_state: 64
his_base_angular_velocity_10frame_step1: 30
his_body_joint_positions_10frame_step1: 290
his_body_joint_velocities_10frame_step1: 290
his_last_actions_10frame_step1: 290
his_gravity_dir_10frame_step1: 30
```

总维度：

```text
994
```

Encoder observation 总维度：

```text
1762
```

其中包括 motion joints、root height、anchor orientation、lower-body、VR 3-point、SMPL joints、wrist joints 等。

## 5. 录制数据的格式与条目，路径位置

### 当前已有数据集

路径：

```text
/home/bob/SonicStar/starVLA/playground/Datasets/sonic_merged_dataset_001
```

元信息：

```text
/home/bob/SonicStar/starVLA/playground/Datasets/sonic_merged_dataset_001/meta/info.json
```

当前统计：

```text
total_episodes: 303
total_frames: 234973
total_tasks: 1
total_videos: 303
fps: 50
```

任务文件：

```text
/home/bob/SonicStar/starVLA/playground/Datasets/sonic_merged_dataset_001/meta/tasks.jsonl
```

当前 task：

```text
pick up the cylinder and throw it into the trash  bin
```

注意原始 task 文本里 `trash  bin` 中间有两个空格。

### episode / video 路径模板

Parquet 数据：

```text
data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
```

视频数据：

```text
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

例如第 0 个 episode 通常对应：

```text
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

### 数据字段

当前 dataset feature schema：

| 字段 | dtype | shape | 含义 |
|---|---|---:|---|
| `observation.images.ego_view` | video | `[480, 640, 3]` | 第一视角图像 |
| `observation.state` | float64 | `[43]` | G1 关节状态 |
| `observation.eef_state` | float64 | `[14]` | 左右 wrist pos/quaternion |
| `action.wbc` | float64 | `[43]` | WBC 关节动作 |
| `observation.root_orientation` | float64 | `[4]` | root orientation quat |
| `observation.projected_gravity` | float64 | `[3]` | 重力方向投影 |
| `observation.cpp_rotation_offset` | float64 | `[4]` | C++ rotation offset |
| `observation.init_base_quat` | float64 | `[4]` | episode 初始 base quat |
| `teleop.delta_heading` | float64 | `[1]` | heading 增量 |
| `action.motion_token` | float64 | `[64]` | Sonic latent motion token |
| `teleop.smpl_joints` | float32 | `[72]` | SMPL joints 展平 |
| `teleop.smpl_pose` | float32 | `[63]` | SMPL pose |
| `teleop.body_quat_w` | float32 | `[4]` | body quaternion |
| `teleop.target_body_orientation` | float32 | `[6]` | 目标身体朝向 6D |
| `teleop.left_hand_joints` | float32 | `[7]` | 左手关节 |
| `teleop.right_hand_joints` | float32 | `[7]` | 右手关节 |
| `teleop.smpl_frame_index` | int64 | `[1]` | SMPL frame index |
| `teleop.left_wrist_joints` | float32 | `[3]` | 左 wrist joints |
| `teleop.right_wrist_joints` | float32 | `[3]` | 右 wrist joints |
| `teleop.stream_mode` | int32 | `[1]` | stream mode |
| `teleop.planner_mode` | int32 | `[1]` | planner mode |
| `teleop.planner_movement` | float32 | `[3]` | planner movement |
| `teleop.planner_facing` | float32 | `[3]` | planner facing |
| `teleop.planner_speed` | float32 | `[1]` | planner speed |
| `teleop.planner_height` | float32 | `[1]` | planner height |
| `teleop.vr_3pt_position` | float32 | `[9]` | VR 三点位置 |
| `teleop.vr_3pt_orientation` | float32 | `[18]` | VR 三点朝向 |
| `timestamp` | float32 | `[1]` | 时间戳 |
| `frame_index` | int64 | `[1]` | episode 内帧号 |
| `episode_index` | int64 | `[1]` | episode id |
| `index` | int64 | `[1]` | 全局 index |
| `task_index` | int64 | `[1]` | task id |

### 43 维关节顺序

`observation.state` 和 `action.wbc` 使用同一套 43 维关节顺序：

```text
left_hip_pitch_joint
left_hip_roll_joint
left_hip_yaw_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
waist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_joint
left_wrist_roll_joint
left_wrist_pitch_joint
left_wrist_yaw_joint
left_hand_index_0_joint
left_hand_index_1_joint
left_hand_middle_0_joint
left_hand_middle_1_joint
left_hand_thumb_0_joint
left_hand_thumb_1_joint
left_hand_thumb_2_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_joint
right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint
right_hand_index_0_joint
right_hand_index_1_joint
right_hand_middle_0_joint
right_hand_middle_1_joint
right_hand_thumb_0_joint
right_hand_thumb_1_joint
right_hand_thumb_2_joint
```

`observation.eef_state` 是：

```text
left_wrist_pos
left_wrist_abs_quat
right_wrist_pos
right_wrist_abs_quat
```

### 录制脚本

数据录制入口：

```text
/home/bob/GR00T-WholeBodyControl/gear_sonic/scripts/run_data_exporter.py
```

示例：

```bash
cd /home/bob/GR00T-WholeBodyControl
PYTHONPATH=$PWD python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "pick up the cylinder and throw it into the trash bin" \
  --dataset-name my_session
```

默认输出根目录：

```text
/home/bob/GR00T-WholeBodyControl/outputs
```

录制脚本的数据源：

| 来源 | 端口/topic | 内容 |
|---|---|---|
| C++ deploy state | `5557/g1_debug` | proprio、base quat、last action 等 |
| C++ deploy config | `5557/robot_config` | robot config，约每 2 秒重发 |
| Sonic/VLA pose | `5556/pose` | SMPL / latent pose / hand joints |
| MuJoCo camera | `5555` | `ego_view` 图像 |

### 当前本地重要改动

1. `run_starvla_inference.py` 已改成自动启动 C++ 控制并切到 `STREAMED_MOTION`，不再依赖手动键盘 `5580`。

2. `scene_43dof.xml` 是当前任务场景，包含桌子、圆柱物体和由 box 组成的 trash bin。

3. `g1_29dof_sonic_model12.yaml` 中 `ENABLE_ELASTIC_BAND=False`，避免机器人被虚拟弹簧悬挂。

4. `base_sim.py` 已给 `elastic_band` 加默认 `None`，避免关闭 elastic band 后异常：

```text
'DefaultEnv' object has no attribute 'elastic_band'
```

5. C++ TensorRT loader 已改为在 hash 不一致时复用现有 `.trt`，避免本地 CUDA/TensorRT 环境触发 rebuild。

