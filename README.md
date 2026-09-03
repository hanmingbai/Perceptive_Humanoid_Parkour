# Perceptive Humanoid Parkour

基于 **Isaac Sim 5.1** + **Isaac Lab 2.3.2** 的感知人形跑酷训练与部署仓库（非最终版）。

支持 **Isaac Lab 训练 / 回放**，以及 **MuJoCo sim2sim** 在本地验证视觉蒸馏策略。

---

## 环境安装

### 1. 创建 Python 环境

Isaac Sim 5.1 需要 **Python 3.11**。

```bash
conda create -n php python=3.11
conda activate php
pip install --upgrade pip
```

### 2. 安装 Isaac Sim 5.1 与 PyTorch

参考官方教程：

[Installation using Isaac Sim Pip Package (Isaac Lab v2.3.2)](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/pip_installation.html)

可用官方示例验证：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

### 3. 安装本仓库

```bash
cd /path/to/perceptive_humanoid_parkour
pip install -e source/perceptive_humanoid_parkour
pip install -e rsl_rl
pip install termcolor
```

### 4. Sim2Sim 额外依赖（可选）

仅在运行 MuJoCo sim2sim 时需要：

```bash
pip install mujoco glfw opencv-python matplotlib
```

---

## 库冲突修复

若启动 Isaac Sim / 训练时出现大量 `omni.graph.core.tests` / `omni.kit.test` 报错，以及：

```text
libstdc++.so.6: version `CXXABI_1.3.15' not found
```

这是 conda 与系统 `libstdc++` 冲突。每次激活环境后先执行：

```bash
conda activate php
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

## 数据与 Checkpoint

| 资源 | 路径 | 说明 |
|------|------|------|
| 动作数据 | `motions/mm_php_post_keep_0812/` | 训练默认加载目录 |
| Motion tracking ckpt | `checkpoints/g1_flat_motion_tracking/model_51000.pt` | 视觉蒸馏 teacher |
| Vision distillation ckpt | `checkpoints/g1_flat_vision_distillation/model_99999.pt` | sim2sim 默认策略 |

---

## Isaac Lab 训练

```bash
# Motion tracking
python scripts/rsl_rl/train.py --task PHP-MotionTracking-Flat-G1-v0 --num_envs=4096 --headless

# Vision distillation（需相机）
python scripts/rsl_rl/train.py --task PHP-VisionDistillation-Flat-G1-v0 --num_envs=1024 --enable_cameras --headless
```

### 回放（Play）

```bash
# Motion tracking
python scripts/rsl_rl/play.py --task PHP-MotionTracking-Flat-G1-v0 --num_envs=16

# Vision distillation（需相机）
python scripts/rsl_rl/play.py --task PHP-VisionDistillation-Flat-G1-v0 --num_envs=16 --enable_cameras
```

通过 `--checkpoint` 指定其他权重，例如：

```bash
python scripts/rsl_rl/play.py \
  --task PHP-VisionDistillation-Flat-G1-v0 \
  --num_envs=16 \
  --enable_cameras \
  --checkpoint logs/rsl_rl/.../model_10000.pt
```

---

## MuJoCo Sim2Sim

`scripts/sim2sim/php_sim2sim.py` 在 MuJoCo 中加载视觉蒸馏策略，观测与 `PHP-VisionDistillation-Flat-G1-v0` 的 policy 输入对齐（含 58×87 深度图）。

### 快速运行

```bash
python scripts/sim2sim/php_sim2sim.py
```

默认使用：

- 策略：`checkpoints/g1_flat_vision_distillation/model_99999.pt`
- 场景：`source/.../assets/unitree_description/mjcf/parkour.xml`

### 指定 Checkpoint

```bash
python scripts/sim2sim/php_sim2sim.py \
  --policy checkpoints/g1_flat_vision_distillation/model_99999.pt
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--policy` | rsl_rl `model_*.pt` 路径 |
| `--xml` | MuJoCo 场景 XML |
| `--device` | `cuda` 或 `cpu`（默认 `cuda`） |
| `--headless` | 无窗口运行 |
| `--record_video` | 录制视频 |
| `--no_depth_window` | 关闭深度图调试窗口 |
| `--num_obstacles` | 障碍物数量（最多 12） |
| `--seed` | 随机赛道种子 |

### 键盘控制

按住生效，松开归零（无 latch）：

| 按键 | 效果 |
|------|------|
| ↑ | 前进（speed=1） |
| ← | 左转（heading=+1） |
| → | 右转（heading=-1） |
| Backspace | 重置机器人姿态 |

策略速度指令为 **0 或 1**，转向为 **-1 / 0 / +1**。

---

## 注册任务一览

| Task ID | 说明 |
|---------|------|
| `PHP-MotionTracking-Flat-G1-v0` | 平地 motion tracking |
| `PHP-VisionDistillation-Flat-G1-v0` | 视觉蒸馏（主任务） |
| `PHP-VisionDistillation-Flat-G1-Wo-State-Estimation-v0` | 无状态估计版本 |
| `PHP-VisionDistillation-Flat-G1-Low-Freq-v0` | 低频控制版本 |
