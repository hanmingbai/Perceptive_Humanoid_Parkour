# Perceptive Humanoid Parkour

基于 **Isaac Sim 5.1** + **Isaac Lab 2.3.2** 的感知人形跑酷训练仓库（非最终版）

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

### 4. 安装本仓库

```bash
cd /path/to/perceptive_humanoid_parkour
pip install -e source/perceptive_humanoid_parkour
pip install -e rsl_rl
pip install termcolor
```

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

## 常用任务

```bash
# motion tracking
python scripts/rsl_rl/train.py --task PHP-MotionTracking-Flat-G1-v0 --num_envs=4096 --headless

# vision distillation（需相机）
python scripts/rsl_rl/train.py --task PHP-VisionDistillation-Flat-G1-v0 --num_envs=1024 --enable_cameras --headless
```

动作数据默认从仓库内 `motions/mm_parkour/` 加载。
