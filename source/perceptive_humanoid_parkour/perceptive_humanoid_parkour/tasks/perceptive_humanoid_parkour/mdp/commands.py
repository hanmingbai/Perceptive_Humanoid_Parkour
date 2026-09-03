from __future__ import annotations

import inspect
import math
import numpy as np
import os
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from dataclasses import MISSING
from datetime import datetime
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, AssetBase, AssetBaseCfg # 新增 AssetBase 导入
import isaaclab.sim as sim_utils # 新增导入用于处理物体尺寸更新
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 只需要把数据集中所有尺寸的obj都列出来，放在这个字典里就行，训练时会根据动作文件里记录的尺寸来匹配使用哪个资产和尺寸
OBJ_SIZE_MAP = {
    "obj_100_40_65": (1.0, 0.40, 0.65),  # climb_09_z_65
    "obj_100_70_35": (1.0, 0.70, 0.35),  # climb_14_z_35
    "obj_100_50_55": (1.0, 0.50, 0.55),  # climb_20_z_55 (npz thickness 0.50; filename 40 is wrong)
    "obj_100_30_45": (1.0, 0.30, 0.45),  # vault_over_z_45
}

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        
        # --- 新增：从数据集中读取 Object 信息 ---
        # 假设 object 信息在数据集中是静态的（每个动作文件对应一个固定的物体位姿/尺寸）
        self.object_size = torch.tensor(data["object_size"], dtype=torch.float32, device=device)
        self.object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
        self.object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
        # ---------------------------------------

        # PHP 二阶段视觉蒸馏：从 npz 读每帧 2D runtime command。
        # ``2d_cmd_rt`` 形状 (T, 2) = [speed_rt, heading_rt_deg_s]
        # heading_rt 是滤波后 velocity_cmd 路径的偏航角速度 (°/s)，不是转向目标角。
        t_total = self.joint_pos.shape[0]
        if "2d_cmd_rt" in data:
            cmd_rt = np.asarray(data["2d_cmd_rt"], dtype=np.float32).reshape(-1, 2)
            assert cmd_rt.shape[0] == t_total, f"{motion_file}: 2d_cmd_rt length mismatch"
            self.cmd_2d_rt = torch.tensor(cmd_rt, dtype=torch.float32, device=device)
        else:
            # 旧数据没有该字段时置零，观测会退化成 [0, 0, 0]
            self.cmd_2d_rt = torch.zeros((t_total, 2), dtype=torch.float32, device=device)

        self._body_indexes = body_indexes
        self.time_step_total = t_total

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        # --- 新增：获取环境中的 Object 资产 ---
        self.asset_names = list(cfg.obj_size_map.keys())
        self.obj_assets = {name: env.scene[name] for name in self.asset_names}
        self.preset_sizes = torch.tensor(
            list(cfg.obj_size_map.values()), device=self.device
        )
        self.hide_state = torch.tensor([0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=self.device)
        # ---------------------------------------
        
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # --- 加载动作 ---
        motion_files = [cfg.motion_file] if isinstance(cfg.motion_file, str) else cfg.motion_file
        self.motion_files = list(motion_files)
        self.motion_names = [os.path.basename(f) for f in self.motion_files]
        self.motions = [MotionLoader(f, self.body_indexes, device=self.device) for f in motion_files]
        self.num_motions = len(self.motions)
        self.all_motion_lengths = torch.tensor([m.time_step_total for m in self.motions], device=self.device)

        # 打印动作信息
        print("-" * 50)
        print(f"[MotionCommand] Loaded {self.num_motions} motions.")
        for i in range(min(self.num_motions, 50)):
            print(f"  [{i:03d}] Path: {os.path.basename(motion_files[i])} | Length: {self.motions[i].time_step_total}")
        if self.num_motions > 50: print(f"  ... and {self.num_motions - 50} more.")
        print("-" * 50)

        # --- 向量化存储优化 ---
        self._setup_tensorized_storage()

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # 采样模式：uniform / sonic / beyondmimic（见 MotionCommandCfg.sampling_mode）。
        # Sonic 全局 bin 布局始终构建（指标兼容）；BeyondMimic 分层采样另建 per-motion bin。
        self.sampling_mode = self._resolve_sampling_mode()
        self._init_adaptive_sampling()
        self._init_beyondmimic_sampling()
        self._stats_env_steps = 0
        self._adaptive_stats_log_interval_steps: int | None = None
        # [SONIC exposure] 首个 _update_command 之后再开始逐步记账，避免初始 reset 污染。
        self._adaptive_stats_warm = False
        self._motion_prob_history_steps: list[int] = []
        self._motion_prob_history: list[np.ndarray] = []
        self._stats_dir = self._resolve_stats_dir()
        if self.sampling_mode == "sonic" and self.cfg.adaptive_stats_enable and self._stats_dir is not None:
            self._write_sampling_meta(self._stats_dir)

        for key in ["error_anchor_pos", "error_anchor_rot", "error_anchor_lin_vel", "error_anchor_ang_vel",
                    "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel",
                    "sampling_entropy", "sampling_top1_prob", "sampling_top1_bin",
                    "sampling_motion_top1_prob", "sampling_motion_top1_id"]:
            self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

        # 物体池信息
        self.obj_size_map = cfg.obj_size_map 
        self.obj_assets = {name: env.scene[name] for name in self.obj_size_map.keys()}
        # 将尺寸转换为 Tensor 方便在 GPU 上批量对比
        self.preset_sizes = torch.tensor(
            list(self.obj_size_map.values()), device=self.device
        ) # 形状为 (Num_Types, 3)
        self.asset_names = list(self.obj_size_map.keys())

    def _setup_tensorized_storage(self):
        """合并 Tensor，干掉 Python 循环"""
        max_len = self.all_motion_lengths.max().item()
        num_bodies = len(self.cfg.body_names)
        num_joints = self.motions[0].joint_pos.shape[-1]

        self.all_joint_pos = torch.zeros((self.num_motions, max_len, num_joints), device=self.device)
        self.all_joint_vel = torch.zeros((self.num_motions, max_len, num_joints), device=self.device)
        self.all_body_pos = torch.zeros((self.num_motions, max_len, num_bodies, 3), device=self.device)
        self.all_body_quat = torch.zeros((self.num_motions, max_len, num_bodies, 4), device=self.device)
        self.all_body_lin_vel = torch.zeros((self.num_motions, max_len, num_bodies, 3), device=self.device)
        self.all_body_ang_vel = torch.zeros((self.num_motions, max_len, num_bodies, 3), device=self.device)

        # --- 新增：物体数据存储 ---
        self.all_object_size = torch.zeros((self.num_motions, 3), device=self.device)
        self.all_object_pos = torch.zeros((self.num_motions, 3), device=self.device)
        self.all_object_quat = torch.zeros((self.num_motions, 4), device=self.device)
        # ------------------------
        # PHP：每帧 2D runtime command [speed_rt, heading_rt_deg_s]，供视觉蒸馏观测用
        self.all_cmd_2d_rt = torch.zeros((self.num_motions, max_len, 2), device=self.device)

        for i, m in enumerate(self.motions):
            l = m.time_step_total
            self.all_joint_pos[i, :l] = m.joint_pos
            self.all_joint_vel[i, :l] = m.joint_vel
            self.all_body_pos[i, :l] = m.body_pos_w
            self.all_body_quat[i, :l] = m.body_quat_w
            self.all_body_lin_vel[i, :l] = m.body_lin_vel_w
            self.all_body_ang_vel[i, :l] = m.body_ang_vel_w
            
            # --- 新增：填充物体数据 ---
            self.all_object_size[i] = m.object_size
            self.all_object_pos[i] = m.object_pos_w
            self.all_object_quat[i] = m.object_quat_w
            # ------------------------
            self.all_cmd_2d_rt[i, :l] = m.cmd_2d_rt

    # ... [property 部分保持不变] ...
    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.all_joint_pos[self.motion_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.all_joint_vel[self.motion_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.all_body_pos[self.motion_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.all_body_quat[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.all_body_lin_vel[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.all_body_ang_vel[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.all_body_pos[self.motion_ids, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.all_body_quat[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.all_body_lin_vel[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.all_body_ang_vel[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    # --- 新增：物体属性获取 ---
    @property
    def object_pos_w(self) -> torch.Tensor:
        return self.all_object_pos[self.motion_ids] + self._env.scene.env_origins

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.all_object_quat[self.motion_ids]

    @property
    def object_size(self) -> torch.Tensor:
        return self.all_object_size[self.motion_ids]

    @property
    def cmd_2d_rt(self) -> torch.Tensor:
        """当前帧数据集 ``2d_cmd_rt``：(num_envs, 2) = [speed_rt, heading_rt_deg_s]。

        从 PHP 移植：按 ``motion_ids`` / ``time_steps`` 索引，给
        ``motion_velocity_command_2d_rt`` 转成 3D 速度指令。
        """
        return self.all_cmd_2d_rt[self.motion_ids, self.time_steps]
    # ------------------------

    @property
    def robot_joint_pos(self) -> torch.Tensor: return self.robot.data.joint_pos
    @property
    def robot_joint_vel(self) -> torch.Tensor: return self.robot.data.joint_vel
    @property
    def robot_body_pos_w(self) -> torch.Tensor: return self.robot.data.body_pos_w[:, self.body_indexes]
    @property
    def robot_body_quat_w(self) -> torch.Tensor: return self.robot.data.body_quat_w[:, self.body_indexes]
    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor: return self.robot.data.body_lin_vel_w[:, self.body_indexes]
    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor: return self.robot.data.body_ang_vel_w[:, self.body_indexes]
    @property
    def robot_anchor_pos_w(self) -> torch.Tensor: return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_quat_w(self) -> torch.Tensor: return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor: return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]
    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor: return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1)
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _resolve_sampling_mode(self) -> str:
        """只认三个模式：``uniform`` / ``sonic`` / ``beyondmimic``。

        ``None`` / ``none`` / 空 / 其它未知字符串 → 均匀采样。
        """
        mode = getattr(self.cfg, "sampling_mode", None)
        if mode is None:
            return "uniform"
        mode = str(mode).strip().lower()
        if mode in ("uniform", "sonic", "beyondmimic"):
            return mode
        if mode not in ("none", "", "auto"):
            print(f"[MotionCommand] unknown sampling_mode={mode!r}, fallback to uniform")
        return "uniform"

    def _init_beyondmimic_sampling(self):
        """BeyondMimic 分层采样：每个 motion 按 ~1s 切 bin（原版 bin_count 公式）。"""
        update_dt = float(self._env.cfg.decimation * self._env.cfg.sim.dt)
        policy_hz = 1.0 / max(update_dt, 1e-8)
        # 原版：int(T // (1 / update_dt)) + 1 ≈ 每秒一个 bin。
        counts = (self.all_motion_lengths.float() / policy_hz).long() + 1
        self.bm_bin_counts = counts.clamp(min=1)
        self.bm_max_bin_count = int(self.bm_bin_counts.max().item())
        self.bm_bin_failed_count = torch.zeros(
            (self.num_motions, self.bm_max_bin_count), dtype=torch.float32, device=self.device
        )
        self.bm_current_bin_failed = torch.zeros_like(self.bm_bin_failed_count)
        arange = torch.arange(self.bm_max_bin_count, device=self.device)
        self.bm_bin_valid = arange[None, :] < self.bm_bin_counts[:, None]
        k = max(int(self.cfg.adaptive_kernel_size), 1)
        kernel = torch.tensor(
            [float(self.cfg.adaptive_lambda) ** i for i in range(k)],
            dtype=torch.float32,
            device=self.device,
        )
        self.bm_kernel = kernel / kernel.sum().clamp_min(1e-12)
        self.bm_motion_prob = self.all_motion_lengths.float()
        self.bm_motion_prob = self.bm_motion_prob / self.bm_motion_prob.sum().clamp_min(1e-12)
        if self.sampling_mode == "beyondmimic":
            print(
                f"[MotionCommand] BeyondMimic 分层采样: 先按 clip 时长抽 motion, "
                f"再按 per-motion bin 失败率抽片段 "
                f"(motions={self.num_motions}, max_bins={self.bm_max_bin_count}, "
                f"policy_hz={policy_hz:.1f})"
            )

    def _init_adaptive_sampling(self):
        """构建跨所有 motion 的全局 bin 池（SONIC / GR00T-WBC 风格）。

        每个 motion 按 ``adaptive_bin_size`` 帧切成固定时长 bin。
        采样概率由「封顶后的 bin 失败率」与均匀先验混合得到，
        从而上调难 motion/难片段，同时避免完全饿死简单样本。
        """
        bin_size = max(int(self.cfg.adaptive_bin_size), 1)
        self.adaptive_bin_size = bin_size

        bin_motion_ids = []
        bin_starts = []
        bin_ends = []
        bin_lengths = []
        num_peer_bins = []
        motion_bin_offsets = []
        motion_num_bins = []

        cur_bin = 0
        for m_id in range(self.num_motions):
            num_frames = int(self.all_motion_lengths[m_id].item())
            starts = torch.arange(0, num_frames, bin_size, device=self.device, dtype=torch.long)
            ends = torch.minimum(starts + bin_size, torch.tensor(num_frames, device=self.device))
            n_bins = int(starts.numel())
            motion_bin_offsets.append(cur_bin)
            motion_num_bins.append(n_bins)
            bin_motion_ids.append(torch.full((n_bins,), m_id, device=self.device, dtype=torch.long))
            bin_starts.append(starts)
            bin_ends.append(ends)
            bin_lengths.append(ends - starts)
            num_peer_bins.append(torch.full((n_bins,), n_bins, device=self.device, dtype=torch.long))
            cur_bin += n_bins

        self.motion_bin_offsets = torch.tensor(motion_bin_offsets, device=self.device, dtype=torch.long)
        self.motion_num_bins = torch.tensor(motion_num_bins, device=self.device, dtype=torch.long)
        self.bin_motion_ids = torch.cat(bin_motion_ids, dim=0)
        self.bin_starts = torch.cat(bin_starts, dim=0)
        self.bin_ends = torch.cat(bin_ends, dim=0)
        self.bin_lengths = torch.cat(bin_lengths, dim=0)
        self.bin_num_peer_bins = torch.cat(num_peer_bins, dim=0)
        self.num_bins = int(self.bin_motion_ids.numel())

        # bin 先验权重：先按长度归一，可选地让每个 motion 总质量相等。
        bin_weights = self.bin_lengths.float() / self.bin_lengths.float().mean().clamp_min(1e-6)
        if self.cfg.adaptive_sequence_length_agnostic:
            bin_weights = bin_weights / self.bin_num_peer_bins.float().clamp_min(1.0)
        self.bin_weights = bin_weights

        init_n = float(self.cfg.adaptive_init_num_failures)
        self.adp_num_failures = torch.full((self.num_bins,), init_n, dtype=torch.float32, device=self.device)
        self.adp_num_episodes = torch.full((self.num_bins,), init_n, dtype=torch.float32, device=self.device)
        self.adp_failure_rate = torch.ones(self.num_bins, dtype=torch.float32, device=self.device)
        self.adp_sampling_prob = torch.ones(self.num_bins, dtype=torch.float32, device=self.device) / max(
            self.num_bins, 1
        )
        self._recompute_sampling_probs()
        if self.sampling_mode == "sonic":
            print(
                f"[MotionCommand] Sonic 自适应采样: {self.num_bins} bins "
                f"(bin_size={bin_size} frames, sequence_length_agnostic={self.cfg.adaptive_sequence_length_agnostic})"
            )
            # resume 友好：可选从先前落盘的 snapshot 恢复 failures/episodes（不写进 RL checkpoint）。
            self._maybe_load_adaptive_sampling_init()
        elif self.sampling_mode == "uniform":
            print(
                f"[MotionCommand] 均匀采样: 每个 skill 等概率, clip 内均匀抽帧 "
                f"({self.num_motions} motions)"
            )

    def _resolve_adaptive_sampling_init_path(self) -> str | None:
        """解析 ``adaptive_sampling_init_path``。

        - ``None`` / 空字符串：不加载，沿用伪计数初始化。
        - 指向 ``.npz`` 文件：直接使用（通常是 ``snapshot_latest.npz``）。
        - 指向目录：优先 ``snapshot_latest.npz``，否则取该目录下最新的 ``snapshot_*.npz``。
        """
        path = self.cfg.adaptive_sampling_init_path
        if path is None:
            return None
        path = str(path).strip()
        if not path:
            return None
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            latest = os.path.join(path, "snapshot_latest.npz")
            if os.path.isfile(latest):
                return latest
            cands = sorted(
                f
                for f in os.listdir(path)
                if f.startswith("snapshot_") and f.endswith(".npz")
            )
            if cands:
                return os.path.join(path, cands[-1])
            raise FileNotFoundError(
                f"[MotionCommand] adaptive_sampling_init_path 目录中找不到 snapshot_*.npz: {path}"
            )
        raise FileNotFoundError(
            f"[MotionCommand] adaptive_sampling_init_path 不存在: {path}"
        )

    def _maybe_load_adaptive_sampling_init(self):
        """若配置了采样权重文件，用其中的 bin 计数覆盖当前自适应采样状态。

        设计说明（刻意不写进 RL checkpoint）：
        - ``adaptive_sampling_init_path is None`` → 保持 ``_init_adaptive_sampling`` 的伪计数初始化。
        - 非 None → 从可视化落盘的 snapshot 读取 ``bin_num_failures`` / ``bin_num_episodes``，
          便于 resume 时跳过“重新摸难点”的冷启动。
        - 会校验 bin 数量；若文件含布局字段（``bin_motion_ids`` / ``bin_starts`` / ``adaptive_bin_size``
          / ``motion_names``），再做一致性检查，避免 glob 顺序变化导致错位。
        """
        if self.sampling_mode != "sonic":
            return
        init_path = self._resolve_adaptive_sampling_init_path()
        if init_path is None:
            return

        data = np.load(init_path, allow_pickle=True)
        required = ("bin_num_failures", "bin_num_episodes")
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(
                f"[MotionCommand] 采样权重文件缺少字段 {missing}；需要含 bin 计数的 snapshot "
                f"(例如 snapshot_latest.npz)。文件: {init_path}"
            )

        failures = np.asarray(data["bin_num_failures"], dtype=np.float32).reshape(-1)
        episodes = np.asarray(data["bin_num_episodes"], dtype=np.float32).reshape(-1)
        if failures.shape[0] != self.num_bins or episodes.shape[0] != self.num_bins:
            raise ValueError(
                f"[MotionCommand] 采样权重 bin 数不匹配: file failures={failures.shape[0]}, "
                f"episodes={episodes.shape[0]}, current num_bins={self.num_bins}。"
                f" 请确认 motion 列表与 adaptive_bin_size 与保存时一致。文件: {init_path}"
            )

        # 可选布局校验：有则必须一致，防止 silent 错位。
        if "adaptive_bin_size" in data:
            file_bin_size = int(np.asarray(data["adaptive_bin_size"]).reshape(-1)[0])
            if file_bin_size != int(self.adaptive_bin_size):
                raise ValueError(
                    f"[MotionCommand] adaptive_bin_size 不一致: file={file_bin_size}, "
                    f"current={self.adaptive_bin_size}. 文件: {init_path}"
                )
        if "bin_motion_ids" in data:
            file_mids = np.asarray(data["bin_motion_ids"], dtype=np.int32).reshape(-1)
            cur_mids = self.bin_motion_ids.detach().cpu().numpy().astype(np.int32)
            if file_mids.shape != cur_mids.shape or not np.array_equal(file_mids, cur_mids):
                raise ValueError(
                    f"[MotionCommand] bin_motion_ids 与当前 motion 布局不一致，拒绝加载。"
                    f" 文件: {init_path}"
                )
        if "bin_starts" in data:
            file_starts = np.asarray(data["bin_starts"], dtype=np.int32).reshape(-1)
            cur_starts = self.bin_starts.detach().cpu().numpy().astype(np.int32)
            if file_starts.shape != cur_starts.shape or not np.array_equal(file_starts, cur_starts):
                raise ValueError(
                    f"[MotionCommand] bin_starts 与当前 bin 切分不一致，拒绝加载。"
                    f" 文件: {init_path}"
                )
        if "motion_names" in data:
            file_names = np.asarray(data["motion_names"]).astype(str).reshape(-1)
            cur_names = np.asarray(self.motion_names).astype(str)
            if file_names.shape != cur_names.shape or not np.array_equal(file_names, cur_names):
                raise ValueError(
                    f"[MotionCommand] motion_names 顺序/集合与当前不一致（常见于未排序 glob）。"
                    f" 拒绝加载。文件: {init_path}"
                )

        self.adp_num_failures = torch.as_tensor(failures, device=self.device, dtype=torch.float32)
        self.adp_num_episodes = torch.as_tensor(episodes, device=self.device, dtype=torch.float32)
        # 数值保护：避免 0/0；与伪计数语义一致，至少保持正的 exposure。
        self.adp_num_episodes.clamp_(min=1e-6)
        self.adp_num_failures.clamp_(min=0.0)
        self._recompute_sampling_probs()
        print(
            f"[MotionCommand] 已从采样权重文件初始化自适应采样: {os.path.abspath(init_path)} "
            f"(bins={self.num_bins}, fail_sum={float(self.adp_num_failures.sum()):.1f}, "
            f"ep_sum={float(self.adp_num_episodes.sum()):.1f})"
        )

    def _motion_time_to_bin(self, motion_ids: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        local_bin = torch.div(time_steps, self.adaptive_bin_size, rounding_mode="floor")
        local_bin = torch.minimum(local_bin, self.motion_num_bins[motion_ids] - 1)
        return self.motion_bin_offsets[motion_ids] + local_bin

    def _recompute_sampling_probs(self):
        """根据封顶后的失败率，重算全局 bin 采样概率（SONIC）。"""
        failure_rate = self.adp_num_failures / self.adp_num_episodes.clamp_min(1e-6)
        self.adp_failure_rate = failure_rate
        mean_fr = failure_rate.mean().clamp_min(1e-12)
        upper = mean_fr * float(self.cfg.adaptive_failure_rate_max_over_mean)
        capped = torch.clamp(failure_rate, min=0.0, max=upper)

        failure_based = capped / capped.sum().clamp_min(1e-12)
        uniform = torch.full_like(failure_based, 1.0 / max(self.num_bins, 1))
        alpha_u = float(self.cfg.adaptive_uniform_ratio)
        probs = failure_based * (1.0 - alpha_u) + uniform * alpha_u
        probs = probs * self.bin_weights
        probs = probs / probs.sum().clamp_min(1e-12)

        # 可选浓度约束，防止概率过度集中到个别 bin/motion（SONIC max_prob_*）。
        max_prob_bin = self.cfg.adaptive_max_prob_per_bin
        if max_prob_bin == "auto":
            max_prob_bin = float(self.cfg.adaptive_failure_rate_max_over_mean) / max(self.num_bins, 1)
        if max_prob_bin is not None and float(max_prob_bin) > 0 and self.num_bins > 1.0 / float(max_prob_bin):
            probs = torch.clamp(probs, max=float(max_prob_bin))
            probs = probs / probs.sum().clamp_min(1e-12)

        max_prob_motion = self.cfg.adaptive_max_prob_per_motion
        if max_prob_motion == "auto":
            max_prob_motion = float(self.cfg.adaptive_failure_rate_max_over_mean) / max(self.num_motions, 1)
        if (
            max_prob_motion is not None
            and float(max_prob_motion) > 0
            and self.num_motions > 1.0 / float(max_prob_motion)
        ):
            for m_id in range(self.num_motions):
                mask = self.bin_motion_ids == m_id
                motion_prob = probs[mask].sum()
                cap = float(max_prob_motion)
                if motion_prob > cap:
                    probs[mask] *= cap / motion_prob
            probs = probs / probs.sum().clamp_min(1e-12)

        self.adp_sampling_prob = probs.float()

    def _update_adaptive_sampling_exposure(self):
        """[SONIC exposure] 每步给当前 bin 记 exposure。

        对齐 Sonic：episodes += 1 / bin_length。
        注意：Isaac Lab 的 step 顺序是
        ``termination.compute → command.reset → termination.reset → command.compute``，
        因此失败计数不能放在这里（此时 terminated 往往已被清掉，且 motion_ids 已是新采样）。
        失败统计见 ``_record_terminal_stats_before_resample``。
        """
        if self.sampling_mode != "sonic" or not self._adaptive_stats_warm:
            return

        t = torch.minimum(self.time_steps, self.all_motion_lengths[self.motion_ids] - 1)
        bin_ids = self._motion_time_to_bin(self.motion_ids, t)

        # [SONIC] exposure：按 bin 长度归一，完整穿过一个 bin 大约贡献 1.0
        counts = torch.bincount(bin_ids, minlength=self.num_bins).float()
        counts = counts / self.bin_lengths.float().clamp_min(1.0)
        self.adp_num_episodes += counts

    def _record_terminal_stats_before_resample(self, env_ids: Sequence[int]):
        """在改写 motion_ids 之前，记录终止 env 的 exposure + failure。

        [Isaac Lab 时序补丁] command.reset 发生时 terminated 仍为 True，
        且尚未切换到新 motion；这是记失败的正确时机。
        自然播完触发的 resample（terminated=False）不会走失败分支。
        """
        if self.sampling_mode != "sonic" or not self._adaptive_stats_warm or len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        terminated = self._env.termination_manager.terminated[env_ids_t]
        if not torch.any(terminated):
            return

        term_env_ids = env_ids_t[terminated]
        t = torch.minimum(
            self.time_steps[term_env_ids],
            self.all_motion_lengths[self.motion_ids[term_env_ids]] - 1,
        )
        bin_ids = self._motion_time_to_bin(self.motion_ids[term_env_ids], t)

        # 失败当帧也补一笔 exposure（该帧尚未被后续 _update_command 记过）
        counts = torch.bincount(bin_ids, minlength=self.num_bins).float()
        counts = counts / self.bin_lengths.float().clamp_min(1.0)
        self.adp_num_episodes += counts

        failure_counts = torch.bincount(bin_ids, minlength=self.num_bins).float()
        self.adp_num_failures += failure_counts * float(self.cfg.adaptive_failure_counts_multiplier)

    def _uniform_sampling(self, env_ids: Sequence[int]):
        """每个 skill 等概率，再在该 clip 内均匀抽起始帧（PHP 蒸馏 / 非自适应）。"""
        n = len(env_ids)
        motion_ids = torch.randint(0, self.num_motions, (n,), device=self.device)
        lengths = self.all_motion_lengths[motion_ids]
        time_steps = (torch.rand(n, device=self.device) * lengths.float()).long()
        time_steps = torch.minimum(time_steps, (lengths - 1).clamp_min(0))
        self.motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = time_steps

        n_m = max(self.num_motions, 1)
        entropy = 1.0 if n_m > 1 else 0.0
        self.metrics["sampling_entropy"][env_ids] = entropy
        self.metrics["sampling_top1_prob"][env_ids] = 1.0 / n_m
        self.metrics["sampling_top1_bin"][env_ids] = 0.0
        self.metrics["sampling_motion_top1_prob"][env_ids] = 1.0 / n_m
        self.metrics["sampling_motion_top1_id"][env_ids] = 0.0

    def _beyondmimic_bin_probs(self) -> torch.Tensor:
        """每个 motion 内的 bin 采样概率，形状 (num_motions, max_bin_count)。

        原版：p ∝ fail_count + uniform_ratio / bin_count，再可选 conv1d 平滑。
        """
        counts = self.bm_bin_counts.float().clamp(min=1.0)
        uniform = float(self.cfg.adaptive_uniform_ratio) / counts
        probs = self.bm_bin_failed_count + uniform[:, None]
        k = int(self.bm_kernel.numel())
        if k > 1:
            x = F.pad(probs.unsqueeze(1), (0, k - 1), mode="replicate")
            probs = F.conv1d(x, self.bm_kernel.view(1, 1, -1)).squeeze(1)
        probs = probs.masked_fill(~self.bm_bin_valid, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probs

    def _record_beyondmimic_failures(self, env_ids: Sequence[int]):
        """终止 env 把失败记到当前 (motion, bin)。时序与 Sonic 补丁相同：resample 前 terminated 仍为 True。"""
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        terminated = self._env.termination_manager.terminated[env_ids_t]
        if not torch.any(terminated):
            return
        term_env_ids = env_ids_t[terminated]
        m_ids = self.motion_ids[term_env_ids]
        t_steps = self.time_steps[term_env_ids]
        m_totals = self.all_motion_lengths[m_ids].clamp(min=1)
        m_bins = self.bm_bin_counts[m_ids]
        bin_indices = torch.div(t_steps * m_bins, m_totals, rounding_mode="floor")
        bin_indices = torch.minimum(bin_indices.clamp(min=0), m_bins - 1)
        ones = torch.ones(term_env_ids.numel(), dtype=torch.float32, device=self.device)
        self.bm_current_bin_failed.index_put_((m_ids, bin_indices), ones, accumulate=True)

    def _beyondmimic_hierarchical_sampling(self, env_ids: Sequence[int]):
        """先按 clip 时长抽 motion，再在该 motion 内按 BeyondMimic bin 难度抽起始帧。"""
        self._record_beyondmimic_failures(env_ids)
        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        motion_ids = torch.multinomial(self.bm_motion_prob, n, replacement=True)
        bin_probs = self._beyondmimic_bin_probs()
        row_probs = bin_probs[motion_ids]
        sampled_bins = torch.multinomial(row_probs, 1).squeeze(-1)
        m_totals = self.all_motion_lengths[motion_ids]
        m_bins = self.bm_bin_counts[motion_ids].float().clamp(min=1.0)
        u = sample_uniform(0.0, 1.0, (n,), device=self.device)
        time_steps = ((sampled_bins.float() + u) / m_bins * (m_totals - 1).float()).long()
        time_steps = torch.minimum(time_steps.clamp(min=0), (m_totals - 1).clamp_min(0))
        self.motion_ids[env_ids_t] = motion_ids
        self.time_steps[env_ids_t] = time_steps

        entropy = -(row_probs * (row_probs + 1e-12).log()).sum(dim=-1)
        entropy = entropy / torch.log(m_bins.clamp(min=2.0))
        pmax, imax = row_probs.max(dim=-1)
        self.metrics["sampling_entropy"][env_ids_t] = entropy
        self.metrics["sampling_top1_prob"][env_ids_t] = pmax
        self.metrics["sampling_top1_bin"][env_ids_t] = imax.float() / m_bins
        mp_max, mp_imax = self.bm_motion_prob.max(dim=0)
        self.metrics["sampling_motion_top1_prob"][env_ids_t] = mp_max
        self.metrics["sampling_motion_top1_id"][env_ids_t] = mp_imax.float()

    def _sonic_adaptive_sampling(self, env_ids: Sequence[int]):
        """从全局失败率加权的 bin 池中，联合采样 (motion, frame)。"""
        # 先按旧 (motion, t) 记账，再重算概率并采样新片段。
        self._record_terminal_stats_before_resample(env_ids)
        self._recompute_sampling_probs()

        n = len(env_ids)
        sampled_bin_ids = torch.multinomial(self.adp_sampling_prob, n, replacement=True)
        motion_ids = self.bin_motion_ids[sampled_bin_ids]
        bin_start = self.bin_starts[sampled_bin_ids]
        bin_end = self.bin_ends[sampled_bin_ids]
        time_steps = (
            torch.rand(n, device=self.device) * (bin_end - bin_start).float()
        ).floor().long() + bin_start

        # 在难点前稍早起步（SONIC pre_failure_sample_window）。
        # 再按 motion 长度封顶，避免短 clip 被整体推到 t=0。
        pre_window = int(self.cfg.adaptive_pre_failure_sample_window)
        if pre_window > 0:
            max_offset = torch.minimum(
                torch.full((n,), pre_window, device=self.device, dtype=torch.long),
                (self.all_motion_lengths[motion_ids] // 4).clamp_min(0),
            )
            # high 必须 > 0；offset ∈ [0, max_offset]。
            high = (max_offset + 1).clamp_min(1)
            offset = (torch.rand(n, device=self.device) * high.float()).long()
            time_steps = (time_steps - offset).clamp_min(0)

        # 保证时间步落在所选 motion 合法范围内。
        max_t = self.all_motion_lengths[motion_ids] - 1
        self.motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = torch.minimum(time_steps, max_t)

        # 采样监控指标
        probs = self.adp_sampling_prob
        entropy = -(probs * (probs + 1e-12).log()).sum() / math.log(max(self.num_bins, 2))
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_entropy"][env_ids] = entropy
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / max(self.num_bins, 1)

        motion_probs = torch.zeros(self.num_motions, device=self.device)
        motion_probs.scatter_add_(0, self.bin_motion_ids, probs)
        mp_max, mp_imax = motion_probs.max(dim=0)
        self.metrics["sampling_motion_top1_prob"][env_ids] = mp_max
        self.metrics["sampling_motion_top1_id"][env_ids] = mp_imax.float()

    def get_motion_sampling_prob(self) -> torch.Tensor:
        """将 bin 概率聚合为每个 motion 的采样质量，形状 ``(num_motions,)``。"""
        motion_probs = torch.zeros(self.num_motions, device=self.device, dtype=self.adp_sampling_prob.dtype)
        motion_probs.scatter_add_(0, self.bin_motion_ids, self.adp_sampling_prob)
        return motion_probs

    def get_sampling_snapshot(self, include_bins: bool = True) -> dict:
        """导出 CPU 侧快照，供日志/可视化（轻量 numpy 字典）。"""
        self._recompute_sampling_probs()
        motion_prob = self.get_motion_sampling_prob().detach().float().cpu().numpy()
        snap = {
            "step": np.int64(self._stats_env_steps),
            "motion_prob": motion_prob.astype(np.float16),
            "motion_names": np.array(self.motion_names),
            "motion_lengths": self.all_motion_lengths.detach().cpu().numpy().astype(np.int32),
        }
        if include_bins:
            snap.update(
                {
                    "bin_prob": self.adp_sampling_prob.detach().float().cpu().numpy().astype(np.float16),
                    "bin_failure_rate": self.adp_failure_rate.detach().float().cpu().numpy().astype(np.float16),
                    "bin_num_failures": self.adp_num_failures.detach().float().cpu().numpy().astype(np.float32),
                    "bin_num_episodes": self.adp_num_episodes.detach().float().cpu().numpy().astype(np.float32),
                    "bin_motion_ids": self.bin_motion_ids.detach().cpu().numpy().astype(np.int32),
                    "bin_starts": self.bin_starts.detach().cpu().numpy().astype(np.int32),
                    "bin_ends": self.bin_ends.detach().cpu().numpy().astype(np.int32),
                    # 供 resume 加载时校验切分是否一致。
                    "adaptive_bin_size": np.int32(self.adaptive_bin_size),
                }
            )
        return snap

    def _resolve_stats_dir(self) -> str | None:
        if self.sampling_mode != "sonic" or not self.cfg.adaptive_stats_enable:
            return None
        if self.cfg.adaptive_stats_save_dir:
            out = self.cfg.adaptive_stats_save_dir
        else:
            # 不放进 rsl_rl 的 log_dir，单独开目录，避免和 checkpoint 混在一起。
            # 默认：logs/adaptive_sampling/<时间戳>/
            root = self.cfg.adaptive_stats_root_dir
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out = os.path.join(root, stamp)
        os.makedirs(out, exist_ok=True)
        print(f"[MotionCommand] 自适应采样统计目录: {os.path.abspath(out)}")
        return out

    def _write_sampling_meta(self, out_dir: str):
        """只写一次：离线画图所需的静态 bin/motion 布局。"""
        path = os.path.join(out_dir, "meta.npz")
        np.savez_compressed(
            path,
            motion_names=np.array(self.motion_names),
            motion_files=np.array(self.motion_files),
            motion_lengths=self.all_motion_lengths.detach().cpu().numpy().astype(np.int32),
            bin_motion_ids=self.bin_motion_ids.detach().cpu().numpy().astype(np.int32),
            bin_starts=self.bin_starts.detach().cpu().numpy().astype(np.int32),
            bin_ends=self.bin_ends.detach().cpu().numpy().astype(np.int32),
            bin_lengths=self.bin_lengths.detach().cpu().numpy().astype(np.int32),
            bin_weights=self.bin_weights.detach().float().cpu().numpy().astype(np.float16),
            adaptive_bin_size=np.int32(self.adaptive_bin_size),
            adaptive_uniform_ratio=np.float32(self.cfg.adaptive_uniform_ratio),
            adaptive_failure_rate_max_over_mean=np.float32(self.cfg.adaptive_failure_rate_max_over_mean),
        )

    def save_sampling_stats(self, out_dir: str | None = None, tag: str = "latest", include_bins: bool = True):
        """把 motion 概率历史，以及可选的 bin 快照，刷写到 ``out_dir``。"""
        out_dir = out_dir or self._stats_dir or self._resolve_stats_dir()
        if out_dir is None:
            return None
        try:
            os.makedirs(out_dir, exist_ok=True)
            if not os.path.isfile(os.path.join(out_dir, "meta.npz")):
                self._write_sampling_meta(out_dir)

            # 紧凑的 motion 概率历史（float16）；长时间训练通常仍远小于 1MB。
            if self._motion_prob_history:
                hist_path = os.path.join(out_dir, "history_motion.npz")
                np.savez_compressed(
                    hist_path,
                    steps=np.asarray(self._motion_prob_history_steps, dtype=np.int64),
                    motion_prob=np.stack(self._motion_prob_history, axis=0),  # (T, M) float16
                    motion_names=np.array(self.motion_names),
                )

            # bin 快照更大：仅在请求时，或非 latest 的显式 dump 时覆盖写入。
            if include_bins or tag != "latest":
                snap = self.get_sampling_snapshot(include_bins=True)
                latest_path = os.path.join(out_dir, f"snapshot_{tag}.npz")
                np.savez_compressed(latest_path, **snap)
                if tag != "latest":
                    np.savez_compressed(os.path.join(out_dir, "snapshot_latest.npz"), **snap)
        except Exception as e:
            # 日志失败绝不能打断 GPU 训练。
            print(f"[MotionCommand] adaptive sampling save failed (ignored): {e}")
            return None
        return out_dir

    def _resolve_num_steps_per_env(self) -> int:
        """读当前训练的 ``num_steps_per_env``（一次 rollout 的 policy step 数）。

        优先从调用栈上的 rsl-rl runner 取实际值（含 hydra/CLI 覆盖）；
        找不到再按 gym 注册表里的 ``rsl_rl_cfg_entry_point`` 读默认配置。
        """
        frame = inspect.currentframe()
        try:
            while frame is not None:
                obj = frame.f_locals.get("self")
                if obj is not None and obj is not self:
                    cfg = getattr(obj, "cfg", None)
                    if isinstance(cfg, dict) and cfg.get("num_steps_per_env") is not None:
                        return max(int(cfg["num_steps_per_env"]), 1)
                    n = getattr(cfg, "num_steps_per_env", None) if cfg is not None else None
                    if n is None:
                        n = getattr(obj, "num_steps_per_env", None)
                    if n is not None:
                        return max(int(n), 1)
                frame = frame.f_back
        finally:
            del frame

        n = self._num_steps_per_env_from_gym_registry()
        if n is not None:
            return max(int(n), 1)
        print("[MotionCommand] 未找到 num_steps_per_env，adaptive stats interval 回退为 24")
        return 24

    def _num_steps_per_env_from_gym_registry(self) -> int | None:
        try:
            import gymnasium as gym
            from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        except Exception:
            return None

        env_cfg_cls = type(self._env.cfg)
        env_id = None
        for spec_id, spec in gym.envs.registry.items():
            kwargs = getattr(spec, "kwargs", None) or {}
            if kwargs.get("rsl_rl_cfg_entry_point") is None:
                continue
            entry = kwargs.get("env_cfg_entry_point")
            if entry is env_cfg_cls:
                env_id = spec_id
                break
            if isinstance(entry, str) and entry.split(":")[-1].split(".")[-1] == env_cfg_cls.__name__:
                env_id = spec_id
                break
        if env_id is None:
            return None
        try:
            agent_cfg = load_cfg_from_registry(env_id, "rsl_rl_cfg_entry_point")
            n = getattr(agent_cfg, "num_steps_per_env", None)
            return int(n) if n is not None else None
        except Exception:
            return None

    def _resolve_stats_log_interval(self) -> int:
        """``interval_env_steps = log_interval_iters * num_steps_per_env``。"""
        if self._adaptive_stats_log_interval_steps is not None:
            return self._adaptive_stats_log_interval_steps
        iters = max(int(self.cfg.adaptive_stats_log_interval_iters), 1)
        steps_per_env = self._resolve_num_steps_per_env()
        interval = iters * steps_per_env
        self._adaptive_stats_log_interval_steps = interval
        print(
            f"[MotionCommand] adaptive stats: every {iters} iters "
            f"(num_steps_per_env={steps_per_env} → {interval} env-steps)"
        )
        return interval

    def maybe_log_sampling_stats(self):
        """在 env step 循环中做周期性轻量落盘。"""
        if self.sampling_mode != "sonic" or not self.cfg.adaptive_stats_enable:
            return
        interval = self._resolve_stats_log_interval()
        if self._stats_env_steps == 0 or (self._stats_env_steps % interval) != 0:
            return

        try:
            motion_prob = self.get_motion_sampling_prob().detach().float().cpu().numpy().astype(np.float16)
            self._motion_prob_history_steps.append(int(self._stats_env_steps))
            self._motion_prob_history.append(motion_prob)

            # 限制内存中历史长度，避免长训膨胀。
            max_hist = max(int(self.cfg.adaptive_stats_history_maxlen), 1)
            if len(self._motion_prob_history) > max_hist:
                drop = len(self._motion_prob_history) - max_hist
                del self._motion_prob_history[:drop]
                del self._motion_prob_history_steps[:drop]

            # motion 历史较勤落盘；完整 bin 快照降频保存。
            bin_every = max(int(self.cfg.adaptive_stats_bin_save_every), 1)
            save_bins = (len(self._motion_prob_history) % bin_every) == 0
            self.save_sampling_stats(tag="latest", include_bins=save_bins)
        except Exception as e:
            print(f"[MotionCommand] adaptive sampling log failed (ignored): {e}")

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0: return
        if self.sampling_mode == "sonic":
            self._sonic_adaptive_sampling(env_ids)
        elif self.sampling_mode == "beyondmimic":
            self._beyondmimic_hierarchical_sampling(env_ids)
        else:
            self._uniform_sampling(env_ids)
        
        # --- 核心新增：更新物体位姿和尺寸 ---
        # 1. 计算当前 batch 要求的尺寸与预设尺寸的匹配关系 (N, M)
        target_sizes = self.object_size[env_ids]
        dists = torch.cdist(target_sizes, self.preset_sizes)
        best_match_indices = torch.argmin(dists, dim=-1)

        # 2. 遍历物体池
        for i, name in enumerate(self.asset_names):
            asset = self.obj_assets[name]
            
            # 找到哪些环境需要这个特定的尺寸
            match_mask = (best_match_indices == i)
            
            # 默认全员“隐藏”
            states_to_write = self.hide_state.repeat(len(env_ids), 1)
            
            if torch.any(match_mask):
                # 1. 获取数量
                num_matches = match_mask.sum()
                
                # 2. 定义角度限制 (15度)
                deg_limit = 5.0
                rad_limit = deg_limit * (np.pi / 180.0)
                
                # 3. 生成 [-rad_limit, +rad_limit] 之间的随机 Yaw 偏移
                # torch.rand 是 [0, 1) -> (2*rand - 1) 是 [-1, 1]
                random_yaw = (torch.rand(int(num_matches.item()), device=self.device) * 2.0 - 1.0) * rad_limit
                
                # 4. 构造增量四元数 (仅绕 Z 轴旋转)
                zeros = torch.zeros_like(random_yaw)
                quat_delta = quat_from_euler_xyz(zeros, zeros, random_yaw)
                
                # 5. 获取原始数据并叠加旋转
                # 假设 self.object_quat_w 的形状是 [num_envs, 4]
                original_quat = self.object_quat_w[env_ids][match_mask]
                new_quat = quat_mul(original_quat, quat_delta)
                
                # 6. 应用到写入状态中
                states_to_write[match_mask, 3:7] = new_quat
                states_to_write[match_mask, :3] = self.object_pos_w[env_ids][match_mask]

                # states_to_write[match_mask, 0] = 4.0 # test
                # states_to_write[match_mask, 1] = 0.0 # test
                # states_to_write[match_mask, 2] = 0.2 # test
            
            # 批量写入物理引擎
            asset.write_root_state_to_sim(states_to_write, env_ids=env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]], device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        root_ori[env_ids] = quat_mul(quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]), root_ori[env_ids])
        
        v_ranges = torch.tensor([self.cfg.velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]], device=self.device)
        v_samples = sample_uniform(v_ranges[:, 0], v_ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += v_samples[:, :3]
        root_ang_vel[env_ids] += v_samples[:, 3:]

        j_pos = self.joint_pos.clone()
        j_pos += sample_uniform(*self.cfg.joint_position_range, j_pos.shape, self.device)
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        j_pos[env_ids] = torch.clip(j_pos[env_ids], limits[:, :, 0], limits[:, :, 1])
        
        self.robot.write_joint_state_to_sim(j_pos[env_ids], self.joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1), env_ids=env_ids)

    def _update_command(self):
        # [SONIC exposure] 先按当前帧记 exposure/failure，再推进时间（与 Sonic 顺序一致）
        self._update_adaptive_sampling_exposure()
        self.time_steps += 1
        self._stats_env_steps += 1
        self._adaptive_stats_warm = True
        done_mask = self.time_steps >= self.all_motion_lengths[self.motion_ids]
        done_env_ids = torch.where(done_mask)[0]
        if len(done_env_ids) > 0: self._resample_command(done_env_ids)

        r_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        r_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        m_anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        m_anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = r_anchor_pos.clone()
        delta_pos_w[..., 2] = m_anchor_pos[..., 2]
        delta_ori_w = yaw_quat(quat_mul(r_anchor_quat, quat_inv(m_anchor_quat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - m_anchor_pos)

        if self.sampling_mode == "beyondmimic":
            alpha = float(self.cfg.adaptive_alpha)
            self.bm_bin_failed_count = (
                alpha * self.bm_current_bin_failed + (1.0 - alpha) * self.bm_bin_failed_count
            )
            self.bm_current_bin_failed.zero_()

        self.maybe_log_sampling_stats()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor"))
                self.goal_anchor_visualizer = VisualizationMarkers(self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor"))
                self.current_body_visualizers = [VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + n)) for n in self.cfg.body_names]
                self.goal_body_visualizers = [VisualizationMarkers(self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + n)) for n in self.cfg.body_names]
            self.current_anchor_visualizer.set_visibility(True); self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)): self.current_body_visualizers[i].set_visibility(True); self.goal_body_visualizers[i].set_visibility(True)
        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False); self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)): self.current_body_visualizers[i].set_visibility(False); self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized: return
        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])

@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = MISSING

    # 数据集配置
    obj_size_map: dict = OBJ_SIZE_MAP # 场景中物体的资产名称和尺寸
    motion_file: str | list[str] = MISSING # 动作文件路径
    anchor_body_name: str = MISSING # 锚点身体名称
    body_names: list[str] = MISSING # 身体名称列表

    # reset 配置
    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    # 动作采样配置
    sampling_mode: str | None = None # 采样模式："uniform"：每个 skill 等概率，clip 内均匀抽帧 / "sonic"：全局 bin 失败率加权（一阶段 / GR00T-WBC） / "beyondmimic"：先按 clip 时长抽 motion，再按原版 BeyondMimic bin 失败率抽片段
    adaptive_bin_size: int = 50 # Sonic：每个 bin 的固定帧长（50Hz 下约 1 秒）
    adaptive_uniform_ratio: float = 0.1 # 与均匀先验的混合权重：p = (1-u)*失败率项 + u*均匀项。
    adaptive_failure_rate_max_over_mean: float = 50.0 # 将每个 bin 的失败率封顶为 beta * mean(failure_rate)。
    adaptive_sequence_length_agnostic: bool = True # True 时每个 motion 先验总质量相等（不再按长度成比例）。
    adaptive_init_num_failures: float = 1.0 # 伪计数，使尚未访问的 bin 初始 failure_rate=1。
    adaptive_pre_failure_sample_window: int = 200  # Sonic 默认；实际还会再 cap 到 motion_length/4，在难点 bin 前再往前采样若干帧，便于策略提前准备，采样时还会再限制为该 motion 长度的 1/4（适配短 clip）
    adaptive_failure_counts_multiplier: float = 1.0
    adaptive_max_prob_per_bin: float | str | None = None # 可选浓度上限：None | "auto" | (0,1] 的浮点数。
    adaptive_max_prob_per_motion: float | str | None = None
    adaptive_sampling_init_path: str | None = None # "logs/adaptive_sampling/2026-08-14_02-15-35/snapshot_latest.npz" 从磁盘恢复采样统计（不写进 RL checkpoint）指向 snapshot npz 文件，或含 snapshot_latest.npz 的 adaptive_sampling 用其中的 bin_num_failures / bin_num_episodes 覆盖初始化，便于接着难 bin 继续训
    
    # 采样统计配置
    adaptive_stats_enable: bool = True # 轻量采样统计（用于可视化）
    adaptive_stats_root_dir: str = "logs/adaptive_sampling" # 统计根目录（不跟 rsl_rl log 混放）。实际写入 root/<时间戳>/
    adaptive_stats_save_dir: str | None = None # 若指定则直接用该目录，不再自动加时间戳子目录。
    adaptive_stats_log_interval_iters: int = 500 # 每隔多少次 RL iteration 记录一次 motion 概率历史，实际 env-step 间隔 = 该值 × runner.num_steps_per_env（训练时自动读取，含 hydra 覆盖）
    adaptive_stats_history_maxlen: int = 200 # 内存/历史文件中最多保留最近 K 条 motion 概率快照。
    adaptive_stats_bin_save_every: int = 5 # 每 N 次 history 采样才保存一次完整 bin 快照（motion 历史仍按 interval）。

    # BeyondMimic 原版 bin 配置
    bin_count: int = 100 # BeyondMimic 原版 bin， 当前无作用
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_alpha: float = 0.001
    
    # 可视化配置
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_anchor")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2) 
    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose_body")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)