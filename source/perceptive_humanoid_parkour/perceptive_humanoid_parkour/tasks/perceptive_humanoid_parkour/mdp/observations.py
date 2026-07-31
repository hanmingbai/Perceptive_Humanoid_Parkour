from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply_inverse, yaw_quat

from perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    ) # 获得姿态的error
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    ) # 获得位置的error

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

# 世界系中的速度，需要改成局部坐标系下的速度，以便学生网络学习到与位置和姿态无关的运动模式
def motion_velocity_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:

    command: MotionCommand = env.command_manager.get_term(command_name)

    linvel = command.anchor_lin_vel_w # 世界系下的线速度
    angvel = command.anchor_ang_vel_w # 世界系下的角速度
    anchor_quat = command.anchor_quat_w
    # 将线速度和角速度从世界系转换到局部坐标系下
    # linvel_local = quat_apply_inverse(anchor_quat, linvel)
    # angvel_local = quat_apply_inverse(anchor_quat, angvel)
    # # 只考虑xy的投影，更适合z轴姿态变化较大的运动
    quat_yaw = yaw_quat(anchor_quat)
    linvel_local = quat_apply_inverse(quat_yaw, linvel)
    angvel_local = quat_apply_inverse(quat_yaw, angvel)

    # linvel = torch.zeros_like(command.anchor_lin_vel_w)
    # angvel = torch.zeros_like(command.anchor_ang_vel_w)
    # linvel[:, 0] = 0.6
    # angvel[:, 2] = 0.5
    # linvel_local[:, 0] = 1.0

    # print(f"command.anchor_lin_vel_w:{command.anchor_lin_vel_w.shape}")
    return torch.cat([linvel_local[:, :2], angvel_local[:, 2:3]], dim=-1)

def motion_velocity_command_bool(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    linvel = command.anchor_lin_vel_w 
    anchor_quat = command.anchor_quat_w
    quat_yaw = yaw_quat(anchor_quat)
    linvel_local = quat_apply_inverse(quat_yaw, linvel)
    # 定义阈值，比如 0.1 m/s (根据你的需求调整)
    threshold = 0.1 
    # 逻辑判断：如果 x 方向速度大于阈值，则为 1，否则为 0，我们创建一个与输入同形状的张量来存储指令
    cmd_x = (linvel_local[:, 0:1] > threshold).float()
    # 构造 [cmd_x, 0, 0] 的指令序列，cmd_x 是 [batch_size, 1]，使用 torch.zeros_like 填充另外两个维度
    zeros = torch.zeros_like(cmd_x)

    return torch.cat([cmd_x, zeros, zeros], dim=-1)

# for perceptive distillation
def get_depth_data(env, sensor_name: str):
    """从指定的 TiledCamera 获取归一化的拉平深度数据"""
    depth = env.scene[sensor_name].data.output["distance_to_image_plane"] # 拿到的是 [num_envs, 58, 87] 的深度 Tensor
    depth = torch.nan_to_num(depth, posinf=5.0) # 预处理：NaN/Inf -> 2.0m, 并归一化到 [0, 1]
    depth = torch.clamp(depth / 5.0, 0.0, 1.0)
    return depth.view(env.num_envs, -1) # 拉平以便 ObservationManager 拼接成 1D Vector (5046维)

def get_depth_obs(env, sensor_name: str):
    depth = env.scene[sensor_name].data.output["distance_to_image_plane"]
    mask = torch.rand_like(depth) > 0.05 # 1. 模拟噪声：随机将 5% 的像素设为 0 (缺失)
    depth *= mask
    depth = torch.round(depth * 100) / 100.0 # 2. 模拟精度：保留两位小数
    return depth

def get_clipped_depth_obs(env, sensor_name: str):
    depth_tensor = env.scene[sensor_name].data.output["distance_to_image_plane"]
    depth = torch.nan_to_num(depth_tensor, posinf=2.0, neginf=0.15) # 极大极小值修正
    depth = torch.where(depth < 0.15, torch.tensor(2.0, device=depth.device), depth)
    depth = torch.clamp(depth, min=0.0, max=2.0) # 裁减视野范围
    depth_normalized = (depth - 0.15) / (2.0 - 0.15) # 归一化处理，显示depth时不处理
    # print(f"depth_size:{depth_normalized.size()}") # torch.Size([num_envs, 58, 87, 1])
    # print(f"depth_view_size:{depth_normalized.view(env.num_envs, -1).size()}") # torch.Size([num_envs, 5046]) 5046-> CNN-GAP -> 32
    return depth_normalized.view(env.num_envs, -1)