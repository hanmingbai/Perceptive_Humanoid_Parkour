"""MuJoCo sim2sim for BeyondMimicPlusVisionDistill (PHP-style 2D command).

Policy observation matches ``ObservationsCfg.PolicyCfg``:
  [base_ang_vel * 0.2 (3), projected_gravity (3),
   motion_velocity_command_2d_rt (3), joint_pos_rel (29),
   joint_vel_rel (29), last_action (29), clipped depth 58x87 (5046)]

Dataset ``2d_cmd_rt[:,0]`` (cmd_speed) is only **0 or 1**. Heading is deg/s,
policy sees ``sign(heading)`` ∈ {-1, 0, +1}. Obs command is therefore
``[0|1, 0, -1|0|+1]``.

Keyboard — hold to set, release to zero (no latch). Avoids MuJoCo WASD /
Space / R / Q bindings (wireframe, shadows, pause, reset, …):
    ↑  hold walk   → [1, 0, turn]
    ←  hold left   → [speed, 0, +1]
    →  hold right  → [speed, 0, -1]
    release        → [0, 0, 0]
    Backspace      reset robot pose

Joint order: policy obs/action are Isaac Lab PhysX order; remapped to MuJoCo/URDF
via ``ISAAC2MUJOCO`` / ``MUJOCO2ISAAC`` (same as ``npz_to_mm_loco`` / deploy
``joint_ids_map``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import glfw
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as mjv
import numpy as np
import torch
from tensordict import TensorDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RSL_RL_ROOT = _REPO_ROOT / "rsl_rl"
if str(_RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_ROOT))

from rsl_rl.models.vision_mlp_model import VisionMLPModel

# ---------------------------------------------------------------------------
# G1 29-DoF constants (same formulas as legged_lab/robots/g1.py)
# ---------------------------------------------------------------------------
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
NUM_ACTIONS = len(JOINT_NAMES)

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# default pose from G1_CYLINDER_CFG.init_state
DEFAULT_DOF_POS = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float32,
)
STIFFNESS = np.array(
    [
        STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
        STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
        STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_4010, STIFFNESS_4010,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_4010, STIFFNESS_4010,
    ],
    dtype=np.float32,
)
DAMPING = np.array(
    [
        DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22,
        2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
        DAMPING_7520_14, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22,
        2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
        DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
        DAMPING_4010, DAMPING_4010,
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
        DAMPING_4010, DAMPING_4010,
    ],
    dtype=np.float32,
)
TORQUE_LIMITS = np.array(
    [
        88, 139, 88, 139, 50, 50,
        88, 139, 88, 139, 50, 50,
        88, 50, 50,
        25, 25, 25, 25, 25, 5, 5,
        25, 25, 25, 25, 25, 5, 5,
    ],
    dtype=np.float32,
)
# G1_ACTION_SCALE = 0.25 * effort / stiffness (MuJoCo / URDF order)
ACTION_SCALE = (0.25 * TORQUE_LIMITS / STIFFNESS).astype(np.float32)

# Isaac Lab PhysX dof order ≠ URDF/MuJoCo. Same maps as npz_to_mm_loco / deploy joint_ids_map.
# isaac[i] = mujoco[ISAAC2MUJOCO[i]] ; mujoco[j] = isaac[MUJOCO2ISAAC[j]]
ISAAC2MUJOCO = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
MUJOCO2ISAAC = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)
DEFAULT_DOF_POS_ISAAC = DEFAULT_DOF_POS[ISAAC2MUJOCO]
ACTION_SCALE_ISAAC = ACTION_SCALE[ISAAC2MUJOCO]


def mujoco_to_isaac(x: np.ndarray) -> np.ndarray:
    return x[ISAAC2MUJOCO]


def isaac_to_mujoco(x: np.ndarray) -> np.ndarray:
    return x[MUJOCO2ISAAC]


NUM_PROPRIO = 96
DEPTH_H, DEPTH_W = 58, 87
DEPTH_DIM = DEPTH_H * DEPTH_W  # 5046
OBS_DIM = NUM_PROPRIO + DEPTH_DIM
DEPTH_NEAR, DEPTH_FAR = 0.15, 2.0
ANG_VEL_SCALE = 0.2

DEFAULT_POLICY = _REPO_ROOT / "checkpoints" / "g1_flat_vision_distillation" / "model_99999.pt"
DEFAULT_XML = (
    _REPO_ROOT
    / "source/perceptive_humanoid_parkour/perceptive_humanoid_parkour/assets/unitree_description/mjcf/parkour.xml"
)

# Isaac OBJ_SIZE_MAP copies in parkour.xml. z = height/2. Geom already yaw-90
# (1 m face toward +X). Unused copies stay buried at z=-10.
OBJ_POOL = [
    {"name": f"obj_100_70_35_{i}", "z": 0.175} for i in range(3)
] + [
    {"name": f"obj_100_30_45_{i}", "z": 0.225} for i in range(3)
] + [
    {"name": f"obj_100_50_55_{i}", "z": 0.275} for i in range(3)
] + [
    {"name": f"obj_100_40_65_{i}", "z": 0.325} for i in range(3)
]
DEFAULT_NUM_OBSTACLES = 12
DEFAULT_GAP = (6.0, 10.0)
DEFAULT_Y_RANGE = (-0.5, 0.5)
DEFAULT_YAW_DEG = 5.0  # training resample uses ±5°


def quat_apply_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of unit quaternion ``q_wxyz`` (Isaac convention)."""
    w = float(q_wxyz[0])
    xyz = q_wxyz[1:4]
    t = 2.0 * np.cross(xyz, v)
    return v - w * t + np.cross(xyz, t)


def clip_depth_meters(depth_raw: np.ndarray) -> np.ndarray:
    """Same clip as ``play_camera.py`` / ``get_clipped_depth_obs`` (meters)."""
    depth = np.nan_to_num(depth_raw, nan=2.0, posinf=2.0, neginf=0.15)
    depth = np.where(depth < 0.15, 2.0, depth)
    return np.clip(depth, 0.0, 2.0).astype(np.float32)


def process_depth(depth_raw: np.ndarray) -> np.ndarray:
    """Normalize clipped meters for the policy CNN."""
    depth = clip_depth_meters(depth_raw)
    return ((depth - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)).astype(np.float32)


def velocity_command_2d_rt(speed: float, heading_deg_s: float) -> np.ndarray:
    """Same mapping as ``motion_velocity_command_2d_rt``: [0|1, 0, sign(heading)]."""
    return np.array([1.0 if speed > 0.5 else 0.0, 0.0, float(np.sign(heading_deg_s))], dtype=np.float32)


def load_vision_policy(policy_path: str, device: str) -> torch.nn.Module:
    """Load VisionMLPModel from an rsl_rl ``model_*.pt`` checkpoint."""
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise ValueError(
            f"{policy_path} is not an rsl_rl checkpoint (missing actor_state_dict). "
            "Export JIT is not used: VisionMLP as_jit() drops the CNN."
        )
    dummy = TensorDict({"policy": torch.zeros(1, OBS_DIM)}, batch_size=[1])
    model = VisionMLPModel(
        obs=dummy,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.01,
            "std_type": "scalar",
        },
        num_proprio=NUM_PROPRIO,
        visual_output_dim=32,
    )
    missing, unexpected = model.load_state_dict(ckpt["actor_state_dict"], strict=True)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    print(f"Loaded VisionMLPModel from {policy_path} (iter={ckpt.get('iter')})")
    return model


class PhpSim2SimController:
    def __init__(
        self,
        policy_path: str,
        xml_path: str = str(DEFAULT_XML),
        device: str = "cuda",
        record_video: bool = False,
        headless: bool = False,
        show_depth: bool = True,
        speed: float = 1.0,
        turn: float = 0.0,
        obstacle_x: float | None = None,
        duration: float = 1000.0,
        num_obstacles: int = DEFAULT_NUM_OBSTACLES,
        gap_min: float = DEFAULT_GAP[0],
        gap_max: float = DEFAULT_GAP[1],
        seed: int | None = None,
    ):
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.policy = load_vision_policy(policy_path, self.device)
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        # Isaac trains at 0.005 with implicit PD. MuJoCo uses explicit torque PD,
        # which is unstable at 0.005 with G1 kp≈99; 0.001 x 20 keeps the 50 Hz policy.
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        self._check_joint_order()

        self.renderer = mujoco.Renderer(self.model, height=DEPTH_H, width=DEPTH_W)
        self.video_renderer = None
        self.record_video = record_video
        self.show_depth = show_depth and not headless
        self.headless = headless
        self.sim_duration = duration
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)  # Isaac order
        self.pd_target = DEFAULT_DOF_POS.copy()  # MuJoCo order
        self._request_reset = False
        self._glfw_window = None

        # Hold-to-command. Interactive starts at zero until a key is held.
        # Headless keeps the CLI values for the whole run.
        self.speed_rt = 1.0 if float(speed) > 0.5 else 0.0
        self.heading_rt_deg_s = float(turn)
        self.velocity_command = velocity_command_2d_rt(self.speed_rt, self.heading_rt_deg_s)

        self._obj_pool = list(OBJ_POOL)
        self._num_obstacles = int(np.clip(num_obstacles, 1, len(self._obj_pool)))
        self._gap_min = float(min(gap_min, gap_max))
        self._gap_max = float(max(gap_min, gap_max))
        self._first_x = 2.15 if obstacle_x is None else float(obstacle_x)
        self._rng = np.random.default_rng(seed)

        self.viewer = None
        if not headless:
            self.viewer = mjv.launch_passive(self.model, self.data, key_callback=self._on_viewer_key)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
            self.viewer.cam.distance = 4.0

        # Depth window: same as scripts/rsl_rl/play_camera.py (gray, meters, 58x87).
        self._depth_fig = None
        self._depth_im = None
        if self.show_depth:
            plt.ion()
            self._depth_fig, ax = plt.subplots(figsize=(6, 4))
            self._depth_im = ax.imshow(np.zeros((DEPTH_H, DEPTH_W)), cmap="gray", vmin=0.1, vmax=2.0)
            ax.set_title("Robot Depth View (87x58 x5)")
            ax.axis("off")
            plt.show(block=False)

    def _check_joint_order(self) -> None:
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.model.nu)]
        if names != JOINT_NAMES:
            raise RuntimeError(f"MuJoCo actuator order != Isaac G1 order.\n  xml={names}\n  expected={JOINT_NAMES}")

    def extract_data(self):
        dof_pos = self.data.qpos.astype(np.float32)[-NUM_ACTIONS:]
        dof_vel = self.data.qvel.astype(np.float32)[-NUM_ACTIONS:]
        quat = self.data.sensor("orientation").data.astype(np.float32)
        ang_vel = self.data.sensor("angular-velocity").data.astype(np.float32)
        return dof_pos, dof_vel, quat, ang_vel

    def render_depth(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="d435_camera")
        self.renderer.enable_depth_rendering()
        depth_raw = self.renderer.render()
        self.renderer.disable_depth_rendering()
        depth_clipped = clip_depth_meters(depth_raw)
        if self._depth_im is not None:
            self._depth_im.set_data(depth_clipped)
            self._depth_fig.canvas.draw()
            self._depth_fig.canvas.flush_events()
        return ((depth_clipped - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)).astype(np.float32)

    def build_obs(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        quat: np.ndarray,
        ang_vel: np.ndarray,
        depth_norm: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        gravity_b = quat_apply_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        joint_pos_rel = mujoco_to_isaac(dof_pos) - DEFAULT_DOF_POS_ISAAC
        joint_vel_rel = mujoco_to_isaac(dof_vel)
        proprio = np.concatenate(
            [
                ang_vel * ANG_VEL_SCALE,
                gravity_b.astype(np.float32),
                command.astype(np.float32),
                joint_pos_rel,
                joint_vel_rel,
                self.last_action,
            ]
        )
        if proprio.shape[0] != NUM_PROPRIO:
            raise RuntimeError(f"proprio dim {proprio.shape[0]} != {NUM_PROPRIO}")
        return np.concatenate([proprio, depth_norm.reshape(-1).astype(np.float32)])

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        td = TensorDict({"policy": obs_t}, batch_size=[1])
        with torch.no_grad():
            action = self.policy(td, stochastic_output=False)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _place_obstacles(self) -> None:
        """Random course: shuffled training sizes, large gaps, small y/yaw jitter."""
        for spec in self._obj_pool:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec["name"])
            if bid < 0:
                continue
            self.model.body_pos[bid] = np.array([0.0, 0.0, -10.0])
            self.model.body_quat[bid] = np.array([1.0, 0.0, 0.0, 0.0])

        order = self._rng.permutation(len(self._obj_pool))[: self._num_obstacles]
        x = self._first_x
        print(f"Obstacles n={self._num_obstacles} first_x={self._first_x:.2f} "
              f"gap=[{self._gap_min:.1f}, {self._gap_max:.1f}]")
        for k, idx in enumerate(order):
            spec = self._obj_pool[idx]
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec["name"])
            if bid < 0:
                continue
            if k > 0:
                x += float(self._rng.uniform(self._gap_min, self._gap_max))
            y = float(self._rng.uniform(*DEFAULT_Y_RANGE))
            yaw = float(self._rng.uniform(-DEFAULT_YAW_DEG, DEFAULT_YAW_DEG)) * (np.pi / 180.0)
            half = 0.5 * yaw
            self.model.body_pos[bid] = np.array([x, y, spec["z"]])
            self.model.body_quat[bid] = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
            print(f"  {spec['name']}: x={x:.2f} y={y:+.2f} yaw={np.degrees(yaw):+.1f}°")

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Motion clips start pelvis at z≈0.776 (Isaac default spawn is 0.76).
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.776])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[-NUM_ACTIONS:] = DEFAULT_DOF_POS
        self.data.qvel[:] = 0.0
        self.last_action[:] = 0.0
        self.pd_target = DEFAULT_DOF_POS.copy()
        self._place_obstacles()
        mujoco.mj_forward(self.model, self.data)

    def _format_cmd(self, command: np.ndarray) -> str:
        return f"cmd [{command[0]:.0f}, {command[1]:.0f}, {command[2]:+.0f}]"

    def _print_cmd(self, command: np.ndarray) -> None:
        key = (float(command[0]), float(command[1]), float(command[2]))
        if key == getattr(self, "_last_printed_cmd", None):
            return
        self._last_printed_cmd = key
        print(self._format_cmd(command), flush=True)

    def _poll_held_keys(self) -> None:
        """Read currently held arrows. Viewer callback has no press/release bit."""
        window = self._glfw_window
        if window is None:
            ctx = glfw.get_current_context()
            if ctx is not None:
                self._glfw_window = ctx
                window = ctx
        if window is None:
            return
        try:
            up = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
            left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
            right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
        except Exception:
            return
        self.speed_rt = 1.0 if up else 0.0
        if left and not right:
            self.heading_rt_deg_s = 1.0
        elif right and not left:
            self.heading_rt_deg_s = -1.0
        else:
            self.heading_rt_deg_s = 0.0
        self.velocity_command = velocity_command_2d_rt(self.speed_rt, self.heading_rt_deg_s)

    def _on_viewer_key(self, keycode: int) -> None:
        """Viewer thread: cache the GLFW window and sample hold state.
        Arrow / Backspace only: WASD/Space/R are MuJoCo viewer shortcuts.
        """
        ctx = glfw.get_current_context()
        if ctx is not None:
            self._glfw_window = ctx
        if keycode == glfw.KEY_BACKSPACE:
            window = self._glfw_window
            if window is not None and glfw.get_key(window, glfw.KEY_BACKSPACE) == glfw.PRESS:
                self._request_reset = True
            return
        self._poll_held_keys()

    def run(self) -> None:
        mp4_writer = None
        if self.record_video:
            import imageio

            video_name = "php_sim2sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
            self.video_renderer = mujoco.Renderer(self.model, height=720, width=1280)

        self.reset()
        steps = int(self.sim_duration / self.sim_dt)
        print(
            "Focus the MuJoCo window (not the depth figure). "
            "Hold ↑ walk, hold ←/→ turn; release → cmd [0, 0, 0]. Backspace reset."
        )
        command = np.zeros(3, dtype=np.float32)
        policy_dt = self.sim_dt * self.sim_decimation
        next_tick = time.perf_counter()

        for i in range(steps):
            if self._request_reset:
                self.reset()
                self._request_reset = False
            if self.viewer is not None and not self.viewer.is_running():
                break
            dof_pos, dof_vel, quat, ang_vel = self.extract_data()

            if i % self.sim_decimation == 0:
                if self.viewer is not None:
                    self._poll_held_keys()
                depth_norm = self.render_depth()
                command = self.velocity_command.copy()
                self._print_cmd(command)

                obs = self.build_obs(dof_pos, dof_vel, quat, ang_vel, depth_norm, command)
                raw_action = np.clip(self.infer(obs), -10.0, 10.0)  # Isaac order
                self.last_action = raw_action
                self.pd_target = isaac_to_mujoco(raw_action * ACTION_SCALE_ISAAC + DEFAULT_DOF_POS_ISAAC)

                if self.viewer is not None:
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()

                if mp4_writer is not None:
                    self.video_renderer.update_scene(self.data, camera=self.viewer.cam if self.viewer else None)
                    rgb_img = self.video_renderer.render()
                    depth_8u = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)
                    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
                    depth_rgb = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
                    main_h, main_w, _ = rgb_img.shape
                    inset_w = int(main_w * 0.3)
                    inset_h = int(inset_w * (depth_rgb.shape[0] / depth_rgb.shape[1]))
                    inset = cv2.resize(depth_rgb, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
                    rgb_img[0:inset_h, -inset_w:] = inset
                    text = f"cmd [{command[0]:.2f}, {command[1]:.1f}, {command[2]:+.0f}]"
                    rgb_img = cv2.putText(
                        rgb_img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    mp4_writer.append_data(rgb_img)

            torque = (self.pd_target - dof_pos) * STIFFNESS - dof_vel * DAMPING
            self.data.ctrl[:] = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
            mujoco.mj_step(self.model, self.data)

            # Pace to 50 Hz policy ticks only. Sleeping every 1 ms physics step
            # plus a heavy policy/depth/cv2 hit every 20 steps is what felt like a freeze.
            if i % self.sim_decimation == self.sim_decimation - 1:
                next_tick += policy_dt
                remain = next_tick - time.perf_counter()
                if remain > 0.0:
                    time.sleep(remain)
                else:
                    next_tick = time.perf_counter()

        print("Simulation finished")
        if mp4_writer is not None:
            mp4_writer.close()
            print("Video saved")
        if self.viewer is not None:
            self.viewer.close()
        if self._depth_fig is not None:
            plt.close(self._depth_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHP vision-distill MuJoCo sim2sim")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="rsl_rl model_*.pt with actor_state_dict")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="MuJoCo XML (29-DoF G1 + d435)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="initial cmd_speed: 0 or 1 (dataset is binary)",
    )
    parser.add_argument(
        "--turn",
        type=float,
        default=0.0,
        help="initial heading (policy uses sign only: +1/0/-1)",
    )
    parser.add_argument(
        "--obstacle_x",
        type=float,
        default=None,
        help="X of the first box (default 2.15 m, inside the 2 m depth clip)",
    )
    parser.add_argument("--num_obstacles", type=int, default=DEFAULT_NUM_OBSTACLES, help="how many boxes to place (max 12)")
    parser.add_argument("--gap_min", type=float, default=DEFAULT_GAP[0], help="min center-to-center gap (m)")
    parser.add_argument("--gap_max", type=float, default=DEFAULT_GAP[1], help="max center-to-center gap (m)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for the random course")
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_depth_window", action="store_true")
    args = parser.parse_args()

    controller = PhpSim2SimController(
        policy_path=args.policy,
        xml_path=args.xml,
        device=args.device,
        record_video=args.record_video,
        headless=args.headless,
        show_depth=not args.no_depth_window,
        speed=args.speed,
        turn=args.turn,
        obstacle_x=args.obstacle_x,
        duration=args.duration,
        num_obstacles=args.num_obstacles,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        seed=args.seed,
    )
    controller.run()


if __name__ == "__main__":
    main()
