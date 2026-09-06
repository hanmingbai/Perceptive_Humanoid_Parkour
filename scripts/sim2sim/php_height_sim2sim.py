"""MuJoCo sim2sim for height-map DAgger (PHP-style 2D command).

Policy observation matches ``HeightMapPolicyCfg``:
  [base_ang_vel * 0.2 (3), projected_gravity (3),
   motion_velocity_command_2d_rt (3), joint_pos_rel (29),
   joint_vel_rel (29), last_action (29), height_scan 17x11 (187)]

Height scan matches Isaac ``height_scan`` + ``height_scanner``:
  GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), ray_alignment=yaw,
  rays from torso + (0, 0, 20) downward; value = torso_z - hit_z - 0.5, clip [-1, 1].
  Only floor / obj geoms (MuJoCo group 0) are hit — same as Isaac ground + obj meshes.

Keyboard is the same as ``php_sim2sim.py``.
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

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_RSL_RL_ROOT = _REPO_ROOT / "rsl_rl"
if str(_RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rsl_rl.models.mlp_model import MLPModel

from php_depth_sim2sim import (  # noqa: E402
    ACTION_SCALE_ISAAC,
    ANG_VEL_SCALE,
    DAMPING,
    DEFAULT_DOF_POS,
    DEFAULT_DOF_POS_ISAAC,
    DEFAULT_GAP,
    DEFAULT_NUM_OBSTACLES,
    DEFAULT_XML,
    DEFAULT_Y_RANGE,
    DEFAULT_YAW_DEG,
    JOINT_NAMES,
    NUM_ACTIONS,
    NUM_PROPRIO,
    OBJ_POOL,
    STIFFNESS,
    TORQUE_LIMITS,
    glfw_window_or_none,
    isaac_to_mujoco,
    mujoco_to_isaac,
    quat_apply_inverse,
    velocity_command_2d_rt,
)

# Isaac GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), ordering="xy"
SCAN_RES = 0.1
SCAN_SIZE_X, SCAN_SIZE_Y = 1.6, 1.0
SCAN_OFFSET_Z = 20.0
HEIGHT_SCAN_OFFSET = 0.5
HEIGHT_CLIP = 1.0
_XS = np.arange(-SCAN_SIZE_X / 2, SCAN_SIZE_X / 2 + 1.0e-9, SCAN_RES)
_YS = np.arange(-SCAN_SIZE_Y / 2, SCAN_SIZE_Y / 2 + 1.0e-9, SCAN_RES)
_GRID_X, _GRID_Y = np.meshgrid(_XS, _YS, indexing="xy")
SCAN_NX, SCAN_NY = _XS.size, _YS.size  # 17 x 11
HEIGHT_DIM = SCAN_NX * SCAN_NY  # 187
OBS_DIM = NUM_PROPRIO + HEIGHT_DIM
LOCAL_STARTS = np.stack(
    [_GRID_X.ravel(), _GRID_Y.ravel(), np.full(HEIGHT_DIM, SCAN_OFFSET_Z)],
    axis=1,
).astype(np.float64)

DEFAULT_POLICY = _REPO_ROOT / "checkpoints" / "g1_flat_height_distillation" / "model_20000.pt"


def load_height_policy(policy_path: str, device: str) -> torch.nn.Module:
    """Load stage-1-style MLPModel from an rsl_rl ``model_*.pt`` checkpoint."""
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    if "actor_state_dict" not in ckpt:
        raise ValueError(f"{policy_path} is not an rsl_rl checkpoint (missing actor_state_dict).")
    dummy = TensorDict({"policy": torch.zeros(1, OBS_DIM)}, batch_size=[1])
    model = MLPModel(
        obs=dummy,
        obs_groups={"actor": ["policy"]},
        obs_set="actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    missing, unexpected = model.load_state_dict(ckpt["actor_state_dict"], strict=True)
    if missing or unexpected:
        print(f"[warn] load_state_dict missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    print(f"Loaded MLPModel from {policy_path} (iter={ckpt.get('iter')})")
    return model


class PhpHeightSim2SimController:
    def __init__(
        self,
        policy_path: str,
        xml_path: str = str(DEFAULT_XML),
        device: str = "cuda",
        record_video: bool = False,
        headless: bool = False,
        show_height: bool = True,
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
        self.policy = load_height_policy(policy_path, self.device)
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.sim_dt = 0.001
        self.sim_decimation = 20
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        self._check_joint_order()
        self._torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self._torso_bid < 0:
            raise RuntimeError("parkour.xml is missing body torso_link")
        # Floor / obstacles stay in default geom group 0; robot visuals=2, collision=3.
        self._scan_geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

        self.video_renderer = None
        self.record_video = record_video
        self.show_height = show_height and not headless
        self.headless = headless
        self.sim_duration = duration
        self.last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.pd_target = DEFAULT_DOF_POS.copy()
        self._request_reset = False
        self._glfw_window = None

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

        self._height_fig = None
        self._height_im = None
        if self.show_height:
            plt.ion()
            self._height_fig, ax = plt.subplots(figsize=(6, 4))
            self._height_im = ax.imshow(
                np.zeros((SCAN_NY, SCAN_NX)),
                cmap="terrain",
                vmin=-HEIGHT_CLIP,
                vmax=HEIGHT_CLIP,
                origin="lower",
            )
            ax.set_title(f"Height scan ({SCAN_NX}x{SCAN_NY}, clip ±{HEIGHT_CLIP})")
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

    def _torso_yaw(self) -> float:
        qw, qx, qy, qz = self.data.xquat[self._torso_bid]
        return float(np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def compute_height_scan(self) -> np.ndarray:
        """Isaac ``height_scan``: torso_z - hit_z - 0.5, clip [-1, 1]."""
        torso_pos = np.array(self.data.xpos[self._torso_bid], dtype=np.float64)
        yaw = self._torso_yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        starts = torso_pos + LOCAL_STARTS @ rot.T
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        heights = np.empty(HEIGHT_DIM, dtype=np.float32)
        torso_z = float(torso_pos[2])
        for i in range(HEIGHT_DIM):
            dist = mujoco.mj_ray(
                self.model, self.data, starts[i], vec, self._scan_geomgroup, 1, -1, geomid
            )
            if dist < 0.0:
                heights[i] = HEIGHT_CLIP
            else:
                hit_z = starts[i, 2] - dist
                heights[i] = np.clip(torso_z - hit_z - HEIGHT_SCAN_OFFSET, -HEIGHT_CLIP, HEIGHT_CLIP)
        if self._height_im is not None:
            self._height_im.set_data(heights.reshape(SCAN_NY, SCAN_NX))
            self._height_fig.canvas.draw()
            self._height_fig.canvas.flush_events()
        return heights

    def build_obs(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        quat: np.ndarray,
        ang_vel: np.ndarray,
        height_scan: np.ndarray,
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
        obs = np.concatenate([proprio, height_scan.reshape(-1).astype(np.float32)])
        if obs.shape[0] != OBS_DIM:
            raise RuntimeError(f"obs dim {obs.shape[0]} != {OBS_DIM}")
        return obs

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        td = TensorDict({"policy": obs_t}, batch_size=[1])
        with torch.no_grad():
            action = self.policy(td, stochastic_output=False)
        return action.detach().cpu().numpy().reshape(-1).astype(np.float32)

    def _place_obstacles(self) -> None:
        for spec in self._obj_pool:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec["name"])
            if bid < 0:
                continue
            self.model.body_pos[bid] = np.array([0.0, 0.0, -10.0])
            self.model.body_quat[bid] = np.array([1.0, 0.0, 0.0, 0.0])

        order = self._rng.permutation(len(self._obj_pool))[: self._num_obstacles]
        x = self._first_x
        print(
            f"Obstacles n={self._num_obstacles} first_x={self._first_x:.2f} "
            f"gap=[{self._gap_min:.1f}, {self._gap_max:.1f}]"
        )
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
        window = glfw_window_or_none(self._glfw_window)
        if window is None:
            return
        try:
            up = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
            left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
            right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
        except Exception:
            self._glfw_window = None
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
        ctx = glfw_window_or_none(glfw.get_current_context())
        if ctx is not None:
            self._glfw_window = ctx
        if keycode == glfw.KEY_BACKSPACE:
            window = glfw_window_or_none(self._glfw_window)
            if window is not None and glfw.get_key(window, glfw.KEY_BACKSPACE) == glfw.PRESS:
                self._request_reset = True
            return
        self._poll_held_keys()

    def run(self) -> None:
        mp4_writer = None
        if self.record_video:
            import imageio

            video_name = "php_height_sim2sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
            self.video_renderer = mujoco.Renderer(self.model, height=720, width=1280)

        self.reset()
        steps = int(self.sim_duration / self.sim_dt)
        print(
            "Focus the MuJoCo window (not the height figure). "
            "Hold ↑ walk, hold ←/→ turn; release → cmd [0, 0, 0]. Backspace reset."
        )
        command = np.zeros(3, dtype=np.float32)
        policy_dt = self.sim_dt * self.sim_decimation
        next_tick = time.perf_counter()
        height_scan = np.zeros(HEIGHT_DIM, dtype=np.float32)

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
                height_scan = self.compute_height_scan()
                command = self.velocity_command.copy()
                self._print_cmd(command)

                obs = self.build_obs(dof_pos, dof_vel, quat, ang_vel, height_scan, command)
                raw_action = np.clip(self.infer(obs), -10.0, 10.0)
                self.last_action = raw_action
                self.pd_target = isaac_to_mujoco(raw_action * ACTION_SCALE_ISAAC + DEFAULT_DOF_POS_ISAAC)

                if self.viewer is not None:
                    pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                    self.viewer.cam.lookat = pelvis_pos
                    self.viewer.sync()

                if mp4_writer is not None:
                    self.video_renderer.update_scene(self.data, camera=self.viewer.cam if self.viewer else None)
                    rgb_img = self.video_renderer.render()
                    hmap = ((np.clip(height_scan.reshape(SCAN_NY, SCAN_NX), -1.0, 1.0) + 1.0) * 0.5 * 255).astype(
                        np.uint8
                    )
                    hmap_bgr = cv2.applyColorMap(hmap, cv2.COLORMAP_VIRIDIS)
                    hmap_rgb = cv2.cvtColor(hmap_bgr, cv2.COLOR_BGR2RGB)
                    main_h, main_w, _ = rgb_img.shape
                    inset_w = int(main_w * 0.3)
                    inset_h = int(inset_w * (hmap_rgb.shape[0] / max(hmap_rgb.shape[1], 1)))
                    inset = cv2.resize(hmap_rgb, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
                    rgb_img[0:inset_h, -inset_w:] = inset
                    text = f"cmd [{command[0]:.2f}, {command[1]:.1f}, {command[2]:+.0f}]"
                    rgb_img = cv2.putText(
                        rgb_img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    mp4_writer.append_data(rgb_img)

            torque = (self.pd_target - dof_pos) * STIFFNESS - dof_vel * DAMPING
            self.data.ctrl[:] = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
            mujoco.mj_step(self.model, self.data)

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
        if self._height_fig is not None:
            plt.close(self._height_fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PHP height-map MuJoCo sim2sim")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="rsl_rl HeightDAgger MLP model_*.pt")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="MuJoCo XML (29-DoF G1)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speed", type=float, default=0.0, help="initial cmd_speed: 0 or 1")
    parser.add_argument("--turn", type=float, default=0.0, help="initial heading (sign only)")
    parser.add_argument("--obstacle_x", type=float, default=None, help="X of the first box")
    parser.add_argument("--num_obstacles", type=int, default=DEFAULT_NUM_OBSTACLES)
    parser.add_argument("--gap_min", type=float, default=DEFAULT_GAP[0])
    parser.add_argument("--gap_max", type=float, default=DEFAULT_GAP[1])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration", type=float, default=1000.0)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no_height_window", action="store_true")
    args = parser.parse_args()

    controller = PhpHeightSim2SimController(
        policy_path=args.policy,
        xml_path=args.xml,
        device=args.device,
        record_video=args.record_video,
        headless=args.headless,
        show_height=not args.no_height_window,
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
