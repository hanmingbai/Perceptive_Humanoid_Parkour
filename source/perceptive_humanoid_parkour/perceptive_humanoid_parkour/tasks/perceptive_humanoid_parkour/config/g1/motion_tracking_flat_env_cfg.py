from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors import TiledCameraCfg

from perceptive_humanoid_parkour.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.config.g1.agents.rsl_rl_vision_distill_cfg import LOW_FREQ_SCALE
from perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.motion_tracking_env_cfg import MotionTrackingEnvCfg

import glob
import os

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.mdp as mdp

@configclass
class G1MotionTrackingFlatEnvCfg(MotionTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]
        ee_body_names = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
        other_body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "right_hip_roll_link",
            "right_knee_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
        ]
        self.rewards.motion_ee_pos.params["body_names"] = ee_body_names
        self.rewards.motion_ee_ori.params["body_names"] = ee_body_names
        self.rewards.motion_body_pos.params["body_names"] = other_body_names
        self.rewards.motion_body_ori.params["body_names"] = other_body_names

        # repo_root/motions/... (this file is .../config/g1/*.py → parents[7] = repo root)
        motion_dir = os.path.join(
            os.path.dirname(__file__), *([".."] * 7), "motions", "mm_php_post_keep_0812"
        )
        self.commands.motion.motion_file = sorted(
            glob.glob(os.path.join(os.path.abspath(motion_dir), "*.npz"))
        )


@configclass
class G1MotionTrackingFlatWoStateEstimationEnvCfg(G1MotionTrackingFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1MotionTrackingFlatLowFreqEnvCfg(G1MotionTrackingFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
