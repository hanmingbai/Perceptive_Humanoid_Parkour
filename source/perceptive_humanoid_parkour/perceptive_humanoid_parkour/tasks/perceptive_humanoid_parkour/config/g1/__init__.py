import gymnasium as gym

from . import agents, motion_tracking_flat_env_cfg, vision_distillation_flat_env_cfg

##
# Register Gym environments.
##

# python scripts/rsl_rl/train.py --task PHP-MotionTracking-Flat-G1-v0 --num_envs=4096 --headless
gym.register(
    id="PHP-MotionTracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": motion_tracking_flat_env_cfg.G1MotionTrackingFlatEnvCfg,
        # "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_distill_cfg:G1FlatDAggerRunnerCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

# python scripts/rsl_rl/train.py --task PHP-VisionDistillation-Flat-G1-v0 --num_envs=1024 --enable_cameras --headless
gym.register(
    id="PHP-VisionDistillation-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vision_distillation_flat_env_cfg.G1VisionDistillationFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_distill_cfg:G1FlatDAggerRunnerCfg",
    },
)

gym.register(
    id="PHP-VisionDistillation-Flat-G1-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vision_distillation_flat_env_cfg.G1VisionDistillationFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_distill_cfg:G1FlatDAggerRunnerCfg",
    },
)


gym.register(
    id="PHP-VisionDistillation-Flat-G1-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": vision_distillation_flat_env_cfg.G1VisionDistillationFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_vision_distill_cfg:G1FlatLowFreqDAggerRunnerCfg",
    },
)
