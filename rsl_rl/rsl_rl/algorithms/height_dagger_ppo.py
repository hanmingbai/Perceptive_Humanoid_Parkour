# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.vision_dagger_ppo import VisionDAggerPPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class HeightDAggerPPO(VisionDAggerPPO):
    """Height-map DAgger-PPO: VisionDAgger curriculum with a stage-1 MLP student.

    Reuses VisionDAggerPPO's λ_D schedule, ``dagger_loss_coef``, and adaptive-KL
    warmup. Construction follows DAggerPPO (no 5046-d depth / CNN split).
    """

    actor: MLPModel

    def __init__(self, actor: MLPModel, critic: MLPModel, storage: RolloutStorage, **kwargs) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        print(
            f"[HeightDAgger] curriculum K=max_iters={self.max_iters}, "
            f"λ_D=max(0.1, 1-k/(K/2)), adaptive KL/LR after iter {self._adaptive_kl_start_iter} (λ_PPO>0.1)"
        )
        print("<< --------------------HeightDAggerPPO Created --------------------- >>")

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
            print("[INFO] Height Actor (MLP) loaded successfully.")
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> HeightDAggerPPO:
        alg_class: type[HeightDAggerPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].get("class_name", "MLPModel"))  # type: ignore

        default_sets = ["actor", "critic", "teacher"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        if cfg["algorithm"].get("max_iters") in (None, 0):
            cfg["algorithm"]["max_iters"] = int(cfg.get("max_iterations", 100000))

        policy_dim = int(obs["policy"].shape[1])
        print(f"[HeightDAgger] policy obs dim={policy_dim} (MLP, no depth CNN split)")

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        cfg["algorithm"].pop("share_cnn_encoders", None)
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

        teacher_cfg = cfg.pop("teacher")
        teacher_model_path = teacher_cfg.pop("model_path", None)
        if isinstance(teacher_cfg, dict):
            teacher_cfg.pop("class_name", None)

        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **teacher_cfg).to(device)
        if teacher_model_path is not None:
            print(f"[DAgger] Loading teacher weights from: {teacher_model_path}")
            loaded_dict = torch.load(teacher_model_path, map_location=device)
            state_dict = loaded_dict.get("model_state_dict", loaded_dict.get("actor_state_dict", loaded_dict))
            teacher.load_state_dict(state_dict, strict=False)
            teacher.eval()

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(
            actor,
            critic,
            storage,
            teacher=teacher,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
