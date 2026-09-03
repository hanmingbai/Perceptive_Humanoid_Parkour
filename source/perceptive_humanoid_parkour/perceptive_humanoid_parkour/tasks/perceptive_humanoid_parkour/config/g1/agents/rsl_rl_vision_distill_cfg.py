from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

import os

# this file is .../config/g1/agents/*.py → 8 levels up = repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 8)))
_DEFAULT_TEACHER_CKPT = os.path.join(
    _REPO_ROOT, "checkpoints", "g1_flat_motion_tracking", "model_51000.pt"
)

@configclass
class DAggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
    dagger_loss_mix_ratio: float = 0.5 # 仅兼容保留；实际混比用 λ_D(k)=max(0.1, 1-k/(K/2))
    # None：用 runner.max_iterations 作为课程总长 K（延长训练时只改那一处即可）
    max_iters: int | None = None
    # PHP Table VI：L_D = coef * MSE，再和 λ_D 做凸组合
    dagger_loss_coef: float = 1.0

@configclass
class G1FlatDAggerRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    experiment_name = "g1_flat_vison_distillation" # 建议修改实验名称以区分
    empirical_normalization = True

    # 配置 Actor (Student)
    actor = RslRlMLPModelCfg(
        class_name="VisionMLPModel",
        hidden_dims=[1024, 512, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.01,           # 代替以前的 init_noise_std 1.0
            std_type="scalar",      # 代替以前的 noise_std_type
        )
    )
    actor.visual_output_dim = 32

    # 显式定义 Critic, Critic 通常不需要视觉，用标准 MLP 即可, Critic 通常不需要分布distribution_cfg，设为 None 即可
    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None
    )

    # 2. DAgger-PPO 算法配置, PPO 基础参数保持不变, 必须指定类名，确保它能被 resolve_callable 找到
    algorithm = DAggerAlgorithmCfg(
        class_name="VisionDAggerPPO", 
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001, # 0.005
        num_learning_epochs=5, # 5
        num_mini_batches=4, # 4
        learning_rate=3.0e-4, # 1e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # Teacher 专家配置, 这里的 class_name 默认是 MLPModel
    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    # 因为 RslRlMLPModelCfg 本身没有 model_path 属性，我们需要手动补上
    teacher.model_path = _DEFAULT_TEACHER_CKPT

    # 观测组映射 (非常关键), 确保 Student 只看 policy 组，Teacher/Critic 看包含特权信息的组
    obs_groups = {
        "actor": ["policy"],      # 学生看本体感知信息
        "critic": ["critic"],     # Critic 看包含参考动作的信息
        "teacher": ["teacher"]     # Teacher 也看包含参考动作的信息
    }


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqDAggerRunnerCfg(G1FlatDAggerRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)