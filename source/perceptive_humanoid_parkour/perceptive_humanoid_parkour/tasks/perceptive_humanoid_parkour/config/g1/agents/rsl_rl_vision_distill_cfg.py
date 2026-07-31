from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

@configclass
class DAggerAlgorithmCfg(RslRlPpoAlgorithmCfg):
    dagger_loss_mix_ratio: float = 0.5

@configclass
class G1FlatDAggerRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 250
    experiment_name = "g1_vision_distillation" # 建议修改实验名称以区分
    empirical_normalization = True

    # 1. 配置 Actor (Student)
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

    critic = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=None
    )

    algorithm = DAggerAlgorithmCfg(
        class_name="VisionDAggerPPO", 
        
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001, # 0.005
        num_learning_epochs=2, # 5
        num_mini_batches=96, # 4
        learning_rate=3.0e-4, # 1e-3
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    teacher = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    teacher.model_path = "/home/hmbai/legged_lab/logs/rsl_rl/g1_flat_plus_obstacle/2026-06-23_00-40-17/model_29999.pt"

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
        "teacher": ["teacher"]
    }


LOW_FREQ_SCALE = 0.5


@configclass
class G1FlatLowFreqDAggerRunnerCfg(G1FlatDAggerRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
