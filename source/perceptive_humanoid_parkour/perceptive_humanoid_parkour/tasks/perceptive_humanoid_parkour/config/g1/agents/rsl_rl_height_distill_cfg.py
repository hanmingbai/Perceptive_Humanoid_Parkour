from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_vision_distill_cfg import DAggerAlgorithmCfg, G1FlatDAggerRunnerCfg


@configclass
class G1FlatHeightDAggerRunnerCfg(G1FlatDAggerRunnerCfg):
    """Stage-2 DAgger with height-map student; actor-critic matches stage-1 MLP."""

    experiment_name = "g1_flat_height_distillation"

    actor = RslRlMLPModelCfg(
        class_name="MLPModel",
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="scalar",
        ),
    )

    algorithm = DAggerAlgorithmCfg(
        class_name="HeightDAggerPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        max_iters=100000,
    )
