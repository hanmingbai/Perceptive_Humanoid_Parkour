from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.mdp as mdp
from perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.config.g1.vision_distillation_flat_env_cfg import (
    G1VisionDistillationFlatEnvCfg,
)
from perceptive_humanoid_parkour.tasks.perceptive_humanoid_parkour.vision_distillation_env_cfg import ObservationsCfg


@configclass
class HeightMapPolicyCfg(ObservationsCfg.PolicyCfg):
    """Stage-2 student: same proprio + 2D cmd as depth distill, height scan instead of depth."""

    depth_flatten = None
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.1, n_max=0.1),
        clip=(-1.0, 1.0),
    )


@configclass
class HeightMapObservationsCfg(ObservationsCfg):
    policy: HeightMapPolicyCfg = HeightMapPolicyCfg()


@configclass
class G1HeightDistillationFlatEnvCfg(G1VisionDistillationFlatEnvCfg):
    """Depth-free stage-2 distillation; critic / teacher unchanged."""

    observations: HeightMapObservationsCfg = HeightMapObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.tiled_camera = None
        self.scene.height_scanner.debug_vis = False
