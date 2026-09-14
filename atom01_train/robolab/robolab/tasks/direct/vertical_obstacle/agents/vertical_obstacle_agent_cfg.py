"""RSL-RL agent configuration for vertical-obstacle training."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from robolab.tasks.direct.base.agents.atom01_agent_cfg import ATOM01FlatAgentCfg, data_augmentation_func


@configclass
class VerticalObstacleAgentCfg(ATOM01FlatAgentCfg):
    """Reuse the 450/840-input policy and reset optimizer for platform traversal."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zky_vertical_obstacle_v3"
        self.wandb_project = "zky_vertical_obstacle_v3"
        self.max_iterations = 60001
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-5,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=0.7,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=self.algorithm.symmetry_cfg,
            rnd_cfg=None,
        )
