from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ZkyGetupBeyondMimicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 35000
    save_interval = 200
    experiment_name = "zky_getup_beyondmimic"
    wandb_project = "zky_getup_beyondmimic"
    logger = "wandb"
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        # Get-up starts from a delicate contact state.  RoboParty's stable
        # runs use a small action envelope so exploration does not destroy the
        # first hand/foot support transition before PPO sees a useful return.
        init_noise_std=0.20,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.0002,
        num_learning_epochs=5,
        num_mini_batches=4,
        # v50 migrates the best v49 actor and learns the appended speed input
        # without erasing the existing 0.5x get-up behavior.
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
    )
