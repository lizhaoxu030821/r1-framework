# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.tasks.direct.base import mdp
from robolab.tasks.direct.base.atom01_env_cfg import ATOM01FlatEnvCfg, ATOM01RewardCfg, COMMAND_DEADZONE


@configclass
class ATOM01SpeedRepairRewardCfg(ATOM01RewardCfg):
    """Reward balance for high-speed forward tracking without large pose drift."""

    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=4.0, params={"std": 0.65})
    lateral_vel_y_l2 = RewTerm(func=mdp.lateral_vel_y_l2, weight=-2.6)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.6, params={"std": 0.40})
    yaw_rate_l2 = RewTerm(func=mdp.yaw_rate_l2, weight=-2.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.30)
    roll_orientation_l2 = RewTerm(func=mdp.roll_orientation_l2, weight=-2.2)
    roll_ang_vel_l2 = RewTerm(func=mdp.roll_ang_vel_l2, weight=-0.18)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.6)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.85,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_heading_l2 = RewTerm(
        func=mdp.feet_heading_l2,
        weight=-1.6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )


@configclass
class ATOM01SpeedRepairEnvCfg(ATOM01FlatEnvCfg):
    """40k curriculum: repair straightness first, then reintroduce 3.5 m/s."""

    reward = ATOM01SpeedRepairRewardCfg()

    def __post_init__(self):
        super().__post_init__()

        # The common_step_counter advances once per policy environment step.
        # With 24 steps/iteration, these boundaries are 0k, 15k, 25k, and 35k
        # training iterations in a 40k continuation run.
        self.commands.velocity_curriculum = (
            (0, (-0.40, 2.60), (-0.25, 0.25), (-0.50, 0.50), (0.82, 0.06, 0.06, 0.04)),
            (360_000, (-0.50, 2.90), (-0.35, 0.35), (-0.70, 0.70), (0.84, 0.05, 0.05, 0.04)),
            (600_000, (-0.70, 3.20), (-0.45, 0.45), (-0.90, 0.90), (0.86, 0.04, 0.04, 0.03)),
            (840_000, (-1.00, 3.50), (-0.55, 0.55), (-1.00, 1.00), (0.88, 0.03, 0.03, 0.03)),
        )
        self.commands.forward_speed_focus_curriculum = (
            (0, 0.25, 1.40),
            (360_000, 0.40, 1.80),
            (600_000, 0.55, 2.20),
            (840_000, 0.65, 2.60),
        )
        self.commands.directional_mode_probabilities = (0.82, 0.06, 0.06, 0.04)
        self.commands.walking_lin_vel_x_targets = ()
        self.commands.deadzone_threshold = COMMAND_DEADZONE
        self.commands.rel_standing_envs = 0.10
