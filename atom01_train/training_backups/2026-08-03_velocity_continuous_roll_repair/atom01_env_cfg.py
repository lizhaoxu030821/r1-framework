# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.tasks.direct.base import mdp
from robolab.assets.robots import ATOM01_CFG
from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg, 
    BaseEnvCfg, 
    RewardCfg, 
    HeightScannerCfg, 
    SceneContextCfg, 
    RobotCfg, 
    ObsScalesCfg, 
    NormalizationCfg, 
    CommandRangesCfg, 
    CommandsCfg, 
    NoiseScalesCfg, 
    NoiseCfg, 
    EventCfg,
    GRAVEL_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG,
    ROUGH_HARD_TERRAINS_CFG,
    SceneCfg
)

COMMAND_DEADZONE = 0.05


@configclass
class ATOM01RewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.2, params={"std": 0.55})
    lateral_vel_y_l2 = RewTerm(func=mdp.lateral_vel_y_l2, weight=-1.5)
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.7, params={"std": 0.45})
    yaw_rate_l2 = RewTerm(func=mdp.yaw_rate_l2, weight=-1.5)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.2)
    root_height_l2 = RewTerm(func=mdp.root_height_l2, weight=-8.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.22)
    roll_orientation_l2 = RewTerm(func=mdp.roll_orientation_l2, weight=-1.5)
    roll_ang_vel_l2 = RewTerm(func=mdp.roll_ang_vel_l2, weight=-0.12)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2.5e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-3.5e-2)
    hip_yaw_action_l2 = RewTerm(
        func=mdp.action_l2_selected,
        weight=-2e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    hip_yaw_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-3e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    hip_yaw_action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2_selected,
        weight=-4e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )
    ankle_roll_action_l2 = RewTerm(
        func=mdp.action_l2_selected,
        weight=-3e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )
    ankle_roll_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-4e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )
    ankle_roll_action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2_selected,
        weight=-5e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.2)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    recovery_upright = RewTerm(
        func=mdp.recovery_upright_exp,
        weight=0.0,
        params={"tilt_threshold": 0.12, "height_threshold": 0.08, "std": 0.35},
    )
    recovery_height = RewTerm(
        func=mdp.recovery_height_exp,
        weight=0.0,
        params={"tilt_threshold": 0.12, "height_threshold": 0.08, "std": 0.12},
    )
    recovery_feet_contact = RewTerm(
        func=mdp.recovery_feet_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "tilt_threshold": 0.12,
            "height_threshold": 0.08,
        },
    )
    recovery_ang_vel_xy_l2 = RewTerm(
        func=mdp.recovery_ang_vel_xy_l2,
        weight=0.0,
        params={"tilt_threshold": 0.12, "height_threshold": 0.08},
    )
    recovery_action_l2 = RewTerm(
        func=mdp.recovery_action_l2,
        weight=0.0,
        params={"tilt_threshold": 0.12, "height_threshold": 0.08},
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 0.28,
            "command_threshold": COMMAND_DEADZONE,
        },
    )
    feet_flight_l2 = RewTerm(
        func=mdp.feet_flight_l2,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.6,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-8e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 350,
            "max_reward": 400,
        },
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.22, "max": 0.42},
    )
    feet_lateral_center_l2 = RewTerm(
        func=mdp.body_lateral_center_l2,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_.*ankle_roll.*", "right_.*ankle_roll.*"],
                preserve_order=True,
            )
        },
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.20, "max": 0.36},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    undesired_foothold = RewTerm(
        func=mdp.undesired_foothold,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]),
            "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
            "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
            "ankle_height": 0.04,
        },
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.6,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )
    feet_orientation_contact_l2 = RewTerm(
        func=mdp.feet_orientation_contact_l2,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll.*"]),
        },
    )
    feet_heading_l2 = RewTerm(
        func=mdp.feet_heading_l2,
        weight=-1.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-3.0)
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*"]
            )
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.12,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_yaw.*", ".*_hip_roll.*"]
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[]
            )
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[],
            )
        },
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_pitch.*", ".*_knee.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"])},
    )
    joint_deviation_ankle_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_ankle_roll.*"]
            )
        },
    )
    left_roll_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1_deadband,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["left_hip_roll_joint", "left_ankle_roll_joint"]
            ),
            "deadband": 0.03,
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["left_.*ankle_roll.*", "right_.*ankle_roll.*"],
                preserve_order=True,
            ),
            "command_threshold": COMMAND_DEADZONE,
        },
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(func=mdp.stand_still, weight=-0.2, params={"pos_cfg": SceneEntityCfg("robot", joint_names=[".*_hip.*", ".*_knee.*", ".*_ankle.*"]),
                                                                     "vel_cfg": SceneEntityCfg("robot", joint_names=[".*_hip.*", ".*_knee.*", ".*_ankle.*"]),
                                                                     "pos_weight": 1.0, "vel_weight": 0.04, "command_threshold": COMMAND_DEADZONE})
    stand_action_l2 = RewTerm(
        func=mdp.stand_action_l2,
        weight=-0.3,
        params={"command_threshold": COMMAND_DEADZONE},
    )
    stand_body_vel_l2 = RewTerm(
        func=mdp.stand_body_vel_l2,
        weight=-1.0,
        params={"command_threshold": COMMAND_DEADZONE},
    )
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.35,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height":0.04,"threshold":0.04, "command_threshold": COMMAND_DEADZONE})


@configclass
class ATOM01FlatEnvCfg(BaseEnvCfg):

    reward = ATOM01RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 12
        self.observation_space = 45
        self.state_space = 84
        self.scene_context.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.height_scanner.prim_body_name = "pelvis"
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene_context.height_scanner.enable_height_scan = False
        self.commands.deadzone_threshold = COMMAND_DEADZONE
        self.commands.ranges = CommandRangesCfg(
            lin_vel_x=(-1.0, 2.5),
            lin_vel_y=(-0.8, 0.8),
            ang_vel_z=(-1.5, 1.5),
            heading=(-3.141592653589793, 3.141592653589793),
        )
        # An empty target set preserves the sampled command instead of snapping vx
        # to gait bins, which makes deployment velocity continuously adjustable.
        self.commands.walking_lin_vel_x_targets = ()
        self.commands.directional_mode_probabilities = (0.35, 0.15, 0.15, 0.15)
        self.commands.velocity_curriculum = (
            (0, (0.05, 0.75), (0.0, 0.0), (0.0, 0.0), (0.90, 0.00, 0.00, 0.00)),
            (45_000, (-0.35, 1.10), (-0.18, 0.18), (-0.35, 0.35), (0.55, 0.10, 0.10, 0.10)),
            (120_000, (-0.60, 1.55), (-0.40, 0.40), (-0.70, 0.70), (0.45, 0.15, 0.15, 0.15)),
            (225_000, (-0.80, 2.00), (-0.60, 0.60), (-1.00, 1.00), (0.40, 0.15, 0.15, 0.15)),
            (330_000, (-1.00, 2.50), (-0.80, 0.80), (-1.50, 1.50), (0.35, 0.15, 0.15, 0.15)),
        )
        self.commands.rel_standing_envs = 0.12
        self.commands.resampling_time_range = (6.0, 10.0)
        self.commands.startup_stand_time_range = (1.0, 2.0)
        self.commands.startup_ramp_time_range = (1.5, 2.0)
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.robot.terminate_contacts_body_names = ["pelvis", ".*_hip_yaw_link", ".*_hip_roll_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["pelvis"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link", "pelvis"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.robot.action_scale_multipliers = {
            "left_ankle_pitch_joint": 0.75,
            "right_ankle_pitch_joint": 0.75,
            "left_ankle_roll_joint": 0.40,
            "right_ankle_roll_joint": 0.40,
        }
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03


@configclass
class ATOM01RoughEnvCfg(ATOM01FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.state_space = 271
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt = self.sim.dt,
            step_dt = self.decimation * self.sim.dt
        )
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.reward.ang_vel_xy_l2.weight = -0.05
        self.reward.lin_vel_z_l2.weight = -0.05
