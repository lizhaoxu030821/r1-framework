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


import isaaclab.sim as sim_utils
from pathlib import Path
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

ATOM01_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/atom01/urdf/atom01.urdf",
        fix_base=False,
        root_link_name="pelvis",
        force_usd_conversion=True,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "left_hip_yaw_joint": 0.,
            "left_hip_roll_joint": 0.,
            "left_hip_pitch_joint": -0.1500,
            "left_knee_joint": 0.3000,
            "left_ankle_pitch_joint": 0.1500,
            "left_ankle_roll_joint": 0.,

            # 右腿关节
            "right_hip_yaw_joint": 0.,
            "right_hip_roll_joint": 0.,
            "right_hip_pitch_joint": 0.1500,
            "right_knee_joint": -0.3000,
            "right_ankle_pitch_joint": -0.1500,
            "right_ankle_roll_joint": 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 3.3,
                ".*_hip_roll_joint": 3.3,
                ".*_hip_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
            },
            armature={
                ".*_hip_yaw_joint": 0.0475276,
                ".*_hip_roll_joint": 0.0703102,
                ".*_hip_pitch_joint": 0.2743039,
                ".*_knee_joint": 0.2743039,
            },
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness={
                ".*_ankle_pitch_joint": 260.0,
                ".*_ankle_roll_joint": 180.0,
            },
            damping={
                ".*_ankle_pitch_joint": 3.0,
                ".*_ankle_roll_joint": 3.0,
            },
            armature={
                ".*_ankle_pitch_joint": 0.0389088,
                ".*_ankle_roll_joint": 0.0389088,
            },
            min_delay=0,
            max_delay=4,
        ),
    },
)


ZKY_19DOF_URDF = (
    Path(__file__).resolve().parents[5]
    / "ZKY_SDK_19Dof/resources/robots/robot_zky_19dof/urdf/robot_zky_fixed_all.urdf"
)

ZKY_19DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ZKY_19DOF_URDF),
        fix_base=False,
        root_link_name="base_link",
        force_usd_conversion=True,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # The CAD collision meshes overlap at the hip and ankle even in
            # nominal poses (up to ~0.17 m across the reference). Keep link-
            # ground contacts for get-up support, but do not generate internal
            # forces from those structural mesh overlaps.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # The visual get-up reference intentionally uses widened simulation
    # ranges. A 0.90 factor would clip reset states while the command and
    # rewards continued to request the unmodified NPZ pose.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint"],
            # DelayedPDActuator is explicit: effort_limit clips the PD output,
            # while effort_limit_sim configures PhysX.  Set both explicitly;
            # otherwise the actuator silently inherits the URDF's 100 Nm and
            # never reaches the intended 150 Nm simulation limit.
            effort_limit=150.0,
            effort_limit_sim=150.0,
            velocity_limit_sim=25.0,
            stiffness={".*_hip_.*_joint": 100.0, ".*_knee_joint": 150.0},
            damping={".*_hip_.*_joint": 3.3, ".*_knee_joint": 5.0},
            armature={".*": 0.05},
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=35.0,
            effort_limit_sim=35.0,
            velocity_limit_sim=8.0,
            stiffness={".*": 40.0},
            damping={".*": 2.0},
            armature={".*": 0.04},
            min_delay=0,
            max_delay=4,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=["body_joint"],
            effort_limit=150.0,
            effort_limit_sim=150.0,
            velocity_limit_sim=25.0,
            stiffness={"body_joint": 150.0},
            damping={"body_joint": 5.0},
            armature={"body_joint": 0.05},
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint"],
            # The frame-zero physics baseline needs 55--66 Nm at the
            # shoulders during the arm-support transition. The previous
            # 35 Nm clip saturated up to 31.6% of all joints at frame 100.
            effort_limit=70.0,
            effort_limit_sim=70.0,
            velocity_limit_sim=8.0,
            stiffness={".*_shoulder_.*_joint": 40.0, ".*_elbow_joint": 30.0},
            damping={".*_shoulder_.*_joint": 2.0, ".*_elbow_joint": 1.5},
            armature={".*": 0.04},
            min_delay=0,
            max_delay=4,
        ),
    },
)
