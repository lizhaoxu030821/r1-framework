
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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from robolab.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_HIGH_KP_REWARD_SCALES = (
    ("ankle", 40.0 / 300.0),
    ("knee", 150.0 / 300.0),
)


def _joint_ids_to_list(asset: Articulation, joint_ids) -> list[int]:
    all_joint_ids = list(range(len(asset.joint_names)))
    if joint_ids is None:
        return all_joint_ids
    if isinstance(joint_ids, slice):
        return all_joint_ids[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.detach().cpu().tolist()
    if isinstance(joint_ids, int):
        return [joint_ids]
    return list(joint_ids)


def _high_kp_reward_scale(asset: Articulation, joint_ids, *, squared: bool) -> torch.Tensor:
    ids = _joint_ids_to_list(asset, joint_ids)
    scale = torch.ones((1, len(ids)), device=asset.data.applied_torque.device, dtype=asset.data.applied_torque.dtype)
    for scale_id, joint_id in enumerate(ids):
        joint_name = asset.joint_names[joint_id]
        for name_fragment, reward_scale in _HIGH_KP_REWARD_SCALES:
            if name_fragment in joint_name:
                scale[:, scale_id] = reward_scale
                break
    return torch.square(scale) if squared else scale


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_scale = _high_kp_reward_scale(asset, asset_cfg.joint_ids, squared=True)
    return torch.sum(torch.square(joint_torque) * torque_scale, dim=1)


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_position_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Non-saturating floating-base position error for recovery shaping."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.linalg.vector_norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)


def motion_anchor_horizontal_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    """Track only horizontal root drift; grounded reference height is handled separately."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]), dim=-1)
    return torch.exp(-error / std**2)


def motion_anchor_horizontal_position_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.linalg.vector_norm(command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2], dim=-1)


def _scaled_reference_height(
    command: MotionCommand,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
) -> torch.Tensor:
    reference_progress = (
        (command.anchor_pos_w[:, 2] - reference_start_height)
        / max(reference_end_height - reference_start_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    return target_start_height + reference_progress * (target_end_height - target_start_height)


def _phase_height_completion(
    command: MotionCommand,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float = 0.0,
) -> torch.Tensor:
    """Fraction of the phase-requested vertical recovery actually achieved."""
    target_height = _scaled_reference_height(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
    )
    target_progress = (target_height - target_start_height).clamp_min(0.0)
    robot_progress = (command.robot_anchor_pos_w[:, 2] - target_start_height).clamp_min(0.0)
    completion = torch.where(
        target_progress > 0.02,
        (robot_progress / target_progress.clamp_min(0.02)).clamp(0.0, 1.0),
        torch.ones_like(target_progress),
    )
    return minimum_gate + (1.0 - minimum_gate) * completion


def motion_phase_height_completion(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _phase_height_completion(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
    )


def motion_phase_height_deficit_squared(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    tolerance: float = 0.03,
) -> torch.Tensor:
    """Strong non-saturating penalty for remaining below the phase target."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_height = _scaled_reference_height(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
    )
    deficit = (target_height - command.robot_anchor_pos_w[:, 2] - tolerance).clamp_min(0.0)
    recovery_range = max(target_end_height - target_start_height, 1.0e-6)
    return torch.square(deficit / recovery_range)


def _height_gate(
    command: MotionCommand,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float,
) -> torch.Tensor:
    return _phase_height_completion(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
        minimum_gate,
    )


def motion_scaled_anchor_height_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
) -> torch.Tensor:
    """Track NPZ height phase while capping the grounded reference at a physical target."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_height = _scaled_reference_height(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
    )
    error = target_height - command.robot_anchor_pos_w[:, 2]
    return torch.exp(-torch.square(error) / std**2)


def motion_scaled_anchor_height_error_abs(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    target_height = _scaled_reference_height(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
    )
    return torch.abs(target_height - command.robot_anchor_pos_w[:, 2])


def motion_anchor_height_error_abs_normalized(
    env: ManagerBasedRLEnv, command_name: str, scale: float = 1.0
) -> torch.Tensor:
    """Absolute base-height error with a bounded, useful gradient."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) / max(scale, 1.0e-6)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float = 0.2,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return motion_global_anchor_orientation_error_exp(env, command_name, std) * _height_gate(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
        minimum_gate,
    )

def motion_special_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_special_body_position_error_exp_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float = 0.2,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_special_body_position_error_exp(env, command_name, std, body_names)
    return reward * _height_gate(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
        minimum_gate,
    )

def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_position_error_exp_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float = 0.2,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_relative_body_position_error_exp(env, command_name, std, body_names)
    return reward * _height_gate(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
        minimum_gate,
    )


def motion_relative_body_position_error_abs(
    env: ManagerBasedRLEnv, command_name: str, body_names: list[str] | None = None
) -> torch.Tensor:
    """Non-saturating mean body-position error for strict imitation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    return torch.linalg.vector_norm(
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1
    ).mean(dim=-1)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float = 0.2,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_relative_body_orientation_error_exp(env, command_name, std, body_names)
    return reward * _height_gate(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
        minimum_gate,
    )


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    pose_gate_std: float | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    if pose_gate_std is not None:
        pose_error = torch.sum(
            torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
            dim=-1,
        ).mean(-1)
        reward *= torch.exp(-pose_error / pose_gate_std**2)
    return reward


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str] | None = None,
    pose_gate_std: float | None = None,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    reward = torch.exp(-error.mean(-1) / std**2)
    if pose_gate_std is not None:
        pose_error = torch.sum(
            torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
            dim=-1,
        ).mean(-1)
        reward *= torch.exp(-pose_error / pose_gate_std**2)
    return reward


def motion_joint_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error / std**2)


def motion_joint_position_error_exp_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    reference_start_height: float,
    reference_end_height: float,
    target_start_height: float,
    target_end_height: float,
    minimum_gate: float = 0.2,
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    reward = motion_joint_position_error_exp(env, command_name, std)
    return reward * _height_gate(
        command,
        reference_start_height,
        reference_end_height,
        target_start_height,
        target_end_height,
        minimum_gate,
    )


def motion_joint_position_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Keep a usable pose-learning gradient after the exponential term saturates."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.mean(torch.abs(command.joint_pos - command.robot_joint_pos), dim=-1)


def motion_lower_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float = 0.20
) -> torch.Tensor:
    """Imitate the NPZ lower-body power stroke independently of arm posture."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = list(command.robot.joint_names)
    lower = [i for i, name in enumerate(names) if any(token in name for token in ("hip", "knee", "ankle"))]
    if not lower:
        raise RuntimeError("Motion command has no lower-body joints")
    error = torch.mean(torch.square(command.joint_pos[:, lower] - command.robot_joint_pos[:, lower]), dim=-1)
    return torch.exp(-error / std**2)


def motion_lower_body_position_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Keep a non-saturating gradient on the NPZ hip/knee/ankle trajectory."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    names = list(command.robot.joint_names)
    lower = [i for i, name in enumerate(names) if any(token in name for token in ("hip", "knee", "ankle"))]
    if not lower:
        raise RuntimeError("Motion command has no lower-body joints")
    return torch.mean(torch.abs(command.joint_pos[:, lower] - command.robot_joint_pos[:, lower]), dim=-1)


def motion_joint_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, pos_gate_std: float | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.mean(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
    reward = torch.exp(-error / std**2)
    if pos_gate_std is not None:
        pos_error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
        reward *= torch.exp(-pos_error / pos_gate_std**2)
    return reward


def motion_anchor_height_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the vertical recovery trajectory explicitly.

    The generic anchor position term averages all three coordinates.  During
    get-up this can be dominated by horizontal/root-frame errors, so keep a
    separate dense signal for the height that actually represents recovery.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
    return torch.exp(-torch.square(error) / std**2)


def motion_anchor_height_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Non-saturating height error that keeps a gradient far from the reference."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])


def motion_anchor_upright_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Non-saturating trunk-up error in the same convention as the dense upright reward."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gravity = command.robot_anchor_quat_w.new_tensor([0.0, 0.0, -1.0]).expand(
        command.robot_anchor_quat_w.shape[0], -1
    )
    target_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, gravity)
    robot_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    return torch.abs(target_gravity_b[:, 2] - robot_gravity_b[:, 2])


def motion_anchor_upright_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Track the reference anchor's up direction, including prone phases."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    gravity = command.robot_anchor_quat_w.new_tensor([0.0, 0.0, -1.0]).expand(command.robot_anchor_quat_w.shape[0], -1)
    target_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, gravity)
    robot_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, gravity)
    error = target_gravity_b[:, 2] - robot_gravity_b[:, 2]
    return torch.exp(-torch.square(error) / std**2)


def getup_height_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_height: float,
    target_height: float,
) -> torch.Tensor:
    """Absolute physical height progress, matching the RoboParty get-up task."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    height = command.robot_anchor_pos_w[:, 2]
    return ((height - start_height) / max(target_height - start_height, 1.0e-6)).clamp(0.0, 1.0)


def getup_upright_progress(
    env: ManagerBasedRLEnv,
    start_upright: float,
    target_upright: float,
) -> torch.Tensor:
    """Absolute physical trunk-up progress, independent of reference error."""
    asset: Articulation = env.scene["robot"]
    gravity = asset.data.root_quat_w.new_tensor([0.0, 0.0, 1.0]).expand(asset.data.root_quat_w.shape[0], -1)
    local_up = math_utils.quat_apply(asset.data.root_quat_w, gravity)
    return ((local_up[:, 2] - start_upright) / max(target_upright - start_upright, 1.0e-6)).clamp(0.0, 1.0)


def getup_coupled_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_height: float,
    target_height: float,
    start_upright: float,
    target_upright: float,
) -> torch.Tensor:
    """Reward recovery only when height and uprightness improve together."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    height = (
        (command.robot_anchor_pos_w[:, 2] - start_height)
        / max(target_height - start_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    gravity = asset.data.root_quat_w.new_tensor([0.0, 0.0, 1.0]).expand(asset.data.root_quat_w.shape[0], -1)
    local_up = math_utils.quat_apply(asset.data.root_quat_w, gravity)[:, 2]
    upright = ((local_up - start_upright) / max(target_upright - start_upright, 1.0e-6)).clamp(0.0, 1.0)
    return height * upright


def getup_terminal_extension(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_height: float,
    target_height: float,
    start_reference_height: float,
    full_reference_height: float,
    minimum_upright: float,
) -> torch.Tensor:
    """Continuously reward straightening after an ordered recovery is upright.

    Unlike a binary stand bonus, this keeps a gradient between the v44
    crouched solution and the full grounded-reference standing height.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    reference_gate = (
        (command.anchor_pos_w[:, 2] - start_reference_height)
        / max(full_reference_height - start_reference_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    height_progress = (
        (command.robot_anchor_pos_w[:, 2] - start_height)
        / max(target_height - start_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    upright = -asset.data.projected_gravity_b[:, 2]
    upright_gate = ((upright - minimum_upright) / max(1.0 - minimum_upright, 1.0e-6)).clamp(0.0, 1.0)
    return reference_gate * upright_gate * height_progress


def terminal_residual_action_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_reference_height: float,
    full_reference_height: float,
    start_robot_height: float,
    full_robot_height: float,
) -> torch.Tensor:
    """Penalize unnecessary residuals only after physical recovery is established."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference_gate = (
        (command.anchor_pos_w[:, 2] - start_reference_height)
        / max(full_reference_height - start_reference_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    robot_gate = (
        (command.robot_anchor_pos_w[:, 2] - start_robot_height)
        / max(full_robot_height - start_robot_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    return reference_gate * robot_gate * torch.sum(torch.square(env.action_manager.action), dim=-1)


def terminal_effort_excess(
    env: ManagerBasedRLEnv,
    command_name: str,
    start_reference_height: float,
    full_reference_height: float,
    start_robot_height: float,
    full_robot_height: float,
    soft_ratio: float = 0.85,
) -> torch.Tensor:
    """Penalize pre-clipping PD effort above a soft actuator limit late in recovery."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    reference_gate = (
        (command.anchor_pos_w[:, 2] - start_reference_height)
        / max(full_reference_height - start_reference_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    robot_gate = (
        (command.robot_anchor_pos_w[:, 2] - start_robot_height)
        / max(full_robot_height - start_robot_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    limits = asset.data.joint_effort_limits.clamp_min(1.0e-6)
    ratio = torch.abs(asset.data.computed_torque) / limits
    excess = torch.relu(ratio - soft_ratio)
    return reference_gate * robot_gate * torch.mean(torch.square(excess), dim=-1)


def getup_height_delta(env: ManagerBasedRLEnv, command_name: str, dt: float) -> torch.Tensor:
    """Reward upward movement and charge back equivalent downward movement."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.clamp(command.metrics["robot_height_delta"] / max(dt, 1.0e-6), min=-1.5, max=1.5)


def getup_upright_delta(env: ManagerBasedRLEnv, command_name: str, dt: float) -> torch.Tensor:
    """Reward orientation improvement without allowing oscillation farming."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.clamp(command.metrics["robot_upright_delta"] / max(dt, 1.0e-6), min=-2.0, max=2.0)


def getup_foot_support(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 20.0,
    start_reference_height: float = 0.22,
    full_reference_height: float = 0.40,
    start_robot_height: float = 0.16,
    full_robot_height: float = 0.30,
    minimum_robot_gate: float = 0.35,
) -> torch.Tensor:
    """Reward the foot-support transition before height recovery is complete.

    Reference phase opens the reward. A small floor on the measured-height
    gate breaks the otherwise circular requirement that the robot must rise
    before it is rewarded for establishing the support needed to rise.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    contacts = forces > force_threshold
    count = contacts.float().sum(dim=1)
    support = torch.clamp(count, max=2.0) / 2.0
    reference_gate = (
        (command.anchor_pos_w[:, 2] - start_reference_height)
        / max(full_reference_height - start_reference_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    measured_gate = (
        (asset.data.root_pos_w[:, 2] - start_robot_height)
        / max(full_robot_height - start_robot_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    measured_gate = minimum_robot_gate + (1.0 - minimum_robot_gate) * measured_gate
    return reference_gate * measured_gate * support


def getup_nonfoot_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 20.0,
    phase_height: float = 0.25,
    full_phase_height: float = 0.50,
) -> torch.Tensor:
    """Penalize remaining on the torso/limbs after the recovery has started.

    Initial prone contact is expected. Gate on reference phase so remaining
    prone late in the clip cannot avoid this penalty merely by staying low.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
    nonfoot_contacts = (forces > force_threshold).float().sum(dim=1)
    gate = (
        (command.anchor_pos_w[:, 2] - phase_height) / max(full_phase_height - phase_height, 1.0e-6)
    ).clamp(0.0, 1.0)
    return gate * nonfoot_contacts


def getup_stable_stand(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_threshold: float,
    upright_threshold: float,
    linear_speed_threshold: float,
    angular_speed_threshold: float,
    require_motion_ended: bool = True,
    joint_error_threshold: float | None = None,
    body_error_threshold: float | None = None,
) -> torch.Tensor:
    """Reward a stable pose after the ordered reference has reached its end."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene["robot"]
    height_ok = asset.data.root_pos_w[:, 2] >= height_threshold
    gravity = asset.data.root_quat_w.new_tensor([0.0, 0.0, -1.0]).expand(asset.data.root_quat_w.shape[0], -1)
    robot_gravity_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, gravity)
    upright_ok = robot_gravity_b[:, 2] <= upright_threshold
    linear_ok = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1) <= linear_speed_threshold
    angular_ok = torch.linalg.norm(asset.data.root_ang_vel_w, dim=1) <= angular_speed_threshold
    reference_in_stand_phase = command.anchor_pos_w[:, 2] >= height_threshold
    stable = reference_in_stand_phase & height_ok & upright_ok & linear_ok & angular_ok
    if joint_error_threshold is not None:
        joint_error = torch.mean(torch.abs(command.joint_pos - command.robot_joint_pos), dim=-1)
        stable &= joint_error <= joint_error_threshold
    if body_error_threshold is not None:
        body_error = torch.linalg.vector_norm(
            command.body_pos_relative_w - command.robot_body_pos_w, dim=-1
        ).mean(dim=-1)
        stable &= body_error <= body_error_threshold
    if require_motion_ended:
        stable &= command.motion_ended
    return stable.float()

def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def hold_final_pose_after_motion(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from the final reference pose after the clip ends."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    
    pos_reward = pos_weight * torch.sum(
        torch.abs(asset.data.joint_pos[:, pos_cfg.joint_ids] - command.joint_pos[:, pos_cfg.joint_ids]), dim=1
    )
    vel_reward = vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1)
    
    reward = pos_reward + vel_reward
    # Only apply when motion has ended
    reward = torch.where(command.motion_ended, reward, torch.zeros_like(reward))
    
    return reward


# Backward compatibility for the existing Atom01 get-up config.
stand_still_after_motion = hold_final_pose_after_motion
