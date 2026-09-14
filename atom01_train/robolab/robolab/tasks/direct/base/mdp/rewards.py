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

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from robolab.envs.base.base_env import BaseEnv


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


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command[:, :2] - vel_yaw[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lateral_vel_y_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    return torch.square(vel_yaw[:, 1] - env.command[:, 1])


def yaw_rate_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_w[:, 2] - env.command[:, 2])


def track_ang_vel_z_world_exp(
    env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def root_height_l2(
    env: BaseEnv,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if target_height is None:
        target_height = asset.data.default_root_state[:, 2]
    reward = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def roll_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize roll independently so fast walking does not trade it for pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def roll_ang_vel_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize body-frame roll rate, the dominant oscillation at high forward speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 0])


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_torque = joint_torque * _high_kp_reward_scale(asset, asset_cfg.joint_ids, squared=False)
    reward = torch.sum(torch.abs(joint_torque * joint_vel), dim=-1)
    return reward


def joint_torques_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_scale = _high_kp_reward_scale(asset, asset_cfg.joint_ids, squared=True)
    return torch.sum(torch.square(joint_torque) * torque_scale, dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, :] - env.action_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def action_l2_selected(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.sum(torch.square(env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]), dim=1)


def action_rate_l2_selected(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]
            - env.action_buffer.buffer[:, -2, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def action_smoothness_l2(env: BaseEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -3, :] - 2*env.action_buffer.buffer[:, -2, :] + env.action_buffer.buffer[:, -1, :] 
        ),
        dim=1,
    )


def action_smoothness_l2_selected(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -3, asset_cfg.joint_ids]
            - 2 * env.action_buffer.buffer[:, -2, asset_cfg.joint_ids]
            + env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def joint_deviation_l1_deadband(
    env: BaseEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    deadband: float = 0.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    reward = torch.sum(torch.clamp(joint_error - deadband, min=0.0), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def _recovery_state(
    env: BaseEnv,
    asset: Articulation,
    tilt_threshold: float,
    height_threshold: float,
    target_height: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    if target_height is None:
        target_height = asset.data.default_root_state[:, 2]
    height_error = torch.clamp(target_height - asset.data.root_pos_w[:, 2], min=0.0)
    recovery_mask = torch.logical_or(tilt > tilt_threshold, height_error > height_threshold)
    return recovery_mask.float(), tilt, height_error


def recovery_upright_exp(
    env: BaseEnv,
    tilt_threshold: float = 0.12,
    height_threshold: float = 0.08,
    std: float = 0.35,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    recovery_mask, tilt, _ = _recovery_state(env, asset, tilt_threshold, height_threshold, target_height)
    return recovery_mask * torch.exp(-torch.square(tilt) / std**2)


def recovery_height_exp(
    env: BaseEnv,
    tilt_threshold: float = 0.12,
    height_threshold: float = 0.08,
    std: float = 0.12,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    recovery_mask, _, height_error = _recovery_state(env, asset, tilt_threshold, height_threshold, target_height)
    return recovery_mask * torch.exp(-torch.square(height_error) / std**2)


def recovery_feet_contact(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    tilt_threshold: float = 0.12,
    height_threshold: float = 0.08,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    recovery_mask, _, _ = _recovery_state(env, asset, tilt_threshold, height_threshold, target_height)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    return recovery_mask * contacts.float().mean(dim=1)


def recovery_ang_vel_xy_l2(
    env: BaseEnv,
    tilt_threshold: float = 0.12,
    height_threshold: float = 0.08,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    recovery_mask, _, _ = _recovery_state(env, asset, tilt_threshold, height_threshold, target_height)
    return recovery_mask * torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def recovery_action_l2(
    env: BaseEnv,
    tilt_threshold: float = 0.12,
    height_threshold: float = 0.08,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    recovery_mask, _, _ = _recovery_state(env, asset, tilt_threshold, height_threshold, target_height)
    return recovery_mask * torch.sum(torch.square(env.action_buffer.buffer[:, -1, :]), dim=1)


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_terminated


def stand_action_l2(env: BaseEnv, command_threshold: float = 0.1) -> torch.Tensor:
    cmd = torch.norm(env.command[:, :2], dim=1) + torch.abs(env.command[:, 2])
    standing = cmd < command_threshold
    action = env.action_buffer.buffer[:, -1, :]
    return standing.float() * torch.sum(torch.square(action), dim=1)


def stand_body_vel_l2(
    env: BaseEnv, command_threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.norm(env.command[:, :2], dim=1) + torch.abs(env.command[:, 2])
    standing = cmd < command_threshold
    body_vel = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    body_vel += torch.square(asset.data.root_ang_vel_b[:, 2])
    return standing.float() * body_vel


def feet_air_time_positive_biped(
    env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    is_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_mode_time = torch.where(is_contact, contact_time, air_time)
    single_stance = torch.sum(is_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, min=0.0, max=threshold)
    moving_command = (torch.norm(env.command[:, :2], dim=1) + torch.abs(env.command[:, 2])) >= command_threshold
    reward *= moving_command
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_flight_l2(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    return (torch.sum(contacts.int(), dim=1) == 0).float()


def feet_slide(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = torch.sum(torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=2), dim=1)
    reward = (reward - threshold).clamp(min=0.0, max=max_reward)
    return reward


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]    
    body_orientation = torch.stack(
        [
            math_utils.quat_apply_inverse(
                asset.data.body_quat_w[:, body_id, :], asset.data.GRAVITY_VEC_W
            )
            for body_id in asset_cfg.body_ids
            if body_id is not None
        ],
        dim=-1,
    )
    return torch.sum(torch.sum(torch.square(body_orientation[:, :2, :]), dim=1), dim=-1)


def feet_orientation_contact_l2(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    foot_orientation = torch.stack(
        [
            math_utils.quat_apply_inverse(
                asset.data.body_quat_w[:, body_id, :], asset.data.GRAVITY_VEC_W
            )
            for body_id in asset_cfg.body_ids
            if body_id is not None
        ],
        dim=-1,
    )
    reward = torch.sum(torch.sum(torch.square(foot_orientation[:, :2, :]), dim=1) * contacts.float(), dim=-1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_heading_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]
    foot_forward_w = math_utils.quat_apply(
        foot_quat_w,
        torch.tensor([1.0, 0.0, 0.0], device=foot_quat_w.device).expand(foot_quat_w.shape[0], foot_quat_w.shape[1], 3),
    )
    root_yaw_quat_w = math_utils.yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand_as(foot_quat_w)
    foot_forward_b = math_utils.quat_apply_inverse(root_yaw_quat_w, foot_forward_w)
    return torch.sum(torch.square(foot_forward_b[:, :, 1]), dim=1)


def feet_stumble(env: BaseEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 3 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def body_distance_y(
    env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min: float = 0.2, max: float = 0.5
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(root_quat_w, feet_pos_w - root_pos_w)
    distance = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1])
    d_min = torch.clamp(distance - min, -0.5, 0.)
    d_max = torch.clamp(distance - max, 0, 0.5)
    return torch.exp(-(torch.abs(d_min) + torch.abs(d_max)) * 100)


def body_lateral_center_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = math_utils.yaw_quat(asset.data.root_quat_w).unsqueeze(1).expand(-1, 2, -1)
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, 2, -1)
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    body_pos_b = math_utils.quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)
    center_y = torch.mean(body_pos_b[:, :, 1], dim=1)
    reward = torch.square(center_y)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(
    env: BaseEnv, sensor_cfg: SceneEntityCfg, command_threshold: float = 0.1
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    reward = (torch.sum(contacts, dim=-1) == 2).float()
    reward *= (
        torch.norm(env.command[:, :2], dim=1) + torch.abs(env.command[:, 2])
    ) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_foothold(env: BaseEnv, sensor_cfg: SceneEntityCfg, sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    undesired_contacts = torch.stack(
        [
            torch.sum(
                (env.scene[sensor.name].data.pos_w[:, 2].unsqueeze(1)
                - env.scene[sensor.name].data.ray_hits_w[..., 2]
                - ankle_height) > 0.01,
                dim=-1
            ) / float(env.scene[sensor.name].data.ray_hits_w.shape[1])
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    reward = torch.where(contacts, undesired_contacts, 0.0)
    return reward.sum(dim=1)

def upward(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reward = -asset.data.projected_gravity_b[:, 2]
    return reward


def stand_still(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command[:, :2], dim=1) + torch.abs(env.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_reward = pos_weight * torch.sum(torch.abs
        (asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1
    )
    vel_reward = vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1)
    reward = torch.where(
        torch.logical_or(cmd >= command_threshold, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(env: BaseEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg1: SceneEntityCfg | None = None,
    sensor_cfg2: SceneEntityCfg | None = None, ankle_height: float = 0.035, threshold: float = 0.05, command_threshold: float = 0.1):
    """
    Calculates reward based on the clearance of the swing leg from the ground during movement.
    Encourages appropriate lift of the feet during the swing phase of the gait.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    feet_height = torch.stack(
        [
            env.scene[sensor.name].data.pos_w[:, 2]
            - env.scene[sensor.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor in [sensor_cfg1, sensor_cfg2]
            if sensor is not None
        ],
        dim=-1,
    )
    feet_height = torch.clamp(feet_height - ankle_height, min=0.0, max=1.0)
    feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
    # Compute single_stance mask
    single_stance = contacts.sum(dim=1) == 1
    # Reward swing clearance continuously up to the target threshold.
    height_score = torch.clamp(feet_height / max(threshold, 1.0e-6), 0.0, 1.0)
    reward = torch.where(torch.logical_and(~contacts, single_stance.unsqueeze(-1)), height_score, 0.0).sum(dim=1)
    moving_command = (torch.norm(env.command[:, :2], dim=1) + torch.abs(env.command[:, 2])) >= command_threshold
    reward *= moving_command
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def joint_deviation_interrupt(env: BaseEnv, asset_cfg1: SceneEntityCfg, asset_cfg2: SceneEntityCfg, weight1: float, weight2: float) -> torch.Tensor:
    """Penalize joint deviation during interruption."""
    # extract the used quantities (to enable type-hinting)
    asset1: Articulation = env.scene[asset_cfg1.name]
    asset2: Articulation = env.scene[asset_cfg2.name]
    angle1 = asset1.data.joint_pos[:, asset_cfg1.joint_ids] - asset1.data.default_joint_pos[:, asset_cfg1.joint_ids]
    angle2 = asset2.data.joint_pos[:, asset_cfg2.joint_ids] - asset2.data.default_joint_pos[:, asset_cfg2.joint_ids]
    reward = weight1 * torch.sum(torch.abs(angle1), dim=1) + weight2 * torch.sum(torch.abs(angle2), dim=1)
    reward *= ~env.interrupt_mask
    return reward

def stand_still_interrupt(
    env: BaseEnv,
    pos_cfg: SceneEntityCfg,
    vel_cfg: SceneEntityCfg,
    interrupt_cfg: SceneEntityCfg,
    pos_weight: float = 1.0,
    vel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene["robot"]
    cmd = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    )
    body_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_ang_vel = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_ang_vel + body_lin_vel
    pos_joint_ids = list(set(pos_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    vel_joint_ids = list(set(vel_cfg.joint_ids) - set(interrupt_cfg.joint_ids))
    pos_reward = torch.where(env.interrupt_mask, 
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos[:, pos_joint_ids] - asset.data.default_joint_pos[:, pos_joint_ids]), dim=1), 
                             pos_weight * torch.sum(torch.abs(asset.data.joint_pos[:, pos_cfg.joint_ids] - asset.data.default_joint_pos[:, pos_cfg.joint_ids]), dim=1))
    vel_reward = torch.where(env.interrupt_mask, 
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_joint_ids]), dim=1), 
                             vel_weight * torch.sum(torch.abs(asset.data.joint_vel[:, vel_cfg.joint_ids]), dim=1))
    reward = torch.where(
        torch.logical_or(cmd > 0.01, body_vel > 0.5),
        0.0,
        pos_reward + vel_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def action_penalty_interrupt(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize action magnitude during interruption."""
    reward = torch.sum(
        torch.square(
            env.action_buffer.buffer[:, -1, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    reward *= env.interrupt_mask
    return reward
