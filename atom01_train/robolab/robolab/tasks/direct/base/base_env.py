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

import numpy as np
import torch
from collections.abc import Sequence
from isaaclab.envs import DirectRLEnv
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.sim as sim_utils

from .base_config import BaseEnvCfg


class DirectionalVelocityCommand(UniformVelocityCommand):
    """Sample directional commands with an optional high-speed forward focus."""

    def __init__(
        self,
        cfg,
        env,
        mode_probabilities: tuple,
        velocity_curriculum: tuple,
        forward_speed_focus_curriculum: tuple,
        min_speed: float,
    ):
        self._validate_mode_probabilities(mode_probabilities)
        previous_step = -1
        for stage in velocity_curriculum:
            if len(stage) != 5:
                raise ValueError("velocity_curriculum stages must contain step, vx, vy, yaw, and mode probabilities")
            if stage[0] <= previous_step:
                raise ValueError("velocity_curriculum steps must be strictly increasing")
            self._validate_mode_probabilities(stage[4])
            previous_step = stage[0]
        self.mode_probabilities = mode_probabilities
        self.velocity_curriculum = velocity_curriculum
        if forward_speed_focus_curriculum:
            if len(forward_speed_focus_curriculum) != len(velocity_curriculum):
                raise ValueError("forward_speed_focus_curriculum must match velocity_curriculum stages")
            for index, focus_stage in enumerate(forward_speed_focus_curriculum):
                if len(focus_stage) != 3:
                    raise ValueError("forward_speed_focus_curriculum stages must contain step, probability, and minimum_vx")
                if focus_stage[0] != velocity_curriculum[index][0]:
                    raise ValueError("forward_speed_focus_curriculum steps must match velocity_curriculum")
                if not 0.0 <= focus_stage[1] <= 1.0:
                    raise ValueError("forward speed focus probability must be in [0, 1]")
                if focus_stage[2] < 0.0:
                    raise ValueError("forward speed focus minimum_vx must be non-negative")
        self.forward_speed_focus_curriculum = forward_speed_focus_curriculum
        self.curriculum_env = env
        self.min_speed = min_speed
        super().__init__(cfg=cfg, env=env)

    @staticmethod
    def _validate_mode_probabilities(probabilities: tuple) -> None:
        if len(probabilities) != 4:
            raise ValueError("mode probabilities must contain forward, backward, lateral, and turn")
        if any(probability < 0.0 for probability in probabilities) or sum(probabilities) > 1.0:
            raise ValueError("mode probabilities must be non-negative and sum to at most 1.0")

    @property
    def curriculum_phase(self) -> int:
        phase = 0
        for index, stage in enumerate(self.velocity_curriculum):
            if self.curriculum_env.common_step_counter < stage[0]:
                break
            phase = index
        return phase

    def _active_sampling_config(self):
        if not self.velocity_curriculum:
            ranges = (self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)
            return ranges, self.mode_probabilities, (0.0, 0.0)
        stage = self.velocity_curriculum[self.curriculum_phase]
        if self.forward_speed_focus_curriculum:
            focus_stage = self.forward_speed_focus_curriculum[self.curriculum_phase]
            return stage[1:4], stage[4], focus_stage[1:]
        return stage[1:4], stage[4], (0.0, 0.0)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ranges, mode_probabilities, forward_focus = self._active_sampling_config()
        if self.velocity_curriculum:
            random_values = torch.empty(len(env_ids), device=self.device)
            for command_index, command_range in enumerate(ranges):
                self.vel_command_b[env_ids, command_index] = random_values.uniform_(*command_range)
            self.is_heading_env[env_ids] = False
            self.is_standing_env[env_ids] = random_values.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
        else:
            super()._resample_command(env_ids)

        probabilities = torch.tensor(
            (*mode_probabilities, 1.0 - sum(mode_probabilities)),
            device=self.device,
        )
        modes = torch.multinomial(probabilities, len(env_ids), replacement=True)
        random_values = torch.empty(len(env_ids), device=self.device)

        forward = modes == 0
        backward = modes == 1
        lateral = modes == 2
        turn = modes == 3

        if torch.any(forward):
            forward_ids = env_ids[forward]
            lower = max(self.min_speed, ranges[0][0])
            self.vel_command_b[forward_ids, 0] = random_values[forward].uniform_(lower, ranges[0][1])
            focus_probability, focus_min_vx = forward_focus
            if focus_probability > 0.0:
                focus_mask = torch.rand(len(forward_ids), device=self.device) < focus_probability
                if torch.any(focus_mask):
                    focus_lower = max(lower, min(focus_min_vx, ranges[0][1]))
                    self.vel_command_b[forward_ids[focus_mask], 0] = torch.empty(
                        int(focus_mask.sum().item()), device=self.device
                    ).uniform_(focus_lower, ranges[0][1])
            self.vel_command_b[forward_ids, 1:] = 0.0
        if torch.any(backward):
            backward_ids = env_ids[backward]
            upper = min(-self.min_speed, ranges[0][1])
            self.vel_command_b[backward_ids, 0] = random_values[backward].uniform_(ranges[0][0], upper)
            self.vel_command_b[backward_ids, 1:] = 0.0
        if torch.any(lateral):
            lateral_ids = env_ids[lateral]
            self.vel_command_b[lateral_ids, 0] = 0.0
            self.vel_command_b[lateral_ids, 2] = 0.0
        if torch.any(turn):
            turn_ids = env_ids[turn]
            self.vel_command_b[turn_ids, :2] = 0.0


class BaseEnv(DirectRLEnv):
    cfg: BaseEnvCfg
    def __init__(self, cfg: BaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.reward_manager = RewardManager(self.cfg.reward, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        self.left_feet_scanner_cfg = SceneEntityCfg("left_feet_scanner")
        self.right_feet_scanner_cfg = SceneEntityCfg("right_feet_scanner")

        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        if self.cfg.commands.directional_mode_probabilities:
            self.command_generator = DirectionalVelocityCommand(
                cfg=command_cfg,
                env=self,
                mode_probabilities=self.cfg.commands.directional_mode_probabilities,
                velocity_curriculum=self.cfg.commands.velocity_curriculum,
                forward_speed_focus_curriculum=self.cfg.commands.forward_speed_focus_curriculum,
                min_speed=self.cfg.commands.deadzone_threshold,
            )
        else:
            self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)

        self.init_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self._reset_idx(env_ids)

    @property
    def command(self) -> torch.Tensor:
        command = self._discretize_walking_command(self.command_generator.command)
        command = command * self._startup_command_ramp()
        deadzone = max(self.cfg.commands.deadzone_threshold, 0.0)
        if deadzone > 0.0:
            command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            command = torch.where(command_norm.unsqueeze(-1) < deadzone, torch.zeros_like(command), command)
        return command

    def _discretize_walking_command(self, command: torch.Tensor) -> torch.Tensor:
        if self.walking_lin_vel_x_targets.numel() == 0:
            return command
        command_x = command[:, 0]
        deadzone = max(self.cfg.commands.deadzone_threshold, 0.0)
        moving = torch.abs(command_x) >= deadzone
        distances = torch.abs(
            torch.abs(command_x).unsqueeze(-1) - torch.abs(self.walking_lin_vel_x_targets).unsqueeze(0)
        )
        target_x = self.walking_lin_vel_x_targets[torch.argmin(distances, dim=1)]
        command = command.clone()
        command[:, 0] = torch.where(moving, torch.sign(command_x) * target_x, command_x)
        return command

    def _startup_command_ramp(self) -> torch.Tensor:
        stand_time = self.startup_stand_time
        ramp_time = torch.clamp(self.startup_ramp_time, min=1.0e-6)
        episode_time = self.episode_length_buf.float() * self.step_dt
        ramp = torch.clamp((episode_time - stand_time) / ramp_time, min=0.0, max=1.0)
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        return ramp.unsqueeze(-1)

    def _sample_startup_times(self, env_ids: Sequence[int]) -> None:
        stand_min, stand_max = self.cfg.commands.startup_stand_time_range
        ramp_min, ramp_max = self.cfg.commands.startup_ramp_time_range
        stand_min = max(float(stand_min), 0.0)
        stand_max = max(float(stand_max), stand_min)
        ramp_min = max(float(ramp_min), 1.0e-6)
        ramp_max = max(float(ramp_max), ramp_min)
        num_resets = len(env_ids)
        self.startup_stand_time[env_ids] = stand_min + (stand_max - stand_min) * torch.rand(
            num_resets, device=self.device
        )
        self.startup_ramp_time[env_ids] = ramp_min + (ramp_max - ramp_min) * torch.rand(
            num_resets, device=self.device
        )

    def init_buffers(self):
        self.extras = {}

        self.episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = torch.full(
            (self.num_actions,),
            self.cfg.robot.action_scale,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.cfg.robot.action_scale_multipliers is not None:
            joint_names = list(self.robot.data.joint_names)
            for joint_name, multiplier in self.cfg.robot.action_scale_multipliers.items():
                if joint_name not in joint_names:
                    raise ValueError(f"Unknown action scale joint name: {joint_name}")
                self.action_scale[joint_names.index(joint_name)] *= multiplier
        stand_min = max(float(self.cfg.commands.startup_stand_time_range[0]), 0.0)
        ramp_min = max(float(self.cfg.commands.startup_ramp_time_range[0]), 1.0e-6)
        self.startup_stand_time = torch.full((self.num_envs,), stand_min, dtype=torch.float, device=self.device)
        self.startup_ramp_time = torch.full((self.num_envs,), ramp_min, dtype=torch.float, device=self.device)
        self.walking_lin_vel_x_targets = torch.tensor(
            self.cfg.commands.walking_lin_vel_x_targets,
            dtype=torch.float,
            device=self.device,
        )
        self.action_buffer = CircularBuffer(
            max_len=self.cfg.robot.action_history_length, batch_size=self.num_envs, device=self.device
        )
        self.action_buffer.append(torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        self.imu_delay_buffer = None
        if self.cfg.noise.imu_delay_max_steps > 0:
            self.imu_delay_buffer = DelayBuffer(
                self.cfg.noise.imu_delay_max_steps,
                batch_size=self.num_envs,
                device=self.device,
            )
            imu_time_lags = torch.randint(
                self.cfg.noise.imu_delay_min_steps,
                self.cfg.noise.imu_delay_max_steps + 1,
                (self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.imu_delay_buffer.set_time_lag(imu_time_lags)

        self.init_obs_buffer()

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            self.noise_scale_vec = noise_vec

            if self.cfg.scene_context.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
                height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
                height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
                height_scan *= self.obs_scales.height_scan
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        if self.imu_delay_buffer is not None:
            imu_obs = self.imu_delay_buffer.compute(torch.cat([ang_vel, projected_gravity], dim=-1))
            ang_vel = imu_obs[:, :3]
            projected_gravity = imu_obs[:, 3:6]
        command = self.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer.buffer[:, -1, :]
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,
                projected_gravity * self.obs_scales.projected_gravity,
                command * self.obs_scales.commands,
                joint_pos * self.obs_scales.joint_pos,
                joint_vel * self.obs_scales.joint_vel,
                action * self.obs_scales.actions,
            ],
            dim=-1,
        )

        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
        feet_contact_force = self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :]
        feet_air_time = self.contact_sensor.data.current_air_time[:, self.feet_cfg.body_ids]
        feet_height = torch.stack(
        [
            self.scene[sensor_cfg.name].data.pos_w[:, 2]
            - self.scene[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=-1)
            for sensor_cfg in [self.left_feet_scanner_cfg, self.right_feet_scanner_cfg]
            if sensor_cfg is not None
        ],
        dim=-1,
        )
        feet_height = torch.clamp(feet_height - 0.04, min=0.0, max=1.0)
        feet_height = torch.nan_to_num(feet_height, nan=1.0, posinf=1.0, neginf=0)
        joint_torque = robot.data.applied_torque
        joint_acc = robot.data.joint_acc
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact.float(), feet_contact_force.flatten(1), feet_air_time.flatten(1), feet_height.flatten(1), joint_acc, joint_torque], dim=-1
        )
        
        return current_actor_obs, current_critic_obs


    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device)

        self._pre_physics_step(actions)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        self.obs_buf = self._get_observations()

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.cfg.scene_context.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

    def _pre_physics_step(self, actions: torch.Tensor):
        self.action_buffer.append(actions)
        self.actions = actions.clone()
        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)
        self.actions = self.actions * self.action_scale + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        if self.cfg.scene_context.height_scanner.enable_height_scan:
            height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                )
            height_scan = torch.clamp(height_scan - self.cfg.normalization.height_scan_offset, min=-1.0, max=1.0)
            height_scan = torch.nan_to_num(height_scan, nan=1.0, posinf=1.0, neginf=-1.0)
            height_scan *= self.obs_scales.height_scan
            current_critic_obs = torch.cat([current_critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            if self.cfg.scene_context.height_scanner.enable_height_scan_actor:
                current_actor_obs = torch.cat([current_actor_obs, height_scan], dim=-1)

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        observations = {"policy": actor_obs, "critic":critic_obs}
        return observations
    
    def _get_rewards(self):
        return self.reward_manager.compute(dt=self.step_dt)
    
    def _get_dones(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        if self.cfg.robot.terminate_contacts_body_names is not None:
            terminated_buf = torch.any(
                torch.max(
                    torch.norm(
                        net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
        if self.cfg.robot.terminate_base_orientation is not None:
            terminated_buf |= torch.acos(-self.robot.data.projected_gravity_b[:, 2]).abs() > self.cfg.robot.terminate_base_orientation
        if self.cfg.robot.terminate_base_height is not None:
            terminated_buf |= self.robot.data.root_pos_w[:, 2] < self.cfg.robot.terminate_base_height
        time_out_buf = self.episode_length_buf >= self.episode_length
        return terminated_buf, time_out_buf

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if len(env_ids) == 0:
            return
        
        if self.cfg.scene_context.height_scanner.enable_height_scan:
            self.height_scanner.reset(env_ids)

        self.extras["log"] = dict()
        if self.cfg.scene_context.terrain_generator is not None:
            if self.cfg.scene_context.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self._sim_step_counter // self.cfg.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.reset_time_outs

        self.command_generator.reset(env_ids)
        if isinstance(self.command_generator, DirectionalVelocityCommand) and self.command_generator.velocity_curriculum:
            ranges, _, forward_focus = self.command_generator._active_sampling_config()
            self.extras["log"].update(
                {
                    "Curriculum/velocity_phase": float(self.command_generator.curriculum_phase),
                    "Curriculum/max_vx": float(ranges[0][1]),
                    "Curriculum/max_abs_vy": float(max(abs(ranges[1][0]), abs(ranges[1][1]))),
                    "Curriculum/max_abs_yaw": float(max(abs(ranges[2][0]), abs(ranges[2][1]))),
                    "Curriculum/high_speed_forward_probability": float(forward_focus[0]),
                    "Curriculum/high_speed_forward_min_vx": float(forward_focus[1]),
                }
            )
        self._sample_startup_times(env_ids)
        if self.imu_delay_buffer is not None:
            imu_time_lags = torch.randint(
                self.cfg.noise.imu_delay_min_steps,
                self.cfg.noise.imu_delay_max_steps + 1,
                (len(env_ids),),
                dtype=torch.int,
                device=self.device,
            )
            self.imu_delay_buffer.set_time_lag(imu_time_lags, env_ids)
            self.imu_delay_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)

        self.scene.write_data_to_sim()
        self.sim.forward()
