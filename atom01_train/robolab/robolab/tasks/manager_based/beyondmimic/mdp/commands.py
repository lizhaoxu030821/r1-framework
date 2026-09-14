
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

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        body_indexes: Sequence[int],
        device: str = "cpu",
        joint_names: Sequence[str] | None = None,
        body_names: Sequence[str] | None = None,
    ):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        if "joint_names" in data and joint_names is not None:
            stored_joint_names = data["joint_names"].tolist()
            joint_indexes = [stored_joint_names.index(name) for name in joint_names]
            joint_pos = joint_pos[:, joint_indexes]
            joint_vel = joint_vel[:, joint_indexes]
        self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        if "ground_correction" in data:
            correction_velocity = np.gradient(data["ground_correction"].astype(np.float32)) * float(self.fps)
        else:
            correction_velocity = np.zeros(self.joint_pos.shape[0], dtype=np.float32)
        self.ground_correction_velocity = torch.tensor(
            correction_velocity, dtype=torch.float32, device=device
        )
        if "body_names" in data and body_names is not None:
            stored_body_names = data["body_names"].tolist()
            self._body_indexes = [stored_body_names.index(name) for name in body_names]
        else:
            self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(
            self.cfg.motion_file,
            self.body_indexes,
            device=self.device,
            joint_names=self.robot.joint_names,
            body_names=self.cfg.body_names,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.playback_speeds = torch.full(
            (self.num_envs,), float(self.cfg.playback_speed), dtype=torch.float32, device=self.device
        )
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        # Expose physical get-up state directly to W&B as command metrics.
        self.metrics["robot_base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_upright"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_height_delta"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_upright_delta"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["robot_min_body_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_min_body_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["max_effort_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["effort_saturation_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frame_zero_robot_base_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frame_zero_robot_upright"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frame_zero_error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frame_zero_physical_success"] = torch.zeros(self.num_envs, device=self.device)
        self._last_robot_height = torch.zeros(self.num_envs, device=self.device)
        self._last_robot_upright = torch.zeros(self.num_envs, device=self.device)
        self._started_at_frame_zero = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_start_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._has_completed_episode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)

        # Track whether motion has ended for each env (used for blend_to_default)
        self.motion_ended = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_after_motion_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["frame_zero_reset_ratio"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["playback_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["playback_speed_min"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["playback_speed_max"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_min_frame"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_max_frame"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def playback_speed(self) -> float:
        """Return the legacy scalar curriculum speed or configured base speed."""
        speed = self.cfg.playback_speed
        current_step = int(self._env.common_step_counter)
        for milestone, scheduled_speed in self.cfg.playback_speed_schedule:
            if current_step < milestone:
                break
            speed = scheduled_speed
        return speed

    @property
    def playback_speed_range(self) -> tuple[float, float]:
        """Return the active per-episode playback-speed range."""
        if not self.cfg.playback_speed_range_schedule:
            speed = float(self.playback_speed)
            return speed, speed
        current_step = int(self._env.common_step_counter)
        minimum, maximum = self.cfg.playback_speed_range_schedule[0][1:]
        for milestone, scheduled_minimum, scheduled_maximum in self.cfg.playback_speed_range_schedule:
            if current_step < milestone:
                break
            minimum, maximum = scheduled_minimum, scheduled_maximum
        minimum = float(minimum)
        maximum = float(maximum)
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError(f"Invalid playback-speed range: {(minimum, maximum)}")
        return minimum, maximum

    @property
    def start_frame_range(self) -> tuple[int, int] | None:
        """Active backward-curriculum reset range, or None for adaptive sampling."""
        if not self.cfg.start_frame_range_schedule:
            return None
        current_step = int(self._env.common_step_counter)
        minimum, maximum = self.cfg.start_frame_range_schedule[0][1:]
        for milestone, scheduled_minimum, scheduled_maximum in self.cfg.start_frame_range_schedule:
            if current_step < milestone:
                break
            minimum, maximum = scheduled_minimum, scheduled_maximum
        end_frame = self.cfg.motion_end_frame if self.cfg.motion_end_frame >= 0 else self.motion.time_step_total - 1
        end_frame = min(end_frame, self.motion.time_step_total - 1)
        minimum = max(0, min(int(minimum), end_frame))
        maximum = max(minimum, min(int(maximum), end_frame))
        return minimum, maximum

    @property
    def target_index(self) -> int:
        return self.motion.time_step_total - 1

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        index = self.time_steps if not self.cfg.goal_only else self.target_index
        return self.motion.joint_pos[index]

    @property
    def joint_vel(self) -> torch.Tensor:
        if self.cfg.goal_only:
            return torch.zeros_like(self.motion.joint_vel[0])
        velocity = self.motion.joint_vel[self.time_steps] * self.playback_speeds[:, None]
        return torch.where(self._reference_ended[:, None], torch.zeros_like(velocity), velocity)

    @property
    def body_pos_w(self) -> torch.Tensor:
        index = self.time_steps if not self.cfg.goal_only else self.target_index
        return self.motion.body_pos_w[index] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        index = self.time_steps if not self.cfg.goal_only else self.target_index
        return self.motion.body_quat_w[index]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        if self.cfg.goal_only:
            return torch.zeros_like(self.motion.body_lin_vel_w[0])
        velocity = self.motion.body_lin_vel_w[self.time_steps] * self.playback_speeds[:, None, None]
        return torch.where(self._reference_ended[:, None, None], torch.zeros_like(velocity), velocity)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        if self.cfg.goal_only:
            return torch.zeros_like(self.motion.body_ang_vel_w[0])
        velocity = self.motion.body_ang_vel_w[self.time_steps] * self.playback_speeds[:, None, None]
        return torch.where(self._reference_ended[:, None, None], torch.zeros_like(velocity), velocity)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        index = self.target_index if self.cfg.goal_only else self.time_steps
        return self.motion.body_pos_w[index, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        index = self.target_index if self.cfg.goal_only else self.time_steps
        return self.motion.body_quat_w[index, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        if self.cfg.goal_only:
            return torch.zeros_like(self.motion.body_lin_vel_w[0, self.motion_anchor_body_index])
        velocity = (
            self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]
            * self.playback_speeds[:, None]
        )
        return torch.where(self._reference_ended[:, None], torch.zeros_like(velocity), velocity)

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        if self.cfg.goal_only:
            return torch.zeros_like(self.motion.body_ang_vel_w[0, self.motion_anchor_body_index])
        velocity = (
            self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]
            * self.playback_speeds[:, None]
        )
        return torch.where(self._reference_ended[:, None], torch.zeros_like(velocity), velocity)

    @property
    def _reference_ended(self) -> torch.Tensor:
        """Return environments whose reference is at its configured terminal frame."""
        end_frame = self.cfg.motion_end_frame if self.cfg.motion_end_frame >= 0 else self.motion.time_step_total - 1
        return self.time_steps >= min(end_frame, self.motion.time_step_total - 1)

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        current_height = self.robot_anchor_pos_w[:, 2]
        self.metrics["robot_base_height"] = current_height
        self.metrics["reference_base_height"] = self.anchor_pos_w[:, 2]
        gravity = self.robot_anchor_quat_w.new_tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1)
        robot_gravity_b = quat_apply_inverse(self.robot_anchor_quat_w, gravity)
        current_upright = -robot_gravity_b[:, 2]
        self.metrics["robot_upright"] = current_upright
        self.metrics["robot_height_delta"] = current_height - self._last_robot_height
        self.metrics["robot_upright_delta"] = current_upright - self._last_robot_upright
        self.metrics["robot_min_body_height"] = self.robot.data.body_pos_w[:, :, 2].amin(dim=1)
        self.metrics["reference_min_body_height"] = self.body_pos_w[:, :, 2].amin(dim=1)
        actuator_effort_limits = torch.empty_like(self.robot.data.applied_torque)
        for actuator in self.robot.actuators.values():
            actuator_effort_limits[:, actuator.joint_indices] = actuator.effort_limit
        effort_ratio = torch.abs(self.robot.data.applied_torque) / actuator_effort_limits.clamp_min(1.0e-6)
        self.metrics["max_effort_ratio"] = effort_ratio.amax(dim=1)
        self.metrics["effort_saturation_fraction"] = (effort_ratio >= 0.98).float().mean(dim=1)

        # Report conditional frame-zero metrics as broadcast scalars so the
        # manager's environment averaging does not dilute them with random-
        # phase episodes. These diagnostics never enter the actor observation.
        frame_zero = self._started_at_frame_zero
        frame_zero_count = frame_zero.float().sum().clamp_min(1.0)

        def frame_zero_mean(values: torch.Tensor) -> torch.Tensor:
            return (values * frame_zero).sum() / frame_zero_count

        self.metrics["frame_zero_robot_base_height"][:] = frame_zero_mean(current_height)
        self.metrics["frame_zero_robot_upright"][:] = frame_zero_mean(current_upright)
        self.metrics["frame_zero_error_joint_pos"][:] = frame_zero_mean(self.metrics["error_joint_pos"])
        joint_speed = torch.linalg.vector_norm(self.robot_joint_vel, dim=-1)
        physical_success = (
            self.motion_ended
            & (current_height > 0.60)
            & (current_upright > 0.80)
            & (joint_speed < 2.0)
        ).float()
        self.metrics["frame_zero_physical_success"][:] = frame_zero_mean(physical_success)
        self.metrics["frame_zero_reset_ratio"][:] = frame_zero.float().mean()
        self.metrics["playback_speed"][:] = self.playback_speeds.mean()
        self.metrics["playback_speed_min"][:] = self.playback_speeds.amin()
        self.metrics["playback_speed_max"][:] = self.playback_speeds.amax()
        active_range = self.start_frame_range
        if active_range is None:
            self.metrics["sampling_min_frame"][:] = 0.0
            self.metrics["sampling_max_frame"][:] = float(self.motion.time_step_total - 1)
        else:
            self.metrics["sampling_min_frame"][:] = float(active_range[0])
            self.metrics["sampling_max_frame"][:] = float(active_range[1])
        self._last_robot_height[:] = current_height
        self._last_robot_upright[:] = current_upright

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # Attribute the previous episode's outcome to its sampled start phase.
        # Timeout-only tasks do not populate termination_manager.terminated,
        # and the current time step at reset is normally the terminal frame.
        completed = self._has_completed_episode[env_ids]
        if torch.any(completed):
            completed_env_ids = env_ids[completed]
            start_steps = self._episode_start_steps[completed_env_ids]
            start_bins = torch.clamp(
                (start_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            gravity = self.robot_anchor_quat_w.new_tensor([0.0, 0.0, -1.0]).expand(len(completed_env_ids), -1)
            robot_gravity_b = quat_apply_inverse(self.robot_anchor_quat_w[completed_env_ids], gravity)
            upright = -robot_gravity_b[:, 2]
            joint_speed = torch.linalg.vector_norm(self.robot_joint_vel[completed_env_ids], dim=-1)
            succeeded = (
                self.motion_ended[completed_env_ids]
                & (self.robot_anchor_pos_w[completed_env_ids, 2] > 0.60)
                & (upright > 0.80)
                & (joint_speed < 2.0)
            )
            attempts = torch.bincount(start_bins, minlength=self.bin_count).float()
            failures = torch.bincount(start_bins[~succeeded], minlength=self.bin_count).float()
            observed = attempts > 0
            failure_rate = failures[observed] / attempts[observed]
            self.bin_failed_count[observed] = (
                self.cfg.adaptive_alpha * failure_rate
                + (1.0 - self.cfg.adaptive_alpha) * self.bin_failed_count[observed]
            )

        if self.cfg.start_at_frame_zero:
            self.time_steps[env_ids] = 0
            self.phase_steps[env_ids] = 0.0
            self._started_at_frame_zero[env_ids] = True
            self._episode_start_steps[env_ids] = 0
            self._has_completed_episode[env_ids] = True
            self.metrics["sampling_entropy"][env_ids] = 0.0
            self.metrics["sampling_top1_prob"][env_ids] = 1.0
            self.metrics["sampling_top1_bin"][env_ids] = 0.0
            self.metrics["frame_zero_reset_ratio"][env_ids] = 1.0
            return
        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        sampled_steps = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        active_range = self.start_frame_range
        if active_range is not None:
            minimum, maximum = active_range
            sampled_steps = torch.randint(
                minimum,
                maximum + 1,
                (len(env_ids),),
                device=self.device,
            )
        frame_zero_ratio = 1.0 if self.cfg.start_at_frame_zero else self.cfg.frame_zero_ratio
        if not self.cfg.start_at_frame_zero and self.cfg.frame_zero_ratio_schedule:
            current_step = int(self._env.common_step_counter)
            for milestone, scheduled_ratio in self.cfg.frame_zero_ratio_schedule:
                if current_step >= milestone:
                    frame_zero_ratio = scheduled_ratio
                else:
                    break
        zero_mask = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device) < frame_zero_ratio
        self.time_steps[env_ids] = torch.where(zero_mask, torch.zeros_like(sampled_steps), sampled_steps)
        self.phase_steps[env_ids] = self.time_steps[env_ids].float()
        self._started_at_frame_zero[env_ids] = zero_mask
        self._episode_start_steps[env_ids] = self.time_steps[env_ids]
        self._has_completed_episode[env_ids] = True

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        if len(env_ids):
            self.metrics["frame_zero_reset_ratio"][env_ids] = zero_mask.float().mean()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)
        self._sample_playback_speeds(env_ids)
        # Reset motion_ended state for resampled envs
        self.motion_ended[env_ids] = False
        self.steps_after_motion_end[env_ids] = 0

        # Keep reset state and sampled command on the same reference frame. A
        # phase-sampled episode must start from the sampled root pose as well
        # as the sampled joints; mixing a late-frame joint pose with frame-0
        # root state creates an impossible penetration/teleport state.
        reset_indices = self.time_steps
        root_pos = self.motion.body_pos_w[reset_indices, 0] + self._env.scene.env_origins
        root_ori = self.motion.body_quat_w[reset_indices, 0]
        playback_speed = self.playback_speeds
        root_lin_vel = self.motion.body_lin_vel_w[reset_indices, 0] * playback_speed[:, None]
        root_ang_vel = self.motion.body_ang_vel_w[reset_indices, 0] * playback_speed[:, None]
        if self.cfg.remove_grounding_vertical_velocity_on_reset:
            root_lin_vel[:, 2] -= (
                self.motion.ground_correction_velocity[reset_indices] * playback_speed
            )
        if self.cfg.phase_reset_height_offset != 0.0:
            phase = reset_indices.float() / max(self.motion.time_step_total - 1, 1)
            root_pos[env_ids, 2] += self.cfg.phase_reset_height_offset * phase[env_ids]

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        if self.cfg.goal_only:
            joint_pos = self.motion.joint_pos[0].repeat(self.num_envs, 1)
            joint_vel = self.motion.joint_vel[0].repeat(self.num_envs, 1)
        else:
            joint_pos = self.joint_pos.clone()
            joint_vel = self.joint_vel.clone()

        if self.cfg.zero_velocity_at_frame_zero:
            frame_zero_ids = env_ids[self._started_at_frame_zero[env_ids]]
            root_lin_vel[frame_zero_ids] = 0.0
            root_ang_vel[frame_zero_ids] = 0.0
            joint_vel[frame_zero_ids] = 0.0
        if self.cfg.zero_velocity_after_frame >= 0:
            settled_ids = env_ids[self.time_steps[env_ids] >= self.cfg.zero_velocity_after_frame]
            root_lin_vel[settled_ids] = 0.0
            root_ang_vel[settled_ids] = 0.0
            joint_vel[settled_ids] = 0.0

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        self._last_robot_height[env_ids] = root_pos[env_ids, 2]
        gravity = root_ori[env_ids].new_tensor([0.0, 0.0, -1.0]).expand(len(env_ids), -1)
        root_gravity = quat_apply_inverse(root_ori[env_ids], gravity)
        self._last_robot_upright[env_ids] = -root_gravity[:, 2]
        self._update_action_offset(env_ids)

    def _sample_playback_speeds(self, env_ids: Sequence[int]) -> None:
        """Sample one fixed speed per episode while retaining base-speed rehearsal."""
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids) == 0:
            return
        minimum, maximum = self.playback_speed_range
        if maximum <= minimum:
            self.playback_speeds[env_ids] = minimum
            return
        samples = sample_uniform(minimum, maximum, (len(env_ids),), device=self.device)
        selector = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        base_ratio = float(self.cfg.playback_speed_base_ratio)
        maximum_ratio = float(self.cfg.playback_speed_max_ratio)
        if base_ratio < 0.0 or maximum_ratio < 0.0 or base_ratio + maximum_ratio > 1.0:
            raise ValueError(
                "playback_speed_base_ratio and playback_speed_max_ratio must be non-negative "
                "and sum to at most 1.0"
            )
        if not minimum <= float(self.cfg.playback_speed) <= maximum:
            raise ValueError(
                f"Base playback speed {self.cfg.playback_speed} is outside active range {(minimum, maximum)}"
            )
        samples[selector < base_ratio] = float(self.cfg.playback_speed)
        maximum_mask = (selector >= base_ratio) & (selector < base_ratio + maximum_ratio)
        samples[maximum_mask] = maximum
        self.playback_speeds[env_ids] = samples

    def _update_action_offset(self, env_ids: Sequence[int] | None = None):
        """Center residual joint actions on the current reference frame."""
        if not self.cfg.use_reference_joint_offset:
            return
        action_term = self._env.action_manager.get_term("joint_pos")
        if not isinstance(action_term._offset, torch.Tensor):
            action_term._offset = torch.zeros_like(action_term.raw_actions)
        action_term._offset[env_ids] = self.joint_pos[env_ids]

    def _update_command(self):
        self.phase_steps += self.playback_speeds
        self.time_steps = torch.floor(self.phase_steps).long()
        end_frame = self.cfg.motion_end_frame if self.cfg.motion_end_frame >= 0 else self.motion.time_step_total - 1
        end_frame = min(end_frame, self.motion.time_step_total - 1)
        env_ids = torch.where(self.time_steps >= end_frame)[0]

        if self.cfg.reset_on_motion_end:
            # Original behavior: reset envs when motion ends
            self._resample_command(env_ids)
        else:
            # New behavior: clamp to last frame and track motion_ended state
            self.time_steps = torch.clamp(self.time_steps, max=end_frame)
            self.phase_steps = torch.clamp(self.phase_steps, max=float(end_frame))
            self.motion_ended[env_ids] = True
            self.steps_after_motion_end[self.motion_ended] += 1

        self._update_action_offset()

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        # Retargeted root translation is only a kinematic visualization path:
        # mesh grounding shifts it independently at every frame. Align the
        # complete reference skeleton to the measured robot anchor so body
        # rewards describe pose shape, not an infeasible floating-base path.
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    start_at_frame_zero: bool = False
    frame_zero_ratio: float = 0.0
    frame_zero_ratio_schedule: tuple[tuple[int, float], ...] = ()
    use_reference_joint_offset: bool = False
    zero_velocity_at_frame_zero: bool = False
    remove_grounding_vertical_velocity_on_reset: bool = False
    phase_reset_height_offset: float = 0.0
    """Vertical reset correction ramped from zero at frame 0 to this value at the final frame."""
    zero_velocity_after_frame: int = -1
    """Set root and joint reset velocities to zero at or after this reference frame."""
    playback_speed: float = 1.0
    playback_speed_schedule: tuple[tuple[int, float], ...] = ()
    playback_speed_range_schedule: tuple[tuple[int, float, float], ...] = ()
    """Environment-step milestones containing (step, minimum_speed, maximum_speed)."""
    playback_speed_base_ratio: float = 0.0
    """Fraction of new episodes fixed at playback_speed for anti-forgetting rehearsal."""
    playback_speed_max_ratio: float = 0.0
    """Fraction of new episodes fixed at the active maximum speed."""
    start_frame_range_schedule: tuple[tuple[int, int, int], ...] = ()
    """Environment-step milestones containing (step, minimum_frame, maximum_frame)."""
    goal_only: bool = False
    motion_end_frame: int = -1

    reset_on_motion_end: bool = True
    """Whether to reset the environment when motion ends. Default is True.
    If False, the motion will stay at the last frame and motion_ended flag will be set."""

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
