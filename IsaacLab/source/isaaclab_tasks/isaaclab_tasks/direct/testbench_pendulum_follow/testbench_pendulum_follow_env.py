# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections.abc import Sequence
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass


@configclass
class TestbenchPendulumFollowEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 10.0
    action_space = 1
    observation_space = 5
    state_space = 0
    action_scale = 0.1

    sim: SimulationCfg = SimulationCfg(dt=1 / 200)

    _urdf_path = "/home/a/lzx/ceshitai1/ce shi tai.stp_fuzhangliang.SLDASM/urdf/ce shi tai.stp_fuzhangliang.SLDASM.urdf"
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_urdf_path,
            fix_base=True,
            self_collision=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=800.0,
                    damping=80.0,
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.8),
            joint_pos={"link1": 0.0, "square": 0.0},
        ),
        actuators={
            "link_actuator": ImplicitActuatorCfg(
                joint_names_expr=["link1"],
                stiffness=800.0,
                damping=80.0,
            ),
            "square_actuator": ImplicitActuatorCfg(
                joint_names_expr=["square"],
                stiffness=10000.0,
                damping=1000.0,
            ),
        },
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3.0)

class TestbenchPendulumFollowEnv(DirectRLEnv):
    cfg: TestbenchPendulumFollowEnvCfg
    def __init__(self, cfg: TestbenchPendulumFollowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._link_dof_idx, _ = self.robot.find_joints("link1")
        self.last_actions = torch.zeros(self.num_envs, 1, device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.articulations["robot"] = self.robot

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self):
        self.robot.set_joint_position_target(self.actions, joint_ids=self._link_dof_idx)

    # ===================== 【终极修复！！！完美 shape (256,5)】=====================
    def _get_observations(self):
        link_pos = self.robot.data.joint_pos[:, self._link_dof_idx]
        link_vel = self.robot.data.joint_vel[:, self._link_dof_idx]
        target_pos = torch.zeros_like(link_pos)
        target_vel = torch.zeros_like(link_vel)
        error_pos = link_pos - target_pos

        obs = torch.cat([link_pos, link_vel, target_pos, target_vel, error_pos], dim=-1)
        # ✅ 【只加这一行！强制变成 (256,5)】
        obs = obs.view(-1, 5)
        return {"policy": obs}

    # ===================== 【奖励：让电机自发正弦运动，你的最终目的！】=====================
    def _get_rewards(self):
        reward = torch.ones(self.num_envs, device=self.device) * 0.8
        reward += 0.4 * torch.tanh(torch.abs(self.robot.data.joint_vel[:, self._link_dof_idx].squeeze()))
        reward -= 0.05 * torch.abs(self.actions.squeeze() - self.last_actions.squeeze())
        self.last_actions = self.actions.clone()
        return reward

    # ===================== 【永不死亡】=====================
    def _get_dones(self):
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self.last_actions[env_ids] = 0.0

