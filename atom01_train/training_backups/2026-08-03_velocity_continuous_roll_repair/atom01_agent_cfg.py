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

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)
import torch
from functools import lru_cache

from robolab.tasks.direct.base import (  # noqa:F401
    BaseAgentCfg,
)


NUM_ACTIONS = 12
POLICY_OBS_DIM = 45
CRITIC_OBS_DIM_FLAT = 84
CRITIC_OBS_DIM_ROUGH = 271
OBS_HISTORY_LENGTH = 10


def generate_height_scan_mirror(start_idx=0, rows=11, cols=17):
    mirror_indices = []
    for row in range(rows):
        mirror_row = rows - 1 - row
        for col in range(cols):
            mirror_idx = start_idx + col + mirror_row * cols
            mirror_indices.append(mirror_idx)
    mirror_signs = [1] * (rows * cols)
    return mirror_indices, mirror_signs


def generate_joint_mirror(start_idx=0):
    """Mirror 12 DoF leg joints in policy order: left/right pairs by joint type."""
    mirror_indices = [start_idx + i for i in (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)]
    mirror_signs = [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1]
    return mirror_indices, mirror_signs


def expand_mirror(indices, signs, frame_dim, history_length=OBS_HISTORY_LENGTH):
    expanded_indices = []
    for i in range(history_length):
        offset = i * frame_dim
        expanded_indices.extend(idx + offset for idx in indices)
    return expanded_indices, signs * history_length


joint_pos_mirror_indices, joint_pos_mirror_signs = generate_joint_mirror(9)
joint_vel_mirror_indices, joint_vel_mirror_signs = generate_joint_mirror(21)
action_mirror_indices, action_mirror_signs = generate_joint_mirror(33)
policy_obs_mirror_indices = [0, 1, 2,
                             3, 4, 5,
                             6, 7, 8]\
                            + joint_pos_mirror_indices + joint_vel_mirror_indices + action_mirror_indices
policy_obs_mirror_signs = [-1, 1, -1,
                           1, -1, 1,
                           1, -1, -1] + joint_pos_mirror_signs + joint_vel_mirror_signs + action_mirror_signs
joint_acc_mirror_indices, joint_acc_mirror_signs = generate_joint_mirror(60)
joint_torques_mirror_indices, joint_torques_mirror_signs = generate_joint_mirror(72)
critic_obs_mirror_indices = policy_obs_mirror_indices + [
                                45, 46, 47,
                                49, 48,
                                53, 54, 55, 50, 51, 52,
                                57, 56,
                                59, 58,
                            ]\
                            + joint_acc_mirror_indices + joint_torques_mirror_indices
critic_obs_mirror_signs = policy_obs_mirror_signs +\
                           [1, -1, 1,
                            1, 1,
                            1, -1, 1, 1, -1, 1,
                            1, 1,
                            1, 1]\
                            + joint_acc_mirror_signs + joint_torques_mirror_signs
height_scan_mirror_indices, height_scan_mirror_signs = generate_height_scan_mirror(84, 11, 17)
critic_obs_mirror_indices_rough = critic_obs_mirror_indices + height_scan_mirror_indices
critic_obs_mirror_signs_rough = critic_obs_mirror_signs + height_scan_mirror_signs
act_mirror_indices, act_mirror_signs = generate_joint_mirror()
policy_obs_mirror_indices_expanded, policy_obs_mirror_signs_expanded = expand_mirror(
    policy_obs_mirror_indices, policy_obs_mirror_signs, POLICY_OBS_DIM
)
critic_obs_mirror_indices_expanded, critic_obs_mirror_signs_expanded = expand_mirror(
    critic_obs_mirror_indices, critic_obs_mirror_signs, CRITIC_OBS_DIM_FLAT
)
critic_obs_mirror_indices_rough_expanded, critic_obs_mirror_signs_rough_expanded = expand_mirror(
    critic_obs_mirror_indices_rough, critic_obs_mirror_signs_rough, CRITIC_OBS_DIM_ROUGH
)


def _validate_mirror(name, indices, signs, dim):
    if len(indices) != dim or len(signs) != dim:
        raise ValueError(f"{name} mirror has {len(indices)} indices/{len(signs)} signs for dim {dim}.")
    if sorted(indices) != list(range(dim)):
        raise ValueError(f"{name} mirror indices must be a permutation of 0..{dim - 1}.")


_validate_mirror("policy", policy_obs_mirror_indices_expanded, policy_obs_mirror_signs_expanded, POLICY_OBS_DIM * OBS_HISTORY_LENGTH)
_validate_mirror("critic-flat", critic_obs_mirror_indices_expanded, critic_obs_mirror_signs_expanded, CRITIC_OBS_DIM_FLAT * OBS_HISTORY_LENGTH)
_validate_mirror("critic-rough", critic_obs_mirror_indices_rough_expanded, critic_obs_mirror_signs_rough_expanded, CRITIC_OBS_DIM_ROUGH * OBS_HISTORY_LENGTH)
_validate_mirror("action", act_mirror_indices, act_mirror_signs, NUM_ACTIONS)

@lru_cache(maxsize=None)
def get_policy_obs_mirror_signs_tensor(device, dtype):
    return torch.tensor(policy_obs_mirror_signs_expanded, device=device, dtype=dtype)

def mirror_policy_observation(policy_obs):
    mirrored_policy_obs = policy_obs[..., policy_obs_mirror_indices_expanded]
    signs = get_policy_obs_mirror_signs_tensor(device=policy_obs.device, dtype=policy_obs.dtype)
    mirrored_policy_obs = mirrored_policy_obs * signs
    return mirrored_policy_obs

def mirror_critic_observation(critic_obs):
    if critic_obs.shape[-1] == CRITIC_OBS_DIM_ROUGH * OBS_HISTORY_LENGTH:
        indices = critic_obs_mirror_indices_rough_expanded
        signs_list = critic_obs_mirror_signs_rough_expanded
    elif critic_obs.shape[-1] == CRITIC_OBS_DIM_FLAT * OBS_HISTORY_LENGTH:
        indices = critic_obs_mirror_indices_expanded
        signs_list = critic_obs_mirror_signs_expanded
    else:
        raise ValueError(
            f"Unsupported critic observation dim {critic_obs.shape[-1]}; "
            f"expected {CRITIC_OBS_DIM_FLAT * OBS_HISTORY_LENGTH} or {CRITIC_OBS_DIM_ROUGH * OBS_HISTORY_LENGTH}."
        )
    mirrored_critic_obs = critic_obs[..., indices]
    signs = torch.tensor(signs_list, device=critic_obs.device, dtype=critic_obs.dtype)
    mirrored_critic_obs = mirrored_critic_obs * signs
    return mirrored_critic_obs

@lru_cache(maxsize=None)
def get_act_mirror_signs_tensor(device, dtype):
    return torch.tensor(act_mirror_signs, device=device, dtype=dtype)

def mirror_actions(actions):
    mirrored_actions = actions[..., act_mirror_indices]
    signs = get_act_mirror_signs_tensor(device=actions.device, dtype=actions.dtype)
    mirrored_actions = mirrored_actions * signs
    return mirrored_actions

def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug


@configclass
class ATOM01FlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "zky"
        self.wandb_project: str = "zky"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 30001
        self.save_interval = 500
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=0.7,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=False,
                mirror_loss_coeff=0.0,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,  # RslRlRndCfg()
        )
        self.clip_actions = 100.0


@configclass
class ATOM01RoughAgentCfg(ATOM01FlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name: str = "zky_rough"
        self.wandb_project: str = "zky_rough"
        self.algorithm = RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.1,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,  # RslRlRndCfg()
        )
