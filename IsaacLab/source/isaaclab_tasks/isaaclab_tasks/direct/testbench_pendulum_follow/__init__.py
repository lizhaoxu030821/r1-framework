# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Testbench pendulum tracking task.

The square joint follows a sinusoidal trajectory. The policy controls the link
joint to track this trajectory as a single pendulum.
"""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Testbench-Pendulum-Follow-Direct-v0",
    entry_point=f"{__name__}.testbench_pendulum_follow_env:TestbenchPendulumFollowEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.testbench_pendulum_follow_env:TestbenchPendulumFollowEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TestbenchPendulumFollowPPORunnerCfg",
    },
)
