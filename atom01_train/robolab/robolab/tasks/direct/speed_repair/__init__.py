# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Straight high-speed repair task for robot_zky."""

import gymnasium as gym

gym.register(
    id="zky_speed_repair",
    entry_point="robolab.tasks.direct.base.base_env:BaseEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.atom01_speed_repair_env_cfg:ATOM01SpeedRepairEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.atom01_speed_repair_agent_cfg:ATOM01SpeedRepairAgentCfg",
    },
)
