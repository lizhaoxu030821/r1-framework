# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from robolab.tasks.direct.base.agents.atom01_agent_cfg import ATOM01FlatAgentCfg


@configclass
class ATOM01SpeedRepairAgentCfg(ATOM01FlatAgentCfg):
    """PPO settings for the 40k straight high-speed repair run."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "zky_speed_repair"
        self.wandb_project = "zky_speed_repair"
        self.max_iterations = 40001
