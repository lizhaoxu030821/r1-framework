"""Direct reinforcement-learning task for curriculum-based vertical obstacles."""

from .vertical_obstacle_env import VerticalObstacleEnv
from .vertical_obstacle_env_cfg import VerticalObstacleEnvCfg
from .agents.vertical_obstacle_agent_cfg import VerticalObstacleAgentCfg

import gymnasium as gym


gym.register(
    id="zky_vertical_obstacle",
    entry_point=f"{__name__}.vertical_obstacle_env:VerticalObstacleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.vertical_obstacle_env_cfg:VerticalObstacleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.vertical_obstacle_agent_cfg:VerticalObstacleAgentCfg",
    },
)
