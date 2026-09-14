"""Environment hooks for the vertical-obstacle curriculum."""

from collections.abc import Sequence

import torch

from robolab.tasks.direct.base import BaseEnv


class VerticalObstacleEnv(BaseEnv):
    """Base locomotion environment with obstacle-aware terrain progression."""

    def init_buffers(self):
        super().init_buffers()
        self.obstacle_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def obstacle_terrain_cfg(self):
        return next(iter(self.cfg.scene_context.terrain_generator.sub_terrains.values()))

    def update_terrain_levels(self, env_ids):
        """Advance only after the robot has crossed the wall; regress on a failed approach."""
        relative_x = self.robot.data.root_pos_w[env_ids, 0] - self.scene.env_origins[env_ids, 0]
        relative_z = self.robot.data.root_pos_w[env_ids, 2] - self.scene.env_origins[env_ids, 2]
        wall_end = self.obstacle_terrain_cfg.obstacle_x + self.obstacle_terrain_cfg.obstacle_thickness / 2
        upright = -self.robot.data.projected_gravity_b[env_ids, 2] > 0.85
        move_up = (
            (relative_x > wall_end + self.cfg.obstacle_landing_margin)
            & (relative_z > 0.72)
            & upright
        )
        move_down = ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)

        levels = self.scene.terrain.terrain_levels[env_ids].float()
        max_level = max(self.cfg.scene_context.terrain_generator.num_rows - 1, 1)
        height_cfg = self.obstacle_terrain_cfg.obstacle_height_range
        target_height = height_cfg[0] + levels / max_level * (height_cfg[1] - height_cfg[0])
        return {
            "Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float()),
            "Curriculum/obstacle_cross_rate": move_up.float().mean(),
            "Curriculum/obstacle_height_target": torch.mean(target_height),
            "Curriculum/obstacle_height_max": torch.tensor(height_cfg[1], device=self.device),
        }

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is not None and len(env_ids) > 0:
            self.obstacle_crossed[env_ids] = False
        super()._reset_idx(env_ids)
