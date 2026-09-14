"""Terrain generator for one full-width vertical obstacle."""

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def vertical_obstacle_terrain(difficulty: float, cfg: "VerticalObstacleTerrainCfg"):
    """Build a flat approach, a vertical wall, and a flat landing area.

    The wall height is controlled by terrain difficulty, so the terrain generator's
    row curriculum directly becomes an obstacle-height curriculum.
    """
    obstacle_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )
    ground_height = 0.20
    ground = trimesh.creation.box(
        (cfg.size[0], cfg.size[1], ground_height),
        trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, -ground_height / 2)),
    )
    # Terrain origins are where the robot is spawned. Keep obstacle_x in that
    # robot-relative frame, then translate it into the sub-terrain mesh frame.
    origin = np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0], dtype=np.float32)
    wall = trimesh.creation.box(
        (cfg.obstacle_thickness, cfg.obstacle_width, obstacle_height),
        trimesh.transformations.translation_matrix(
            (origin[0] + cfg.obstacle_x, cfg.size[1] / 2, obstacle_height / 2)
        ),
    )
    return [ground, wall], origin


@configclass
class VerticalObstacleTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a wall that cannot be bypassed laterally."""

    function = vertical_obstacle_terrain
    obstacle_height_range: tuple[float, float] = MISSING
    obstacle_x: float = 1.50
    obstacle_width: float = 8.0
    obstacle_thickness: float = 0.60
