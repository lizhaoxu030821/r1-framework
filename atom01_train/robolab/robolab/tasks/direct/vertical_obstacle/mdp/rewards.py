"""Reward shaping for learning a high-clearance obstacle-crossing gait."""

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def vertical_obstacle_clearance(
    env,
    contact_sensor_cfg: SceneEntityCfg,
    feet_asset_cfg: SceneEntityCfg,
    obstacle_x: float,
    obstacle_height_range: tuple[float, float],
    obstacle_thickness: float = 0.60,
    approach_margin: float = 0.75,
    landing_margin: float = 0.30,
    clearance_margin: float = 0.06,
) -> torch.Tensor:
    """Reward sufficient swing-foot height across the platform corridor."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    single_stance = contacts.sum(dim=1) == 1

    foot_z = asset.data.body_pos_w[:, feet_asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    level = env.scene.terrain.terrain_levels.float()
    max_level = max(env.cfg.scene_context.terrain_generator.num_rows - 1, 1)
    obstacle_height = obstacle_height_range[0] + level / max_level * (
        obstacle_height_range[1] - obstacle_height_range[0]
    )
    target = obstacle_height + clearance_margin
    clearance_score = torch.clamp((foot_z - target.unsqueeze(-1) + 0.12) / 0.12, 0.0, 1.0)
    swing_score = torch.where(~contacts, clearance_score, torch.zeros_like(clearance_score)).sum(dim=1)

    relative_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obstacle_front = obstacle_x - obstacle_thickness / 2
    obstacle_back = obstacle_x + obstacle_thickness / 2
    active = (relative_x > obstacle_front - approach_margin) & (relative_x < obstacle_back + landing_margin)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7
    return swing_score * single_stance.float() * active.float() * upright


def vertical_obstacle_top_contact(
    env,
    contact_sensor_cfg: SceneEntityCfg,
    feet_asset_cfg: SceneEntityCfg,
    obstacle_x: float,
    obstacle_height_range: tuple[float, float],
    obstacle_thickness: float = 0.60,
    x_tolerance: float = 0.10,
) -> torch.Tensor:
    """Reward stable foot contacts on the obstacle top surface."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    asset: Articulation = env.scene["robot"]
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    foot_pos = asset.data.body_pos_w[:, feet_asset_cfg.body_ids, :] - env.scene.env_origins.unsqueeze(1)

    level = env.scene.terrain.terrain_levels.float()
    max_level = max(env.cfg.scene_context.terrain_generator.num_rows - 1, 1)
    obstacle_height = obstacle_height_range[0] + level / max_level * (
        obstacle_height_range[1] - obstacle_height_range[0]
    )
    obstacle_front = obstacle_x - obstacle_thickness / 2
    obstacle_back = obstacle_x + obstacle_thickness / 2
    over_top = (foot_pos[..., 0] > obstacle_front - x_tolerance) & (
        foot_pos[..., 0] < obstacle_back + x_tolerance
    )
    at_top_height = (foot_pos[..., 2] > obstacle_height.unsqueeze(-1) + 0.02) & (
        foot_pos[..., 2] < obstacle_height.unsqueeze(-1) + 0.16
    )
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.8) / 0.8
    return (contacts & over_top & at_top_height).float().sum(dim=1) * upright


def vertical_obstacle_forward_progress(
    env,
    obstacle_x: float,
    obstacle_thickness: float = 0.60,
    corridor_margin: float = 0.75,
) -> torch.Tensor:
    """Reward upright forward motion through the obstacle corridor."""
    asset: Articulation = env.scene["robot"]
    relative_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    obstacle_front = obstacle_x - obstacle_thickness / 2
    obstacle_back = obstacle_x + obstacle_thickness / 2
    active = (relative_x > obstacle_front - corridor_margin) & (relative_x < obstacle_back + corridor_margin)
    forward_velocity = torch.clamp(asset.data.root_lin_vel_w[:, 0], min=0.0, max=0.6)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.8) / 0.8
    return forward_velocity * active.float() * upright


def vertical_obstacle_success(
    env,
    contact_sensor_cfg: SceneEntityCfg,
    obstacle_x: float,
    obstacle_thickness: float = 0.60,
    landing_margin: float = 0.45,
    minimum_root_height: float = 0.72,
) -> torch.Tensor:
    """Emit a one-step bonus after an upright, supported landing beyond the platform."""
    asset: Articulation = env.scene["robot"]
    relative_x = asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    relative_z = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    supported = contacts.any(dim=1)
    upright = -asset.data.projected_gravity_b[:, 2] > 0.85
    crossed_now = (
        (relative_x > obstacle_x + obstacle_thickness / 2 + landing_margin)
        & (relative_z > minimum_root_height)
        & upright
        & supported
    )
    newly_crossed = crossed_now & ~env.obstacle_crossed
    env.obstacle_crossed |= crossed_now
    return newly_crossed.float()
