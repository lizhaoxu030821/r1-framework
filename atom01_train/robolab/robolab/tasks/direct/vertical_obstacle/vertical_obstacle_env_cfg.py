"""Configuration for the ZKY vertical-obstacle task.

The actor and critic observation dimensions intentionally remain 45 and 84 per
frame, matching the existing ``zky`` task and its checkpoints.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from robolab.assets.robots import ATOM01_CFG
from robolab.tasks.direct.base.atom01_env_cfg import ATOM01FlatEnvCfg, ATOM01RewardCfg
from robolab.tasks.direct.base.base_config import CommandRangesCfg
from robolab.tasks.direct.base.scene_cfg import SceneCfg
from . import mdp
from .vertical_obstacle_terrain import VerticalObstacleTerrainCfg


VERTICAL_OBSTACLE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    use_cache=False,
    sub_terrains={
        "vertical_wall": VerticalObstacleTerrainCfg(
            proportion=1.0,
            obstacle_height_range=(0.04, 0.30),
            obstacle_x=1.50,
            obstacle_width=8.0,
            obstacle_thickness=0.60,
        ),
    },
)


@configclass
class VerticalObstacleRewardCfg(ATOM01RewardCfg):
    """Prioritize clearance, upright recovery, and crossing over speed tracking."""

    vertical_clearance = RewTerm(
        func=mdp.vertical_obstacle_clearance,
        weight=1.5,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "obstacle_x": 1.50,
            "obstacle_height_range": (0.04, 0.30),
            "obstacle_thickness": 0.60,
            "approach_margin": 0.75,
            "landing_margin": 0.30,
            "clearance_margin": 0.06,
        },
    )
    obstacle_top_contact = RewTerm(
        func=mdp.vertical_obstacle_top_contact,
        # This is deliberately only a weak shaping signal.  A large per-step
        # contact reward makes standing on the platform preferable to crossing it.
        weight=0.5,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "obstacle_x": 1.50,
            "obstacle_height_range": (0.04, 0.30),
            "obstacle_thickness": 0.60,
        },
    )
    obstacle_forward_progress = RewTerm(
        func=mdp.vertical_obstacle_forward_progress,
        weight=3.0,
        params={"obstacle_x": 1.50, "obstacle_thickness": 0.60, "corridor_margin": 0.75},
    )
    obstacle_success = RewTerm(
        func=mdp.vertical_obstacle_success,
        weight=60.0,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "obstacle_x": 1.50,
            "obstacle_thickness": 0.60,
            "landing_margin": 0.45,
            "minimum_root_height": 0.72,
        },
    )


@configclass
class VerticalObstacleEnvCfg(ATOM01FlatEnvCfg):
    """Fixed-timing, high-step traversal over a 4 cm -> 30 cm platform."""

    reward = VerticalObstacleRewardCfg()
    obstacle_landing_margin: float = 0.45
    foot_length_m: float = 0.266
    obstacle_front_distance_range: tuple[float, float] = (0.236, 0.296)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.scene_context.terrain_generator = VERTICAL_OBSTACLE_TERRAINS_CFG
        self.scene_context.max_init_terrain_level = 0
        self.scene_context.height_scanner.enable_height_scan = False

        # Keep the command tensor and policy input exactly unchanged, while making
        # speed a fixed transport command rather than the objective.
        self.commands.ranges = CommandRangesCfg(
            lin_vel_x=(0.30, 0.30),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        )
        self.commands.directional_mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        self.commands.velocity_curriculum = (
            (0, (0.30, 0.30), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        )
        self.commands.forward_speed_focus_curriculum = ()
        self.commands.rel_standing_envs = 0.0
        self.commands.resampling_time_range = (12.0, 12.0)
        self.commands.startup_stand_time_range = (0.5, 0.5)
        self.commands.startup_ramp_time_range = (1.0, 1.0)

        # Increase the clearance signal and reduce velocity incentives inherited
        # from the forward-speed task.
        self.reward.track_lin_vel_xy_exp.weight = 0.75
        self.reward.track_lin_vel_xy_exp.params["std"] = 0.35
        self.reward.track_ang_vel_z_exp.weight = 0.15
        self.reward.lateral_vel_y_l2.weight = -0.35
        self.reward.yaw_rate_l2.weight = -0.25
        self.reward.feet_height.weight = 0.9
        self.reward.feet_height.params["threshold"] = 0.32
        self.reward.feet_air_time.weight = 0.55
        self.reward.feet_air_time.params["threshold"] = 0.45
        self.reward.feet_stumble.weight = -2.0
        self.reward.root_height_l2.weight = -4.0

        # The task has no terrain observation, so randomize the robot's reset
        # position instead of moving the terrain mesh.  This keeps all terrain
        # and reward coordinates consistent while placing the wall front roughly
        # one ZKY foot length (+/- 3 cm) in front of the ankle at reset.
        terrain_cfg = next(iter(self.scene_context.terrain_generator.sub_terrains.values()))
        obstacle_front_x = terrain_cfg.obstacle_x - terrain_cfg.obstacle_thickness / 2
        min_front_distance, max_front_distance = self.obstacle_front_distance_range
        reset_x_range = (
            obstacle_front_x - max_front_distance,
            obstacle_front_x - min_front_distance,
        )

        # Keep small reset noise but remove mid-episode pushes.
        self.events.reset_base.params["pose_range"] = {
            "x": reset_x_range,
            "y": (-0.03, 0.03),
            "z": (-0.005, 0.015),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.03, 0.03),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.03, 0.03),
            "z": (-0.02, 0.02),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
        }
        self.events.push_robot = None

        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )
        self.robot.terminate_contacts_body_names = ["pelvis", ".*_hip_yaw_link", ".*_hip_roll_link"]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["pelvis"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["left_.*_link", "right_.*_link", "pelvis"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.scene_context.robot = ATOM01_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
