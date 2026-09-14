import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots import ZKY_19DOF_CFG
from robolab.tasks.manager_based.beyondmimic.beyondmimic_env_cfg import BeyondMimicEnvCfg
import robolab.tasks.manager_based.beyondmimic.mdp as mdp


ZKY_GETUP_MOTION = os.path.abspath(
    os.path.join(
        ROBOLAB_ROOT_DIR,
        "..",
        "..",
        "ZKY_SDK_19Dof",
        "resources",
        "motion",
        "zky",
        "npz",
        "r1_standup_reference_full26_50hz_zky_ik_hand_visual_grounded_full19.npz",
    )
)

# Names are resolved against NPZ metadata, so this order does not depend on
# Isaac's internal articulation topology.
ZKY_BODY_NAMES = [
    "base_link",
    "left_hip_yaw_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_yaw_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "body_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
]


@configclass
class ZkyGetupBeyondMimicEnvCfg(BeyondMimicEnvCfg):
    """BeyondMimic tracking task for the constrained ZKY 19-DoF get-up clip."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ZKY_19DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Remove command latency while learning the nominal reference. Delay
        # randomization can be restored after fixed-frame-zero success.
        for actuator_cfg in self.scene.robot.actuators.values():
            actuator_cfg.min_delay = 0
            actuator_cfg.max_delay = 0
        # Preserve the control authority under which the v44 checkpoint
        # learned its ordered frame-zero get-up. Reducing this to 0.20 changed
        # the physical meaning of the migrated actor outputs and destroyed the
        # previously verified recovery before PPO could adapt.
        self.actions.joint_pos.scale = 0.50
        self.commands.motion.motion_file = ZKY_GETUP_MOTION
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = ZKY_BODY_NAMES
        self.commands.motion.reset_on_motion_end = False
        # v51 learned the final pose but accumulated its largest tracking
        # errors through frames 20--180. Train those support/turn/rise states
        # directly, then progressively reconnect them to the frame-zero task.
        self.commands.motion.start_at_frame_zero = False
        self.commands.motion.start_frame_range_schedule = (
            (0, 20, 180),
            (240_000, 20, 60),
            (480_000, 60, 120),
            (720_000, 20, 180),
        )
        self.commands.motion.frame_zero_ratio = 1.0
        self.commands.motion.frame_zero_ratio_schedule = (
            (0, 1.0),
            (240_000, 0.80),
            (480_000, 0.60),
            (720_000, 0.90),
        )
        self.commands.motion.adaptive_uniform_ratio = 1.0
        self.commands.motion.adaptive_alpha = 0.0
        self.commands.motion.use_reference_joint_offset = True
        # The deployed get-up starts stationary. Random-phase resets keep
        # reference velocities, while frame-zero resets must not inject the
        # clip's non-zero initial momentum.
        self.commands.motion.zero_velocity_at_frame_zero = True
        self.commands.motion.remove_grounding_vertical_velocity_on_reset = True
        # The official reference is already grounded. Mid-phase reset states
        # must exactly match the NPZ instead of receiving the old late-stand
        # vertical correction.
        self.commands.motion.phase_reset_height_offset = 0.0
        self.commands.motion.zero_velocity_after_frame = 210
        self.commands.motion.playback_speed = 0.50
        self.commands.motion.playback_speed_schedule = ()
        self.commands.motion.playback_speed_range_schedule = ((0, 0.50, 0.50),)
        self.commands.motion.playback_speed_base_ratio = 1.0
        self.commands.motion.playback_speed_max_ratio = 0.0
        self.commands.motion.goal_only = False
        # Expose the complete 229-frame fall-to-stand sequence.  The prior
        # frame-40 curriculum only reached the low recovery pose (~0.19 m).
        self.commands.motion.motion_end_frame = -1
        # Match the NPZ frame-0 state exactly for nominal skill learning.
        # Randomization can be added only after the reference sequence is
        # reproduced reliably.
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False

        # Keep the actor exactly compatible with the RoboParty deployment
        # interface (101-D): reference joint command, IMU angular velocity
        # and projected gravity, joint position/velocity, and previous action.
        # Global root/height/phase and base linear velocity remain privileged
        # training information only; they are unavailable on the real robot.

        # Keep the first training run focused on reference tracking. Push and
        # wide dynamics randomization can be introduced after nominal success.
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.scale_link_mass = None
        self.events.scale_actuator_gains = None
        self.events.randomize_joint_default_pos = None

        # Grounded reference height is handled by a phase-scaled physical
        # target below. Keep only a weak horizontal root-drift constraint.
        self.rewards.motion_global_anchor_pos = RewTerm(
            func=mdp.motion_anchor_horizontal_position_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 0.40},
        )
        self.rewards.motion_global_anchor_pos_abs = RewTerm(
            func=mdp.motion_anchor_horizontal_position_error_abs,
            weight=-0.1,
            params={"command_name": "motion"},
        )
        self.rewards.motion_global_anchor_ori = RewTerm(
            func=mdp.motion_global_anchor_orientation_error_exp_height_gated,
            weight=1.5,
            params={
                "command_name": "motion", "std": 0.35,
                "reference_start_height": 0.160455, "reference_end_height": 0.949483,
                "target_start_height": 0.160455, "target_end_height": 0.90, "minimum_gate": 0.20,
            },
        )
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp_height_gated,
            weight=1.5,
            params={
                "command_name": "motion", "std": 0.20,
                "reference_start_height": 0.160455, "reference_end_height": 0.949483,
                "target_start_height": 0.160455, "target_end_height": 0.90, "minimum_gate": 0.20,
            },
        )
        self.rewards.motion_body_ori = RewTerm(
            func=mdp.motion_relative_body_orientation_error_exp_height_gated,
            weight=1.0,
            params={
                "command_name": "motion", "std": 0.30,
                "reference_start_height": 0.160455, "reference_end_height": 0.949483,
                "target_start_height": 0.160455, "target_end_height": 0.90, "minimum_gate": 0.20,
            },
        )
        self.rewards.motion_body_pos_abs = RewTerm(
            func=mdp.motion_relative_body_position_error_abs,
            weight=-0.30,
            params={"command_name": "motion"},
        )
        self.rewards.motion_body_lin_vel.weight = 0.25
        self.rewards.motion_body_lin_vel.params.update({"std": 1.5, "pose_gate_std": 0.25})
        self.rewards.motion_body_ang_vel.weight = 0.25
        self.rewards.motion_body_ang_vel.params.update({"std": 2.0, "pose_gate_std": 0.25})
        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.motion_joint_position_error_exp_height_gated,
            weight=3.0,
            params={
                "command_name": "motion", "std": 0.30,
                "reference_start_height": 0.160455, "reference_end_height": 0.949483,
                "target_start_height": 0.160455, "target_end_height": 0.90, "minimum_gate": 0.25,
            },
        )
        self.rewards.motion_joint_pos_abs = RewTerm(
            func=mdp.motion_joint_position_error_abs,
            weight=-0.5,
            params={"command_name": "motion"},
        )
        self.rewards.motion_lower_body_pos = RewTerm(
            func=mdp.motion_lower_body_position_error_exp,
            weight=4.0,
            params={"command_name": "motion", "std": 0.20},
        )
        self.rewards.motion_lower_body_pos_abs = RewTerm(
            func=mdp.motion_lower_body_position_error_abs,
            weight=-1.0,
            params={"command_name": "motion"},
        )
        self.rewards.motion_joint_vel = RewTerm(
            func=mdp.motion_joint_velocity_error_exp,
            weight=0.5,
            params={"command_name": "motion", "std": 2.0, "pos_gate_std": 0.45},
        )
        self.rewards.motion_key_body_pos = RewTerm(
            func=mdp.motion_special_body_position_error_exp_height_gated,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": 0.15,
                "reference_start_height": 0.160455,
                "reference_end_height": 0.949483,
                "target_start_height": 0.160455,
                "target_end_height": 0.90,
                "minimum_gate": 0.35,
                "body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_elbow_link",
                    "right_elbow_link",
                    "body_link",
                ],
            },
        )
        # Get-up-specific shaping adapted from the successful Atom01/UFO-RPO
        # experiments: make vertical recovery and final support visible to PPO
        # instead of relying only on averaged whole-body tracking errors.
        self.rewards.motion_anchor_height = RewTerm(
            func=mdp.motion_scaled_anchor_height_error_exp,
            weight=5.0,
            params={
                "command_name": "motion",
                "std": 0.12,
                "reference_start_height": 0.160455,
                "reference_end_height": 0.949483,
                "target_start_height": 0.160455,
                "target_end_height": 0.90,
            },
        )
        self.rewards.motion_anchor_upright = RewTerm(
            func=mdp.motion_anchor_upright_error_exp,
            weight=1.0,
            params={"command_name": "motion", "std": 0.20},
        )
        self.rewards.motion_anchor_height_abs = RewTerm(
            func=mdp.motion_scaled_anchor_height_error_abs,
            weight=-5.0,
            params={
                "command_name": "motion",
                "reference_start_height": 0.160455,
                "reference_end_height": 0.949483,
                "target_start_height": 0.160455,
                "target_end_height": 0.90,
            },
        )
        self.rewards.motion_anchor_upright_abs = RewTerm(
            func=mdp.motion_anchor_upright_error_abs,
            weight=-0.25,
            params={"command_name": "motion"},
        )
        self.rewards.getup_height_progress = RewTerm(
            func=mdp.getup_height_progress,
            weight=3.0,
            params={"command_name": "motion", "start_height": 0.16, "target_height": 0.90},
        )
        self.rewards.getup_upright_progress = RewTerm(
            func=mdp.getup_upright_progress,
            weight=0.5,
            params={"start_upright": 0.0, "target_upright": 0.90},
        )
        self.rewards.getup_coupled_progress = RewTerm(
            func=mdp.getup_coupled_progress,
            weight=10.0,
            params={
                "command_name": "motion",
                "start_height": 0.16,
                "target_height": 0.90,
                "start_upright": 0.0,
                "target_upright": 0.90,
            },
        )
        self.rewards.getup_terminal_extension = RewTerm(
            func=mdp.getup_terminal_extension,
            weight=15.0,
            params={
                "command_name": "motion",
                "start_height": 0.65,
                "target_height": 0.90,
                "start_reference_height": 0.65,
                "full_reference_height": 0.85,
                "minimum_upright": 0.80,
            },
        )
        self.rewards.getup_height_delta = RewTerm(
            func=mdp.getup_height_delta,
            weight=6.0,
            params={"command_name": "motion", "dt": 0.02},
        )
        self.rewards.getup_upright_delta = RewTerm(
            func=mdp.getup_upright_delta,
            weight=3.0,
            params={"command_name": "motion", "dt": 0.02},
        )
        self.rewards.getup_foot_support = RewTerm(
            func=mdp.getup_foot_support,
            weight=2.0,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]
                ),
                "asset_cfg": SceneEntityCfg("robot"),
                "force_threshold": 20.0,
                "start_reference_height": 0.22,
                "full_reference_height": 0.40,
                "start_robot_height": 0.16,
                "full_robot_height": 0.30,
                "minimum_robot_gate": 0.35,
            },
        )
        # RoboParty uses a contact-conditioned foot-slip penalty. Keep it
        # small here so feet can still reposition during the prone phase,
        # while discouraging lateral skating once support is established.
        self.rewards.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.05,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                ),
            },
        )
        self.rewards.getup_nonfoot_contact = RewTerm(
            func=mdp.getup_nonfoot_contact,
            weight=-2.0,
            params={
                "command_name": "motion",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["base_link", "body_link", ".*_hip_.*", ".*_knee.*", ".*_elbow.*", ".*_shoulder_.*"],
                ),
                "force_threshold": 20.0,
                "phase_height": 0.25,
                "full_phase_height": 0.50,
            },
        )
        self.rewards.getup_physical_stand = RewTerm(
            func=mdp.getup_stable_stand,
            weight=8.0,
            params={
                "command_name": "motion",
                "height_threshold": 0.65,
                "upright_threshold": -0.80,
                "linear_speed_threshold": 0.35,
                "angular_speed_threshold": 0.75,
                "require_motion_ended": False,
            },
        )
        self.rewards.getup_stable_stand = RewTerm(
            func=mdp.getup_stable_stand,
            weight=30.0,
            params={
                "command_name": "motion",
                "height_threshold": 0.88,
                "upright_threshold": -0.90,
                "linear_speed_threshold": 0.35,
                "angular_speed_threshold": 0.75,
                "require_motion_ended": True,
                "joint_error_threshold": 0.35,
                "body_error_threshold": 0.15,
            },
        )
        self.rewards.hold_final_pose = RewTerm(
            func=mdp.hold_final_pose_after_motion,
            weight=-0.2,
            params={
                "command_name": "motion",
                "pos_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "vel_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "pos_weight": 1.0,
                "vel_weight": 0.03,
            },
        )
        self.rewards.joint_pos_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*"])
        # Root recovery needs a substantial residual at native playback speed;
        # keep this small enough that PPO cannot choose the zero-motion pose.
        self.rewards.residual_action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
        self.rewards.terminal_residual_action_l2 = RewTerm(
            func=mdp.terminal_residual_action_l2,
            weight=-0.02,
            params={
                "command_name": "motion",
                "start_reference_height": 0.70,
                "full_reference_height": 0.85,
                "start_robot_height": 0.55,
                "full_robot_height": 0.75,
            },
        )
        self.rewards.terminal_effort_excess = RewTerm(
            func=mdp.terminal_effort_excess,
            weight=-5.0,
            params={
                "command_name": "motion",
                "start_reference_height": 0.70,
                "full_reference_height": 0.85,
                "start_robot_height": 0.55,
                "full_robot_height": 0.75,
                "soft_ratio": 0.85,
            },
        )

        self.rewards.motion_phase_height_completion = RewTerm(
            func=mdp.motion_phase_height_completion,
            weight=10.0,
            params={
                "command_name": "motion",
                "reference_start_height": 0.160455,
                "reference_end_height": 0.949483,
                "target_start_height": 0.160455,
                "target_end_height": 0.90,
            },
        )
        self.rewards.motion_phase_height_deficit = RewTerm(
            func=mdp.motion_phase_height_deficit_squared,
            weight=-16.0,
            params={
                "command_name": "motion",
                "reference_start_height": 0.160455,
                "reference_end_height": 0.949483,
                "target_start_height": 0.160455,
                "target_end_height": 0.90,
                "tolerance": 0.03,
            },
        )

        # Nominal learning needs enough recovery horizon before adaptive
        # sampling has concentrated on difficult phases.
        # Give the controller time to use the elbows/feet before terminating;
        # this is still below a full-body fall and avoids learning an early
        # reset shortcut.
        # Keep the full episode available while PPO discovers contact-based
        # recovery. Physical success is assessed separately at the final frame.
        # RoboParty's successful get-up runs expose the complete clip and use
        # physical success metrics instead of terminating on motion mismatch.
        # A mismatch termination at frame ~147 was the dominant v11 failure.
        self.terminations.anchor_pos = None
        self.terminations.anchor_ori = None
        # End every sampled phase after the same 90-step stabilization window.
        self.terminations.motion_complete = DoneTerm(
            func=mdp.motion_hold_complete,
            time_out=True,
            params={"command_name": "motion", "hold_steps": 90},
        )

        # The initial 0.5x stage needs 9.16 s for the clip plus stabilization.
        self.episode_length_s = 11.0
