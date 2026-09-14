# Copyright (c) 2025-2026, The RoboLab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import csv
import contextlib
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    import mujoco_viewer
except ImportError:
    mujoco_viewer = None

try:
    import mujoco.viewer as mujoco_builtin_viewer
except ImportError:
    mujoco_builtin_viewer = None


ROOT = Path(__file__).resolve().parents[3]  # .../modules/atom01_train
for path in (ROOT / "robolab", ROOT / "rsl_rl", ROOT.parent.parent / "IsaacLab" / "source"):
    if path.exists():
        sys.path.insert(0, str(path))

from robolab.assets import ISAAC_DATA_DIR  # noqa: E402


class PassiveMujocoViewer:
    """Small adapter for MuJoCo's built-in passive viewer."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        if mujoco_builtin_viewer is None:
            raise RuntimeError("MuJoCo built-in viewer is unavailable.")
        self._handle = mujoco_builtin_viewer.launch_passive(model, data)
        self._overlay = {}

    def render(self):
        if self.is_alive:
            self._handle.sync()

    @property
    def is_alive(self) -> bool:
        return self._handle.is_running()

    def close(self):
        self._handle.close()


POLICY_JOINT_NAMES = [
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

POLICY_DEFAULT_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, -0.15, 0.15, 0.3, -0.3, 0.15, -0.15, 0.0, 0.0],
    dtype=np.float64,
)

POLICY_JOINT_INDEX = {name: i for i, name in enumerate(POLICY_JOINT_NAMES)}

KP_BY_JOINT = {
    "hip_yaw": 100.0,
    "hip_roll": 100.0,
    "hip_pitch": 100.0,
    "knee": 300.0,
    "ankle_pitch": 260.0,
    "ankle_roll": 180.0,
}

KD_BY_JOINT = {
    "hip_yaw": 3.3,
    "hip_roll": 3.3,
    "hip_pitch": 3.3,
    "knee": 5.0,
    "ankle_pitch": 3.0,
    "ankle_roll": 3.0,
}

TAU_LIMIT_BY_JOINT = {
    "hip_yaw": 120.0,
    "hip_roll": 120.0,
    "hip_pitch": 120.0,
    "knee": 120.0,
    "ankle_pitch": 27.0,
    "ankle_roll": 27.0,
}


def joint_kind(name: str) -> str:
    for kind in KP_BY_JOINT:
        if kind in name:
            return kind
    raise KeyError(f"Unsupported joint name for gains: {name}")


def load_policy(checkpoint_path: str, verbose: bool = False):
    checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    def mlp_dims(prefix: str):
        layers = []
        for key, value in state_dict.items():
            if key.startswith(prefix) and key.endswith(".weight"):
                layer = key[len(prefix) :].split(".", 1)[0]
                if layer.isdigit():
                    layers.append((int(layer), value.shape))
        layers.sort(key=lambda item: item[0])
        if not layers:
            raise RuntimeError(f"Cannot infer MLP dims from checkpoint prefix {prefix!r}")
        return layers[0][1][1], [shape[0] for _, shape in layers[:-1]], layers[-1][1][0]

    actor_obs_dim, actor_hidden_dims, actor_out_dim = mlp_dims("actor.")
    critic_obs_dim, critic_hidden_dims, _ = mlp_dims("critic.")
    actor_obs_normalization = any(key.startswith("actor_obs_normalizer.") for key in state_dict)
    critic_obs_normalization = any(key.startswith("critic_obs_normalizer.") for key in state_dict)
    noise_std_type = "log" if "log_std" in state_dict else "scalar"
    state_dependent_std = "std" not in state_dict and "log_std" not in state_dict
    num_actions = actor_out_dim // 2 if state_dependent_std else actor_out_dim
    if actor_obs_dim % 45 == 0:
        single_obs = 45
    elif actor_obs_dim % 48 == 0:
        single_obs = 48
    else:
        raise RuntimeError(f"Expected zky policy actor obs dim divisible by 45 or 48, got {actor_obs_dim}.")
    if num_actions != 12:
        raise RuntimeError(f"Expected 12 policy actions, got {num_actions}.")

    from rsl_rl.modules import ActorCritic

    dummy_obs = {
        "policy": torch.zeros((1, actor_obs_dim), dtype=torch.float32),
        "critic": torch.zeros((1, critic_obs_dim), dtype=torch.float32),
    }
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    kwargs = dict(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=num_actions,
        actor_obs_normalization=actor_obs_normalization,
        critic_obs_normalization=critic_obs_normalization,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type=noise_std_type,
        state_dependent_std=state_dependent_std,
    )
    if verbose:
        policy = ActorCritic(**kwargs)
    else:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            policy = ActorCritic(**kwargs)
    policy.load_state_dict(state_dict)
    policy.eval()

    return policy, actor_obs_dim // single_obs, single_obs


def quat_xyzw_from_mujoco(data: mujoco.MjData) -> np.ndarray:
    return data.qpos[3:7][[1, 2, 3, 0]].copy()


def root_obs(data: mujoco.MjData):
    rot = R.from_quat(quat_xyzw_from_mujoco(data))
    omega = data.sensor("angular-velocity").data.astype(np.float64).copy()
    gravity_b = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True)
    return omega, gravity_b


def geom_vertices_world(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        local = np.array(
            [[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=np.float64,
        )
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        count = model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[start : start + count]
    else:
        center = data.geom_xpos[geom_id]
        radius = model.geom_rbound[geom_id]
        return np.array([center - radius, center + radius], dtype=np.float64)
    return data.geom_xpos[geom_id] + local @ data.geom_xmat[geom_id].reshape(3, 3).T


def foot_geom_ids(model: mujoco.MjModel) -> dict[str, list[int]]:
    result = {"left": [], "right": []}
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name and name.startswith("left_foot_collision"):
            result["left"].append(geom_id)
        elif name and name.startswith("right_foot_collision"):
            result["right"].append(geom_id)
    if not result["left"] or not result["right"]:
        raise RuntimeError("Cannot find left_foot_collision*/right_foot_collision* in MJCF.")
    return result


def configure_foot_contact(
    model: mujoco.MjModel,
    ids_by_side: dict[str, list[int]],
    sliding: float | None,
    torsional: float | None,
    rolling: float | None,
    solref_timeconst: float | None,
    solref_dampratio: float | None,
):
    geom_ids = [geom_id for ids in ids_by_side.values() for geom_id in ids]
    for geom_id in geom_ids:
        if sliding is not None:
            model.geom_friction[geom_id, 0] = sliding
        if torsional is not None:
            model.geom_friction[geom_id, 1] = torsional
        if rolling is not None:
            model.geom_friction[geom_id, 2] = rolling
        if solref_timeconst is not None:
            model.geom_solref[geom_id, 0] = solref_timeconst
        if solref_dampratio is not None:
            model.geom_solref[geom_id, 1] = solref_dampratio


def foot_corner_heights(vertices: np.ndarray, side: str) -> dict[str, float]:
    x_mid = 0.5 * (vertices[:, 0].min() + vertices[:, 0].max())
    y_mid = 0.5 * (vertices[:, 1].min() + vertices[:, 1].max())
    if side == "left":
        lateral_masks = (("inner", vertices[:, 1] < y_mid), ("outer", vertices[:, 1] >= y_mid))
    elif side == "right":
        lateral_masks = (("inner", vertices[:, 1] >= y_mid), ("outer", vertices[:, 1] < y_mid))
    else:
        raise ValueError(f"Unsupported foot side: {side}")
    corners = {}
    for x_name, x_mask in (("rear", vertices[:, 0] < x_mid), ("toe", vertices[:, 0] >= x_mid)):
        for y_name, y_mask in lateral_masks:
            mask = x_mask & y_mask
            corners[f"{x_name}_{y_name}"] = float(vertices[mask, 2].min()) if np.any(mask) else float("nan")
    return corners


def foot_summary(model: mujoco.MjModel, data: mujoco.MjData, ids_by_side: dict[str, list[int]]) -> str:
    parts = []
    for side, ids in ids_by_side.items():
        vertices = np.concatenate([geom_vertices_world(model, data, geom_id) for geom_id in ids], axis=0)
        corners = foot_corner_heights(vertices, side)
        heel_z = min(corners["rear_inner"], corners["rear_outer"])
        toe_z = min(corners["toe_inner"], corners["toe_outer"])
        inner_z = min(corners["rear_inner"], corners["toe_inner"])
        outer_z = min(corners["rear_outer"], corners["toe_outer"])
        min_z = float(vertices[:, 2].min())
        max_z = float(vertices[:, 2].max())
        contact_count = 0
        force_z = 0.0
        geom_set = set(ids)
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 in geom_set or contact.geom2 in geom_set:
                force = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(model, data, i, force)
                force_z += float(force[0])
                contact_count += 1
        parts.append(
            f"{side[0]}:n{contact_count},fz{force_z:.0f},z{min_z:.3f}/{max_z:.3f},"
            f"toe-heel={toe_z - heel_z:+.3f},outer-inner={outer_z - inner_z:+.3f}"
        )
    return " ".join(parts)


def foot_metrics(model: mujoco.MjModel, data: mujoco.MjData, ids: list[int], side: str) -> dict[str, float]:
    vertices = np.concatenate([geom_vertices_world(model, data, geom_id) for geom_id in ids], axis=0)
    corners = foot_corner_heights(vertices, side)
    heel_z = min(corners["rear_inner"], corners["rear_outer"])
    toe_z = min(corners["toe_inner"], corners["toe_outer"])
    inner_z = min(corners["rear_inner"], corners["toe_inner"])
    outer_z = min(corners["rear_outer"], corners["toe_outer"])
    contact_count = 0
    force_z = 0.0
    geom_set = set(ids)
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 in geom_set or contact.geom2 in geom_set:
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, force)
            force_z += float(force[0])
            contact_count += 1
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return {
        "contact_count": float(contact_count),
        "force_z": force_z,
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "min_z": float(vertices[:, 2].min()),
        "max_z": float(vertices[:, 2].max()),
        "toe_minus_heel": toe_z - heel_z,
        "outer_minus_inner": outer_z - inner_z,
    }


def foot_corner_summary(model: mujoco.MjModel, data: mujoco.MjData, ids_by_side: dict[str, list[int]]) -> str:
    parts = []
    for side, ids in ids_by_side.items():
        vertices = np.concatenate([geom_vertices_world(model, data, geom_id) for geom_id in ids], axis=0)
        corners = foot_corner_heights(vertices, side)
        parts.append(
            f"{side[0]}:"
            f"ri={corners['rear_inner']:.3f},ro={corners['rear_outer']:.3f},"
            f"ti={corners['toe_inner']:.3f},to={corners['toe_outer']:.3f}"
        )
    return " ".join(parts)


def foot_center_world(model: mujoco.MjModel, data: mujoco.MjData, ids: list[int]) -> np.ndarray:
    vertices = np.concatenate([geom_vertices_world(model, data, geom_id) for geom_id in ids], axis=0)
    return 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))


def robot_com_world(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if body_id < 0:
        body_id = 1
    return data.subtree_com[body_id].copy()


def com_summary(model: mujoco.MjModel, data: mujoco.MjData, ids_by_side: dict[str, list[int]]) -> str:
    left = foot_center_world(model, data, ids_by_side["left"])
    right = foot_center_world(model, data, ids_by_side["right"])
    feet_center = 0.5 * (left + right)
    com = robot_com_world(model, data)
    rpy = R.from_quat(quat_xyzw_from_mujoco(data)).as_euler("xyz")
    yaw = rpy[2]
    rel = com[:2] - feet_center[:2]
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    left_axis = np.array([-np.sin(yaw), np.cos(yaw)])
    return (
        f"com=({com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f}) "
        f"feet_c=({feet_center[0]:+.4f},{feet_center[1]:+.4f}) "
        f"com-feet_body=({float(rel @ forward):+.4f},{float(rel @ left_axis):+.4f}) "
        f"foot_y=({left[1]:+.4f},{right[1]:+.4f})"
    )


def build_model_state(model: mujoco.MjModel):
    joint_names_mj = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    actuator_names_mj = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    missing = [name for name in POLICY_JOINT_NAMES if name not in joint_names_mj]
    if missing:
        raise RuntimeError(f"MJCF is missing policy joints: {missing}")
    missing_actuators = [name for name in POLICY_JOINT_NAMES if name not in actuator_names_mj]
    if missing_actuators:
        raise RuntimeError(f"MJCF is missing policy actuators: {missing_actuators}")

    policy_to_mj_joint_id = np.array([joint_names_mj.index(name) for name in POLICY_JOINT_NAMES], dtype=np.int64)
    policy_qpos_adr = np.array([model.jnt_qposadr[joint_id] for joint_id in policy_to_mj_joint_id], dtype=np.int64)
    policy_dof_adr = np.array([model.jnt_dofadr[joint_id] for joint_id in policy_to_mj_joint_id], dtype=np.int64)
    policy_to_actuator_id = np.array([actuator_names_mj.index(name) for name in POLICY_JOINT_NAMES], dtype=np.int64)

    default_qpos = np.zeros(model.nq, dtype=np.float64)
    default_qpos[:3] = np.array([0.0, 0.0, 0.9])
    default_qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    default_qpos[policy_qpos_adr] = POLICY_DEFAULT_POS

    kp = np.array([KP_BY_JOINT[joint_kind(name)] for name in POLICY_JOINT_NAMES], dtype=np.float64)
    kd = np.array([KD_BY_JOINT[joint_kind(name)] for name in POLICY_JOINT_NAMES], dtype=np.float64)
    tau_limit = np.array([TAU_LIMIT_BY_JOINT[joint_kind(name)] for name in POLICY_JOINT_NAMES], dtype=np.float64)
    return joint_names_mj, actuator_names_mj, policy_qpos_adr, policy_dof_adr, policy_to_actuator_id, default_qpos, kp, kd, tau_limit


def auto_align_root_height(model: mujoco.MjModel, data: mujoco.MjData, ids_by_side: dict[str, list[int]], clearance: float):
    mujoco.mj_forward(model, data)
    min_z = min(float(geom_vertices_world(model, data, geom_id)[:, 2].min()) for ids in ids_by_side.values() for geom_id in ids)
    data.qpos[2] += clearance - min_z
    mujoco.mj_forward(model, data)


def recenter_root_pose(data: mujoco.MjData, preserve_tilt: bool):
    if preserve_tilt:
        rpy = R.from_quat(quat_xyzw_from_mujoco(data)).as_euler("xyz")
        rpy[2] = 0.0
        quat_xyzw = R.from_euler("xyz", rpy).as_quat()
    else:
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    data.qpos[0:2] = 0.0
    data.qpos[3:7] = quat_xyzw[[3, 0, 1, 2]]
    data.qvel[:6] = 0.0


def settle_default_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_adr: np.ndarray,
    dof_adr: np.ndarray,
    actuator_ids: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_limit: np.ndarray,
    duration: float,
    recenter: bool,
    preserve_tilt: bool,
):
    if duration <= 0.0:
        return
    settle_steps = int(np.ceil(duration / model.opt.timestep))
    for _ in range(settle_steps):
        q = data.qpos[qpos_adr]
        dq = data.qvel[dof_adr]
        tau_policy = kp * (POLICY_DEFAULT_POS - q) - kd * dq
        data.ctrl[:] = 0.0
        data.ctrl[actuator_ids] = np.clip(tau_policy, -tau_limit, tau_limit)
        mujoco.mj_step(model, data)
    if recenter:
        recenter_root_pose(data, preserve_tilt)
        mujoco.mj_forward(model, data)


def render_pause(model: mujoco.MjModel, viewer, duration: float):
    if duration <= 0.0:
        return
    pause_steps = int(np.ceil(duration / model.opt.timestep))
    for _ in range(pause_steps):
        viewer.render()
        if not viewer.is_alive:
            break


def set_status_overlay(
    viewer,
    world_velocity: np.ndarray,
    policy_command: np.ndarray,
    policy_time: float,
    schedule_stage: int,
    schedule_size: int,
) -> None:
    """Add a display-only HUD; none of these values are fed back to the policy."""
    stage = "constant"
    if schedule_stage >= 0:
        stage = f"{schedule_stage + 1}/{schedule_size}"
    labels = (
        "PURE POLICY (no pose feedback)\n"
        "Test time\n"
        "Speed stage\n"
        "WORLD vx\n"
        "WORLD vy\n"
        "WORLD yaw rate\n"
        "CMD vx\n"
        "CMD vy\n"
        "CMD yaw rate"
    )
    values = (
        "\n"
        f"{policy_time:7.2f} s\n"
        f"{stage}\n"
        f"{world_velocity[0]:+7.3f} m/s\n"
        f"{world_velocity[1]:+7.3f} m/s\n"
        f"{world_velocity[2]:+7.3f} rad/s ({np.degrees(world_velocity[2]):+6.1f} deg/s)\n"
        f"{policy_command[0]:+7.3f} m/s\n"
        f"{policy_command[1]:+7.3f} m/s\n"
        f"{policy_command[2]:+7.3f} rad/s"
    )
    viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = [labels, values]


def main():
    parser = argparse.ArgumentParser(description="Clean robot_zky MuJoCo sim2sim based on the direct training env.")
    parser.add_argument("--load_model", default=None)
    parser.add_argument("--terrain", action="store_true")
    parser.add_argument("--vx", type=float, default=0.4)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--dyaw", type=float, default=0.0)
    parser.add_argument(
        "--vx_schedule",
        default=None,
        help="Comma-separated pure-policy vx commands, applied in order without pose feedback.",
    )
    parser.add_argument("--vx_hold_time", type=float, default=8.0, help="Seconds assigned to each vx schedule stage.")
    parser.add_argument(
        "--vx_transition_time",
        type=float,
        default=1.0,
        help="Seconds used to smoothly interpolate into each vx schedule stage.",
    )
    parser.add_argument("--no_render", action="store_true")
    parser.add_argument("--gif_path", default=None, help="Write an off-screen MuJoCo replay GIF to this path.")
    parser.add_argument("--mp4_path", default=None, help="Write an off-screen MuJoCo replay MP4 to this path.")
    parser.add_argument("--gif_fps", type=int, default=15, help="Frame rate used when writing --gif_path.")
    parser.add_argument("--mp4_bitrate", default="16M", help="Target H.264 bitrate used when writing --mp4_path.")
    parser.add_argument("--gif_width", type=int, default=480, help="Off-screen GIF frame width in pixels.")
    parser.add_argument("--gif_height", type=int, default=360, help="Off-screen GIF frame height in pixels.")
    parser.add_argument("--no_policy", action="store_true")
    parser.add_argument("--no_plots", action="store_true")  # accepted for command compatibility
    parser.add_argument("--foot_friction", type=float, default=None, help="Override foot sliding friction in MuJoCo.")
    parser.add_argument("--foot_torsional_friction", type=float, default=None, help="Override foot torsional friction in MuJoCo.")
    parser.add_argument("--foot_rolling_friction", type=float, default=None, help="Override foot rolling friction in MuJoCo.")
    parser.add_argument("--foot_solref_timeconst", type=float, default=None, help="Override foot contact solref time constant.")
    parser.add_argument("--foot_solref_dampratio", type=float, default=None, help="Override foot contact solref damping ratio.")
    parser.add_argument("--max_steps", type=int, default=None, help="Low-level MuJoCo steps.")
    parser.add_argument("--debug_steps", type=int, default=0, help="Policy frames to print.")
    parser.add_argument("--debug_interval", type=int, default=0, help="Print compact joint/foot debug every N policy frames.")
    parser.add_argument("--debug_vectors", action="store_true", help="Print q/obs/action vectors during debug frames.")
    parser.add_argument("--debug_com", action="store_true", help="Print robot COM relative to the foot support center.")
    parser.add_argument("--joint_log_csv", default=None, help="Write compact per-policy-frame joint/foot diagnostics to this CSV.")
    parser.add_argument("--joint_log_interval", type=int, default=1, help="Write CSV diagnostics every N policy frames.")
    parser.add_argument(
        "--history_init",
        choices=("zero", "repeat"),
        default="repeat",
        help="Initialize stacked policy history from reset-like zeros or by repeating the first observation.",
    )
    parser.add_argument("--action_scale", type=float, default=0.25)
    parser.add_argument("--action_clip", type=float, default=100.0)
    parser.add_argument(
        "--use_training_action_scale_multipliers",
        action="store_true",
        default=True,
        help="Apply the ATOM01 training per-joint action multipliers in MuJoCo.",
    )
    parser.add_argument(
        "--no_training_action_scale_multipliers",
        dest="use_training_action_scale_multipliers",
        action="store_false",
        help="Disable the ATOM01 training per-joint action multipliers in MuJoCo.",
    )
    parser.add_argument("--hip_roll_action_scale", type=float, default=1.0, help="Extra multiplier for hip-roll policy actions.")
    parser.add_argument("--ankle_roll_action_scale", type=float, default=1.0, help="Extra multiplier for ankle-roll policy actions.")
    parser.add_argument("--left_hip_roll_action_scale", type=float, default=1.0, help="Extra multiplier for left hip-roll policy action.")
    parser.add_argument("--right_hip_roll_action_scale", type=float, default=1.0, help="Extra multiplier for right hip-roll policy action.")
    parser.add_argument("--left_ankle_roll_action_scale", type=float, default=1.0, help="Extra multiplier for left ankle-roll policy action.")
    parser.add_argument("--right_ankle_roll_action_scale", type=float, default=1.0, help="Extra multiplier for right ankle-roll policy action.")
    parser.add_argument("--left_hip_roll_action_clip", type=float, default=None, help="Optional symmetric clip for left hip-roll action after scaling.")
    parser.add_argument("--right_hip_roll_action_clip", type=float, default=None, help="Optional symmetric clip for right hip-roll action after scaling.")
    parser.add_argument("--left_ankle_roll_action_clip", type=float, default=None, help="Optional symmetric clip for left ankle-roll action after scaling.")
    parser.add_argument("--right_ankle_roll_action_clip", type=float, default=None, help="Optional symmetric clip for right ankle-roll action after scaling.")
    parser.add_argument("--flip_left_hip_roll_action", action="store_true", help="Flip left hip-roll policy action before MuJoCo PD.")
    parser.add_argument("--flip_right_hip_roll_action", action="store_true", help="Flip right hip-roll policy action before MuJoCo PD.")
    parser.add_argument("--flip_left_ankle_roll_action", action="store_true", help="Flip left ankle-roll policy action before MuJoCo PD.")
    parser.add_argument("--flip_right_ankle_roll_action", action="store_true", help="Flip right ankle-roll policy action before MuJoCo PD.")
    parser.add_argument(
        "--flip_left_hip_roll_convention",
        action="store_true",
        help="Flip left hip-roll q/dq/last_action observations and flip policy action back for MuJoCo.",
    )
    parser.add_argument(
        "--flip_right_hip_roll_convention",
        action="store_true",
        help="Flip right hip-roll q/dq/last_action observations and flip policy action back for MuJoCo.",
    )
    parser.add_argument(
        "--flip_left_ankle_roll_convention",
        action="store_true",
        help="Flip left ankle-roll q/dq/last_action observations and flip policy action back for MuJoCo.",
    )
    parser.add_argument(
        "--flip_right_ankle_roll_convention",
        action="store_true",
        help="Flip right ankle-roll q/dq/last_action observations and flip policy action back for MuJoCo.",
    )
    parser.add_argument("--actuator_delay_steps", type=int, default=0, help="Fixed low-level PD target delay; training randomizes actuator delay.")
    parser.add_argument("--warmup_time", type=float, default=5.0)
    parser.add_argument("--ramp_time", type=float, default=2.0)
    parser.add_argument(
        "--policy_action_ramp_time",
        type=float,
        default=0.0,
        help="Seconds to blend policy actions in from the default pose at startup.",
    )
    parser.add_argument(
        "--warmup_action_scale",
        type=float,
        default=1.0,
        help="Extra action multiplier while command warmup is active; reduces startup lunges.",
    )
    parser.add_argument("--gait_phase_offset", type=float, default=-0.5 * np.pi, help="Initial gait phase offset in radians.")
    parser.add_argument("--gait_period", type=float, default=0.8)
    parser.add_argument("--root_z", type=float, default=0.9)
    parser.add_argument("--foot_ground_clearance", type=float, default=0.001)
    parser.add_argument("--settle_time", type=float, default=0.0, help="Seconds to settle contacts at the default pose before policy starts.")
    parser.add_argument("--stand_time", type=float, default=1.0, help="Visible default-pose standing time before policy starts.")
    parser.add_argument("--no_settle_recenter", action="store_true", help="Do not zero root xy/yaw/velocity after startup settling.")
    parser.add_argument("--preserve_settle_tilt", action="store_true", help="Keep settle roll/pitch when recentering; off matches the training reset pose.")
    parser.add_argument("--stop_root_z", type=float, default=0.5)
    parser.add_argument("--verbose_model", action="store_true")
    args = parser.parse_args()
    if args.actuator_delay_steps < 0:
        raise ValueError("--actuator_delay_steps must be non-negative.")
    if args.vx_hold_time <= 0.0:
        raise ValueError("--vx_hold_time must be positive.")
    if not 0.0 <= args.vx_transition_time <= args.vx_hold_time:
        raise ValueError("--vx_transition_time must be between zero and --vx_hold_time.")
    vx_schedule = None
    if args.vx_schedule:
        try:
            vx_schedule = tuple(float(value.strip()) for value in args.vx_schedule.split(","))
        except ValueError as error:
            raise ValueError("--vx_schedule must be a comma-separated list of numbers.") from error
        if not vx_schedule:
            raise ValueError("--vx_schedule must contain at least one speed.")

    xml_name = "atom01_terrain.xml" if args.terrain else "atom01.xml"
    xml_path = Path(ISAAC_DATA_DIR) / "robots" / "roboparty" / "atom01" / "mjcf" / xml_name
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    joint_names_mj, actuator_names_mj, qpos_adr, dof_adr, actuator_ids, default_qpos, kp, kd, tau_limit = build_model_state(model)
    default_qpos[2] = args.root_z
    data.qpos[:] = default_qpos
    data.qvel[:] = 0.0

    feet = foot_geom_ids(model)
    configure_foot_contact(
        model,
        feet,
        args.foot_friction,
        args.foot_torsional_friction,
        args.foot_rolling_friction,
        args.foot_solref_timeconst,
        args.foot_solref_dampratio,
    )
    auto_align_root_height(model, data, feet, args.foot_ground_clearance)
    settle_default_pose(
        model,
        data,
        qpos_adr,
        dof_adr,
        actuator_ids,
        kp,
        kd,
        tau_limit,
        args.settle_time,
        not args.no_settle_recenter,
        args.preserve_settle_tilt,
    )

    policy = None
    frame_stack = 10
    single_obs = 45
    if not args.no_policy:
        if args.load_model is None:
            raise ValueError("--load_model is required unless --no_policy is set.")
        policy, frame_stack, single_obs = load_policy(args.load_model, verbose=args.verbose_model)
        if frame_stack != 10:
            raise RuntimeError(f"Expected policy frame stack 10 from training, got {frame_stack}.")

    hist_obs = np.zeros((frame_stack, single_obs), dtype=np.float32)
    obs_history_initialized = False
    convention_sign = np.ones(12, dtype=np.float64)
    convention_flag_to_joint = {
        "flip_left_hip_roll_convention": "left_hip_roll_joint",
        "flip_right_hip_roll_convention": "right_hip_roll_joint",
        "flip_left_ankle_roll_convention": "left_ankle_roll_joint",
        "flip_right_ankle_roll_convention": "right_ankle_roll_joint",
    }
    for flag_name, joint_name in convention_flag_to_joint.items():
        if getattr(args, flag_name):
            convention_sign[POLICY_JOINT_INDEX[joint_name]] = -1.0

    prev_policy_action = np.zeros(12, dtype=np.float64)
    current_policy_action = np.zeros(12, dtype=np.float64)
    current_env_action = np.zeros(12, dtype=np.float64)
    base_action_multiplier = np.ones(12, dtype=np.float64)
    if args.use_training_action_scale_multipliers:
        base_action_multiplier[POLICY_JOINT_INDEX["left_ankle_pitch_joint"]] *= 0.75
        base_action_multiplier[POLICY_JOINT_INDEX["right_ankle_pitch_joint"]] *= 0.75
        base_action_multiplier[POLICY_JOINT_INDEX["left_ankle_roll_joint"]] *= 0.40
        base_action_multiplier[POLICY_JOINT_INDEX["right_ankle_roll_joint"]] *= 0.40
    target_policy_pos = POLICY_DEFAULT_POS.copy()
    delayed_target_policy_pos = target_policy_pos.copy()
    target_delay_buffer = [target_policy_pos.copy() for _ in range(args.actuator_delay_steps)]

    viewer = None
    if not args.no_render:
        if mujoco_viewer is not None:
            viewer = mujoco_viewer.MujocoViewer(model, data)
        elif mujoco_builtin_viewer is not None:
            print("[INFO] mujoco_viewer is not installed; using MuJoCo built-in passive viewer.")
            viewer = PassiveMujocoViewer(model, data)
        else:
            raise RuntimeError("No MuJoCo viewer is installed; use --no_render.")
        render_pause(model, viewer, args.stand_time)

    gif_path = None
    gif_renderer = None
    gif_camera = None
    gif_frames = []
    next_capture_time = 0.0
    mp4_path = None
    mp4_writer = None
    if args.gif_path is not None or args.mp4_path is not None:
        if args.gif_fps <= 0 or args.gif_width <= 0 or args.gif_height <= 0:
            raise ValueError("--gif_fps, --gif_width, and --gif_height must be positive.")
        import imageio.v2 as imageio

        if args.gif_path is not None:
            gif_path = Path(args.gif_path).expanduser()
            gif_path.parent.mkdir(parents=True, exist_ok=True)
        if args.mp4_path is not None:
            mp4_path = Path(args.mp4_path).expanduser()
            mp4_path.parent.mkdir(parents=True, exist_ok=True)
            mp4_writer = imageio.get_writer(
                mp4_path,
                fps=args.gif_fps,
                codec="libx264",
                bitrate=args.mp4_bitrate,
                macro_block_size=1,
            )
        # The source MJCF defaults to a 640x480 off-screen framebuffer.  Resize
        # it before constructing the renderer so high-definition recording works.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.gif_width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.gif_height)
        gif_renderer = mujoco.Renderer(model, height=args.gif_height, width=args.gif_width)
        gif_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(gif_camera)
        gif_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        gif_camera.distance = 2.4
        gif_camera.azimuth = 90.0
        gif_camera.elevation = -18.0

    if vx_schedule is None:
        print(f"[INFO] clean sim2sim: vx={args.vx:.3f}, vy={args.vy:.3f}, dyaw={args.dyaw:.3f}")
    else:
        print(
            "[INFO] clean sim2sim pure-policy vx schedule: "
            f"values={vx_schedule}, hold={args.vx_hold_time:.1f}s, transition={args.vx_transition_time:.1f}s, "
            f"vy={args.vy:.3f}, dyaw={args.dyaw:.3f}"
        )
    print("[DEBUG] Isaac policy joint order:", POLICY_JOINT_NAMES)
    print("[DEBUG] Mujoco joint order:", joint_names_mj)
    print("[DEBUG] Mujoco actuator order:", actuator_names_mj)
    print("[DEBUG] initial_foot:", foot_summary(model, data, feet))
    if args.debug_com:
        print("[DEBUG] initial_com:", com_summary(model, data, feet))
    print(
        "[DEBUG] deploy_limits:",
        {
            "action_scale": args.action_scale,
            "action_clip": args.action_clip,
            "single_obs": single_obs,
            "base_action_multiplier": base_action_multiplier.tolist(),
            "actuator_delay_steps": args.actuator_delay_steps,
            "convention_sign": convention_sign.tolist(),
            "foot_friction": args.foot_friction,
            "foot_torsional_friction": args.foot_torsional_friction,
            "foot_rolling_friction": args.foot_rolling_friction,
            "foot_solref_timeconst": args.foot_solref_timeconst,
            "foot_solref_dampratio": args.foot_solref_dampratio,
            "max_kp": float(kp.max()),
            "max_tau_limit": float(tau_limit.max()),
        },
    )

    fall_reason = None
    max_abs_raw_action = 0.0
    max_abs_env_action = 0.0
    max_abs_tau = 0.0
    hud_world_velocity = np.zeros(3, dtype=np.float64)
    hud_policy_command = np.zeros(3, dtype=np.float64)
    hud_policy_time = 0.0
    hud_schedule_stage = -1
    if args.max_steps is not None:
        max_low_steps = args.max_steps
    elif vx_schedule is not None:
        schedule_duration = args.warmup_time + len(vx_schedule) * args.vx_hold_time
        max_low_steps = int(np.ceil(schedule_duration / model.opt.timestep))
    else:
        max_low_steps = 120000
    decimation = 4
    joint_log_file = None
    joint_log_writer = None
    if args.joint_log_csv is not None:
        joint_log_path = Path(args.joint_log_csv).expanduser()
        joint_log_path.parent.mkdir(parents=True, exist_ok=True)
        joint_log_file = joint_log_path.open("w", newline="")
        joint_log_writer = csv.DictWriter(
            joint_log_file,
            fieldnames=[
                "frame",
                "time",
                "root_x",
                "root_y",
                "root_z",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "world_vx",
                "world_vy",
                "world_yaw_rate",
                "cmd_vx",
                "cmd_vy",
                "cmd_dyaw",
                "gait_sin",
                "gait_cos",
                "left_hip_yaw_q",
                "right_hip_yaw_q",
                "left_hip_roll_q",
                "right_hip_roll_q",
                "left_ankle_roll_q",
                "right_ankle_roll_q",
                "left_hip_yaw_target",
                "right_hip_yaw_target",
                "left_hip_roll_target",
                "right_hip_roll_target",
                "left_ankle_roll_target",
                "right_ankle_roll_target",
                "left_hip_yaw_policy_action",
                "right_hip_yaw_policy_action",
                "left_hip_roll_policy_action",
                "right_hip_roll_policy_action",
                "left_ankle_roll_policy_action",
                "right_ankle_roll_policy_action",
                "left_hip_yaw_action",
                "right_hip_yaw_action",
                "left_hip_roll_action",
                "right_hip_roll_action",
                "left_ankle_roll_action",
                "right_ankle_roll_action",
                "left_foot_y",
                "right_foot_y",
                "left_foot_fz",
                "right_foot_fz",
                "left_contacts",
                "right_contacts",
                "left_outer_minus_inner",
                "right_outer_minus_inner",
                "left_toe_minus_heel",
                "right_toe_minus_heel",
            ],
        )
        joint_log_writer.writeheader()

    try:
        for low_step in range(max_low_steps):
            policy_frame = low_step // decimation
            if low_step % decimation == 0:
                omega, gravity_b = root_obs(data)
                q_policy = data.qpos[qpos_adr].copy()
                dq_policy = data.qvel[dof_adr].copy()

                policy_time = policy_frame * decimation * model.opt.timestep
                warmup_phase = 1.0 if policy_time < args.warmup_time else 0.0
                walk_alpha = np.clip((policy_time - args.warmup_time) / max(args.ramp_time, 1e-6), 0.0, 1.0)
                if args.policy_action_ramp_time > 0.0:
                    policy_action_alpha = np.clip(policy_time / args.policy_action_ramp_time, 0.0, 1.0)
                else:
                    policy_action_alpha = 1.0
                if warmup_phase > 0.5:
                    policy_action_alpha *= args.warmup_action_scale
                gait_phase = 2.0 * np.pi * policy_time / max(args.gait_period, 1e-6) + args.gait_phase_offset
                schedule_stage = -1
                command_vx = args.vx
                if vx_schedule is not None:
                    schedule_time = max(policy_time - args.warmup_time, 0.0)
                    schedule_stage = min(int(schedule_time / args.vx_hold_time), len(vx_schedule) - 1)
                    command_vx = vx_schedule[schedule_stage]
                    if schedule_stage > 0 and args.vx_transition_time > 0.0:
                        transition_time = schedule_time - schedule_stage * args.vx_hold_time
                        transition_alpha = np.clip(transition_time / args.vx_transition_time, 0.0, 1.0)
                        transition_alpha = transition_alpha * transition_alpha * (3.0 - 2.0 * transition_alpha)
                        previous_vx = vx_schedule[schedule_stage - 1]
                        command_vx = previous_vx + transition_alpha * (command_vx - previous_vx)

                obs = np.zeros((1, single_obs), dtype=np.float32)
                obs[0, 0:3] = omega
                obs[0, 3:6] = gravity_b
                obs[0, 6:9] = np.array(
                    [command_vx * walk_alpha, args.vy * walk_alpha, args.dyaw * walk_alpha], dtype=np.float64
                )
                world_ang_vel = R.from_quat(quat_xyzw_from_mujoco(data)).apply(omega)
                instantaneous_world_velocity = np.array(
                    [data.qvel[0], data.qvel[1], world_ang_vel[2]], dtype=np.float64
                )
                instantaneous_world_speed_xy = float(np.linalg.norm(instantaneous_world_velocity[:2]))
                velocity_filter_alpha = 0.2
                if policy_frame == 0:
                    hud_world_velocity = instantaneous_world_velocity
                else:
                    hud_world_velocity = (
                        (1.0 - velocity_filter_alpha) * hud_world_velocity
                        + velocity_filter_alpha * instantaneous_world_velocity
                    )
                hud_policy_command = obs[0, 6:9].astype(np.float64, copy=True)
                hud_policy_time = policy_time
                hud_schedule_stage = schedule_stage
                q_policy_obs = (q_policy - POLICY_DEFAULT_POS) * convention_sign
                dq_policy_obs = dq_policy * convention_sign
                if single_obs == 48:
                    obs[0, 9] = warmup_phase
                    obs[0, 10] = np.sin(gait_phase)
                    obs[0, 11] = np.cos(gait_phase)
                    obs[0, 12:24] = q_policy_obs
                    obs[0, 24:36] = dq_policy_obs
                    obs[0, 36:48] = prev_policy_action
                else:
                    obs[0, 9:21] = q_policy_obs
                    obs[0, 21:33] = dq_policy_obs
                    obs[0, 33:45] = prev_policy_action

                if args.no_policy:
                    raw_action = np.zeros(12, dtype=np.float64)
                else:
                    if not obs_history_initialized:
                        hist_obs[:] = 0.0 if args.history_init == "zero" else obs
                        hist_obs[-1] = obs
                        obs_history_initialized = True
                    else:
                        hist_obs = np.concatenate((hist_obs[1:], obs), axis=0)
                    policy_obs = hist_obs.reshape(1, -1).astype(np.float32)
                    with torch.inference_mode():
                        raw_action = policy.act_inference({"policy": torch.from_numpy(policy_obs)})[0].cpu().numpy().astype(np.float64)

                clipped_raw_action = np.clip(raw_action, -args.action_clip, args.action_clip)
                action_multiplier = base_action_multiplier.copy()
                action_multiplier[[2, 3]] *= args.hip_roll_action_scale
                action_multiplier[[10, 11]] *= args.ankle_roll_action_scale
                action_multiplier[2] *= args.left_hip_roll_action_scale
                action_multiplier[3] *= args.right_hip_roll_action_scale
                action_multiplier[10] *= args.left_ankle_roll_action_scale
                action_multiplier[11] *= args.right_ankle_roll_action_scale
                if args.flip_left_hip_roll_action:
                    action_multiplier[2] *= -1.0
                if args.flip_right_hip_roll_action:
                    action_multiplier[3] *= -1.0
                if args.flip_left_ankle_roll_action:
                    action_multiplier[10] *= -1.0
                if args.flip_right_ankle_roll_action:
                    action_multiplier[11] *= -1.0
                clipped_raw_action *= action_multiplier
                joint_action_clips = {
                    2: args.left_hip_roll_action_clip,
                    3: args.right_hip_roll_action_clip,
                    10: args.left_ankle_roll_action_clip,
                    11: args.right_ankle_roll_action_clip,
                }
                for joint_index, joint_clip in joint_action_clips.items():
                    if joint_clip is not None:
                        clipped_raw_action[joint_index] = np.clip(clipped_raw_action[joint_index], -joint_clip, joint_clip)
                current_policy_action = clipped_raw_action * policy_action_alpha
                current_env_action = current_policy_action * convention_sign
                target_policy_pos = POLICY_DEFAULT_POS + current_env_action * args.action_scale
                joint_ranges = model.jnt_range[[joint_names_mj.index(name) for name in POLICY_JOINT_NAMES]]
                target_policy_pos = np.clip(target_policy_pos, joint_ranges[:, 0], joint_ranges[:, 1])
                prev_policy_action = current_policy_action.copy()

                max_abs_raw_action = max(max_abs_raw_action, float(np.abs(raw_action).max()))
                max_abs_env_action = max(max_abs_env_action, float(np.abs(current_env_action).max()))

                should_print_debug = (args.debug_steps > 0 and policy_frame < args.debug_steps) or (
                    args.debug_interval > 0 and policy_frame % args.debug_interval == 0
                )
                if should_print_debug:
                    quat = quat_xyzw_from_mujoco(data)
                    rpy = R.from_quat(quat).as_euler("xyz", degrees=True)
                    left_metrics = foot_metrics(model, data, feet["left"], "left")
                    right_metrics = foot_metrics(model, data, feet["right"], "right")
                    print(
                        "debug "
                        f"frame={policy_frame} "
                        f"pos=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
                        f"rpy=({rpy[0]:.1f},{rpy[1]:.1f},{rpy[2]:.1f}) "
                        f"cmd=({obs[0, 6]:.2f},{obs[0, 7]:.2f},{obs[0, 8]:.2f}) "
                        f"world_v=({instantaneous_world_velocity[0]:+.3f},{instantaneous_world_velocity[1]:+.3f},{instantaneous_world_velocity[2]:+.3f}) "
                        f"world_xy_speed={instantaneous_world_speed_xy:.3f} "
                        f"speed_stage={schedule_stage} "
                        f"warmup={warmup_phase:.0f} "
                        f"act_alpha={policy_action_alpha:.2f} "
                        f"l_roll(q/t/p/m)=({q_policy[2]:+.3f},{target_policy_pos[2]:+.3f},{current_policy_action[2]:+.3f},{current_env_action[2]:+.3f}) "
                        f"r_roll(q/t/p/m)=({q_policy[3]:+.3f},{target_policy_pos[3]:+.3f},{current_policy_action[3]:+.3f},{current_env_action[3]:+.3f}) "
                        f"l_ank_roll(q/t/p/m)=({q_policy[10]:+.3f},{target_policy_pos[10]:+.3f},{current_policy_action[10]:+.3f},{current_env_action[10]:+.3f}) "
                        f"r_ank_roll(q/t/p/m)=({q_policy[11]:+.3f},{target_policy_pos[11]:+.3f},{current_policy_action[11]:+.3f},{current_env_action[11]:+.3f}) "
                        f"foot_y=({left_metrics['center_y']:+.3f},{right_metrics['center_y']:+.3f}) "
                        f"foot_fz=({left_metrics['force_z']:.0f},{right_metrics['force_z']:.0f}) "
                        f"outer-inner=({left_metrics['outer_minus_inner']:+.3f},{right_metrics['outer_minus_inner']:+.3f}) "
                        f"raw_max={float(np.abs(raw_action).max()):.3f} "
                        f"env_act_max={float(np.abs(current_env_action).max()):.3f}"
                    )
                    if args.debug_com:
                        print("  com         ", com_summary(model, data, feet))
                    if args.debug_vectors:
                        print("  omega       ", np.round(omega, 4).tolist())
                        print("  gravity_b   ", np.round(gravity_b, 4).tolist())
                        print("  q_rel       ", np.round(q_policy_obs, 4).tolist())
                        print("  dq          ", np.round(dq_policy_obs, 4).tolist())
                        print("  raw_action  ", np.round(raw_action, 4).tolist())
                        print("  policy_action", np.round(current_policy_action, 4).tolist())
                        print("  env_action  ", np.round(current_env_action, 4).tolist())
                        print("  target_q    ", np.round(target_policy_pos, 4).tolist())
                        print("  delayed_q   ", np.round(delayed_target_policy_pos, 4).tolist())
                        print("  foot_corners", foot_corner_summary(model, data, feet))

                if joint_log_writer is not None and policy_frame % max(args.joint_log_interval, 1) == 0:
                    left_metrics = foot_metrics(model, data, feet["left"], "left")
                    right_metrics = foot_metrics(model, data, feet["right"], "right")
                    quat = quat_xyzw_from_mujoco(data)
                    rpy = R.from_quat(quat).as_euler("xyz", degrees=True)
                    gait_sin_log = np.sin(gait_phase) if single_obs == 48 else float("nan")
                    gait_cos_log = np.cos(gait_phase) if single_obs == 48 else float("nan")
                    joint_log_writer.writerow(
                        {
                            "frame": policy_frame,
                            "time": policy_time,
                            "root_x": data.qpos[0],
                            "root_y": data.qpos[1],
                            "root_z": data.qpos[2],
                            "roll_deg": rpy[0],
                            "pitch_deg": rpy[1],
                            "yaw_deg": rpy[2],
                            "world_vx": hud_world_velocity[0],
                            "world_vy": hud_world_velocity[1],
                            "world_yaw_rate": hud_world_velocity[2],
                            "cmd_vx": obs[0, 6],
                            "cmd_vy": obs[0, 7],
                            "cmd_dyaw": obs[0, 8],
                            "gait_sin": gait_sin_log,
                            "gait_cos": gait_cos_log,
                            "left_hip_yaw_q": q_policy[0],
                            "right_hip_yaw_q": q_policy[1],
                            "left_hip_roll_q": q_policy[2],
                            "right_hip_roll_q": q_policy[3],
                            "left_ankle_roll_q": q_policy[10],
                            "right_ankle_roll_q": q_policy[11],
                            "left_hip_yaw_target": target_policy_pos[0],
                            "right_hip_yaw_target": target_policy_pos[1],
                            "left_hip_roll_target": target_policy_pos[2],
                            "right_hip_roll_target": target_policy_pos[3],
                            "left_ankle_roll_target": target_policy_pos[10],
                            "right_ankle_roll_target": target_policy_pos[11],
                            "left_hip_yaw_policy_action": current_policy_action[0],
                            "right_hip_yaw_policy_action": current_policy_action[1],
                            "left_hip_roll_policy_action": current_policy_action[2],
                            "right_hip_roll_policy_action": current_policy_action[3],
                            "left_ankle_roll_policy_action": current_policy_action[10],
                            "right_ankle_roll_policy_action": current_policy_action[11],
                            "left_hip_yaw_action": current_env_action[0],
                            "right_hip_yaw_action": current_env_action[1],
                            "left_hip_roll_action": current_env_action[2],
                            "right_hip_roll_action": current_env_action[3],
                            "left_ankle_roll_action": current_env_action[10],
                            "right_ankle_roll_action": current_env_action[11],
                            "left_foot_y": left_metrics["center_y"],
                            "right_foot_y": right_metrics["center_y"],
                            "left_foot_fz": left_metrics["force_z"],
                            "right_foot_fz": right_metrics["force_z"],
                            "left_contacts": left_metrics["contact_count"],
                            "right_contacts": right_metrics["contact_count"],
                            "left_outer_minus_inner": left_metrics["outer_minus_inner"],
                            "right_outer_minus_inner": right_metrics["outer_minus_inner"],
                            "left_toe_minus_heel": left_metrics["toe_minus_heel"],
                            "right_toe_minus_heel": right_metrics["toe_minus_heel"],
                        }
                    )

            q = data.qpos[qpos_adr]
            dq = data.qvel[dof_adr]
            target_delay_buffer.append(target_policy_pos.copy())
            delayed_target_policy_pos = target_delay_buffer.pop(0)
            tau_policy = kp * (delayed_target_policy_pos - q) - kd * dq
            tau_policy = np.clip(tau_policy, -tau_limit, tau_limit)
            data.ctrl[:] = 0.0
            data.ctrl[actuator_ids] = tau_policy
            max_abs_tau = max(max_abs_tau, float(np.abs(tau_policy).max()))

            mujoco.mj_step(model, data)
            if gif_renderer is not None and data.time + 0.5 * model.opt.timestep >= next_capture_time:
                # Follow the robot while retaining the wall and the landing area in view.
                gif_camera.lookat[:] = (data.qpos[0] + 0.28, data.qpos[1], 0.38)
                gif_renderer.update_scene(data, camera=gif_camera)
                frame = gif_renderer.render().copy()
                if mp4_writer is not None:
                    mp4_writer.append_data(frame)
                if gif_path is not None:
                    gif_frames.append(frame)
                next_capture_time += 1.0 / args.gif_fps
            if viewer is not None:
                set_status_overlay(
                    viewer,
                    hud_world_velocity,
                    hud_policy_command,
                    hud_policy_time,
                    hud_schedule_stage,
                    len(vx_schedule) if vx_schedule is not None else 0,
                )
                viewer.render()
                if not viewer.is_alive:
                    fall_reason = "viewer window closed"
                    break
            if args.stop_root_z > 0.0 and data.qpos[2] < args.stop_root_z:
                fall_reason = f"root_z {data.qpos[2]:.3f} below {args.stop_root_z:.3f}"
                break
    finally:
        if joint_log_file is not None:
            joint_log_file.close()
        if gif_renderer is not None:
            gif_renderer.close()
        if mp4_writer is not None:
            mp4_writer.close()
            print(f"[INFO] Wrote replay MP4: {mp4_path}")
        if gif_path is not None and gif_frames:
            imageio.mimsave(gif_path, gif_frames, format="GIF", duration=1.0 / args.gif_fps, loop=0)
            print(f"[INFO] Wrote replay GIF with {len(gif_frames)} frames: {gif_path}")

    quat = quat_xyzw_from_mujoco(data)
    rpy = R.from_quat(quat).as_euler("xyz", degrees=True)
    print(
        "[SUMMARY] "
        f"fall_reason={fall_reason} "
        f"root_pos={np.round(data.qpos[:3], 4).tolist()} "
        f"root_euler_deg={np.round(rpy, 3).tolist()} "
        f"max_abs_raw_action={max_abs_raw_action:.3f} "
        f"max_abs_env_action={max_abs_env_action:.3f} "
        f"max_abs_tau={max_abs_tau:.3f}"
    )
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
