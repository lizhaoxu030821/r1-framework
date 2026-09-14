"""Replay the 19-DoF ZKY BeyondMimic get-up policy in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

try:
    import mujoco.viewer
except ImportError:
    mujoco.viewer = None


POLICY_JOINT_NAMES = [
    "body_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--action-scale", type=float, default=0.50)
    parser.add_argument("--playback-speed", type=float, default=0.50)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument("--episode-length", type=float, default=11.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=50)
    parser.add_argument("--dump-initial-policy-io", action="store_true")
    parser.add_argument(
        "--reference-visualization",
        action="store_true",
        help="Visualize the NPZ kinematically without running the policy or MuJoCo dynamics.",
    )
    parser.add_argument(
        "--foot-only-contact",
        action="store_true",
        help="Keep the source MJCF's foot-only collision model instead of matching IsaacLab ground contact.",
    )
    return parser.parse_args()


def load_actor(checkpoint_path: Path) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    weights = []
    for key, value in state.items():
        if key.startswith("actor.") and key.endswith(".weight"):
            weights.append((int(key.split(".")[1]), value))
    weights.sort()
    if not weights or weights[0][1].shape[1] != 101 or weights[-1][1].shape[0] != 19:
        raise RuntimeError("Checkpoint must contain a 101-D observation, 19-D action actor.")

    modules: list[torch.nn.Module] = []
    for layer_index, (state_index, weight) in enumerate(weights):
        linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(state[f"actor.{state_index}.bias"])
        modules.append(linear)
        if layer_index != len(weights) - 1:
            modules.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*modules)
    actor.eval()
    return actor


def configure_ground_contacts(model: mujoco.MjModel) -> None:
    """Match IsaacLab: explicit link collisions contact ground, without self-collision."""
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    if ground_id < 0:
        raise RuntimeError("MJCF has no geom named 'ground'.")
    # A dedicated pair of bits makes robot-ground pairs collide while all
    # robot-robot pairs fail MuJoCo's collision mask test.
    model.geom_contype[ground_id] = 1
    model.geom_conaffinity[ground_id] = 0
    collision_count = 0
    for geom_id in range(model.ngeom):
        if geom_id == ground_id:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name.endswith("_collision"):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 1
            model.geom_friction[geom_id] = (1.0, 0.3, 0.3)
            model.geom_solref[geom_id] = (0.02, 1.0)
            model.geom_solimp[geom_id] = (0.90, 0.95, 0.001, 0.5, 2.0)
            collision_count += 1
        elif model.geom_bodyid[geom_id] != 0:
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
    if collision_count != 20:
        raise RuntimeError(
            f"Expected 20 explicit *_collision geoms, found {collision_count}. "
            "Run build_zky_isaac_collision_mjcf.py first."
        )


def model_mapping(model: mujoco.MjModel, joint_names: list[str]):
    qpos_ids, qvel_ids, actuator_ids = [], [], []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id < 0 or actuator_id < 0:
            raise RuntimeError(f"MJCF is missing joint/actuator {name!r}.")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        qvel_ids.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(actuator_id)
    return np.asarray(qpos_ids), np.asarray(qvel_ids), np.asarray(actuator_ids)


def gains_and_limits(joint_names: list[str]):
    kp, kd, effort = [], [], []
    for name in joint_names:
        if "hip" in name:
            kp.append(100.0); kd.append(3.3); effort.append(150.0)
        elif "knee" in name:
            kp.append(150.0); kd.append(5.0); effort.append(150.0)
        elif "ankle" in name:
            kp.append(40.0); kd.append(2.0); effort.append(35.0)
        elif name == "body_joint":
            kp.append(150.0); kd.append(5.0); effort.append(150.0)
        elif "shoulder" in name:
            kp.append(40.0); kd.append(2.0); effort.append(70.0)
        elif "elbow" in name:
            kp.append(30.0); kd.append(1.5); effort.append(70.0)
        else:
            raise RuntimeError(f"No actuator parameters for {name!r}.")
    return np.asarray(kp), np.asarray(kd), np.asarray(effort)


def configure_joint_armature(model: mujoco.MjModel, joint_names: list[str]) -> None:
    """Match the explicit actuator armatures used by the IsaacLab task."""
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_id = int(model.jnt_dofadr[joint_id])
        model.dof_armature[dof_id] = 0.04 if ("ankle" in name or "shoulder" in name or "elbow" in name) else 0.05


def projected_gravity_and_angular_velocity(data: mujoco.MjData):
    quat = data.qpos[3:7]
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    gravity_b = rotation.T @ np.array([0.0, 0.0, -1.0])
    angular_velocity_b = rotation.T @ data.qvel[3:6]
    return gravity_b, angular_velocity_b


def run_episode(args, model, data, actor, motion, joint_names, motion_joint_ids, qpos_ids, qvel_ids, actuator_ids, viewer):
    joint_pos_ref = motion["joint_pos"][:, motion_joint_ids]
    joint_vel_ref = motion["joint_vel"][:, motion_joint_ids]
    body_pos = motion["body_pos_w"]
    body_quat = motion["body_quat_w"]
    kp, kd, effort = gains_and_limits(joint_names)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = body_pos[0, 0]
    data.qpos[3:7] = body_quat[0, 0]
    data.qpos[qpos_ids] = joint_pos_ref[0]
    mujoco.mj_forward(model, data)

    last_action = np.zeros(19, dtype=np.float32)
    control_steps = int(round(args.episode_length / args.control_dt))
    physics_steps = int(round(args.control_dt / model.opt.timestep))
    if not np.isclose(physics_steps * model.opt.timestep, args.control_dt):
        raise ValueError("--control-dt must be an integer multiple of the MJCF timestep.")

    start_wall = time.monotonic()
    for control_step in range(control_steps):
        phase = min(control_step * args.playback_speed, len(joint_pos_ref) - 1)
        frame = int(np.floor(phase))
        if args.reference_visualization:
            data.qpos[:3] = body_pos[frame, 0]
            data.qpos[3:7] = body_quat[frame, 0]
            data.qpos[qpos_ids] = joint_pos_ref[frame]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            if viewer is not None:
                viewer.sync()
                deadline = start_wall + (control_step + 1) * args.control_dt / max(args.realtime_factor, 1.0e-6)
                time.sleep(max(0.0, deadline - time.monotonic()))
                if not viewer.is_running():
                    return False
            continue
        command = np.concatenate((joint_pos_ref[frame], joint_vel_ref[frame] * args.playback_speed))
        gravity_b, angular_velocity_b = projected_gravity_and_angular_velocity(data)
        q = data.qpos[qpos_ids].copy()
        dq = data.qvel[qvel_ids].copy()
        observation = np.concatenate((command, angular_velocity_b, gravity_b, q, dq, last_action)).astype(np.float32)
        with torch.inference_mode():
            action = actor(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
        if control_step == 0 and args.dump_initial_policy_io:
            print(f"initial_policy_observation={observation.tolist()}")
            print(f"initial_policy_action={action.tolist()}")
        target = joint_pos_ref[frame] + args.action_scale * action
        target = np.clip(target, model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names], 0],
                         model.jnt_range[[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names], 1])
        last_action = action.astype(np.float32)

        saturated = 0.0
        for _ in range(physics_steps):
            q = data.qpos[qpos_ids]
            dq = data.qvel[qvel_ids]
            raw_torque = kp * (target - q) - kd * dq
            saturated = max(saturated, float(np.mean(np.abs(raw_torque) >= effort)))
            data.ctrl[actuator_ids] = np.clip(raw_torque, -effort, effort)
            mujoco.mj_step(model, data)

        if args.debug_interval > 0 and control_step % args.debug_interval == 0:
            upright = -projected_gravity_and_angular_velocity(data)[0][2]
            print(
                f"step={control_step:03d} frame={frame:03d} base_z={data.qpos[2]:.3f} "
                f"upright={upright:.3f} joint_speed={np.linalg.norm(data.qvel[qvel_ids]):.3f} "
                f"saturation={saturated:.3f}"
            )
        if not np.all(np.isfinite(data.qpos)):
            raise RuntimeError(f"MuJoCo state became non-finite at control step {control_step}.")
        if viewer is not None:
            viewer.sync()
            deadline = start_wall + (control_step + 1) * args.control_dt / max(args.realtime_factor, 1.0e-6)
            time.sleep(max(0.0, deadline - time.monotonic()))
            if not viewer.is_running():
                return False

    upright = -projected_gravity_and_angular_velocity(data)[0][2]
    print(
        f"final_base_height={data.qpos[2]:.6f} final_upright={upright:.6f} "
        f"final_joint_speed={np.linalg.norm(data.qvel[qvel_ids]):.6f}"
    )
    return True


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.xml, args.npz):
        if not path.is_file():
            raise FileNotFoundError(path)
    actor = load_actor(args.checkpoint)
    motion = np.load(args.npz)
    stored_joint_names = motion["joint_names"].tolist()
    if set(stored_joint_names) != set(POLICY_JOINT_NAMES):
        missing = sorted(set(POLICY_JOINT_NAMES) - set(stored_joint_names))
        extra = sorted(set(stored_joint_names) - set(POLICY_JOINT_NAMES))
        raise RuntimeError(f"NPZ joint names mismatch: missing={missing}, extra={extra}.")
    joint_names = POLICY_JOINT_NAMES
    motion_joint_ids = np.asarray([stored_joint_names.index(name) for name in joint_names])

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    if not args.foot_only_contact:
        configure_ground_contacts(model)
    configure_joint_armature(model, joint_names)
    data = mujoco.MjData(model)
    qpos_ids, qvel_ids, actuator_ids = model_mapping(model, joint_names)
    print("joint_order=" + ",".join(joint_names))
    print(f"contacts={'foot-only' if args.foot_only_contact else 'whole-body-ground'} action_scale={args.action_scale}")

    if args.headless:
        run_episode(args, model, data, actor, motion, joint_names, motion_joint_ids, qpos_ids, qvel_ids, actuator_ids, None)
        return
    if mujoco.viewer is None:
        raise RuntimeError("MuJoCo passive viewer is unavailable; install the mujoco package with viewer support.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if not run_episode(
                args, model, data, actor, motion, joint_names, motion_joint_ids,
                qpos_ids, qvel_ids, actuator_ids, viewer
            ):
                break
            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                break


if __name__ == "__main__":
    main()
