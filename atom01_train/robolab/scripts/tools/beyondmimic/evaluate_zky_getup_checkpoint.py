"""Evaluate a ZKY get-up policy from reference frame zero without randomization."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
for path in (ROOT / "robolab", ROOT / "rsl_rl", ROOT.parent.parent / "IsaacLab" / "source"):
    if path.exists():
        sys.path.insert(0, str(path))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--zero-actions", action="store_true", help="Evaluate reference-offset PD without a policy.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", default="ZKY-Getup-BeyondMimic-v0")
parser.add_argument("--start_frame", type=int, default=0)
parser.add_argument(
    "--root_height_offset",
    type=float,
    default=0.0,
    help="Diagnostic vertical offset applied to the reference reset root.",
)
parser.add_argument("--dump_initial_policy_io", action="store_true")
parser.add_argument("--export_rollout_npz", type=Path)
parser.add_argument(
    "--clip_actions",
    type=float,
    default=None,
    help="Override the runner raw-action clip for deployment-envelope diagnostics.",
)
parser.add_argument(
    "--playback_speed",
    type=float,
    default=None,
    help="Override the task reference playback speed for timing diagnostics.",
)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply, quat_error_magnitude, quat_inv, quat_mul, yaw_quat
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

import robolab.tasks  # noqa: F401


def disable_events(event_cfg) -> None:
    for name in vars(event_cfg):
        if not name.startswith("_"):
            setattr(event_cfg, name, None)


def set_reference_frame(env, command, frame: int) -> None:
    """Put every robot at one reference frame and synchronize the command."""
    env_ids = torch.arange(env.num_envs, device=env.device)
    if not 0 <= frame < command.motion.time_step_total:
        raise ValueError(f"--start_frame must be in [0, {command.motion.time_step_total - 1}]")
    root_pos = command.motion.body_pos_w[frame, 0].repeat(env.num_envs, 1) + env.scene.env_origins
    root_pos[:, 2] += args.root_height_offset
    root_quat = command.motion.body_quat_w[frame, 0].repeat(env.num_envs, 1)
    root_lin_vel = (
        command.motion.body_lin_vel_w[frame, 0].repeat(env.num_envs, 1) * command.playback_speeds[:, None]
    )
    root_ang_vel = (
        command.motion.body_ang_vel_w[frame, 0].repeat(env.num_envs, 1) * command.playback_speeds[:, None]
    )
    joint_pos = command.motion.joint_pos[frame].repeat(env.num_envs, 1)
    joint_vel = command.motion.joint_vel[frame].repeat(env.num_envs, 1) * command.playback_speeds[:, None]
    if frame == 0 and command.cfg.zero_velocity_at_frame_zero:
        root_lin_vel.zero_()
        root_ang_vel.zero_()
        joint_vel.zero_()

    command.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    command.robot.write_root_state_to_sim(
        torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1), env_ids=env_ids
    )
    env.scene.write_data_to_sim()
    env.sim.forward()

    command.time_steps.fill_(frame)
    command.phase_steps.fill_(float(frame))
    command.motion_ended.zero_()
    command.steps_after_motion_end.zero_()
    command._started_at_frame_zero.fill_(frame == 0)
    command._episode_start_steps.fill_(frame)
    command._has_completed_episode.fill_(True)
    command._update_action_offset()
    anchor_pos = command.anchor_pos_w[:, None, :].expand(-1, len(command.cfg.body_names), -1)
    anchor_quat = command.anchor_quat_w[:, None, :].expand(-1, len(command.cfg.body_names), -1)
    robot_anchor_pos = command.robot_anchor_pos_w[:, None, :].expand_as(anchor_pos)
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand_as(anchor_quat)
    delta_pos = robot_anchor_pos.clone()
    delta_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
    command.body_pos_relative_w = delta_pos + quat_apply(delta_quat, command.body_pos_w - anchor_pos)
    command.body_quat_relative_w = quat_mul(delta_quat, command.body_quat_w)


def capture_motion_state(command) -> dict[str, float | str]:
    command._update_metrics()
    sensor = command._env.scene.sensors["contact_forces"]
    contact_force = torch.linalg.vector_norm(sensor.data.net_forces_w_history, dim=-1).amax(dim=1)
    foot_ids = [sensor.body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    nonfoot_ids = [
        index
        for index, name in enumerate(sensor.body_names)
        if name in ("base_link", "body_link")
        or "hip_" in name
        or "knee" in name
        or "elbow" in name
        or "shoulder_" in name
    ]
    applied_effort = torch.abs(command.robot.data.applied_torque)
    computed_effort = torch.abs(command.robot.data.computed_torque)
    actuator_effort_limits = torch.empty_like(applied_effort)
    actuator_effort_limits.fill_(float("nan"))
    simulator_effort_limits = torch.empty_like(applied_effort)
    simulator_effort_limits.fill_(float("nan"))
    for actuator in command.robot.actuators.values():
        joint_ids = actuator.joint_indices
        actuator_effort_limits[:, joint_ids] = actuator.effort_limit
        simulator_effort_limits[:, joint_ids] = actuator.effort_limit_sim

    actuator_ratio = applied_effort / actuator_effort_limits.clamp_min(1.0e-6)
    mean_actuator_ratio = actuator_ratio.mean(dim=0)
    max_effort_joint_id = int(mean_actuator_ratio.argmax().item())
    return {
        "reference_height": command.anchor_pos_w[:, 2].mean().item(),
        "base_height": command.robot_anchor_pos_w[:, 2].mean().item(),
        "upright": (-command.robot.data.projected_gravity_b[:, 2]).mean().item(),
        "joint_error": command.metrics["error_joint_pos"].mean().item(),
        "joint_speed": torch.linalg.vector_norm(command.robot_joint_vel, dim=-1).mean().item(),
        "foot_contacts": (contact_force[:, foot_ids] > 20.0).float().sum(dim=1).mean().item(),
        "nonfoot_contacts": (contact_force[:, nonfoot_ids] > 20.0).float().sum(dim=1).mean().item(),
        "max_effort_ratio": mean_actuator_ratio[max_effort_joint_id].item(),
        "max_effort_joint": command.robot.joint_names[max_effort_joint_id],
        "computed_effort": computed_effort[:, max_effort_joint_id].mean().item(),
        "applied_effort": applied_effort[:, max_effort_joint_id].mean().item(),
        "actuator_effort_limit": actuator_effort_limits[:, max_effort_joint_id].mean().item(),
        "simulator_effort_limit": simulator_effort_limits[:, max_effort_joint_id].mean().item(),
        "data_effort_limit": command.robot.data.joint_effort_limits[:, max_effort_joint_id].mean().item(),
        "saturation_fraction": (
            computed_effort >= 0.98 * actuator_effort_limits
        ).float().mean().item(),
    }


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    agent_cfg.device = args.device
    if args.clip_actions is not None:
        if args.clip_actions <= 0.0:
            raise ValueError("--clip_actions must be positive")
        agent_cfg.clip_actions = args.clip_actions
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.motion.pose_range = {key: (0.0, 0.0) for key in env_cfg.commands.motion.pose_range}
    env_cfg.commands.motion.velocity_range = {key: (0.0, 0.0) for key in env_cfg.commands.motion.velocity_range}
    env_cfg.commands.motion.joint_position_range = (0.0, 0.0)
    if args.playback_speed is not None:
        if args.playback_speed <= 0.0:
            raise ValueError("--playback_speed must be positive")
        env_cfg.commands.motion.playback_speed = args.playback_speed
        env_cfg.commands.motion.playback_speed_schedule = ()
        env_cfg.commands.motion.playback_speed_range_schedule = ()
    # Keep the strict evaluator alive for the full configured horizon so its
    # final snapshot is captured before Gym auto-resets a completed episode.
    if hasattr(env_cfg.terminations, "motion_complete"):
        env_cfg.terminations.motion_complete = None
    # The ZKY task already disables all stochastic events. Keep its remaining
    # deterministic (1.0, 1.0) joint-parameter event so scene startup follows
    # the same manager lifecycle as training.

    raw_env = gym.make(args.task, cfg=env_cfg)
    raw_env.reset()
    command = raw_env.unwrapped.command_manager.get_term("motion")

    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    if args.zero_actions:
        policy = lambda obs: torch.zeros((obs.shape[0], env.num_actions), device=obs.device)
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --zero-actions is used")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Runner construction may reset the wrapped environment. Force the exact
    # deployment initial state only after the policy has been constructed.
    set_reference_frame(env.unwrapped, command, args.start_frame)
    env.unwrapped.episode_length_buf.zero_()
    command._update_metrics()
    print(f"initial_anchor_error_mean={command.metrics['error_anchor_pos'].mean().item():.6f}")
    print(f"initial_body_error_mean={command.metrics['error_body_pos'].mean().item():.6f}")
    print(f"initial_joint_error_mean={command.metrics['error_joint_pos'].mean().item():.6f}")
    print("policy_joint_order=" + ",".join(command.robot.joint_names))
    obs = env.get_observations()
    if args.dump_initial_policy_io:
        with torch.inference_mode():
            initial_action = policy(obs)
        actor_obs = obs[0] if isinstance(obs, tuple) else obs
        actor_obs = actor_obs["policy"] if isinstance(actor_obs, Mapping) else actor_obs
        print("initial_policy_observation=" + repr(actor_obs[0].detach().cpu().tolist()))
        print("initial_policy_action=" + repr(initial_action[0].detach().cpu().tolist()))

    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
    failed_anchor_pos = torch.zeros_like(active)
    failed_anchor_ori = torch.zeros_like(active)
    term_names = set(env.unwrapped.termination_manager.active_terms)
    snapshot = {}
    milestone_frames = (0, 50, 100, 150, 200, 228)
    milestone_states = {args.start_frame: capture_motion_state(command)}
    max_steps = int(env.unwrapped.max_episode_length)
    rollout_root_pos = []
    rollout_root_quat = []
    rollout_joint_pos = []
    rollout_body_pos = []
    rollout_body_quat = []
    rollout_reference_frame = []
    rollout_reference_root_pos = []
    rollout_reference_root_quat = []
    rollout_reference_joint_pos = []
    rollout_reference_body_pos = []
    rollout_reference_body_quat = []
    rollout_action = []
    rollout_joint_mae = []
    rollout_body_rmse = []
    rollout_root_pos_error = []
    rollout_root_ori_error = []
    rollout_contact_force = []
    rollout_effort_ratio = []

    def capture_rollout_frame(action: torch.Tensor | None = None) -> None:
        env_origin = env.unwrapped.scene.env_origins[0]
        reference_frame = int(command.time_steps[0].item())
        actual_body_pos = command.robot_body_pos_w[0] - env_origin
        reference_body_pos = command.body_pos_w[0] - env_origin
        reference_root_pos = command.anchor_pos_w[0] - env_origin
        joint_error = torch.abs(command.robot_joint_pos[0] - command.joint_pos[0])
        body_error = command.robot_body_pos_w[0] - command.body_pos_relative_w[0]
        sensor = env.unwrapped.scene.sensors["contact_forces"]
        contact_force = torch.linalg.vector_norm(sensor.data.net_forces_w_history[0], dim=-1).amax(dim=0)
        effort_limits = torch.empty_like(command.robot.data.applied_torque[0])
        for actuator in command.robot.actuators.values():
            actuator_limits = actuator.effort_limit
            if isinstance(actuator_limits, torch.Tensor) and actuator_limits.ndim > 1:
                actuator_limits = actuator_limits[0]
            effort_limits[actuator.joint_indices] = actuator_limits
        rollout_root_pos.append(
            (command.robot.data.root_pos_w[0] - env_origin).detach().cpu().numpy().copy()
        )
        rollout_root_quat.append(command.robot.data.root_quat_w[0].detach().cpu().numpy().copy())
        rollout_joint_pos.append(command.robot.data.joint_pos[0].detach().cpu().numpy().copy())
        rollout_body_pos.append(actual_body_pos.detach().cpu().numpy().copy())
        rollout_body_quat.append(command.robot_body_quat_w[0].detach().cpu().numpy().copy())
        rollout_reference_frame.append(reference_frame)
        rollout_reference_root_pos.append(reference_root_pos.detach().cpu().numpy().copy())
        rollout_reference_root_quat.append(command.anchor_quat_w[0].detach().cpu().numpy().copy())
        rollout_reference_joint_pos.append(command.joint_pos[0].detach().cpu().numpy().copy())
        rollout_reference_body_pos.append(reference_body_pos.detach().cpu().numpy().copy())
        rollout_reference_body_quat.append(command.body_quat_w[0].detach().cpu().numpy().copy())
        if action is None:
            action = torch.zeros(env.num_actions, device=env.unwrapped.device)
        elif action.ndim == 2:
            action = action[0]
        rollout_action.append(action.detach().cpu().numpy().copy())
        rollout_joint_mae.append(joint_error.mean().item())
        rollout_body_rmse.append(torch.sqrt(torch.mean(torch.square(body_error))).item())
        rollout_root_pos_error.append(torch.linalg.vector_norm(command.robot_anchor_pos_w[0] - command.anchor_pos_w[0]).item())
        rollout_root_ori_error.append(
            quat_error_magnitude(command.robot_anchor_quat_w[0:1], command.anchor_quat_w[0:1]).item()
        )
        rollout_contact_force.append(contact_force.detach().cpu().numpy().copy())
        rollout_effort_ratio.append(
            (torch.abs(command.robot.data.applied_torque[0]) / effort_limits.clamp_min(1.0e-6))
            .detach().cpu().numpy().copy()
        )

    # Always retain the single-environment diagnostic trace. It is small and
    # lets headless checkpoint monitoring report effort safety without having
    # to write a rollout archive.
    capture_rollout_frame()

    for step in range(max_steps):
        if step == max_steps - 2:
            command._update_metrics()
            snapshot = {
                "base_height": command.robot_anchor_pos_w[:, 2].clone(),
                "upright": (-command.robot.data.projected_gravity_b[:, 2]).clone(),
                "joint_mae": torch.mean(torch.abs(command.joint_pos - command.robot_joint_pos), dim=-1),
                "body_rmse": torch.sqrt(
                    torch.mean(torch.square(command.body_pos_relative_w - command.robot_body_pos_w), dim=(1, 2))
                ),
                "anchor_error": command.metrics["error_anchor_pos"].clone(),
                "anchor_ori_error": quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w),
                "joint_speed": torch.linalg.vector_norm(command.robot_joint_vel, dim=-1).clone(),
            }
        with torch.inference_mode():
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
        # The final env.step() reaches the time limit and auto-resets the
        # environment.  Do not append that reset state to the rollout.
        if step < max_steps - 1:
            capture_rollout_frame(action)
        current_frame = int(command.time_steps[0].item())
        for milestone_frame in milestone_frames:
            if milestone_frame not in milestone_states and current_frame >= milestone_frame:
                milestone_states[milestone_frame] = capture_motion_state(command)
        newly_done = active & dones.bool()
        if newly_done.any():
            if "anchor_pos" in term_names:
                failed_anchor_pos |= newly_done & env.unwrapped.termination_manager.get_term("anchor_pos")
            if "anchor_ori" in term_names:
                failed_anchor_ori |= newly_done & env.unwrapped.termination_manager.get_term("anchor_ori")
            active &= ~newly_done

    termination_success = ~(failed_anchor_pos | failed_anchor_ori)
    physical_success = (
        (snapshot["base_height"] >= 0.88)
        & (snapshot["upright"] >= 0.90)
        & (snapshot["joint_speed"] < 2.0)
    )
    tracking_success = (
        (snapshot["joint_mae"] <= 0.50)
        & (snapshot["body_rmse"] <= 0.15)
        & (snapshot["anchor_error"] <= 0.12)
        & (snapshot["anchor_ori_error"] <= 0.55)
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"playback_speed={command.playback_speeds.mean().item():.6f}")
    print(f"environments={env.num_envs} episode_steps={max_steps} reference_frames={command.motion.time_step_total}")
    print(f"termination_success_rate={termination_success.float().mean().item():.6f}")
    print(f"physical_success_rate={physical_success.float().mean().item():.6f}")
    print(f"tracking_success_rate={tracking_success.float().mean().item():.6f}")
    print(f"anchor_pos_failure_rate={failed_anchor_pos.float().mean().item():.6f}")
    print(f"anchor_ori_failure_rate={failed_anchor_ori.float().mean().item():.6f}")
    for name, values in snapshot.items():
        print(f"final_{name}_mean={values.mean().item():.6f} final_{name}_max={values.max().item():.6f}")
    effort_ratio_history = np.asarray(rollout_effort_ratio)
    print(f"rollout_max_effort_ratio={effort_ratio_history.max():.6f}")
    print(
        "rollout_effort_saturation_fraction="
        f"{np.mean(effort_ratio_history >= 0.98):.6f}"
    )
    for frame in milestone_frames:
        state = milestone_states.get(frame)
        if state is not None:
            values = " ".join(
                f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}"
                for name, value in state.items()
            )
            print(f"frame_{frame}: {values}")
    if args.export_rollout_npz is not None:
        joint_mae = np.asarray(rollout_joint_mae)
        per_joint_mae = np.mean(
            np.abs(np.asarray(rollout_joint_pos) - np.asarray(rollout_reference_joint_pos)), axis=0
        )
        print(f"rollout_joint_mae={joint_mae.mean():.6f}")
        print(f"rollout_body_rmse={np.mean(rollout_body_rmse):.6f}")
        print(f"rollout_root_pos_error={np.mean(rollout_root_pos_error):.6f}")
        print(f"rollout_root_ori_error={np.mean(rollout_root_ori_error):.6f}")
        first_bad = np.flatnonzero(joint_mae > 0.20)
        print(f"first_joint_mae_over_0.20_frame={int(first_bad[0]) if len(first_bad) else -1}")
        for joint_id in np.argsort(per_joint_mae)[::-1]:
            print(f"joint_mae/{command.robot.joint_names[joint_id]}={per_joint_mae[joint_id]:.6f}")
        output_path = args.export_rollout_npz.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            fps=np.asarray(round(1.0 / env.unwrapped.step_dt)),
            root_pos=np.asarray(rollout_root_pos),
            root_quat=np.asarray(rollout_root_quat),
            joint_pos=np.asarray(rollout_joint_pos),
            body_pos=np.asarray(rollout_body_pos),
            body_quat=np.asarray(rollout_body_quat),
            reference_frame=np.asarray(rollout_reference_frame),
            reference_root_pos=np.asarray(rollout_reference_root_pos),
            reference_root_quat=np.asarray(rollout_reference_root_quat),
            reference_joint_pos=np.asarray(rollout_reference_joint_pos),
            reference_body_pos=np.asarray(rollout_reference_body_pos),
            reference_body_quat=np.asarray(rollout_reference_body_quat),
            action=np.asarray(rollout_action),
            joint_mae=joint_mae,
            body_rmse=np.asarray(rollout_body_rmse),
            root_pos_error=np.asarray(rollout_root_pos_error),
            root_ori_error=np.asarray(rollout_root_ori_error),
            contact_force=np.asarray(rollout_contact_force),
            effort_ratio=np.asarray(rollout_effort_ratio),
            joint_names=np.asarray(command.robot.joint_names),
            body_names=np.asarray(command.cfg.body_names),
            contact_body_names=np.asarray(env.unwrapped.scene.sensors["contact_forces"].body_names),
            source_checkpoint=np.asarray(str(args.checkpoint)),
        )
        print(f"exported_rollout={output_path} frames={len(rollout_joint_pos)}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
