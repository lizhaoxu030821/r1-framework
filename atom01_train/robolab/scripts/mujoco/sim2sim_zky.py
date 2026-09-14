# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import mujoco, mujoco_viewer
import sys
import contextlib
from pathlib import Path
from itertools import count
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[3]  # .../modules/atom01_train
for p in [
    ROOT / "robolab",
    ROOT / "rsl_rl",
    ROOT.parent.parent / "IsaacLab" / "source",
]:
    if p.exists():
        sys.path.insert(0, str(p))

from robolab.assets import ISAAC_DATA_DIR
import torch
import os
import cv2
import matplotlib.pyplot as plt # Import matplotlib

class cmd:
    vx = 0.5
    vy = 0
    dyaw = 0.0

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def _geom_vertices_world(model, data, geom_id):
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        vert_adr = model.mesh_vertadr[mesh_id]
        vert_num = model.mesh_vertnum[mesh_id]
        local = model.mesh_vert[vert_adr : vert_adr + vert_num]
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        size = model.geom_size[geom_id]
        local = np.array(
            [
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.double,
        )
    else:
        center = data.geom_xpos[geom_id].copy()
        radius = model.geom_rbound[geom_id]
        return np.array([center - radius, center + radius], dtype=np.double)

    rot = data.geom_xmat[geom_id].reshape(3, 3)
    return data.geom_xpos[geom_id] + local @ rot.T

def _foot_contact_summary(model, data, foot_geom_ids):
    summary = {}
    for side, geom_ids in foot_geom_ids.items():
        normal_force = 0.0
        count = 0
        positions = []
        min_z = np.inf
        max_z = -np.inf
        centers = []
        geom_id_set = set(geom_ids)
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 not in geom_id_set and contact.geom2 not in geom_id_set:
                continue
            force = np.zeros(6, dtype=np.double)
            mujoco.mj_contactForce(model, data, i, force)
            normal_force += float(force[0])
            count += 1
            positions.append(contact.pos.copy())
        for geom_id in geom_ids:
            vertices = _geom_vertices_world(model, data, geom_id)
            min_z = min(min_z, float(np.min(vertices[:, 2])))
            max_z = max(max_z, float(np.max(vertices[:, 2])))
            centers.append(data.geom_xpos[geom_id].copy())
        summary[side] = {
            "count": count,
            "fz": normal_force,
            "min_z": min_z,
            "max_z": max_z,
            "center": np.mean(centers, axis=0),
            "contact_center": np.mean(positions, axis=0) if positions else None,
        }
    return summary

def _find_foot_geom_ids(model, side):
    ids = []
    prefix = f"{side}_foot_collision"
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is not None and name.startswith(prefix):
            ids.append(geom_id)
    if not ids:
        raise ValueError(f"Cannot find Mujoco foot collision geoms with prefix '{prefix}'.")
    return ids

def _compact_foot_summary(foot_summary):
    return " ".join(
        f"{side[0]}:n{info['count']},fz{info['fz']:.0f},z{info['min_z']:.3f}/{info['max_z']:.3f}"
        for side, info in foot_summary.items()
    )

#跟随机器人的摄像机
def update_camera_lookat(camera, data, height_offset):
    base_pos = data.qpos[:3].astype(np.float64)
    camera.lookat[:] = [base_pos[0], base_pos[1], base_pos[2] + height_offset]

def run_mujoco(policy, cfg, headless=False, debug_steps=0, no_policy=False, max_steps=None, no_render=False):
    """
    Run the Mujoco simulation using the provided policy and configuration.

    Args:
        policy: The policy used for controlling the simulation.
        cfg: The configuration object containing simulation settings.
        headless: If True, run without GUI and save video.

    Returns:
        None
    """
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[:3] = cfg.robot_config.default_root_pos
    data.qpos[3:7] = cfg.robot_config.default_root_quat
    data.qpos[-cfg.robot_config.num_actions:] = cfg.robot_config.default_pos
    mujoco.mj_forward(model, data)
    foot_geom_ids = {
        "left": _find_foot_geom_ids(model, "left"),
        "right": _find_foot_geom_ids(model, "right"),
    }
    if cfg.robot_config.auto_root_height:
        foot_min_z = min(
            np.min(_geom_vertices_world(model, data, geom_id)[:, 2])
            for geom_ids in foot_geom_ids.values()
            for geom_id in geom_ids
        )
        data.qpos[2] += cfg.robot_config.foot_ground_clearance - foot_min_z
        mujoco.mj_forward(model, data)
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    actuated_joint_names = joint_names[-cfg.robot_config.num_actions:]
    urdf2usd = [0] * cfg.robot_config.num_actions
    for policy_idx, mujoco_idx in enumerate(cfg.robot_config.usd2urdf):
        urdf2usd[mujoco_idx] = policy_idx
    if debug_steps > 0:
        print(
            "[DEBUG] model "
            f"joints={model.njnt} actuators={model.nu} "
            f"root_z={data.qpos[2]:.3f} "
            f"default_pos={np.round(cfg.robot_config.default_pos, 3).tolist()}"
        )
        if cfg.sim_config.verbose_debug:
            print("[DEBUG] Mujoco joint_names:", joint_names)
            print("[DEBUG] Mujoco actuator_names:", actuator_names)
            print("[DEBUG] coord_signs:", cfg.robot_config.coord_signs.tolist())
            print("[DEBUG] action_signs:", cfg.robot_config.action_signs.tolist())
        print(
            "[DEBUG] deploy_limits:",
            {
                "action_scale": cfg.robot_config.action_scale,
                "action_clip": cfg.robot_config.action_clip,
                "action_filter": cfg.robot_config.action_filter,
                "max_kp": float(np.max(cfg.robot_config.kps)),
                "max_kd": float(np.max(cfg.robot_config.kds)),
                "max_tau_limit": float(np.max(cfg.robot_config.tau_limit)),
            },
        )
        print("[DEBUG] initial_foot:", _compact_foot_summary(_foot_contact_summary(model, data, foot_geom_ids)))
    
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    os.environ['MUJOCO_GL'] = 'glfw'
    # 根据 headless 参数选择渲染模式
    if no_render:
        renderer = None
        out = None
        cam = None
        viewer = None
    elif headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        # 设置视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 创建并配置相机
        cam = mujoco.MjvCamera()
        cam.distance = 4.0      # 增加距离以获得更好的视角
        cam.azimuth = 45.0     # 水平旋转角度
        cam.elevation = -20.0   # 垂直俯仰角度
        update_camera_lookat(cam, data, cfg.camera_config.lookat_height_offset)
        out = cv2.VideoWriter('simulation.mp4', fourcc, 1.0/cfg.sim_config.dt/cfg.sim_config.decimation, (1920, 1080))
    else:
        mode = 'window'
        viewer = mujoco_viewer.MujocoViewer(model, data, mode=mode, width=1920, height=1080)
        # 设置窗口模式下的相机参数
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        update_camera_lookat(viewer.cam, data, cfg.camera_config.lookat_height_offset)


    target_pos = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    last_policy_action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    raw_action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
    policy_action = np.zeros((cfg.robot_config.num_actions), dtype=np.double)

    if no_policy:
        policy_frame_stack = cfg.robot_config.frame_stack
    else:
        actor_input_dim = policy.actor[0].in_features
        if actor_input_dim % cfg.robot_config.num_single_obs != 0:
            raise ValueError(
                f"Actor input dim {actor_input_dim} is not compatible with "
                f"single observation dim {cfg.robot_config.num_single_obs}."
            )
        policy_frame_stack = actor_input_dim // cfg.robot_config.num_single_obs
    hist_obs = np.zeros((policy_frame_stack, cfg.robot_config.num_single_obs), dtype=np.float32)
    hist_obs.fill(0.0)

    count_lowlevel = 0
    max_abs_omega = 0.0
    max_abs_action = 0.0
    max_abs_tau = 0.0
    fall_reason = None
    printed_first_policy_obs = False

    # --- Data collection lists for plotting (LOW FREQUENCY ONLY) ---
    time_data = []
    commanded_joint_pos_data = []
    actual_joint_pos_data = []
    tau = np.zeros((cfg.robot_config.num_actions), dtype=np.double)  # Initialize tau
    tau_data = []
    commanded_lin_vel_x_data = []
    commanded_lin_vel_y_data = []
    commanded_ang_vel_z_data = []
    actual_lin_vel_data = [] # Store [vx, vy] at low freq
    actual_ang_vel_data = [] # Store [wz] at low freq
    # -------------------------------------------------------------
    progress = tqdm(count(), desc="Simulating...", unit="step")
    interrupted = False
    start_root_pos = data.qpos[:3].copy()
    try:
        for step in progress:

            # Obtain an observation
            q, dq, quat, v, omega, gvec = get_obs(data)
            root_euler = R.from_quat(quat).as_euler("xyz", degrees=True)
            max_abs_omega = max(max_abs_omega, float(np.max(np.abs(omega))))
            q = q[-cfg.robot_config.num_actions:]
            dq = dq[-cfg.robot_config.num_actions:]

            # 1000hz -> 100hz/50hz
            if count_lowlevel % cfg.sim_config.decimation == 0:
                q_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
                dq_obs = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
                q_ = q - cfg.robot_config.default_pos
                for i in range(len(cfg.robot_config.usd2urdf)):
                    q_obs[i] = cfg.robot_config.coord_signs[i] * q_[cfg.robot_config.usd2urdf[i]]
                    dq_obs[i] = cfg.robot_config.coord_signs[i] * dq[cfg.robot_config.usd2urdf[i]]

                obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)

                obs[0, 0:3] = cfg.robot_config.ang_vel_signs * omega
                obs[0, 3:6] = cfg.robot_config.gravity_signs * gvec
                sim_time = count_lowlevel * cfg.sim_config.dt
                settling = sim_time < cfg.sim_config.settle_time
                # IsaacLab starts the policy episode at t=0 with warmup_phase=1.
                # The MuJoCo settle phase is only a deployment convenience, so it
                # must not shift gait/warmup time or fill the policy history.
                policy_time = max(sim_time - cfg.sim_config.settle_time, 0.0)
                walk_ramp = np.clip(
                    (policy_time - cfg.sim_config.warmup_time) / max(cfg.sim_config.ramp_time, 1e-6),
                    0.0,
                    1.0,
                )
                warmup_phase = 1.0 if policy_time < cfg.sim_config.warmup_time else 0.0
                gait_phase = 2.0 * np.pi * policy_time / max(cfg.sim_config.gait_period, 1e-6)
                cmd_vx = cfg.robot_config.command_signs[0] * cmd.vx * walk_ramp
                cmd_vy = cfg.robot_config.command_signs[1] * cmd.vy * walk_ramp
                cmd_dyaw = cfg.robot_config.command_signs[2] * cmd.dyaw * walk_ramp

                obs[0, 6] = cmd_vx
                obs[0, 7] = cmd_vy
                obs[0, 8] = cmd_dyaw
                obs[0, 9] = warmup_phase
                obs[0, 10] = np.sin(gait_phase)
                obs[0, 11] = np.cos(gait_phase)
                obs[0, 12:24] = q_obs
                obs[0, 24:36] = dq_obs
                # IsaacLab stores the raw previous policy action in the observation
                # buffer. Do not feed the filtered/clipped deployment command back
                # to the policy, otherwise the recurrent history seen by the actor
                # no longer matches training.
                obs[0, 36:48] = last_policy_action

                if settling or no_policy:
                    policy_input = None
                    policy_action[:] = 0.0
                    raw_action[:] = 0.0
                else:
                    if policy_frame_stack > 1:
                        hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)
                        policy_input = hist_obs.reshape(1, -1).astype(np.float32)
                    else:
                        policy_input = obs
                    with torch.inference_mode():
                        obs_dict = {"policy": torch.from_numpy(policy_input)}
                        policy_action[:] = policy.act_inference(obs_dict)[0].cpu().numpy()
                        raw_action[:] = policy_action
                        if debug_steps > 0 and not printed_first_policy_obs:
                            actor_obs = obs_dict["policy"]
                            norm_obs = policy.actor_obs_normalizer(actor_obs)
                            progress.write(
                                "[DEBUG] first_policy "
                                f"policy_time={policy_time:.3f} "
                                f"obs_minmax=({float(actor_obs.min()):.3f},{float(actor_obs.max()):.3f}) "
                                f"norm_minmax=({float(norm_obs.min()):.3f},{float(norm_obs.max()):.3f}) "
                                f"raw_max={float(np.max(np.abs(raw_action))):.3f}"
                            )
                            printed_first_policy_obs = True
                last_policy_action[:] = policy_action
                deploy_action = np.clip(raw_action, -cfg.robot_config.action_clip, cfg.robot_config.action_clip)
                action[:] = (
                    cfg.robot_config.action_filter * deploy_action
                    + (1.0 - cfg.robot_config.action_filter) * action
                )
                max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))

                if debug_steps > 0 and count_lowlevel // cfg.sim_config.decimation < debug_steps:
                    foot_summary = _foot_contact_summary(model, data, foot_geom_ids)
                    progress.write(
                        "debug "
                        f"frame={count_lowlevel // cfg.sim_config.decimation} "
                        f"pos=({data.qpos[0]:.3f},{data.qpos[1]:.3f},{data.qpos[2]:.3f}) "
                        f"rpy=({root_euler[0]:.1f},{root_euler[1]:.1f},{root_euler[2]:.1f}) "
                        f"vel=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f}) "
                        f"foot={_compact_foot_summary(foot_summary)} "
                        f"cmd=({cmd_vx:.2f},{cmd_vy:.2f},{cmd_dyaw:.2f}) "
                        f"warmup={warmup_phase:.0f} "
                        f"walk_ramp={walk_ramp:.3f} "
                        f"act_max={float(np.max(np.abs(action))):.3f} "
                        f"raw_max={float(np.max(np.abs(raw_action))):.3f}"
                    )

                target_q = action * cfg.robot_config.action_scale * cfg.robot_config.action_scale_multipliers
                for i in range(len(cfg.robot_config.usd2urdf)):
                    target_pos[cfg.robot_config.usd2urdf[i]] = cfg.robot_config.action_signs[i] * target_q[i]
                target_pos = target_pos + cfg.robot_config.default_pos

                q_low_freq = q.copy()
                v_low_freq = v[:2].copy()
                omega_low_freq = omega[2].copy()

                time_data.append(step * cfg.sim_config.dt)
                commanded_joint_pos_data.append(target_pos.copy())
                actual_joint_pos_data.append(q_low_freq)
                tau_data.append(tau.copy())
                commanded_lin_vel_x_data.append(cmd_vx)
                commanded_lin_vel_y_data.append(cmd_vy)
                commanded_ang_vel_z_data.append(cmd_dyaw)
                actual_lin_vel_data.append(v_low_freq)
                actual_ang_vel_data.append(omega_low_freq)

                if no_render:
                    pass
                elif headless:
                    update_camera_lookat(cam, data, cfg.camera_config.lookat_height_offset)
                    renderer.update_scene(data, camera=cam)
                    img = renderer.render()
                    out.write(img)
                else:
                    update_camera_lookat(viewer.cam, data, cfg.camera_config.lookat_height_offset)
                    viewer.render()
                    if not viewer.is_alive:
                        fall_reason = "viewer window closed"
                        break

            target_vel = np.zeros((cfg.robot_config.num_actions), dtype=np.double)
            tau = pd_control(target_pos, q, cfg.robot_config.kps,
                            target_vel, dq, cfg.robot_config.kds)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            max_abs_tau = max(max_abs_tau, float(np.max(np.abs(tau))))
            data.ctrl = tau
            mujoco.mj_step(model, data)

            count_lowlevel += 1
            if cfg.sim_config.stop_pitch_deg > 0.0 and abs(root_euler[1]) > cfg.sim_config.stop_pitch_deg:
                fall_reason = f"pitch {root_euler[1]:.2f} deg exceeded {cfg.sim_config.stop_pitch_deg:.2f} deg"
                break
            if cfg.sim_config.stop_root_z > 0.0 and data.qpos[2] < cfg.sim_config.stop_root_z:
                fall_reason = f"root_z {data.qpos[2]:.3f} below {cfg.sim_config.stop_root_z:.3f}"
                break
            if max_steps is not None and count_lowlevel >= max_steps:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nSimulation interrupted by user.")
    finally:
        progress.close()
        root_quat = data.qpos[3:7]
        root_euler = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]]).as_euler("xyz", degrees=True)
        print(
            "[SUMMARY] "
            f"fall_reason={fall_reason} "
            f"root_pos={np.round(data.qpos[:3], 4).tolist()} "
            f"root_euler_deg={np.round(root_euler, 3).tolist()} "
            f"avg_world_vel={np.round((data.qpos[:3] - start_root_pos) / max(count_lowlevel * cfg.sim_config.dt, 1e-9), 4).tolist()} "
            f"max_abs_omega={max_abs_omega:.3f} "
            f"max_abs_action={max_abs_action:.3f} "
            f"max_abs_tau={max_abs_tau:.3f}"
        )
        if no_render:
            pass
        elif headless:
            out.release()
        else:
            viewer.close()

     # --- Plotting Section (Using only low-frequency data) ---

    if interrupted:
        print("Generating plots from collected data...")
    else:
        print("Simulation finished. Generating plots...")

    # Convert collected data to numpy arrays
    time_data = np.array(time_data)
    commanded_joint_pos_data = np.array(commanded_joint_pos_data)
    actual_joint_pos_data = np.array(actual_joint_pos_data)
    tau_data = np.array(tau_data)
    commanded_lin_vel_x_data = np.array(commanded_lin_vel_x_data)
    commanded_lin_vel_y_data = np.array(commanded_lin_vel_y_data)
    commanded_ang_vel_z_data = np.array(commanded_ang_vel_z_data)
    actual_lin_vel_data = np.array(actual_lin_vel_data)
    actual_ang_vel_data = np.array(actual_ang_vel_data)


    # Plot 1: Commanded vs Actual Joint Positions
    if cfg.sim_config.no_plots:
        print("Plotting skipped.")
        return

    # Plot 1: Commanded vs Actual Joint Positions
    num_joints = cfg.robot_config.num_actions
    n_cols = 4 # Or adjust based on num_joints
    n_rows = (num_joints + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)
    axes1 = axes1.flatten()

    joint_names = [f'Joint {i+1}' for i in range(num_joints)] # Generic names (consider using specific robot joint names if available)

    for i in range(num_joints):
        ax = axes1[i]
        # Plotting low-frequency commanded and actual joint positions
        ax.plot(time_data, commanded_joint_pos_data[:, i], label='Commanded', linestyle='--')
        ax.plot(time_data, actual_joint_pos_data[:, i], label='Actual')
        ax.set_title(joint_names[i])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for i in range(num_joints, len(axes1)):
        fig1.delaxes(axes1[i])

    fig1.suptitle("Commanded vs Actual Joint Positions", fontsize=16)
    plt.tight_layout()


    # Plot 2: Commanded vs Actual Base Velocities
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Linear Velocity X
    # Plotting low-frequency commanded and actual velocities
    axes2[0].plot(time_data, commanded_lin_vel_x_data, label='Commanded Vx', linestyle='--')
    axes2[0].plot(time_data, actual_lin_vel_data[:, 0], label='Actual Vx')
    axes2[0].set_title("Base Linear Velocity X")
    axes2[0].set_xlabel("Time [s]")
    axes2[0].set_ylabel("Velocity [m/s]")
    axes2[0].legend()
    axes2[0].grid(True)

    # Linear Velocity Y
    axes2[1].plot(time_data, commanded_lin_vel_y_data, label='Commanded Vy', linestyle='--')
    axes2[1].plot(time_data, actual_lin_vel_data[:, 1], label='Actual Vy')
    axes2[1].set_title("Base Linear Velocity Y")
    axes2[1].set_xlabel("Time [s]")
    axes2[1].set_ylabel("Velocity [m/s]")
    axes2[1].legend()
    axes2[1].grid(True)

    # Angular Velocity Z
    axes2[2].plot(time_data, commanded_ang_vel_z_data, label='Commanded Dyaw', linestyle='--')
    axes2[2].plot(time_data, actual_ang_vel_data, label='Actual Dyaw') # actual_ang_vel_data is already 1D
    axes2[2].set_title("Base Angular Velocity Z (Dyaw)")
    axes2[2].set_xlabel("Time [s]")
    axes2[2].set_ylabel("Angular Velocity [rad/s]")
    axes2[2].legend()
    axes2[2].grid(True)

    fig2.suptitle("Commanded vs Actual Base Velocities", fontsize=16)
    plt.tight_layout()

    # plt.show()
    fig1.savefig("joint_positions.png")
    fig2.savefig("base_velocities.png")

    print("Plots finished.")
    # --- End Plotting Section ---

    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    parser.add_argument('--headless', action='store_true',
                      help='Run without GUI and save video')
    parser.add_argument('--vx', type=float, default=0.4, help='Forward velocity command.')
    parser.add_argument('--vy', type=float, default=0.0, help='Lateral velocity command.')
    parser.add_argument('--dyaw', type=float, default=0.0, help='Yaw rate command.')
    parser.add_argument('--debug_steps', type=int, default=0, help='Print low-level sim2sim diagnostics for N policy frames.')
    parser.add_argument('--no_policy', action='store_true', help='Hold the default pose with PD and do not query the policy.')
    parser.add_argument('--max_steps', type=int, default=None, help='Stop Mujoco after this many low-level steps.')
    parser.add_argument('--no_render', action='store_true', help='Run without viewer or video renderer.')
    parser.add_argument('--action_scale', type=float, default=0.25, help='Joint target scale used by the policy in Mujoco.')
    parser.add_argument('--action_clip', type=float, default=100.0, help='Clamp raw policy actions before sending them to Mujoco PD.')
    parser.add_argument('--action_filter', type=float, default=1.0, help='Low-pass factor for policy actions. 1.0 disables filtering.')
    parser.add_argument('--kp_scale', type=float, default=1.0, help='Scale deployment PD stiffness from IsaacLab training gains.')
    parser.add_argument('--settle_time', type=float, default=0.0, help='Seconds to hold the default pose before enabling the policy.')
    parser.add_argument('--warmup_time', type=float, default=5.0, help='Seconds to command in-place stepping before walking.')
    parser.add_argument('--gait_period', type=float, default=0.8, help='Seconds for one full left/right gait cycle.')
    parser.add_argument('--ramp_time', type=float, default=2.0, help='Seconds to ramp command after warmup.')
    parser.add_argument('--stop_pitch_deg', type=float, default=0.0, help='Stop when absolute root pitch exceeds this angle. 0 disables.')
    parser.add_argument('--stop_root_z', type=float, default=0.5, help='Stop when root z drops below this height. 0 disables.')
    parser.add_argument('--verbose_model', action='store_true', help='Print ActorCritic network structure when loading the checkpoint.')
    parser.add_argument('--verbose_debug', action='store_true', help='Print full sim2sim debug metadata such as joint names and sign maps.')
    parser.add_argument('--flip_cmd_vx', action='store_true', help='Flip the x velocity command seen by the policy.')
    parser.add_argument('--flip_right_sagittal', action='store_true', help='Flip right hip/knee/ankle pitch policy coordinates for Mujoco.')
    parser.add_argument('--flip_right_roll', action='store_true', help='Flip right hip/ankle roll policy coordinates for Mujoco.')
    parser.add_argument('--flip_obs_omega_y', action='store_true', help='Flip observed base pitch angular velocity.')
    parser.add_argument('--flip_obs_gravity_x', action='store_true', help='Flip observed projected gravity x component.')
    parser.add_argument('--flip_action_sagittal', action='store_true', help='Flip all hip/knee/ankle pitch actions sent to Mujoco.')
    parser.add_argument('--root_z', type=float, default=0.9, help='Initial Mujoco pelvis z before optional foot-ground auto alignment.')
    parser.add_argument('--no_auto_root_height', action='store_true', help='Do not adjust initial pelvis z from foot collision bottom.')
    parser.add_argument('--foot_ground_clearance', type=float, default=0.001, help='Target initial foot collision clearance above ground.')
    parser.add_argument('--no_plots', action='store_true', help='Skip diagnostic plot generation.')
    args = parser.parse_args()
    cmd.vx = args.vx
    cmd.vy = args.vy
    cmd.dyaw = args.dyaw
    print(f"[INFO] Sim2sim command: vx={cmd.vx:.3f}, vy={cmd.vy:.3f}, dyaw={cmd.dyaw:.3f}")

    class Sim2simCfg():

        class sim_config:
            if args.terrain:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/atom01/mjcf/atom01_terrain.xml'
            else:
                mujoco_model_path = f'{ISAAC_DATA_DIR}/robots/roboparty/atom01/mjcf/atom01.xml'
            dt = 0.005
            decimation = 4
            no_plots = args.no_plots
            settle_time = args.settle_time
            warmup_time = args.warmup_time
            gait_period = args.gait_period
            ramp_time = args.ramp_time
            stop_pitch_deg = args.stop_pitch_deg
            stop_root_z = args.stop_root_z
            verbose_debug = args.verbose_debug

        class camera_config:
            lookat_height_offset = 0.6

        class robot_config:
            # Match the IsaacLab training actuator gains before applying any
            # deployment-scale tuning.
            kps = args.kp_scale * np.array([100, 100, 100, 300, 260, 180, 100, 100, 100, 300, 260, 180], dtype=np.double)
            kds = np.array([3.3, 3.3, 3.3, 5.0, 3.0, 3.0, 3.3, 3.3, 3.3, 5.0, 3.0, 3.0], dtype=np.double)
            default_root_pos = np.array([0.0, 0.0, args.root_z], dtype=np.double)
            default_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.double)
            auto_root_height = not args.no_auto_root_height
            foot_ground_clearance = args.foot_ground_clearance
            default_pos = np.array([0, 0, -0.15, 0.3, 0.15, 0, 0, 0, 0.15, -0.3, -0.15, 0], dtype=np.double)
            tau_limit = np.array([120, 120, 120, 120, 27, 27, 120, 120, 120, 120, 27, 27], dtype=np.double)
            frame_stack = 10
            num_single_obs = 48
            num_observations = 480
            num_actions = 12
            action_scale = args.action_scale
            action_scale_multipliers = np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.40, 0.40],
                dtype=np.double,
            )
            action_clip = args.action_clip
            action_filter = args.action_filter
            command_signs = np.array([-1.0 if args.flip_cmd_vx else 1.0, 1.0, 1.0], dtype=np.double)
            ang_vel_signs = np.ones(3, dtype=np.double)
            gravity_signs = np.ones(3, dtype=np.double)
            if args.flip_obs_omega_y:
                ang_vel_signs[1] = -1.0
            if args.flip_obs_gravity_x:
                gravity_signs[0] = -1.0
            # Policy/Isaac order:
            # left/right hip_yaw, left/right hip_roll, left/right hip_pitch,
            # left/right knee, left/right ankle_pitch, left/right ankle_roll.
            #
            # Mujoco order:
            # all left leg joints, then all right leg joints.
            usd2urdf = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
            coord_signs = np.ones(12, dtype=np.double)
            if args.flip_right_sagittal:
                coord_signs[[5, 7, 9]] = -1.0
            if args.flip_right_roll:
                coord_signs[[3, 11]] = -1.0
            action_signs = coord_signs.copy()
            if args.flip_action_sagittal:
                action_signs[[4, 5, 6, 7, 8, 9]] *= -1.0

    def infer_mlp_dims(state_dict, prefix):
        linear_layers = []
        for key, value in state_dict.items():
            if key.startswith(prefix) and key.endswith(".weight"):
                layer_name = key[len(prefix):].split(".", 1)[0]
                if layer_name.isdigit():
                    linear_layers.append((int(layer_name), value.shape))

        if not linear_layers:
            raise ValueError(f"Cannot infer network structure from prefix '{prefix}'.")

        linear_layers.sort(key=lambda item: item[0])
        input_dim = linear_layers[0][1][1]
        hidden_dims = [shape[0] for _, shape in linear_layers[:-1]]
        output_dim = linear_layers[-1][1][0]
        return input_dim, hidden_dims, output_dim

    policy = None
    if not args.no_policy:
        # Load policy weights (state_dict format as saved by training).
        load_path = os.path.expanduser(args.load_model)
        if not os.path.isabs(load_path):
            load_path = os.path.join(os.getcwd(), load_path)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Cannot find model file: {load_path}")

        ckpt = torch.load(load_path, map_location='cpu')
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            raise ValueError("Loaded checkpoint does not contain 'model_state_dict'.")

        # Build policy network matching the training configuration.
        from rsl_rl.modules import ActorCritic

        actor_obs_dim, actor_hidden_dims, actor_output_dim = infer_mlp_dims(state_dict, "actor.")
        critic_obs_dim, critic_hidden_dims, _ = infer_mlp_dims(state_dict, "critic.")
        actor_obs_normalization = any(key.startswith("actor_obs_normalizer.") for key in state_dict)
        critic_obs_normalization = any(key.startswith("critic_obs_normalizer.") for key in state_dict)
        noise_std_type = "log" if "log_std" in state_dict else "scalar"
        state_dependent_std = "std" not in state_dict and "log_std" not in state_dict
        num_actions = actor_output_dim // 2 if state_dependent_std else actor_output_dim

        dummy_obs = {
            "policy": torch.zeros((1, actor_obs_dim), dtype=torch.float32),
            "critic": torch.zeros((1, critic_obs_dim), dtype=torch.float32),
        }
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        if args.verbose_model:
            policy = ActorCritic(
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
        else:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
                policy = ActorCritic(
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
        policy.load_state_dict(state_dict)
        policy.eval()

    run_mujoco(policy, Sim2simCfg(), args.headless, args.debug_steps, args.no_policy, args.max_steps, args.no_render)
