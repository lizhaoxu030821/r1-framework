"""Visualize an IsaacLab-exported ZKY policy rollout in MuJoCo."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="Open synchronized policy and reference windows (requires an extended evaluator export).",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate the rollout and exit without opening a window.")
    args = parser.parse_args()

    rollout = np.load(args.rollout)
    root_pos = rollout["root_pos"]
    root_quat = rollout["root_quat"]
    joint_pos = rollout["joint_pos"]
    joint_names = rollout["joint_names"].tolist()
    fps = args.fps or float(rollout["fps"])
    if root_pos.shape != (len(joint_pos), 3) or root_quat.shape != (len(joint_pos), 4):
        raise RuntimeError("Invalid rollout root/joint array shapes.")
    if joint_pos.shape[1] != len(joint_names):
        raise RuntimeError("Joint position width does not match joint_names.")
    if not all(np.isfinite(values).all() for values in (root_pos, root_quat, joint_pos)):
        raise RuntimeError("Rollout contains NaN or infinite values.")

    # Older exports include the state produced by IsaacLab's automatic reset
    # after the timeout. Trim a terminal discontinuity so looping and holding
    # the last frame show the completed policy rollout instead of frame zero.
    if len(root_pos) > 1:
        terminal_jump = np.linalg.norm(root_pos[-1] - root_pos[-2])
        if terminal_jump > 0.20:
            print(f"trimming terminal reset frame (root jump={terminal_jump:.3f} m)")
            root_pos = root_pos[:-1]
            root_quat = root_quat[:-1]
            joint_pos = joint_pos[:-1]

    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    qpos_ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MJCF is missing joint {name!r}.")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))

    source = str(rollout["source_checkpoint"]) if "source_checkpoint" in rollout.files else "unknown"
    print(
        f"frames={len(joint_pos)} fps={fps:g} root_height={root_pos[0, 2]:.3f}->{root_pos[-1, 2]:.3f} "
        f"source={source}"
    )
    if args.check_only:
        return

    reference_data = None
    reference_viewer = None
    if args.compare_reference:
        required = {"reference_root_pos", "reference_root_quat", "reference_joint_pos"}
        missing = required.difference(rollout.files)
        if missing:
            raise RuntimeError(
                f"Rollout lacks reference comparison fields {sorted(missing)}; export it with the updated evaluator."
            )
        reference_data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if reference_data is not None:
            reference_viewer = mujoco.viewer.launch_passive(model, reference_data)
            print("policy window and synchronized reference window opened")
        while viewer.is_running():
            start = time.monotonic()
            for frame in range(len(joint_pos)):
                if not viewer.is_running() or (reference_viewer is not None and not reference_viewer.is_running()):
                    if reference_viewer is not None:
                        reference_viewer.close()
                    return
                data.qpos[:3] = root_pos[frame]
                data.qpos[3:7] = root_quat[frame]
                data.qpos[qpos_ids] = joint_pos[frame]
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                if reference_viewer is not None and reference_data is not None:
                    reference_data.qpos[:3] = rollout["reference_root_pos"][frame]
                    reference_data.qpos[3:7] = rollout["reference_root_quat"][frame]
                    reference_data.qpos[qpos_ids] = rollout["reference_joint_pos"][frame]
                    reference_data.qvel[:] = 0.0
                    mujoco.mj_forward(model, reference_data)
                    reference_viewer.sync()
                deadline = start + (frame + 1) / fps
                time.sleep(max(0.0, deadline - time.monotonic()))
            if not args.loop:
                while viewer.is_running():
                    viewer.sync()
                    if reference_viewer is not None and reference_viewer.is_running():
                        reference_viewer.sync()
                    time.sleep(0.02)
                if reference_viewer is not None:
                    reference_viewer.close()
                return
        if reference_viewer is not None:
            reference_viewer.close()


if __name__ == "__main__":
    main()
