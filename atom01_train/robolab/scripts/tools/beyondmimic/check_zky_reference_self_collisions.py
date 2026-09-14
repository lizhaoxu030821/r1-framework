#!/usr/bin/env python3
"""Check a ZKY NPZ against the full-body collision meshes in the training URDF."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


def main() -> int:
    workspace = Path(__file__).resolve().parents[5]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "npz",
        type=Path,
        help="BeyondMimic motion NPZ containing joint_names, body_pos_w, and body_quat_w",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=workspace
        / "ZKY_SDK_19Dof/resources/robots/robot_zky_19dof/urdf/robot_zky_fixed_all.urdf",
    )
    args = parser.parse_args()

    motion = np.load(args.npz)
    joint_names = [str(name) for name in motion["joint_names"]]
    body_names = [str(name) for name in motion["body_names"]]
    root_index = body_names.index("base_link")

    urdf = args.urdf.resolve()
    xml = re.sub(
        r'<robot\s+name="robot_zky">',
        '<robot name="robot_zky"><mujoco><compiler balanceinertia="true"/></mujoco>',
        urdf.read_text(),
        count=1,
    )
    assets = {path.name: path.read_bytes() for path in (urdf.parent.parent / "meshes").glob("*.STL")}
    model = mujoco.MjModel.from_xml_string(xml, assets)
    data = mujoco.MjData(model)

    joint_qpos_addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"training URDF is missing joint {name}")
        joint_qpos_addresses.append(model.jnt_qposadr[joint_id])

    pair_stats: dict[tuple[str, str], list[float | int]] = defaultdict(lambda: [0, 0.0])
    collision_frames = 0
    deepest_distance = 0.0
    for frame in range(motion["joint_pos"].shape[0]):
        data.qpos[:3] = motion["body_pos_w"][frame, root_index]
        data.qpos[3:7] = motion["body_quat_w"][frame, root_index]
        data.qpos[joint_qpos_addresses] = motion["joint_pos"][frame]
        mujoco.mj_forward(model, data)

        if data.ncon:
            collision_frames += 1
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body_a = model.geom_bodyid[contact.geom1]
            body_b = model.geom_bodyid[contact.geom2]
            name_a = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_a)
            name_b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_b)
            pair = tuple(sorted((name_a, name_b)))
            pair_stats[pair][0] += 1
            pair_stats[pair][1] = min(float(pair_stats[pair][1]), float(contact.dist))
            deepest_distance = min(deepest_distance, float(contact.dist))

    print(f"frames: {motion['joint_pos'].shape[0]}")
    print(f"frames_with_self_collision: {collision_frames}")
    print(f"deepest_penetration_m: {deepest_distance:.6f}")
    for pair, (count, minimum) in sorted(pair_stats.items(), key=lambda item: item[1][1]):
        print(f"{pair[0]} <-> {pair[1]}: contacts={count}, min_distance={minimum:.6f} m")
    return 1 if collision_frames else 0


if __name__ == "__main__":
    raise SystemExit(main())
