#!/usr/bin/env python3
"""Validate the ZKY get-up reference before launching Isaac Sim training."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import struct
import zipfile
from pathlib import Path


JOINT_NAMES = (
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "body_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
)

BODY_NAMES = (
    "base_link",
    "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "body_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_elbow_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_elbow_link",
)

HARDWARE_LIMITS = (
    (-1.0, 1.1),
    (-0.35, 0.91),
    (-1.0, 0.45),
    (-0.79, 1.77),
    (-0.32, 0.85),
    (-0.9, 0.5),
    (-1.0, 1.1),
    (-0.91, 0.35),
    (-0.45, 1.0),
    (-1.81, 0.65),
    (-0.89, 0.38),
    (-0.5, 0.9),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-1.57, 1.57),
    (-1.57, 1.57),
)

EXPECTED_SHAPES = {
    "fps": (),
    "joint_pos": (229, 19),
    "joint_vel": (229, 19),
    "body_pos_w": (229, 20, 3),
    "body_quat_w": (229, 20, 4),
    "body_lin_vel_w": (229, 20, 3),
    "body_ang_vel_w": (229, 20, 3),
    "joint_names": (19,),
    "body_names": (20,),
}


def load_npy(archive: zipfile.ZipFile, key: str) -> tuple[tuple[int, ...], list[float | int | str]]:
    with archive.open(f"{key}.npy") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"{key}: invalid NPY magic")
        major, _minor = stream.read(2)
        header_size = struct.unpack("<H" if major == 1 else "<I", stream.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(stream.read(header_size).decode("latin1").strip())
        if header["fortran_order"]:
            raise ValueError(f"{key}: Fortran-order arrays are unsupported")
        if header["descr"].startswith("<U"):
            item_size = int(header["descr"][2:]) * 4
            payload = stream.read()
            values = [payload[i : i + item_size].decode("utf-32le").rstrip("\0") for i in range(0, len(payload), item_size)]
            return tuple(header["shape"]), values
        formats = {"<f4": "<f", "<f8": "<d", "<i8": "<q", "<i4": "<i"}
        try:
            value_format = formats[header["descr"]]
        except KeyError as exc:
            raise ValueError(f"{key}: unsupported dtype {header['descr']}") from exc
        item_size = struct.calcsize(value_format)
        payload = stream.read()
        if len(payload) % item_size:
            raise ValueError(f"{key}: truncated payload")
        values = [item[0] for item in struct.iter_unpack(value_format, payload)]
        return tuple(header["shape"]), values


def load_csv(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="") as stream:
        lines = (line for line in stream if not line.startswith("#"))
        reader = csv.reader(lines)
        header = next(reader)
        return header, [[float(value) for value in row] for row in reader]


def main() -> int:
    workspace = Path(__file__).resolve().parents[5]
    default_stem = workspace / "ZKY_SDK_19Dof/resources/motion/zky/npz/r1_standup_reference_full26_50hz_zky_ik_hand_limited_full19"
    default_csv = workspace / "ZKY_SDK_19Dof/resources/motion/zky/npz/r1_standup_reference_full26_50hz_zky_ik_hand_limited.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, default=default_stem.with_suffix(".npz"))
    parser.add_argument("--csv", type=Path, default=default_csv)
    args = parser.parse_args()

    errors: list[str] = []
    with zipfile.ZipFile(args.npz) as archive:
        arrays = {key: load_npy(archive, key) for key in EXPECTED_SHAPES}

    for key, expected_shape in EXPECTED_SHAPES.items():
        shape, values = arrays[key]
        if shape != expected_shape:
            errors.append(f"{key}: expected {expected_shape}, got {shape}")
        if key not in ("joint_names", "body_names") and any(not math.isfinite(float(value)) for value in values):
            errors.append(f"{key}: contains non-finite values")

    if tuple(arrays["joint_names"][1]) != JOINT_NAMES:
        errors.append("joint_names metadata does not match the ZKY 19-DoF contract")
    if tuple(arrays["body_names"][1]) != BODY_NAMES:
        errors.append("body_names metadata does not match the ZKY 20-body contract")

    fps = int(arrays["fps"][1][0])
    if fps != 50:
        errors.append(f"fps: expected 50, got {fps}")

    header, rows = load_csv(args.csv)
    expected_columns = [f"dof_{name}" for name in JOINT_NAMES]
    indexes = []
    for column in expected_columns:
        if column not in header:
            errors.append(f"CSV missing column: {column}")
        else:
            indexes.append(header.index(column))
    if len(rows) != EXPECTED_SHAPES["joint_pos"][0]:
        errors.append(f"CSV: expected 229 frames, got {len(rows)}")

    joint_values = arrays["joint_pos"][1]
    max_delta = 0.0
    if len(indexes) == len(JOINT_NAMES) and len(rows) * len(JOINT_NAMES) == len(joint_values):
        for frame, row in enumerate(rows):
            for joint, csv_index in enumerate(indexes):
                max_delta = max(max_delta, abs(float(joint_values[frame * 19 + joint]) - row[csv_index]))
        if max_delta > 1.0e-5:
            errors.append(f"CSV/NPZ joint mismatch: max delta {max_delta:.6g} rad")

    violations = []
    for joint, (name, (lower, upper)) in enumerate(zip(JOINT_NAMES, HARDWARE_LIMITS)):
        values = joint_values[joint::19]
        minimum, maximum = min(values), max(values)
        if minimum < lower - 1.0e-5 or maximum > upper + 1.0e-5:
            violations.append(f"{name}: [{minimum:.3f}, {maximum:.3f}] vs [{lower:.3f}, {upper:.3f}]")

    print(f"motion: {args.npz}")
    print(f"schema: {arrays['joint_pos'][0][0]} frames, {fps} Hz, 19 joints, 20 bodies")
    print(f"CSV/NPZ joint max delta: {max_delta:.3g} rad")
    if violations:
        print("hardware-limit warnings:")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print("hardware limits: all reference positions are within configured limits")

    if errors:
        print("FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("PASS: motion data is structurally ready for the 19-DoF BeyondMimic task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
