#!/usr/bin/env python3

import argparse
import sys

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


HARDWARE_ORDER = [
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
]

KP_10 = {
    "left_hip_yaw_joint": 20.0,
    "left_hip_roll_joint": 30.0,
    "left_hip_pitch_joint": 30.0,
    "left_knee_joint": 30.0,
    "left_ankle_pitch_joint": 20.0,
    "left_ankle_roll_joint": 20.0,
    "right_hip_yaw_joint": 20.0,
    "right_hip_roll_joint": 30.0,
    "right_hip_pitch_joint": 30.0,
    "right_knee_joint": 30.0,
    "right_ankle_pitch_joint": 20.0,
    "right_ankle_roll_joint": 20.0,
}

KD_10 = {
    "left_hip_yaw_joint": 0.0037,
    "left_hip_roll_joint": 0.0123,
    "left_hip_pitch_joint": 0.0120,
    "left_knee_joint": 0.0120,
    "left_ankle_pitch_joint": 0.0031,
    "left_ankle_roll_joint": 0.0031,
    "right_hip_yaw_joint": 0.0037,
    "right_hip_roll_joint": 0.0123,
    "right_hip_pitch_joint": 0.0120,
    "right_knee_joint": 0.0120,
    "right_ankle_pitch_joint": 0.0031,
    "right_ankle_roll_joint": 0.0031,
}

POSITION_PD_MODE = 2.0
DISABLED_MODE = 0.0
MAX_SAFE_DELTA = 0.02


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish a temporary one-joint /motor_params probe for Stage 3 validation."
    )
    parser.add_argument(
        "--joint",
        default="left_hip_yaw_joint",
        choices=HARDWARE_ORDER,
        help="Joint to probe. Default is the first Stage 3 validation joint.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
        help="Joint position increment in rad. Keep this at 0.0 for the first run.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.8,
        help="How long to publish the probe command in seconds.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Publish rate in Hz.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="Delay before publishing, to let subscribers connect.",
    )
    parser.add_argument(
        "--zero-cycles",
        type=int,
        default=10,
        help="How many zero-output cycles to publish after the probe.",
    )
    parser.add_argument(
        "--list-joints",
        action="store_true",
        help="Print the hardware-order joint list and exit.",
    )
    parser.add_argument(
        "--kp-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the Stage 3 baseline kp for the target joint.",
    )
    parser.add_argument(
        "--kd-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the Stage 3 baseline kd for the target joint.",
    )
    parser.add_argument(
        "--kp-override",
        type=float,
        default=None,
        help="Explicit kp override for the target joint. If set, it takes precedence over --kp-scale.",
    )
    parser.add_argument(
        "--kd-override",
        type=float,
        default=None,
        help="Explicit kd override for the target joint. If set, it takes precedence over --kd-scale.",
    )
    parser.add_argument(
        "--allow-unsafe-delta",
        action="store_true",
        help="Explicitly allow |delta| beyond the built-in %.3f rad safety cap." % MAX_SAFE_DELTA,
    )
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def validate_args(args):
    if args.list_joints:
        return

    if abs(args.delta) > MAX_SAFE_DELTA and not args.allow_unsafe_delta:
        raise ValueError(
            "Refusing to publish |delta| > %.3f rad without --allow-unsafe-delta; requested %.6f rad"
            % (MAX_SAFE_DELTA, args.delta)
        )
    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0")
    if args.rate <= 0.0:
        raise ValueError("--rate must be > 0")
    if args.settle_time < 0.0:
        raise ValueError("--settle-time must be >= 0")
    if args.zero_cycles < 0:
        raise ValueError("--zero-cycles must be >= 0")
    if args.kp_scale <= 0.0:
        raise ValueError("--kp-scale must be > 0")
    if args.kd_scale <= 0.0:
        raise ValueError("--kd-scale must be > 0")
    if args.kp_override is not None and args.kp_override < 0.0:
        raise ValueError("--kp-override must be >= 0")
    if args.kd_override is not None and args.kd_override < 0.0:
        raise ValueError("--kd-override must be >= 0")


def wait_for_joint_state():
    joint_state = rospy.wait_for_message("/joint_states", JointState, timeout=5.0)
    pos_by_name = dict(zip(joint_state.name, joint_state.position))

    missing = [name for name in HARDWARE_ORDER if name not in pos_by_name]
    if missing:
        raise RuntimeError("joint_states missing joints: %s" % missing)

    return [float(pos_by_name[name]) for name in HARDWARE_ORDER]


def effective_target_gains(args):
    base_kp = KP_10[args.joint]
    base_kd = KD_10[args.joint]
    kp = args.kp_override if args.kp_override is not None else base_kp * args.kp_scale
    kd = args.kd_override if args.kd_override is not None else base_kd * args.kd_scale
    return kp, kd


def build_probe_motor_params(target_joint, delta, base_positions, kp, kd):
    target_positions = list(base_positions)
    target_positions[HARDWARE_ORDER.index(target_joint)] += delta

    data = []
    for name, position in zip(HARDWARE_ORDER, target_positions):
        if name == target_joint:
            data.extend([kp, kd, position, 0.0, 0.0, POSITION_PD_MODE])
        else:
            data.extend([0.0, 0.0, position, 0.0, 0.0, DISABLED_MODE])
    return data


def build_zero_motor_params(base_positions):
    data = []
    for position in base_positions:
        data.extend([0.0, 0.0, position, 0.0, 0.0, DISABLED_MODE])
    return data


def main():
    args = parse_args()
    validate_args(args)

    if args.list_joints:
        for index, name in enumerate(HARDWARE_ORDER, start=1):
            print("%02d %s" % (index, name))
        return

    rospy.init_node("zky_stage3_one_joint_probe", anonymous=False)

    kp, kd = effective_target_gains(args)
    rospy.logwarn(
        "Stage 3 one-joint probe: joint=%s delta=%.6f kp=%.6f kd=%.6f duration=%.2fs rate=%.1fHz",
        args.joint,
        args.delta,
        kp,
        kd,
        args.duration,
        args.rate,
    )

    base_positions = wait_for_joint_state()
    probe_data = build_probe_motor_params(args.joint, args.delta, base_positions, kp, kd)
    zero_data = build_zero_motor_params(base_positions)

    publisher = rospy.Publisher("/motor_params", Float64MultiArray, queue_size=1)
    rospy.sleep(args.settle_time)

    rate = rospy.Rate(args.rate)
    deadline = rospy.Time.now() + rospy.Duration(args.duration)
    while not rospy.is_shutdown() and rospy.Time.now() < deadline:
        publisher.publish(Float64MultiArray(data=probe_data))
        rate.sleep()

    for _ in range(args.zero_cycles):
        if rospy.is_shutdown():
            break
        publisher.publish(Float64MultiArray(data=zero_data))
        rate.sleep()

    rospy.logwarn("Stage 3 one-joint probe complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr(str(exc))
        raise
