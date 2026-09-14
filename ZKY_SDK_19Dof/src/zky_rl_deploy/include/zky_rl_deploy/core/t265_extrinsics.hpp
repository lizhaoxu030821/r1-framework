#pragma once

#include <array>

namespace zky_rl_deploy {

struct T265ExtrinsicConfig {
  // R_pelvis_t265：把 canonical T265 FLU frame 的向量旋转到 pelvis frame。
  std::array<double, 4> rotation_quat_xyzw{0.0, 0.0, 0.0, 1.0};
  // t_pelvis_t265：T265 原点在 pelvis frame 下的位置，单位 m。
  std::array<double, 3> translation_xyz_m{0.0, 0.0, 0.0};
};

class T265Extrinsics {
 public:
  using QuaternionXyzw = std::array<double, 4>;
  using Vector3 = std::array<double, 3>;

  // /camera/imu 的 frame_id 是 camera_imu_optical_frame，仍需先转到
  // canonical T265 FLU（x forward, y left, z up）。
  static QuaternionXyzw ImuOpticalFrameToCanonicalT265Flu();

  // /camera/odom/sample 的 pose 已由 realsense2_camera 转成 pose_frame，
  // position 方向与机器人 FLU 一致，因此这里不再额外施加 optical->FLU 旋转，
  // 只应用 T265->pelvis 的固定外参。
  static QuaternionXyzw TransformPoseFrameOrientationToPelvis(
      const QuaternionXyzw& world_t265_pose_frame_orientation_xyzw,
      const T265ExtrinsicConfig& extrinsic);

  static Vector3 TransformImuOpticalAngularVelocityToPelvis(
      const Vector3& imu_optical_angular_velocity_rad_s,
      const T265ExtrinsicConfig& extrinsic);

  static Vector3 TransformPoseFramePositionToPelvis(
      const Vector3& world_t265_position_flu_m,
      const QuaternionXyzw& world_pelvis_orientation_xyzw,
      const T265ExtrinsicConfig& extrinsic);

 private:
  static QuaternionXyzw NormalizeQuaternion(const QuaternionXyzw& quaternion,
                                            const char* label);
  static QuaternionXyzw Conjugate(const QuaternionXyzw& quaternion);
  static QuaternionXyzw Multiply(const QuaternionXyzw& lhs, const QuaternionXyzw& rhs);
  static Vector3 RotateVector(const QuaternionXyzw& rotation_xyzw, const Vector3& vector);
};

}  // namespace zky_rl_deploy
