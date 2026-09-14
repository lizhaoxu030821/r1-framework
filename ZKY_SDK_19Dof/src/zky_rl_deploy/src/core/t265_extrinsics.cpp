#include "zky_rl_deploy/core/t265_extrinsics.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace zky_rl_deploy {
namespace {

bool AllFinite(const T265Extrinsics::QuaternionXyzw& quaternion) {
  for (double value : quaternion) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

bool AllFinite(const T265Extrinsics::Vector3& vector) {
  for (double value : vector) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  return true;
}

}  // namespace

T265Extrinsics::QuaternionXyzw T265Extrinsics::ImuOpticalFrameToCanonicalT265Flu() {
  // optical/camera-centric -> FLU:
  // x_flu = z_raw, y_flu = -x_raw, z_flu = -y_raw.
  return {0.5, -0.5, 0.5, -0.5};
}

T265Extrinsics::QuaternionXyzw T265Extrinsics::TransformPoseFrameOrientationToPelvis(
    const QuaternionXyzw& world_t265_pose_frame_orientation_xyzw,
    const T265ExtrinsicConfig& extrinsic) {
  const QuaternionXyzw world_t265 =
      NormalizeQuaternion(world_t265_pose_frame_orientation_xyzw,
                          "world_t265_pose_frame_orientation_xyzw");
  const QuaternionXyzw pelvis_from_canonical =
      NormalizeQuaternion(extrinsic.rotation_quat_xyzw, "t265_to_pelvis.rotation_quat_xyzw");
  return NormalizeQuaternion(Multiply(world_t265, Conjugate(pelvis_from_canonical)),
                             "world_pelvis_orientation_xyzw");
}

T265Extrinsics::Vector3 T265Extrinsics::TransformImuOpticalAngularVelocityToPelvis(
    const Vector3& imu_optical_angular_velocity_rad_s, const T265ExtrinsicConfig& extrinsic) {
  if (!AllFinite(imu_optical_angular_velocity_rad_s)) {
    throw std::invalid_argument("imu_optical_angular_velocity_rad_s contains non-finite value");
  }

  const QuaternionXyzw raw_to_canonical =
      NormalizeQuaternion(ImuOpticalFrameToCanonicalT265Flu(), "raw_to_canonical");
  const QuaternionXyzw pelvis_from_canonical =
      NormalizeQuaternion(extrinsic.rotation_quat_xyzw, "t265_to_pelvis.rotation_quat_xyzw");
  const Vector3 canonical = RotateVector(raw_to_canonical, imu_optical_angular_velocity_rad_s);
  return RotateVector(pelvis_from_canonical, canonical);
}

T265Extrinsics::Vector3 T265Extrinsics::TransformPoseFramePositionToPelvis(
    const Vector3& world_t265_position_flu_m,
    const QuaternionXyzw& world_pelvis_orientation_xyzw,
    const T265ExtrinsicConfig& extrinsic) {
  if (!AllFinite(world_t265_position_flu_m)) {
    throw std::invalid_argument("world_t265_position_flu_m contains non-finite value");
  }
  if (!AllFinite(extrinsic.translation_xyz_m)) {
    throw std::invalid_argument("t265_to_pelvis.translation_xyz_m contains non-finite value");
  }

  const QuaternionXyzw world_pelvis =
      NormalizeQuaternion(world_pelvis_orientation_xyzw, "world_pelvis_orientation_xyzw");
  const Vector3 world_translation =
      RotateVector(world_pelvis, extrinsic.translation_xyz_m);
  return {world_t265_position_flu_m[0] - world_translation[0],
          world_t265_position_flu_m[1] - world_translation[1],
          world_t265_position_flu_m[2] - world_translation[2]};
}

T265Extrinsics::QuaternionXyzw T265Extrinsics::NormalizeQuaternion(
    const QuaternionXyzw& quaternion, const char* label) {
  if (!AllFinite(quaternion)) {
    throw std::invalid_argument(std::string(label) + " contains non-finite value");
  }

  const double norm =
      std::sqrt(quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
  if (norm <= 1e-12) {
    throw std::invalid_argument(std::string(label) + " has near-zero norm");
  }

  return {quaternion[0] / norm,
          quaternion[1] / norm,
          quaternion[2] / norm,
          quaternion[3] / norm};
}

T265Extrinsics::QuaternionXyzw T265Extrinsics::Conjugate(const QuaternionXyzw& quaternion) {
  return {-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]};
}

T265Extrinsics::QuaternionXyzw T265Extrinsics::Multiply(const QuaternionXyzw& lhs,
                                                        const QuaternionXyzw& rhs) {
  return {
      lhs[3] * rhs[0] + lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1],
      lhs[3] * rhs[1] - lhs[0] * rhs[2] + lhs[1] * rhs[3] + lhs[2] * rhs[0],
      lhs[3] * rhs[2] + lhs[0] * rhs[1] - lhs[1] * rhs[0] + lhs[2] * rhs[3],
      lhs[3] * rhs[3] - lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2],
  };
}

T265Extrinsics::Vector3 T265Extrinsics::RotateVector(const QuaternionXyzw& rotation_xyzw,
                                                      const Vector3& vector) {
  const QuaternionXyzw rotation = NormalizeQuaternion(rotation_xyzw, "rotation_xyzw");
  const QuaternionXyzw pure_vector = {vector[0], vector[1], vector[2], 0.0};
  const QuaternionXyzw rotated =
      Multiply(Multiply(rotation, pure_vector), Conjugate(rotation));
  return {rotated[0], rotated[1], rotated[2]};
}

}  // namespace zky_rl_deploy
