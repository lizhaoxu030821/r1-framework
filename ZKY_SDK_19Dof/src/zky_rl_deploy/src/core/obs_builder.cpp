#include "zky_rl_deploy/core/obs_builder.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace zky_rl_deploy {
namespace {

using QuaternionWxyz = ObsBuilder::QuaternionWxyz;
using QuaternionXyzw = ObsBuilder::QuaternionXyzw;
using MotionRefOriB = ObsBuilder::MotionRefOriB;

QuaternionWxyz NormalizeQuaternion(const QuaternionWxyz& quaternion,
                                   const char* field_name) {
  const double norm_squared = quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1] +
                              quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3];
  if (!(norm_squared > 0.0)) {
    throw std::invalid_argument(std::string(field_name) +
                                " must have non-zero quaternion norm.");
  }

  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  return {quaternion[0] * inverse_norm,
          quaternion[1] * inverse_norm,
          quaternion[2] * inverse_norm,
          quaternion[3] * inverse_norm};
}

QuaternionWxyz InverseQuaternion(const QuaternionWxyz& quaternion) {
  return {quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]};
}

QuaternionWxyz MultiplyQuaternions(const QuaternionWxyz& lhs, const QuaternionWxyz& rhs) {
  return {
      lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
      lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2],
      lhs[0] * rhs[2] - lhs[1] * rhs[3] + lhs[2] * rhs[0] + lhs[3] * rhs[1],
      lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1] + lhs[3] * rhs[0],
  };
}

std::array<double, 9> MatrixFromQuaternion(const QuaternionWxyz& quaternion) {
  // 这里严格对齐 ref/sim2sim_mimic.py::matrix_from_quat 的 [w, x, y, z] 展开公式。
  const double r = quaternion[0];
  const double i = quaternion[1];
  const double j = quaternion[2];
  const double k = quaternion[3];
  const double two_s =
      2.0 / (r * r + i * i + j * j + k * k);

  return {
      1.0 - two_s * (j * j + k * k),
      two_s * (i * j - k * r),
      two_s * (i * k + j * r),
      two_s * (i * j + k * r),
      1.0 - two_s * (i * i + k * k),
      two_s * (j * k - i * r),
      two_s * (i * k - j * r),
      two_s * (j * k + i * r),
      1.0 - two_s * (i * i + j * j),
  };
}

void ValidateExpectedVectorSize(const std::vector<double>& values,
                                std::size_t expected_size,
                                const char* field_name) {
  if (values.size() != expected_size) {
    std::ostringstream stream;
    stream << field_name << " size mismatch: expected " << expected_size << ", got "
           << values.size();
    throw std::invalid_argument(stream.str());
  }
}

}  // namespace

ObsBuilder::ObsBuilder(std::vector<double> default_joint_pos_policy_order)
    : default_joint_pos_policy_order_(std::move(default_joint_pos_policy_order)),
      last_action_(kJointCount, 0.0) {
  ValidateExpectedVectorSize(default_joint_pos_policy_order_,
                             kJointCount,
                             "default_joint_pos_policy_order");
  ValidateFiniteVector(default_joint_pos_policy_order_, "default_joint_pos_policy_order");
}

ObservationBuildResult ObsBuilder::BuildObservation(const ObservationInput& input) {
  ValidateExpectedVectorSize(input.motion_command.joint_pos_policy_order,
                             kJointCount,
                             "motion_command.joint_pos_policy_order");
  ValidateExpectedVectorSize(input.motion_command.joint_vel_policy_order,
                             kJointCount,
                             "motion_command.joint_vel_policy_order");
  ValidateExpectedVectorSize(input.measured_joint_pos_policy_order,
                             kJointCount,
                             "measured_joint_pos_policy_order");
  ValidateExpectedVectorSize(input.measured_joint_vel_policy_order,
                             kJointCount,
                             "measured_joint_vel_policy_order");
  ValidateExpectedVectorSize(last_action_, kJointCount, "last_action");

  ValidateFiniteVector(input.motion_command.joint_pos_policy_order,
                       "motion_command.joint_pos_policy_order");
  ValidateFiniteVector(input.motion_command.joint_vel_policy_order,
                       "motion_command.joint_vel_policy_order");
  ValidateFiniteQuaternionXyzw(input.current_pelvis_orientation_xyzw,
                               "current_pelvis_orientation_xyzw");
  ValidateFiniteQuaternion(input.reference_pelvis_orientation_wxyz,
                           "reference_pelvis_orientation_wxyz");
  ValidateFiniteArray3(input.base_ang_vel_body, "base_ang_vel_body");
  ValidateFiniteVector(input.measured_joint_pos_policy_order, "measured_joint_pos_policy_order");
  ValidateFiniteVector(input.measured_joint_vel_policy_order, "measured_joint_vel_policy_order");
  ValidateFiniteVector(default_joint_pos_policy_order_, "default_joint_pos_policy_order");
  ValidateFiniteVector(last_action_, "last_action");

  ObservationBuildResult result;
  result.observation.assign(kObservationDim, 0.0);
  result.motion_ref_ori_b = ComputeMotionRefOriB(input.current_pelvis_orientation_xyzw,
                                                 input.reference_pelvis_orientation_wxyz);
  result.last_action_used = last_action_;

  std::size_t offset = 0U;
  for (double value : input.motion_command.joint_pos_policy_order) {
    result.observation[offset++] = value;
  }
  for (double value : input.motion_command.joint_vel_policy_order) {
    result.observation[offset++] = value;
  }
  for (double value : result.motion_ref_ori_b) {
    result.observation[offset++] = value;
  }
  for (double value : input.base_ang_vel_body) {
    result.observation[offset++] = value;
  }
  for (std::size_t joint_index = 0; joint_index < kJointCount; ++joint_index) {
    result.observation[offset++] =
        input.measured_joint_pos_policy_order[joint_index] -
        default_joint_pos_policy_order_[joint_index];
  }
  for (double value : input.measured_joint_vel_policy_order) {
    result.observation[offset++] = value;
  }
  for (double value : last_action_) {
    result.observation[offset++] = value;
  }

  if (offset != kObservationDim) {
    throw std::logic_error("internal observation packing bug: final dimension is not 69.");
  }

  return result;
}

void ObsBuilder::UpdateLastAction(const std::vector<double>& action_policy_order) {
  ValidateExpectedVectorSize(action_policy_order, kJointCount, "action_policy_order");
  ValidateFiniteVector(action_policy_order, "action_policy_order");
  last_action_ = action_policy_order;
}

void ObsBuilder::ResetLastActionToZeros() { last_action_.assign(kJointCount, 0.0); }

ObsBuilder::QuaternionWxyz ObsBuilder::ConvertRosXyzwToInternalWxyz(
    const QuaternionXyzw& quaternion_xyzw) {
  ValidateFiniteQuaternionXyzw(quaternion_xyzw, "quaternion_xyzw");
  return {quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]};
}

ObsBuilder::MotionRefOriB ObsBuilder::ComputeMotionRefOriB(
    const QuaternionXyzw& current_pelvis_orientation_xyzw,
    const QuaternionWxyz& reference_pelvis_orientation_wxyz) {
  const QuaternionWxyz q_cur_w =
      NormalizeQuaternion(ConvertRosXyzwToInternalWxyz(current_pelvis_orientation_xyzw),
                          "current_pelvis_orientation_xyzw");
  const QuaternionWxyz q_ref_w =
      NormalizeQuaternion(reference_pelvis_orientation_wxyz,
                          "reference_pelvis_orientation_wxyz");

  const QuaternionWxyz q_rel = MultiplyQuaternions(InverseQuaternion(q_cur_w), q_ref_w);
  const std::array<double, 9> rotation_matrix = MatrixFromQuaternion(q_rel);

  // 这里显式按 [R00, R01, R10, R11, R20, R21] 写出前两列。
  // 不能依赖 Eigen 默认列主序 flatten，否则 reshape 后的 6 维顺序会和 sim2sim 不一致。
  return {rotation_matrix[0],
          rotation_matrix[1],
          rotation_matrix[3],
          rotation_matrix[4],
          rotation_matrix[6],
          rotation_matrix[7]};
}

void ObsBuilder::ValidateFiniteVector(const std::vector<double>& values, const char* field_name) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      std::ostringstream stream;
      stream << field_name << " contains non-finite value at index " << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

void ObsBuilder::ValidateFiniteQuaternion(const QuaternionWxyz& quaternion,
                                          const char* field_name) {
  for (std::size_t index = 0; index < quaternion.size(); ++index) {
    if (!std::isfinite(quaternion[index])) {
      std::ostringstream stream;
      stream << field_name << " contains non-finite value at index " << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

void ObsBuilder::ValidateFiniteQuaternionXyzw(const QuaternionXyzw& quaternion,
                                              const char* field_name) {
  for (std::size_t index = 0; index < quaternion.size(); ++index) {
    if (!std::isfinite(quaternion[index])) {
      std::ostringstream stream;
      stream << field_name << " contains non-finite value at index " << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

void ObsBuilder::ValidateFiniteArray3(const std::array<double, 3>& values,
                                      const char* field_name) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      std::ostringstream stream;
      stream << field_name << " contains non-finite value at index " << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

}  // namespace zky_rl_deploy
