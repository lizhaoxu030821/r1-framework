#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "zky_rl_deploy/core/motion_clip.hpp"

namespace zky_rl_deploy {

struct ObservationInput {
  MotionClipCommand motion_command;
  std::array<double, 4> current_pelvis_orientation_xyzw{};
  std::array<double, 4> reference_pelvis_orientation_wxyz{};
  // 这里的 base_ang_vel 必须已经是 pelvis/body frame 角速度，单位 rad/s，
  // 且三个轴的正方向要和训练仿真完全一致；不能把未做外参转换的 T265 原始 IMU 角速度直接塞进来。
  std::array<double, 3> base_ang_vel_body{};
  std::vector<double> measured_joint_pos_policy_order;
  std::vector<double> measured_joint_vel_policy_order;
};

struct ObservationBuildResult {
  std::vector<double> observation;
  std::array<double, 6> motion_ref_ori_b{};
  std::vector<double> last_action_used;
};

class ObsBuilder {
 public:
  using QuaternionXyzw = std::array<double, 4>;
  using QuaternionWxyz = std::array<double, 4>;
  using MotionRefOriB = std::array<double, 6>;

  static constexpr std::size_t kJointCount = 12U;
  static constexpr std::size_t kMotionCommandDim = 24U;
  static constexpr std::size_t kMotionRefOriBDim = 6U;
  static constexpr std::size_t kBaseAngVelDim = 3U;
  static constexpr std::size_t kObservationDim = 69U;

  // default_joint_pos 来自 ONNX metadata 的权威副本，用于构造 joint_pos_rel。
  // last_action 在构造时显式初始化为全零，保持与 sim2sim_mimic.py 的 action_buffer 初始化一致。
  explicit ObsBuilder(std::vector<double> default_joint_pos_policy_order);

  ObservationBuildResult BuildObservation(const ObservationInput& input);

  void UpdateLastAction(const std::vector<double>& action_policy_order);
  void ResetLastActionToZeros();
  const std::vector<double>& last_action() const { return last_action_; }

  // ROS/T265 输入通常是 [x, y, z, w]；控制核心内部统一使用 [w, x, y, z]，
  // 避免把四元数顺序混在公式实现里，导致相对姿态和观测 silently 错掉。
  static QuaternionWxyz ConvertRosXyzwToInternalWxyz(const QuaternionXyzw& quaternion_xyzw);

  // motion_ref_ori_b 的物理含义是“参考 pelvis 的前两根坐标轴在当前 pelvis 坐标系下的投影”。
  // 这里必须先做相对姿态，再显式抽出矩阵前两列；不能直接把 T265 原始姿态拼进观测，
  // 也不能依赖 Eigen 默认列主序去 reshape，否则会和 sim2sim 的语义不一致。
  static MotionRefOriB ComputeMotionRefOriB(const QuaternionXyzw& current_pelvis_orientation_xyzw,
                                            const QuaternionWxyz& reference_pelvis_orientation_wxyz);

 private:
  static void ValidateFiniteVector(const std::vector<double>& values, const char* field_name);
  static void ValidateFiniteQuaternion(const QuaternionWxyz& quaternion,
                                       const char* field_name);
  static void ValidateFiniteQuaternionXyzw(const QuaternionXyzw& quaternion,
                                           const char* field_name);
  static void ValidateFiniteArray3(const std::array<double, 3>& values,
                                   const char* field_name);

  std::vector<double> default_joint_pos_policy_order_;
  std::vector<double> last_action_;
};

}  // namespace zky_rl_deploy
