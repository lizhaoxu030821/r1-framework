#include "zky_rl_deploy/core/json_motion_frame_provider.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "zky_rl_deploy/core/motor_command.hpp"

namespace zky_rl_deploy {

JsonMotionFrameProvider::JsonMotionFrameProvider(const std::string& json_path)
    : source_path_(json_path) {
  const YAML::Node root = YAML::LoadFile(json_path);
  const YAML::Node joint_pos_node = root["joint_pos"];
  const YAML::Node joint_vel_node = root["joint_vel"];
  const YAML::Node body_quat_node = root["body_quat_w"];
  if (!joint_pos_node || !joint_vel_node || !joint_pos_node.IsSequence() ||
      !joint_vel_node.IsSequence()) {
    throw std::invalid_argument("motion clip JSON must contain joint_pos/joint_vel sequences");
  }
  if (joint_pos_node.size() != joint_vel_node.size()) {
    throw std::invalid_argument("motion clip JSON joint_pos/joint_vel frame counts differ");
  }

  frames_.reserve(joint_pos_node.size());
  reference_pelvis_orientation_wxyz_.reserve(joint_pos_node.size());
  for (std::size_t frame_index = 0; frame_index < joint_pos_node.size(); ++frame_index) {
    MotionFrame frame;
    frame.joint_pos_npz_order = joint_pos_node[frame_index].as<std::vector<double>>();
    frame.joint_vel_npz_order = joint_vel_node[frame_index].as<std::vector<double>>();
    if (frame.joint_pos_npz_order.size() != kMotorCommandJointCount ||
        frame.joint_vel_npz_order.size() != kMotorCommandJointCount) {
      throw std::invalid_argument("motion clip JSON joint vectors must be 12-dimensional");
    }
    frames_.push_back(std::move(frame));

    std::array<double, 4> reference_orientation = {1.0, 0.0, 0.0, 0.0};
    if (body_quat_node && body_quat_node.IsSequence() && frame_index < body_quat_node.size() &&
        body_quat_node[frame_index].IsSequence() && body_quat_node[frame_index].size() > 0U &&
        body_quat_node[frame_index][0].IsSequence() &&
        body_quat_node[frame_index][0].size() == 4U) {
      // 这里先把 body_quat_w[frame][0] 视作 pelvis 参考姿态，只用于 dry_run/shadow 的观测闭环检查；
      // 真正上机前仍需把 body name -> index 的权威映射接进来，不能长期依赖这个简化假设。
      const std::vector<double> quaternion =
          body_quat_node[frame_index][0].as<std::vector<double>>();
      reference_orientation = {quaternion[0], quaternion[1], quaternion[2], quaternion[3]};
    } else {
      warnings_.push_back(
          "motion clip JSON is missing body_quat_w[frame][0]; using identity reference orientation");
    }
    reference_pelvis_orientation_wxyz_.push_back(reference_orientation);
  }
}

std::size_t JsonMotionFrameProvider::FrameCount() const { return frames_.size(); }

MotionFrame JsonMotionFrameProvider::GetFrame(std::size_t frame_index) const {
  if (frame_index >= frames_.size()) {
    throw std::out_of_range("motion clip frame index out of range");
  }
  return frames_[frame_index];
}

const std::array<double, 4>& JsonMotionFrameProvider::ReferencePelvisOrientationWxyz(
    std::size_t frame_index) const {
  if (frame_index >= reference_pelvis_orientation_wxyz_.size()) {
    throw std::out_of_range("motion clip reference quaternion index out of range");
  }
  return reference_pelvis_orientation_wxyz_[frame_index];
}

}  // namespace zky_rl_deploy
