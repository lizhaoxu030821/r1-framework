#include "zky_rl_deploy/core/joint_mapper.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace zky_rl_deploy {
namespace {

using JointOrder = JointMapper::JointOrder;
using SignConvention = JointMapper::SignConvention;
using SignVerification = JointMapper::SignVerification;

std::unordered_map<std::string, std::size_t> BuildUniqueIndexMap(
    const JointOrder& order, const std::string& order_name) {
  if (order.empty()) {
    throw std::invalid_argument(order_name + " must not be empty.");
  }

  std::unordered_map<std::string, std::size_t> index_map;
  index_map.reserve(order.size());
  for (std::size_t index = 0; index < order.size(); ++index) {
    const std::string& joint_name = order[index];
    if (joint_name.empty()) {
      throw std::invalid_argument(order_name + " contains an empty joint name.");
    }

    const auto [_, inserted] = index_map.emplace(joint_name, index);
    if (!inserted) {
      throw std::invalid_argument(order_name + " contains duplicated joint: " + joint_name);
    }
  }

  return index_map;
}

std::vector<int> BuildOrderedSigns(const JointOrder& order, const SignConvention& sign_convention) {
  if (sign_convention.size() != order.size()) {
    throw std::invalid_argument(
        "sign_convention size must exactly match policy_order size.");
  }

  std::vector<int> ordered_signs;
  ordered_signs.reserve(order.size());
  for (const std::string& joint_name : order) {
    const auto sign_it = sign_convention.find(joint_name);
    if (sign_it == sign_convention.end()) {
      throw std::invalid_argument("sign_convention is missing joint: " + joint_name);
    }
    if (sign_it->second != 1 && sign_it->second != -1) {
      throw std::invalid_argument(
          "sign_convention must be +1 or -1 for joint: " + joint_name);
    }
    ordered_signs.push_back(sign_it->second);
  }

  return ordered_signs;
}

std::vector<bool> BuildOrderedSignVerification(const JointOrder& order,
                                               const SignVerification& sign_verified) {
  // 空验证表时默认全部视为“未验证”，这是比默认 true 更保守的安全策略，
  // 可以避免开发阶段因为忘记填配置而把未验证关节误放进功能态。
  if (sign_verified.empty()) {
    return std::vector<bool>(order.size(), false);
  }

  if (sign_verified.size() != order.size()) {
    throw std::invalid_argument(
        "sign_verified size must exactly match policy_order size.");
  }

  std::vector<bool> ordered_verified;
  ordered_verified.reserve(order.size());
  for (const std::string& joint_name : order) {
    const auto verified_it = sign_verified.find(joint_name);
    if (verified_it == sign_verified.end()) {
      throw std::invalid_argument("sign_verified is missing joint: " + joint_name);
    }
    ordered_verified.push_back(verified_it->second);
  }

  return ordered_verified;
}

std::vector<int> ReorderSigns(const std::vector<int>& source_signs,
                              const std::vector<std::size_t>& target_from_source_indices) {
  std::vector<int> target_signs(target_from_source_indices.size(), 1);
  for (std::size_t target_index = 0; target_index < target_from_source_indices.size();
       ++target_index) {
    target_signs[target_index] = source_signs[target_from_source_indices[target_index]];
  }
  return target_signs;
}

std::string JoinJointNames(const std::vector<std::string>& joint_names) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < joint_names.size(); ++index) {
    if (index != 0U) {
      stream << ", ";
    }
    stream << joint_names[index];
  }
  return stream.str();
}

}  // namespace

JointMapper::JointMapper(JointOrder policy_order,
                         JointOrder hardware_order,
                         JointOrder npz_order,
                         SignConvention sign_convention,
                         SignVerification sign_verified)
    : policy_order_(std::move(policy_order)),
      hardware_order_(std::move(hardware_order)),
      npz_order_(std::move(npz_order)),
      hardware_from_policy_indices_(BuildTargetFromSourceIndices(
          policy_order_, hardware_order_, "policy_order", "hardware_order")),
      policy_from_hardware_indices_(BuildTargetFromSourceIndices(
          hardware_order_, policy_order_, "hardware_order", "policy_order")),
      policy_from_npz_indices_(
          BuildTargetFromSourceIndices(npz_order_, policy_order_, "npz_order", "policy_order")),
      policy_signs_(BuildOrderedSigns(policy_order_, sign_convention)),
      hardware_signs_(ReorderSigns(policy_signs_, hardware_from_policy_indices_)),
      policy_sign_verified_(BuildOrderedSignVerification(policy_order_, sign_verified)) {}

JointMapper::JointValues JointMapper::PolicyToHardwareCommand(
    const JointValues& policy_values) const {
  return ReorderValues(policy_values,
                       policy_order_.size(),
                       hardware_from_policy_indices_,
                       &hardware_signs_,
                       "policy_order",
                       "hardware_order");
}

JointMapper::JointValues JointMapper::HardwareToPolicyFeedback(
    const JointValues& hardware_values) const {
  return ReorderValues(hardware_values,
                       hardware_order_.size(),
                       policy_from_hardware_indices_,
                       &policy_signs_,
                       "hardware_order",
                       "policy_order");
}

JointMapper::JointValues JointMapper::NpzToPolicyReference(const JointValues& npz_values) const {
  return ReorderValues(npz_values,
                       npz_order_.size(),
                       policy_from_npz_indices_,
                       nullptr,
                       "npz_order",
                       "policy_order");
}

bool JointMapper::AllSignsVerifiedForActiveFunction() const {
  for (bool verified : policy_sign_verified_) {
    if (!verified) {
      return false;
    }
  }
  return true;
}

std::vector<JointMapper::JointName> JointMapper::GetUnverifiedSignJoints() const {
  std::vector<JointName> unverified_joints;
  for (std::size_t index = 0; index < policy_order_.size(); ++index) {
    if (!policy_sign_verified_[index]) {
      unverified_joints.push_back(policy_order_[index]);
    }
  }
  return unverified_joints;
}

void JointMapper::ThrowIfSignsUnverifiedForActiveFunction() const {
  if (AllSignsVerifiedForActiveFunction()) {
    return;
  }

  throw std::logic_error(
      "Unverified sign_convention joints cannot enter ACTIVE_FUNCTION: " +
      JoinJointNames(GetUnverifiedSignJoints()));
}

JointMapper::JointValues JointMapper::ReorderValues(
    const JointValues& source_values,
    std::size_t expected_size,
    const std::vector<std::size_t>& target_from_source_indices,
    const std::vector<int>* target_signs,
    const std::string& source_name,
    const std::string& target_name) const {
  if (source_values.size() != expected_size) {
    std::ostringstream stream;
    stream << source_name << " values size mismatch: expected " << expected_size << ", got "
           << source_values.size();
    throw std::invalid_argument(stream.str());
  }

  JointValues target_values(target_from_source_indices.size(), 0.0);
  for (std::size_t target_index = 0; target_index < target_from_source_indices.size();
       ++target_index) {
    const double value = source_values[target_from_source_indices[target_index]];
    if (target_signs == nullptr) {
      target_values[target_index] = value;
    } else {
      target_values[target_index] = static_cast<double>((*target_signs)[target_index]) * value;
    }
  }

  (void)target_name;
  return target_values;
}

std::vector<std::size_t> JointMapper::BuildTargetFromSourceIndices(
    const JointOrder& source_order,
    const JointOrder& target_order,
    const std::string& source_name,
    const std::string& target_name) {
  const auto source_indices = BuildUniqueIndexMap(source_order, source_name);
  const auto target_indices = BuildUniqueIndexMap(target_order, target_name);

  if (source_indices.size() != target_indices.size()) {
    std::ostringstream stream;
    stream << source_name << " and " << target_name
           << " do not contain the same number of joints.";
    throw std::invalid_argument(stream.str());
  }

  std::vector<std::size_t> target_from_source_indices;
  target_from_source_indices.reserve(target_order.size());

  for (const std::string& joint_name : target_order) {
    const auto source_it = source_indices.find(joint_name);
    if (source_it == source_indices.end()) {
      throw std::invalid_argument(source_name + " is missing joint required by " + target_name +
                                  ": " + joint_name);
    }
    target_from_source_indices.push_back(source_it->second);
  }

  for (const auto& [joint_name, _] : source_indices) {
    if (target_indices.find(joint_name) == target_indices.end()) {
      throw std::invalid_argument(target_name + " is missing joint required by " + source_name +
                                  ": " + joint_name);
    }
  }

  return target_from_source_indices;
}

}  // namespace zky_rl_deploy
