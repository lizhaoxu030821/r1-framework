#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace zky_rl_deploy {

constexpr std::size_t kMotorCommandJointCount = 12U;
constexpr std::size_t kMotorParamsFieldsPerJoint = 6U;
constexpr std::size_t kMotorParamsLength =
    kMotorCommandJointCount * kMotorParamsFieldsPerJoint;

enum class MotorMode : int {
  Disable = 0,
  Passive = 1,
  PositionPd = 2,
  Reserved = 99,
};

struct MotorCommand {
  double kp{0.0};
  double kd{0.0};
  double joint_position{0.0};
  double joint_velocity{0.0};
  double joint_torque{0.0};
  // 即使当前底层可能还没有真正消费 mode，
  // 上层也必须继续发布并记录它，保证 CSV、状态机和后续重构都沿用同一份模式语义。
  MotorMode mode{MotorMode::Disable};
};

using MotorCommandBuffer = std::vector<MotorCommand>;

inline MotorCommandBuffer BuildDisableCommandBuffer() {
  return MotorCommandBuffer(kMotorCommandJointCount, MotorCommand{});
}

inline MotorCommandBuffer BuildPassiveCommandBuffer() {
  return MotorCommandBuffer(
      kMotorCommandJointCount,
      MotorCommand{0.0, 0.0, 0.0, 0.0, 0.0, MotorMode::Passive});
}

inline std::vector<double> PackMotorParams(const MotorCommandBuffer& joint_commands) {
  if (joint_commands.size() != kMotorCommandJointCount) {
    throw std::invalid_argument(
        "PackMotorParams expects exactly 12 joint-space commands for /motor_params.");
  }

  std::vector<double> packed;
  packed.reserve(kMotorParamsLength);
  for (const MotorCommand& command : joint_commands) {
    // /motor_params 的字段顺序必须固定为 [kp, kd, pos, vel, tau, mode] * 12，
    // 这样上层日志、单元测试和底层 CSV 才能对齐到同一份关节空间命令语义。
    packed.push_back(command.kp);
    packed.push_back(command.kd);
    packed.push_back(command.joint_position);
    packed.push_back(command.joint_velocity);
    packed.push_back(command.joint_torque);
    packed.push_back(static_cast<double>(static_cast<int>(command.mode)));
  }
  return packed;
}

enum class CommandMuxRoute {
  kZeroOutput,
  kShadowOnly,
  kActiveHardware,
};

inline const char* ToString(CommandMuxRoute route) {
  switch (route) {
    case CommandMuxRoute::kZeroOutput:
      return "zero_output";
    case CommandMuxRoute::kShadowOnly:
      return "shadow_only";
    case CommandMuxRoute::kActiveHardware:
      return "active_hardware";
  }

  return "unknown";
}

struct CommandMuxOutput {
  CommandMuxRoute route{CommandMuxRoute::kZeroOutput};
  // gated_* 表示真正允许流向物理输出链的安全命令。
  // 在当前阶段它始终保持零输出，避免 checklist 未完成前误发真实 /motor_params。
  MotorCommandBuffer gated_joint_commands{BuildDisableCommandBuffer()};
  std::vector<double> gated_motor_params{PackMotorParams(BuildDisableCommandBuffer())};
  std::optional<MotorCommandBuffer> shadow_joint_commands;
  std::optional<std::vector<double>> shadow_motor_params;
  bool publish_motor_params{false};
  bool publish_shadow_motor_params{false};
  bool publish_zero_motor_params_on_active_hardware{false};
  std::vector<std::string> reasons;
};

class CommandMux {
 public:
  static CommandMuxOutput BuildZeroOutput(
      std::vector<std::string> reasons = {}) {
    CommandMuxOutput output;
    output.route = CommandMuxRoute::kZeroOutput;
    output.reasons = std::move(reasons);
    return output;
  }

  static CommandMuxOutput BuildZeroHardwareOutput(
      std::vector<std::string> reasons = {}) {
    CommandMuxOutput output = BuildZeroOutput(std::move(reasons));
    output.publish_motor_params = true;
    output.publish_zero_motor_params_on_active_hardware = true;
    return output;
  }

  static CommandMuxOutput BuildShadowOutput(
      const MotorCommandBuffer& candidate_joint_commands,
      std::vector<std::string> reasons = {}) {
    CommandMuxOutput output;
    output.route = CommandMuxRoute::kShadowOnly;
    output.shadow_joint_commands = candidate_joint_commands;
    // 当前阶段只允许 shadow 记录候选 active 命令；
    // 即使未来底层开始消费 mode，这里也必须把 mode 一并打包记录，避免日志与真实意图脱节。
    output.shadow_motor_params = PackMotorParams(candidate_joint_commands);
    output.publish_motor_params = false;
    output.publish_shadow_motor_params = true;
    output.reasons = std::move(reasons);
    return output;
  }

  static CommandMuxOutput BuildActiveHardwareOutput(
      const MotorCommandBuffer& accepted_joint_commands,
      std::vector<std::string> reasons = {}) {
    CommandMuxOutput output;
    output.route = CommandMuxRoute::kActiveHardware;
    output.gated_joint_commands = accepted_joint_commands;
    output.gated_motor_params = PackMotorParams(accepted_joint_commands);
    output.publish_motor_params = true;
    output.publish_shadow_motor_params = false;
    output.reasons = std::move(reasons);
    return output;
  }
};

}  // namespace zky_rl_deploy
