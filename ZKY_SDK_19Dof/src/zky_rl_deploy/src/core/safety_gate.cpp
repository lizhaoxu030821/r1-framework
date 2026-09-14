#include "zky_rl_deploy/core/safety_gate.hpp"

#include <cmath>
#include <sstream>
#include <utility>

namespace zky_rl_deploy {
namespace {

using Milliseconds = SafetyGate::Milliseconds;

std::string JoinReasons(const std::vector<std::string>& reasons) {
  if (reasons.empty()) {
    return "none";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < reasons.size(); ++index) {
    if (index != 0U) {
      stream << "; ";
    }
    stream << reasons[index];
  }
  return stream.str();
}

Milliseconds AbsMilliseconds(Milliseconds value) {
  return value >= Milliseconds::zero() ? value : -value;
}

void AppendDimensionReason(const char* label,
                           std::size_t actual_size,
                           std::vector<std::string>* reasons) {
  std::ostringstream stream;
  stream << label << " dimension mismatch: expected " << kMotorCommandJointCount
         << ", got " << actual_size;
  reasons->push_back(stream.str());
}

void AppendNonFiniteValueReason(const char* label,
                                std::size_t index,
                                std::vector<std::string>* reasons) {
  std::ostringstream stream;
  stream << label << " contains NaN/Inf at joint index " << index;
  reasons->push_back(stream.str());
}

void AppendNonFiniteContextReason(const char* label,
                                  std::size_t index,
                                  std::vector<std::string>* reasons) {
  std::ostringstream stream;
  stream << label << " contains NaN/Inf at index " << index;
  reasons->push_back(stream.str());
}

void AppendLimitReason(std::size_t index,
                       double target_position,
                       double lower_limit,
                       double upper_limit,
                       std::vector<std::string>* reasons) {
  std::ostringstream stream;
  stream << "joint_position exceeds limit at joint index " << index << ": target="
         << target_position << ", allowed=[" << lower_limit << ", " << upper_limit << "]";
  reasons->push_back(stream.str());
}

void AppendTargetStepReason(std::size_t index,
                            double current_target,
                            double previous_target,
                            double limit,
                            std::vector<std::string>* reasons) {
  std::ostringstream stream;
  stream << "single-step target jump exceeded at joint index " << index
         << ": |" << current_target << " - " << previous_target << "| > " << limit;
  reasons->push_back(stream.str());
}

void AppendJointErrorReason(std::size_t index,
                            double current_target,
                            double measured_position,
                            double limit,
                            std::vector<std::string>* reasons) {
  std::ostringstream stream;
  stream << "joint tracking error exceeded at joint index " << index
         << ": |" << current_target << " - " << measured_position << "| > " << limit;
  reasons->push_back(stream.str());
}

void AppendNonFiniteCommandReasons(const MotorCommandBuffer& commands,
                                   std::vector<std::string>* reasons) {
  for (std::size_t index = 0; index < commands.size(); ++index) {
    const MotorCommand& command = commands[index];
    if (!std::isfinite(command.kp)) {
      AppendNonFiniteValueReason("kp", index, reasons);
    }
    if (!std::isfinite(command.kd)) {
      AppendNonFiniteValueReason("kd", index, reasons);
    }
    if (!std::isfinite(command.joint_position)) {
      AppendNonFiniteValueReason("joint_position", index, reasons);
    }
    if (!std::isfinite(command.joint_velocity)) {
      AppendNonFiniteValueReason("joint_velocity", index, reasons);
    }
    if (!std::isfinite(command.joint_torque)) {
      AppendNonFiniteValueReason("joint_torque", index, reasons);
    }
  }
}

void AppendNonFiniteContextVectorReasons(const std::vector<double>& values,
                                         const char* label,
                                         std::vector<std::string>* reasons) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      AppendNonFiniteContextReason(label, index, reasons);
    }
  }
}

}  // namespace

SafetyGate::SafetyGate(Config config) : config_(std::move(config)) {
  ValidateConfig(config_);
}

SafetyGate::Decision SafetyGate::EvaluateActiveCommand(
    const MotorCommandBuffer& candidate_joint_commands,
    const RuntimeContext& context) const {
  Decision decision;

  const bool candidate_size_ok =
      candidate_joint_commands.size() == kMotorCommandJointCount;
  if (!candidate_size_ok) {
    AppendDimensionReason(
        "candidate_joint_commands", candidate_joint_commands.size(), &decision.blocking_reasons);
  } else {
    AppendNonFiniteCommandReasons(candidate_joint_commands, &decision.blocking_reasons);
  }

  const bool measured_size_ok =
      context.measured_joint_position.size() == kMotorCommandJointCount;
  if (!measured_size_ok) {
    AppendDimensionReason(
        "measured_joint_position", context.measured_joint_position.size(), &decision.blocking_reasons);
  } else {
    AppendNonFiniteContextVectorReasons(
        context.measured_joint_position, "measured_joint_position", &decision.blocking_reasons);
  }

  const bool previous_target_missing = context.previous_target_joint_position.empty();
  const bool previous_target_size_ok =
      previous_target_missing ||
      context.previous_target_joint_position.size() == kMotorCommandJointCount;
  if (!previous_target_size_ok) {
    AppendDimensionReason("previous_target_joint_position",
                          context.previous_target_joint_position.size(),
                          &decision.blocking_reasons);
  } else if (!previous_target_missing) {
    AppendNonFiniteContextVectorReasons(context.previous_target_joint_position,
                                        "previous_target_joint_position",
                                        &decision.blocking_reasons);
  }

  if (candidate_size_ok) {
    for (std::size_t index = 0; index < kMotorCommandJointCount; ++index) {
      const MotorCommand& command = candidate_joint_commands[index];

      if (std::isfinite(command.joint_position) &&
          (command.joint_position < config_.joint_lower_limits[index] ||
           command.joint_position > config_.joint_upper_limits[index])) {
        AppendLimitReason(index,
                          command.joint_position,
                          config_.joint_lower_limits[index],
                          config_.joint_upper_limits[index],
                          &decision.blocking_reasons);
      }

      if (!previous_target_missing && previous_target_size_ok &&
          std::isfinite(command.joint_position) &&
          std::isfinite(context.previous_target_joint_position[index]) &&
          std::fabs(command.joint_position - context.previous_target_joint_position[index]) >
              config_.max_target_step_rad) {
        AppendTargetStepReason(index,
                               command.joint_position,
                               context.previous_target_joint_position[index],
                               config_.max_target_step_rad,
                               &decision.blocking_reasons);
      }

      if (measured_size_ok && std::isfinite(command.joint_position) &&
          std::isfinite(context.measured_joint_position[index]) &&
          std::fabs(command.joint_position - context.measured_joint_position[index]) >
              config_.max_joint_error_rad) {
        AppendJointErrorReason(index,
                               command.joint_position,
                               context.measured_joint_position[index],
                               config_.max_joint_error_rad,
                               &decision.blocking_reasons);
      }
    }
  }

  if (config_.require_t265_verified_for_active && !context.t265_verified) {
    decision.blocking_reasons.push_back(
        "T265 calibration verified=false; active command can only stay in shadow.");
  }

  if (AbsMilliseconds(context.control_loop_jitter) > config_.max_control_loop_jitter) {
    std::ostringstream stream;
    stream << "control loop jitter exceeded: "
           << AbsMilliseconds(context.control_loop_jitter).count() << " ms > "
           << config_.max_control_loop_jitter.count() << " ms";
    decision.blocking_reasons.push_back(stream.str());
  }

  if (context.consecutive_overruns > config_.max_consecutive_overruns) {
    std::ostringstream stream;
    stream << "control loop overrun count exceeded: " << context.consecutive_overruns << " > "
           << config_.max_consecutive_overruns;
    decision.blocking_reasons.push_back(stream.str());
  }

  if (context.stale_frame_count > config_.max_stale_frame_count) {
    std::ostringstream stream;
    stream << "stale frame count exceeded: " << context.stale_frame_count << " > "
           << config_.max_stale_frame_count;
    decision.blocking_reasons.push_back(stream.str());
  }

  if (context.stale_hold_time > config_.max_stale_hold_time) {
    std::ostringstream stream;
    stream << "stale hold time exceeded: " << context.stale_hold_time.count() << " ms > "
           << config_.max_stale_hold_time.count() << " ms";
    decision.blocking_reasons.push_back(stream.str());
  }

  if (decision.blocking_reasons.empty()) {
    decision.active_command_accepted = true;
    decision.mux_output = context.allow_active_hardware_output
                              ? CommandMux::BuildActiveHardwareOutput(candidate_joint_commands)
                              : CommandMux::BuildShadowOutput(candidate_joint_commands);
  } else {
    decision.active_command_accepted = false;
    decision.mux_output = context.allow_active_hardware_output
                              ? CommandMux::BuildZeroHardwareOutput(decision.blocking_reasons)
                              : CommandMux::BuildZeroOutput(decision.blocking_reasons);
  }

  return decision;
}

std::string SafetyGate::Decision::Summary() const {
  std::ostringstream stream;
  stream << "active_command_accepted=" << (active_command_accepted ? "true" : "false")
         << ", blocking_reasons=" << JoinReasons(blocking_reasons)
         << ", mux_route=" << ToString(mux_output.route)
         << ", publish_motor_params=" << (mux_output.publish_motor_params ? "true" : "false")
         << ", publish_shadow_motor_params="
         << (mux_output.publish_shadow_motor_params ? "true" : "false");
  return stream.str();
}

void SafetyGate::ValidateConfig(const Config& config) {
  if (config.joint_lower_limits.size() != kMotorCommandJointCount) {
    throw std::invalid_argument(
        "joint_lower_limits must contain exactly 12 entries.");
  }
  if (config.joint_upper_limits.size() != kMotorCommandJointCount) {
    throw std::invalid_argument(
        "joint_upper_limits must contain exactly 12 entries.");
  }
  if (config.max_target_step_rad <= 0.0) {
    throw std::invalid_argument("max_target_step_rad must be > 0.");
  }
  if (config.max_joint_error_rad <= 0.0) {
    throw std::invalid_argument("max_joint_error_rad must be > 0.");
  }
  if (config.max_control_loop_jitter < Milliseconds::zero()) {
    throw std::invalid_argument("max_control_loop_jitter must be >= 0 ms.");
  }
  if (config.max_stale_hold_time < Milliseconds::zero()) {
    throw std::invalid_argument("max_stale_hold_time must be >= 0 ms.");
  }

  for (std::size_t index = 0; index < kMotorCommandJointCount; ++index) {
    if (!std::isfinite(config.joint_lower_limits[index]) ||
        !std::isfinite(config.joint_upper_limits[index])) {
      throw std::invalid_argument(
          "joint limits must be finite for every joint.");
    }
    if (config.joint_upper_limits[index] < config.joint_lower_limits[index]) {
      throw std::invalid_argument(
          "joint_upper_limits must be >= joint_lower_limits for every joint.");
    }
  }
}

}  // namespace zky_rl_deploy
