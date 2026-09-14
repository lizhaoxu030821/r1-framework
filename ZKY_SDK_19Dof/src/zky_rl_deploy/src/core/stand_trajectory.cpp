#include "zky_rl_deploy/core/stand_trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace zky_rl_deploy {
namespace {

constexpr double kEpsilon = 1e-12;

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

double ClampUnitInterval(double value) {
  return std::clamp(value, 0.0, 1.0);
}

double AbsOrZero(double value) {
  return value >= 0.0 ? value : -value;
}

}  // namespace

StandTrajectory::StandTrajectory(Config config,
                                 JointMapper joint_mapper,
                                 std::vector<double> stand_pose_policy_order)
    : config_(std::move(config)),
      joint_mapper_(std::move(joint_mapper)),
      stand_pose_hardware_order_(
          joint_mapper_.PolicyToHardwareCommand(stand_pose_policy_order)) {
  ValidateConfig(config_);
  ValidateJointVectorSize(stand_pose_hardware_order_, "stand_pose_hardware_order");
  ValidateFiniteVector(stand_pose_hardware_order_, "stand_pose_hardware_order");
}

void StandTrajectory::Start(
    TimePoint start_time, const std::vector<double>& measured_joint_position_hardware_order) {
  ValidateJointVectorSize(measured_joint_position_hardware_order,
                          "measured_joint_position_hardware_order");
  ValidateFiniteVector(measured_joint_position_hardware_order,
                       "measured_joint_position_hardware_order");

  // baseline 明确禁止电机刚使能后直接阶跃到 stand_pose：
  // 刚使能时机械链路、吊轨受力和关节跟踪状态都还没稳定，直接跳到目标位会把静载和惯性冲击一次性打进结构。
  // 因此这里必须锁存进入 STAND_INIT 那一刻的实测关节角，后续所有命令都从“机器人此刻真实在的位置”平滑插值过去。
  start_time_ = start_time;
  start_joint_position_hardware_order_ = measured_joint_position_hardware_order;
  effective_duration_s_ = std::max(
      std::chrono::duration<double>(config_.nominal_duration).count(),
      ComputeMinimumDurationSForVelocityLimits(start_joint_position_hardware_order_));
  started_ = true;
}

StandTrajectory::SampleResult StandTrajectory::Sample(
    TimePoint now, const RuntimeSafetyContext& safety_context) const {
  if (!started_) {
    throw std::logic_error("StandTrajectory must be started before sampling.");
  }

  ValidateJointVectorSize(safety_context.measured_joint_position_hardware_order,
                          "measured_joint_position_hardware_order");
  ValidateFiniteVector(safety_context.measured_joint_position_hardware_order,
                       "measured_joint_position_hardware_order");

  SampleResult result;
  result.joint_position_hardware_order.resize(kJointCount, 0.0);
  result.elapsed_s =
      std::max(0.0, std::chrono::duration<double>(now - start_time_).count());
  result.planned_duration_s = effective_duration_s_;
  result.alpha = effective_duration_s_ <= kEpsilon
                     ? 1.0
                     : ClampUnitInterval(result.elapsed_s / effective_duration_s_);
  result.smoothstep_alpha = ComputeSmoothstep(result.alpha);
  result.completed = result.alpha >= 1.0 - kEpsilon;

  for (std::size_t joint_index = 0; joint_index < kJointCount; ++joint_index) {
    result.joint_position_hardware_order[joint_index] =
        result.smoothstep_alpha * stand_pose_hardware_order_[joint_index] +
        (1.0 - result.smoothstep_alpha) * start_joint_position_hardware_order_[joint_index];
  }

  if (!std::isfinite(safety_context.body_roll_rad)) {
    result.safe_exit_reasons.push_back("body_roll_rad is NaN/Inf");
  } else if (AbsOrZero(safety_context.body_roll_rad) > config_.max_abs_roll_rad) {
    std::ostringstream stream;
    stream << "body roll exceeded limit: |" << safety_context.body_roll_rad << "| > "
           << config_.max_abs_roll_rad;
    result.safe_exit_reasons.push_back(stream.str());
  }

  if (!std::isfinite(safety_context.body_pitch_rad)) {
    result.safe_exit_reasons.push_back("body_pitch_rad is NaN/Inf");
  } else if (AbsOrZero(safety_context.body_pitch_rad) > config_.max_abs_pitch_rad) {
    std::ostringstream stream;
    stream << "body pitch exceeded limit: |" << safety_context.body_pitch_rad << "| > "
           << config_.max_abs_pitch_rad;
    result.safe_exit_reasons.push_back(stream.str());
  }

  for (std::size_t joint_index = 0; joint_index < kJointCount; ++joint_index) {
    const double tracking_error =
        std::fabs(safety_context.measured_joint_position_hardware_order[joint_index] -
                  result.joint_position_hardware_order[joint_index]);
    if (tracking_error > config_.max_joint_error_rad) {
      std::ostringstream stream;
      stream << "joint tracking error exceeded at joint index " << joint_index << ": "
             << tracking_error << " > " << config_.max_joint_error_rad;
      result.safe_exit_reasons.push_back(stream.str());
    }
  }

  result.request_safe_exit = !result.safe_exit_reasons.empty();
  return result;
}

double StandTrajectory::ComputeSmoothstep(double alpha) {
  const double clamped_alpha = ClampUnitInterval(alpha);
  return clamped_alpha * clamped_alpha * (3.0 - 2.0 * clamped_alpha);
}

void StandTrajectory::ValidateConfig(const Config& config) {
  if (config.nominal_duration <= Duration::zero()) {
    throw std::invalid_argument("nominal_duration must be > 0");
  }
  ValidateJointVectorSize(config.max_joint_velocity_rad_s, "max_joint_velocity_rad_s");
  ValidateFiniteVector(config.max_joint_velocity_rad_s, "max_joint_velocity_rad_s");
  for (double limit : config.max_joint_velocity_rad_s) {
    if (limit <= 0.0) {
      throw std::invalid_argument("max_joint_velocity_rad_s must be > 0 for every joint");
    }
  }
  if (!std::isfinite(config.max_joint_error_rad) || config.max_joint_error_rad <= 0.0) {
    throw std::invalid_argument("max_joint_error_rad must be finite and > 0");
  }
  if (!std::isfinite(config.max_abs_roll_rad) || config.max_abs_roll_rad <= 0.0) {
    throw std::invalid_argument("max_abs_roll_rad must be finite and > 0");
  }
  if (!std::isfinite(config.max_abs_pitch_rad) || config.max_abs_pitch_rad <= 0.0) {
    throw std::invalid_argument("max_abs_pitch_rad must be finite and > 0");
  }
}

void StandTrajectory::ValidateFiniteVector(const std::vector<double>& values,
                                           const char* field_name) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      std::ostringstream stream;
      stream << field_name << " contains NaN/Inf at index " << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

void StandTrajectory::ValidateJointVectorSize(const std::vector<double>& values,
                                              const char* field_name) {
  if (values.size() != kJointCount) {
    std::ostringstream stream;
    stream << field_name << " must contain exactly " << kJointCount
           << " entries, got " << values.size();
    throw std::invalid_argument(stream.str());
  }
}

double StandTrajectory::ComputeMinimumDurationSForVelocityLimits(
    const std::vector<double>& start_joint_position_hardware_order) const {
  double minimum_duration_s = 0.0;
  for (std::size_t joint_index = 0; joint_index < kJointCount; ++joint_index) {
    const double delta =
        std::fabs(stand_pose_hardware_order_[joint_index] -
                  start_joint_position_hardware_order[joint_index]);
    const double required_duration_s =
        (1.5 * delta) / config_.max_joint_velocity_rad_s[joint_index];
    minimum_duration_s = std::max(minimum_duration_s, required_duration_s);
  }
  return minimum_duration_s;
}

std::string StandTrajectory::SampleResult::Summary() const {
  std::ostringstream stream;
  stream << "alpha=" << alpha << ", smoothstep_alpha=" << smoothstep_alpha
         << ", elapsed_s=" << elapsed_s
         << ", planned_duration_s=" << planned_duration_s
         << ", completed=" << (completed ? "true" : "false")
         << ", request_safe_exit=" << (request_safe_exit ? "true" : "false")
         << ", safe_exit_reasons=" << JoinReasons(safe_exit_reasons);
  return stream.str();
}

}  // namespace zky_rl_deploy
