#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/joint_mapper.hpp"

namespace zky_rl_deploy {

class StandTrajectory {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  using Duration = Clock::duration;

  static constexpr std::size_t kJointCount = 12U;

  struct Config {
    Duration nominal_duration{std::chrono::seconds(5)};
    std::vector<double> max_joint_velocity_rad_s;
    double max_joint_error_rad{0.35};
    double max_abs_roll_rad{0.35};
    double max_abs_pitch_rad{0.35};
  };

  struct RuntimeSafetyContext {
    std::vector<double> measured_joint_position_hardware_order;
    double body_roll_rad{0.0};
    double body_pitch_rad{0.0};
  };

  struct SampleResult {
    std::vector<double> joint_position_hardware_order;
    double alpha{0.0};
    double smoothstep_alpha{0.0};
    double elapsed_s{0.0};
    double planned_duration_s{0.0};
    bool completed{false};
    bool request_safe_exit{false};
    std::vector<std::string> safe_exit_reasons;

    std::string Summary() const;
  };

  StandTrajectory(Config config,
                  JointMapper joint_mapper,
                  std::vector<double> stand_pose_policy_order);

  void Start(TimePoint start_time,
             const std::vector<double>& measured_joint_position_hardware_order);

  SampleResult Sample(TimePoint now, const RuntimeSafetyContext& safety_context) const;

  bool started() const { return started_; }
  double effective_duration_s() const { return effective_duration_s_; }
  const std::vector<double>& stand_pose_hardware_order() const {
    return stand_pose_hardware_order_;
  }
  const std::vector<double>& start_joint_position_hardware_order() const {
    return start_joint_position_hardware_order_;
  }

  static double ComputeSmoothstep(double alpha);

 private:
  static void ValidateConfig(const Config& config);
  static void ValidateFiniteVector(const std::vector<double>& values, const char* field_name);
  static void ValidateJointVectorSize(const std::vector<double>& values, const char* field_name);

  double ComputeMinimumDurationSForVelocityLimits(
      const std::vector<double>& start_joint_position_hardware_order) const;

  Config config_;
  JointMapper joint_mapper_;
  std::vector<double> stand_pose_hardware_order_;

  bool started_{false};
  TimePoint start_time_{};
  std::vector<double> start_joint_position_hardware_order_;
  double effective_duration_s_{0.0};
};

}  // namespace zky_rl_deploy
