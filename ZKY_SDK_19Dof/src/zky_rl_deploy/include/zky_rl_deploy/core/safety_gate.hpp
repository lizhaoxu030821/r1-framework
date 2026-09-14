#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/motor_command.hpp"

namespace zky_rl_deploy {

// SafetyGate 只负责部署层的“上层放行”判断：
// 它会在 shadow/active 候选命令进入输出链前做维度、跳变和传感器条件检查，
// 但绝不替代 motor_control 里的最终硬件限幅与 EtherCAT 侧安全保护。
class SafetyGate {
 public:
  using Milliseconds = std::chrono::milliseconds;

  struct Config {
    std::vector<double> joint_lower_limits;
    std::vector<double> joint_upper_limits;
    double max_target_step_rad{0.25};
    double max_joint_error_rad{0.35};
    Milliseconds max_control_loop_jitter{5};
    std::size_t max_consecutive_overruns{3U};
    std::size_t max_stale_frame_count{3U};
    Milliseconds max_stale_hold_time{100};
    bool require_t265_verified_for_active{true};
  };

  struct RuntimeContext {
    std::vector<double> measured_joint_position;
    std::vector<double> previous_target_joint_position;
    bool t265_verified{false};
    bool allow_active_hardware_output{false};
    Milliseconds control_loop_jitter{0};
    std::size_t consecutive_overruns{0U};
    std::size_t stale_frame_count{0U};
    Milliseconds stale_hold_time{0};
  };

  struct Decision {
    bool active_command_accepted{false};
    std::vector<std::string> blocking_reasons;
    CommandMuxOutput mux_output;

    std::string Summary() const;
  };

  explicit SafetyGate(Config config);

  Decision EvaluateActiveCommand(
      const MotorCommandBuffer& candidate_joint_commands,
      const RuntimeContext& context) const;

  const Config& config() const { return config_; }

 private:
  static void ValidateConfig(const Config& config);

  Config config_;
};

}  // namespace zky_rl_deploy
