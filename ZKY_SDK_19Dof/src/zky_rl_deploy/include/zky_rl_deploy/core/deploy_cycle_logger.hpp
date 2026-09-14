#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/motor_command.hpp"

namespace zky_rl_deploy {

struct DeployCycleLogRecord {
  double ros_time_s{0.0};
  std::size_t loop_index{0U};

  std::string fsm_state;
  std::string fsm_request_trace;
  std::string current_policy_trace;
  std::string stage_name;
  std::string candidate_source;

  bool joy_received{false};
  bool joy_semantic_mapping_ready{false};
  bool joy_start_pressed{false};
  bool joy_back_pressed{false};
  bool joy_home_pressed{false};
  bool joy_rb_pressed{false};
  bool joy_a_pressed{false};
  bool joy_b_pressed{false};
  bool joy_x_pressed{false};
  bool joy_y_pressed{false};
  std::string joy_summary;

  bool beyond_mimic_action_valid{false};
  std::string beyond_mimic_action_source{"unavailable"};
  double inference_time_ms{0.0};
  std::vector<double> beyond_mimic_action;

  std::vector<double> target_joint_position;
  std::string limited_motor_params_route{"zero_output"};
  bool publish_motor_params{false};
  bool publish_shadow_motor_params{false};
  std::vector<double> limited_motor_params;
  std::vector<int> joint_modes;

  bool joint_feedback_received{false};
  std::vector<double> joint_feedback_position;
  std::vector<double> joint_feedback_velocity;

  bool t265_odom_received{false};
  bool t265_imu_received{false};
  std::array<double, 4> t265_orientation_xyzw{};
  std::array<double, 3> t265_base_ang_vel_body{};
  std::array<double, 3> t265_odom_position{};
  double joint_state_age_ms{-1.0};
  double t265_odom_age_ms{-1.0};
  double t265_imu_age_ms{-1.0};
  double joint_state_t265_odom_dt_ms{-1.0};
  double joint_state_t265_imu_dt_ms{-1.0};
  double t265_odom_t265_imu_dt_ms{-1.0};

  double dt_ms{0.0};
  double jitter_ms{0.0};
  std::size_t overrun_count{0U};
  std::size_t stale_frame_count{0U};
  int stale_hold_time_ms{0};
  std::size_t max_frame_jump_observed{0U};
  std::size_t shadow_cycles_without_fault{0U};
  std::string safety_trigger_reason{"none"};
};

// 该 CSV logger 只做周期级复盘记录，专门补齐 baseline 要求的安全门控字段。
// 它不会参与命令裁决，也不会改变 /motor_params、FSM 或硬件输出行为。
class DeployCycleCsvLogger {
 public:
  explicit DeployCycleCsvLogger(const std::string& workspace_root);

  const std::filesystem::path& file_path() const { return file_path_; }
  void Append(const DeployCycleLogRecord& record);

 private:
  static std::filesystem::path BuildOutputPath(const std::string& workspace_root);
  static void WriteHeader(std::ofstream* stream);
  static void ValidateRecord(const DeployCycleLogRecord& record);

  std::filesystem::path file_path_;
  std::ofstream stream_;
};

}  // namespace zky_rl_deploy
