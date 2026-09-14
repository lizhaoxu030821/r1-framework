#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include "zky_rl_deploy/core/deploy_cycle_logger.hpp"

namespace zky_rl_deploy {
namespace {

std::vector<double> FilledDoubleVector(std::size_t size, double value) {
  return std::vector<double>(size, value);
}

std::vector<int> FilledIntVector(std::size_t size, int value) {
  return std::vector<int>(size, value);
}

TEST(DeployCycleCsvLoggerTest, WritesHeaderAndEscapedRow) {
  const std::filesystem::path workspace_root =
      std::filesystem::temp_directory_path() / "zky_deploy_cycle_logger_test";
  std::filesystem::remove_all(workspace_root);

  std::filesystem::path log_path;
  {
    DeployCycleCsvLogger logger(workspace_root.string());
    log_path = logger.file_path();
  DeployCycleLogRecord record;
  record.ros_time_s = 12.5;
  record.loop_index = 7U;
  record.fsm_state = "PASSIVE";
  record.fsm_request_trace = "PASSIVE";
  record.current_policy_trace = "stand_hold";
  record.stage_name = "stage6_shadow";
  record.candidate_source = "motion_clip_shadow_reference";
  record.joy_summary = "A=1,B=0";
  record.beyond_mimic_action_valid = true;
  record.beyond_mimic_action_source = "onnx_shadow";
  record.inference_time_ms = 1.25;
  record.beyond_mimic_action = FilledDoubleVector(kMotorCommandJointCount, 0.1);
  record.target_joint_position = FilledDoubleVector(kMotorCommandJointCount, 0.2);
  record.limited_motor_params_route = "shadow_only";
  record.publish_shadow_motor_params = true;
  record.limited_motor_params = FilledDoubleVector(kMotorParamsLength, 0.3);
  record.joint_modes = FilledIntVector(kMotorCommandJointCount, 2);
  record.joint_feedback_position = FilledDoubleVector(kMotorCommandJointCount, 0.4);
  record.joint_feedback_velocity = FilledDoubleVector(kMotorCommandJointCount, 0.5);
  record.t265_odom_received = true;
  record.t265_imu_received = true;
  record.t265_orientation_xyzw = {0.0, 0.0, 0.0, 1.0};
  record.t265_base_ang_vel_body = {0.1, 0.2, 0.3};
  record.t265_odom_position = {1.0, 2.0, 3.0};
  record.dt_ms = 20.0;
  record.jitter_ms = 0.5;
  record.overrun_count = 1U;
  record.stale_frame_count = 0U;
  record.stale_hold_time_ms = 0;
  record.max_frame_jump_observed = 1U;
  record.shadow_cycles_without_fault = 42U;
  record.safety_trigger_reason = "none,still shadow";

    logger.Append(record);
  }

  std::ifstream stream(log_path);
  ASSERT_TRUE(stream.is_open());

  std::string header;
  std::string row;
  ASSERT_TRUE(static_cast<bool>(std::getline(stream, header)));
  ASSERT_TRUE(static_cast<bool>(std::getline(stream, row)));

  EXPECT_NE(header.find("limited_motor_param_j01_mode"), std::string::npos);
  EXPECT_NE(header.find("limited_mode_j01"), std::string::npos);
  EXPECT_NE(header.find("safety_trigger_reason"), std::string::npos);
  EXPECT_NE(row.find("\"A=1,B=0\""), std::string::npos);
  EXPECT_NE(row.find("\"none,still shadow\""), std::string::npos);

  std::filesystem::remove_all(workspace_root);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
