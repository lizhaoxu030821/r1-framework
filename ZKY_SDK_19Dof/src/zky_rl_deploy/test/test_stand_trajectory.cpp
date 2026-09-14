#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "zky_rl_deploy/core/joint_mapper.hpp"
#include "zky_rl_deploy/core/stand_trajectory.hpp"

namespace zky_rl_deploy {
namespace {

using Milliseconds = std::chrono::milliseconds;

using JointOrder = JointMapper::JointOrder;
using SignConvention = JointMapper::SignConvention;
using SignVerification = JointMapper::SignVerification;

JointOrder MakePolicyOrder() {
  return {"left_hip_yaw_joint",      "right_hip_yaw_joint",      "left_hip_roll_joint",
          "right_hip_roll_joint",    "left_hip_pitch_joint",    "right_hip_pitch_joint",
          "left_knee_joint",         "right_knee_joint",        "left_ankle_pitch_joint",
          "right_ankle_pitch_joint", "left_ankle_roll_joint",   "right_ankle_roll_joint"};
}

JointOrder MakeHardwareOrder() {
  return {"left_hip_yaw_joint",   "left_hip_roll_joint",     "left_hip_pitch_joint",
          "left_knee_joint",      "left_ankle_pitch_joint",  "left_ankle_roll_joint",
          "right_hip_yaw_joint",  "right_hip_roll_joint",    "right_hip_pitch_joint",
          "right_knee_joint",     "right_ankle_pitch_joint", "right_ankle_roll_joint"};
}

SignConvention MakeAllPositiveSigns() {
  SignConvention signs;
  for (const std::string& joint_name : MakePolicyOrder()) {
    signs.emplace(joint_name, 1);
  }
  return signs;
}

SignVerification MakeAllVerified() {
  SignVerification verified;
  for (const std::string& joint_name : MakePolicyOrder()) {
    verified.emplace(joint_name, true);
  }
  return verified;
}

JointMapper MakeJointMapper() {
  return JointMapper(
      MakePolicyOrder(), MakeHardwareOrder(), MakePolicyOrder(), MakeAllPositiveSigns(),
      MakeAllVerified());
}

std::vector<double> MakeDefaultJointPosPolicyOrder() {
  return {0.0, 0.0, 0.0, 0.0, -0.2, 0.2, 0.5, -0.5, 0.3, -0.3, 0.0, 0.0};
}

StandTrajectory::Config MakeConfig() {
  StandTrajectory::Config config;
  config.nominal_duration = std::chrono::seconds(4);
  config.max_joint_velocity_rad_s = std::vector<double>(StandTrajectory::kJointCount, 10.0);
  config.max_joint_error_rad = 0.2;
  config.max_abs_roll_rad = 0.35;
  config.max_abs_pitch_rad = 0.35;
  return config;
}

StandTrajectory::RuntimeSafetyContext MakeSafetyContext(
    const std::vector<double>& measured_joint_position_hardware_order) {
  StandTrajectory::RuntimeSafetyContext context;
  context.measured_joint_position_hardware_order = measured_joint_position_hardware_order;
  context.body_roll_rad = 0.0;
  context.body_pitch_rad = 0.0;
  return context;
}

TEST(StandTrajectoryTest, ReordersDefaultStandPoseIntoHardwareOrder) {
  StandTrajectory trajectory(MakeConfig(), MakeJointMapper(), MakeDefaultJointPosPolicyOrder());

  const std::vector<double> expected_hardware_order = {
      0.0, 0.0, -0.2, 0.5, 0.3, 0.0, 0.0, 0.0, 0.2, -0.5, -0.3, 0.0};
  EXPECT_EQ(trajectory.stand_pose_hardware_order(), expected_hardware_order);
}

TEST(StandTrajectoryTest, ReturnsStartAndEndPoseAtTrajectoryBoundaries) {
  StandTrajectory trajectory(MakeConfig(), MakeJointMapper(), MakeDefaultJointPosPolicyOrder());
  const StandTrajectory::TimePoint start_time(Milliseconds(1000));
  const std::vector<double> measured_start = {
      0.1, -0.1, -0.1, 0.4, 0.2, 0.0, 0.1, -0.1, 0.1, -0.4, -0.2, 0.0};

  trajectory.Start(start_time, measured_start);

  const StandTrajectory::SampleResult start_sample =
      trajectory.Sample(start_time, MakeSafetyContext(measured_start));
  EXPECT_EQ(start_sample.joint_position_hardware_order, measured_start);
  EXPECT_DOUBLE_EQ(start_sample.alpha, 0.0);
  EXPECT_DOUBLE_EQ(start_sample.smoothstep_alpha, 0.0);
  EXPECT_FALSE(start_sample.completed);

  const StandTrajectory::TimePoint end_time =
      start_time + std::chrono::seconds(4);
  const StandTrajectory::SampleResult end_sample = trajectory.Sample(
      end_time, MakeSafetyContext(trajectory.stand_pose_hardware_order()));
  EXPECT_EQ(end_sample.joint_position_hardware_order, trajectory.stand_pose_hardware_order());
  EXPECT_DOUBLE_EQ(end_sample.alpha, 1.0);
  EXPECT_DOUBLE_EQ(end_sample.smoothstep_alpha, 1.0);
  EXPECT_TRUE(end_sample.completed);
}

TEST(StandTrajectoryTest, UsesSmoothstepMidpointAtHalfDuration) {
  StandTrajectory trajectory(MakeConfig(), MakeJointMapper(), MakeDefaultJointPosPolicyOrder());
  const StandTrajectory::TimePoint start_time(Milliseconds(0));
  const std::vector<double> measured_start(StandTrajectory::kJointCount, 0.0);
  trajectory.Start(start_time, measured_start);

  const StandTrajectory::TimePoint mid_time = start_time + std::chrono::seconds(2);
  const StandTrajectory::SampleResult mid_sample =
      trajectory.Sample(mid_time, MakeSafetyContext(measured_start));

  EXPECT_DOUBLE_EQ(mid_sample.alpha, 0.5);
  EXPECT_DOUBLE_EQ(mid_sample.smoothstep_alpha, 0.5);
  for (std::size_t joint_index = 0; joint_index < StandTrajectory::kJointCount; ++joint_index) {
    EXPECT_DOUBLE_EQ(mid_sample.joint_position_hardware_order[joint_index],
                     0.5 * trajectory.stand_pose_hardware_order()[joint_index]);
  }
}

TEST(StandTrajectoryTest, ExpandsDurationToRespectMaxJointVelocityLimit) {
  StandTrajectory::Config config = MakeConfig();
  config.nominal_duration = std::chrono::seconds(4);
  config.max_joint_velocity_rad_s = std::vector<double>(StandTrajectory::kJointCount, 0.1);
  StandTrajectory trajectory(config, MakeJointMapper(), MakeDefaultJointPosPolicyOrder());

  const StandTrajectory::TimePoint start_time(Milliseconds(0));
  const std::vector<double> measured_start(StandTrajectory::kJointCount, 0.0);
  trajectory.Start(start_time, measured_start);

  EXPECT_NEAR(trajectory.effective_duration_s(), 7.5, 1e-9);

  const double delta_t_s = 0.01;
  const StandTrajectory::TimePoint sample_time = start_time + Milliseconds(3750);
  const StandTrajectory::SampleResult sample_before =
      trajectory.Sample(sample_time, MakeSafetyContext(measured_start));
  const StandTrajectory::SampleResult sample_after = trajectory.Sample(
      sample_time + Milliseconds(10), MakeSafetyContext(measured_start));

  for (std::size_t joint_index = 0; joint_index < StandTrajectory::kJointCount; ++joint_index) {
    const double velocity =
        std::fabs(sample_after.joint_position_hardware_order[joint_index] -
                  sample_before.joint_position_hardware_order[joint_index]) /
        delta_t_s;
    EXPECT_LE(velocity, config.max_joint_velocity_rad_s[joint_index] + 1e-6);
  }
}

TEST(StandTrajectoryTest, RequestsSafeExitOnTrackingErrorViolation) {
  StandTrajectory trajectory(MakeConfig(), MakeJointMapper(), MakeDefaultJointPosPolicyOrder());
  const StandTrajectory::TimePoint start_time(Milliseconds(0));
  const std::vector<double> measured_start(StandTrajectory::kJointCount, 0.0);
  trajectory.Start(start_time, measured_start);

  StandTrajectory::RuntimeSafetyContext context = MakeSafetyContext(measured_start);
  context.measured_joint_position_hardware_order[2] = 1.0;

  const StandTrajectory::SampleResult sample =
      trajectory.Sample(start_time + Milliseconds(1000), context);

  EXPECT_TRUE(sample.request_safe_exit);
  EXPECT_FALSE(sample.safe_exit_reasons.empty());
  EXPECT_NE(sample.Summary().find("request_safe_exit=true"), std::string::npos);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
