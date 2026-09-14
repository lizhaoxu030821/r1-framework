#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "zky_rl_deploy/core/obs_builder.hpp"

namespace zky_rl_deploy {
namespace {

std::vector<double> MakeDefaultJointPos() {
  return {0.0, 0.0, 0.0, 0.0, -0.2, 0.2, 0.5, -0.5, 0.3, -0.3, 0.0, 0.0};
}

MotionClipCommand MakeMotionCommand() {
  MotionClipCommand command;
  command.source_frame_index = 7U;
  command.joint_pos_policy_order = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
                                    7.0,  8.0,  9.0,  10.0, 11.0, 12.0};
  command.joint_vel_policy_order = {101.0, 102.0, 103.0, 104.0, 105.0, 106.0,
                                    107.0, 108.0, 109.0, 110.0, 111.0, 112.0};
  return command;
}

ObservationInput MakeObservationInput() {
  ObservationInput input;
  input.motion_command = MakeMotionCommand();
  input.current_pelvis_orientation_xyzw = {0.0, 0.0, 0.0, 1.0};
  input.reference_pelvis_orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
  input.base_ang_vel_body = {0.1, 0.2, 0.3};
  input.measured_joint_pos_policy_order = {0.1,  0.2,  0.3,  0.4,  0.5,  0.6,
                                           0.7,  0.8,  0.9,  1.0,  1.1,  1.2};
  input.measured_joint_vel_policy_order = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                                           1.7, 1.8, 1.9, 2.0, 2.1, 2.2};
  return input;
}

ObsBuilder::QuaternionWxyz QuaternionFromRollPitchYaw(double roll,
                                                      double pitch,
                                                      double yaw) {
  const double cy = std::cos(yaw * 0.5);
  const double sy = std::sin(yaw * 0.5);
  const double cp = std::cos(pitch * 0.5);
  const double sp = std::sin(pitch * 0.5);
  const double cr = std::cos(roll * 0.5);
  const double sr = std::sin(roll * 0.5);
  return {
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
  };
}

ObsBuilder::MotionRefOriB ExpectedMotionRefOriBFromRollPitchYaw(double roll,
                                                                double pitch,
                                                                double yaw) {
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);

  const double r00 = cy * cp;
  const double r01 = cy * sp * sr - sy * cr;
  const double r10 = sy * cp;
  const double r11 = sy * sp * sr + cy * cr;
  const double r20 = -sp;
  const double r21 = cp * sr;
  return {r00, r01, r10, r11, r20, r21};
}

TEST(ObsBuilderTest, Builds69DimObservationAndUsesZeroLastActionOnFirstFrame) {
  ObsBuilder builder(MakeDefaultJointPos());
  const ObservationBuildResult result = builder.BuildObservation(MakeObservationInput());

  ASSERT_EQ(result.observation.size(), ObsBuilder::kObservationDim);
  EXPECT_EQ(result.motion_ref_ori_b, ObsBuilder::MotionRefOriB({1.0, 0.0, 0.0, 1.0, 0.0, 0.0}));

  for (std::size_t index = 0; index < ObsBuilder::kJointCount; ++index) {
    EXPECT_DOUBLE_EQ(result.observation[index], static_cast<double>(index + 1));
    EXPECT_DOUBLE_EQ(result.observation[12U + index], 101.0 + static_cast<double>(index));
    EXPECT_DOUBLE_EQ(result.observation[57U + index], 0.0);
  }

  EXPECT_DOUBLE_EQ(result.observation[30], 0.1);
  EXPECT_DOUBLE_EQ(result.observation[31], 0.2);
  EXPECT_DOUBLE_EQ(result.observation[32], 0.3);

  const std::vector<double> defaults = MakeDefaultJointPos();
  const ObservationInput input = MakeObservationInput();
  for (std::size_t index = 0; index < ObsBuilder::kJointCount; ++index) {
    EXPECT_DOUBLE_EQ(result.observation[33U + index],
                     input.measured_joint_pos_policy_order[index] - defaults[index]);
    EXPECT_DOUBLE_EQ(result.observation[45U + index], input.measured_joint_vel_policy_order[index]);
  }
}

TEST(ObsBuilderTest, ComputesIdentityMotionRefOriBForUnitQuaternions) {
  const ObsBuilder::MotionRefOriB motion_ref =
      ObsBuilder::ComputeMotionRefOriB({0.0, 0.0, 0.0, 1.0}, {1.0, 0.0, 0.0, 0.0});
  EXPECT_EQ(motion_ref, ObsBuilder::MotionRefOriB({1.0, 0.0, 0.0, 1.0, 0.0, 0.0}));
}

TEST(ObsBuilderTest, ComputesKnownRelativeYawPitchRollMotionRefOriB) {
  const double roll = 0.3;
  const double pitch = -0.2;
  const double yaw = 0.4;
  const ObsBuilder::QuaternionWxyz q_ref = QuaternionFromRollPitchYaw(roll, pitch, yaw);
  const ObsBuilder::MotionRefOriB motion_ref =
      ObsBuilder::ComputeMotionRefOriB({0.0, 0.0, 0.0, 1.0}, q_ref);
  const ObsBuilder::MotionRefOriB expected =
      ExpectedMotionRefOriBFromRollPitchYaw(roll, pitch, yaw);

  for (std::size_t index = 0; index < motion_ref.size(); ++index) {
    EXPECT_NEAR(motion_ref[index], expected[index], 1e-9);
  }
}

TEST(ObsBuilderTest, UsesPreviousActionOnSecondFrame) {
  ObsBuilder builder(MakeDefaultJointPos());
  (void)builder.BuildObservation(MakeObservationInput());

  const std::vector<double> previous_action = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                               0.07, 0.08, 0.09, 0.10, 0.11, 0.12};
  builder.UpdateLastAction(previous_action);
  const ObservationBuildResult result = builder.BuildObservation(MakeObservationInput());

  for (std::size_t index = 0; index < ObsBuilder::kJointCount; ++index) {
    EXPECT_DOUBLE_EQ(result.observation[57U + index], previous_action[index]);
  }
}

TEST(ObsBuilderTest, RejectsNanOrInfInput) {
  ObsBuilder builder(MakeDefaultJointPos());
  ObservationInput input = MakeObservationInput();
  input.base_ang_vel_body[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(builder.BuildObservation(input), std::invalid_argument);

  input = MakeObservationInput();
  input.measured_joint_vel_policy_order[3] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(builder.BuildObservation(input), std::invalid_argument);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
