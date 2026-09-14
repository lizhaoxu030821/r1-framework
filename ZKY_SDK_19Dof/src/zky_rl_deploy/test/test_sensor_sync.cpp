#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/sensor_sync.hpp"

namespace zky_rl_deploy {
namespace {

using Milliseconds = std::chrono::milliseconds;

SensorSync::Config MakeVerifiedConfig() {
  SensorSync::Config config;
  config.t265_extrinsic_verified = true;
  return config;
}

SensorSync::JointFeedbackSnapshot MakeJointSnapshot(SensorSync::TimePoint stamp) {
  SensorSync::JointFeedbackSnapshot snapshot;
  snapshot.stamp = stamp;
  snapshot.joint_pos_policy_order = {0.1, 0.2, 0.3};
  snapshot.joint_vel_policy_order = {1.1, 1.2, 1.3};
  return snapshot;
}

SensorSync::T265OdomSnapshot MakeOdomSnapshot(SensorSync::TimePoint stamp) {
  SensorSync::T265OdomSnapshot snapshot;
  snapshot.stamp = stamp;
  snapshot.position_flu_m = {1.0, 2.0, 3.0};
  snapshot.pelvis_orientation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return snapshot;
}

SensorSync::T265ImuSnapshot MakeImuSnapshot(SensorSync::TimePoint stamp) {
  SensorSync::T265ImuSnapshot snapshot;
  snapshot.stamp = stamp;
  snapshot.base_ang_vel_body_rad_s = {0.1, -0.2, 0.3};
  return snapshot;
}

bool ContainsReason(const std::vector<std::string>& reasons, const std::string& needle) {
  return std::any_of(reasons.begin(), reasons.end(), [&](const std::string& reason) {
    return reason.find(needle) != std::string::npos;
  });
}

TEST(SensorSyncTest, ReportsOkForFreshAlignedSnapshots) {
  SensorSync sync(MakeVerifiedConfig());
  const SensorSync::TimePoint now(Milliseconds(1000));

  sync.UpdateJointFeedback(MakeJointSnapshot(now - Milliseconds(5)));
  sync.UpdateT265Odom(MakeOdomSnapshot(now - Milliseconds(4)));
  sync.UpdateT265Imu(MakeImuSnapshot(now - Milliseconds(3)));

  const SensorSync::Evaluation evaluation =
      sync.Evaluate(now, SensorSync::OutputMode::kActiveHardware);

  EXPECT_TRUE(sync.HasCompleteSnapshot());
  EXPECT_EQ(evaluation.severity, SensorSync::Severity::kOk);
  EXPECT_TRUE(evaluation.warn_reasons.empty());
  EXPECT_TRUE(evaluation.passive_reasons.empty());
  EXPECT_TRUE(evaluation.has_complete_snapshot);
  ASSERT_TRUE(evaluation.latest_complete_snapshot.has_value());
  EXPECT_EQ(evaluation.latest_complete_snapshot->joint_feedback.joint_pos_policy_order.size(), 3U);
  EXPECT_TRUE(evaluation.output_gate.allowed);
}

TEST(SensorSyncTest, ReportsWarnWhenTimestampSkewExceedsWarnThreshold) {
  SensorSync sync(MakeVerifiedConfig());
  const SensorSync::TimePoint now(Milliseconds(1000));

  sync.UpdateJointFeedback(MakeJointSnapshot(now - Milliseconds(5)));
  sync.UpdateT265Odom(MakeOdomSnapshot(now - Milliseconds(26)));
  sync.UpdateT265Imu(MakeImuSnapshot(now - Milliseconds(5)));

  const SensorSync::Evaluation evaluation =
      sync.Evaluate(now, SensorSync::OutputMode::kActiveHardware);

  EXPECT_EQ(evaluation.severity, SensorSync::Severity::kWarn);
  EXPECT_FALSE(evaluation.warn_reasons.empty());
  EXPECT_TRUE(evaluation.passive_reasons.empty());
  EXPECT_TRUE(ContainsReason(evaluation.warn_reasons, "joint_state<->t265_odom"));
  EXPECT_TRUE(evaluation.output_gate.allowed);
}

TEST(SensorSyncTest, ReportsPassiveAndBlocksActiveWhenTimestampSkewExceedsPassiveThreshold) {
  SensorSync::Config config = MakeVerifiedConfig();
  config.joint_state_timeout = Milliseconds(100);
  config.t265_odom_timeout = Milliseconds(100);
  config.t265_imu_timeout = Milliseconds(100);

  SensorSync sync(config);
  const SensorSync::TimePoint now(Milliseconds(1100));

  sync.UpdateJointFeedback(MakeJointSnapshot(now - Milliseconds(10)));
  sync.UpdateT265Odom(MakeOdomSnapshot(now - Milliseconds(55)));
  sync.UpdateT265Imu(MakeImuSnapshot(now - Milliseconds(10)));

  const SensorSync::Evaluation evaluation =
      sync.Evaluate(now, SensorSync::OutputMode::kActiveHardware);

  EXPECT_EQ(evaluation.severity, SensorSync::Severity::kPassive);
  EXPECT_TRUE(evaluation.warn_reasons.empty());
  EXPECT_FALSE(evaluation.passive_reasons.empty());
  EXPECT_TRUE(ContainsReason(evaluation.passive_reasons, "joint_state<->t265_odom"));
  EXPECT_FALSE(evaluation.output_gate.allowed);
  EXPECT_TRUE(ContainsReason(evaluation.output_gate.blocking_reasons, "joint_state<->t265_odom"));
}

TEST(SensorSyncTest, RejectsActiveHardwareOutputWhenT265ExtrinsicIsUnverified) {
  SensorSync::Config config;
  config.t265_extrinsic_verified = false;
  config.allow_dry_run_with_unverified_extrinsic = true;
  config.allow_shadow_with_unverified_extrinsic = true;
  config.allow_motor_output_with_unverified_extrinsic = false;

  SensorSync sync(config);
  const SensorSync::TimePoint now(Milliseconds(1000));

  sync.UpdateJointFeedback(MakeJointSnapshot(now - Milliseconds(5)));
  sync.UpdateT265Odom(MakeOdomSnapshot(now - Milliseconds(4)));
  sync.UpdateT265Imu(MakeImuSnapshot(now - Milliseconds(3)));

  const SensorSync::Evaluation active_evaluation =
      sync.Evaluate(now, SensorSync::OutputMode::kActiveHardware);
  const SensorSync::Evaluation shadow_evaluation =
      sync.Evaluate(now, SensorSync::OutputMode::kShadow);

  EXPECT_EQ(active_evaluation.severity, SensorSync::Severity::kOk);
  EXPECT_FALSE(active_evaluation.output_gate.allowed);
  EXPECT_TRUE(
      ContainsReason(active_evaluation.output_gate.blocking_reasons, "T265 extrinsic is unverified"));
  EXPECT_TRUE(shadow_evaluation.output_gate.allowed);
  EXPECT_FALSE(sync.AllowsActiveHardwareOutputWithCurrentCalibration());
  EXPECT_THROW(sync.ThrowIfT265ExtrinsicUnverifiedForActiveHardwareOutput(), std::logic_error);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
