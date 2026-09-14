#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "zky_rl_deploy/core/joint_mapper.hpp"
#include "zky_rl_deploy/core/motion_clip.hpp"

namespace zky_rl_deploy {
namespace {

using JointOrder = JointMapper::JointOrder;
using SignConvention = JointMapper::SignConvention;
using SignVerification = JointMapper::SignVerification;

class InMemoryMotionFrameProvider final : public MotionFrameProvider {
 public:
  explicit InMemoryMotionFrameProvider(std::vector<MotionFrame> frames)
      : frames_(std::move(frames)) {}

  std::size_t FrameCount() const override { return frames_.size(); }

  MotionFrame GetFrame(std::size_t frame_index) const override { return frames_.at(frame_index); }

 private:
  std::vector<MotionFrame> frames_;
};

JointOrder MakePolicyOrder() {
  return {
      "left_hip_yaw_joint",   "right_hip_yaw_joint",   "left_hip_roll_joint",
      "right_hip_roll_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
      "left_knee_joint",      "right_knee_joint",      "left_ankle_pitch_joint",
      "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"};
}

JointOrder MakeHardwareOrder() {
  return {"left_hip_yaw_joint",    "left_hip_roll_joint",      "left_hip_pitch_joint",
          "left_knee_joint",       "left_ankle_pitch_joint",   "left_ankle_roll_joint",
          "right_hip_yaw_joint",   "right_hip_roll_joint",     "right_hip_pitch_joint",
          "right_knee_joint",      "right_ankle_pitch_joint",  "right_ankle_roll_joint"};
}

JointOrder MakeCustomNpzOrder() {
  return {"left_knee_joint",        "right_knee_joint",       "left_hip_yaw_joint",
          "right_hip_yaw_joint",    "left_hip_roll_joint",    "right_hip_roll_joint",
          "left_hip_pitch_joint",   "right_hip_pitch_joint",  "left_ankle_pitch_joint",
          "right_ankle_pitch_joint","left_ankle_roll_joint",  "right_ankle_roll_joint"};
}

SignConvention MakeAllPositiveSigns(const JointOrder& policy_order) {
  SignConvention signs;
  for (const std::string& joint_name : policy_order) {
    signs.emplace(joint_name, 1);
  }
  return signs;
}

SignVerification MakeAllVerified(const JointOrder& policy_order) {
  SignVerification verified;
  for (const std::string& joint_name : policy_order) {
    verified.emplace(joint_name, true);
  }
  return verified;
}

std::shared_ptr<const MotionFrameProvider> MakeProvider() {
  std::vector<MotionFrame> frames;
  frames.push_back(MotionFrame{{70.0, 80.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0, 100.0, 110.0, 120.0},
                               {700.0, 800.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 900.0, 1000.0, 1100.0, 1200.0}});
  frames.push_back(MotionFrame{{71.0, 81.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 91.0, 101.0, 111.0, 121.0},
                               {701.0, 801.0, 101.0, 201.0, 301.0, 401.0, 501.0, 601.0, 901.0, 1001.0, 1101.0, 1201.0}});
  frames.push_back(MotionFrame{{72.0, 82.0, 12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 92.0, 102.0, 112.0, 122.0},
                               {702.0, 802.0, 102.0, 202.0, 302.0, 402.0, 502.0, 602.0, 902.0, 1002.0, 1102.0, 1202.0}});
  return std::make_shared<InMemoryMotionFrameProvider>(std::move(frames));
}

JointMapper MakeJointMapper() {
  const JointOrder policy_order = MakePolicyOrder();
  return JointMapper(policy_order,
                     MakeHardwareOrder(),
                     MakeCustomNpzOrder(),
                     MakeAllPositiveSigns(policy_order),
                     MakeAllVerified(policy_order));
}

TEST(MotionClipTest, PlaysWindowAndReordersNpzDataToPolicyOrder) {
  MotionClipConfig config;
  config.start_frame = 0U;
  config.end_frame = 2U;
  config.loop = false;
  config.playback_rate = 1.0;
  config.max_frame_jump = 1U;

  MotionClip clip(config, MakeJointMapper(), MakeProvider());

  const MotionClipStepResult first = clip.ResetToWindowStart();
  EXPECT_EQ(first.command.source_frame_index, 0U);
  EXPECT_EQ(first.command.joint_pos_policy_order.front(), 10.0);
  EXPECT_EQ(first.command.joint_pos_policy_order[6], 70.0);
  EXPECT_EQ(first.command.joint_vel_policy_order[6], 700.0);

  const MotionClipStepResult second = clip.Advance(false);
  EXPECT_EQ(second.command.source_frame_index, 1U);
  EXPECT_EQ(second.command.joint_pos_policy_order.front(), 11.0);
  EXPECT_EQ(second.frame_jump, 1U);

  const MotionClipStepResult third = clip.Advance(false);
  EXPECT_EQ(third.command.source_frame_index, 2U);
  EXPECT_EQ(third.command.joint_pos_policy_order.front(), 12.0);
}

TEST(MotionClipTest, LoopsBackToWindowStart) {
  MotionClipConfig config;
  config.start_frame = 0U;
  config.end_frame = 1U;
  config.loop = true;
  config.playback_rate = 1.0;
  config.max_frame_jump = 1U;

  MotionClip clip(config, MakeJointMapper(), MakeProvider());
  clip.ResetToWindowStart();
  const MotionClipStepResult second = clip.Advance(false);
  EXPECT_EQ(second.command.source_frame_index, 1U);

  const MotionClipStepResult looped = clip.Advance(false);
  EXPECT_TRUE(looped.looped);
  EXPECT_EQ(looped.command.source_frame_index, 0U);
}

TEST(MotionClipTest, HoldsOnStaleAndRequestsPassiveAfterConsecutiveLimit) {
  MotionClipConfig config;
  config.loop = true;
  config.playback_rate = 1.0;
  config.max_frame_jump = 1U;
  config.stale_frame_behavior = StaleFrameBehavior::Hold;
  config.max_consecutive_stale_frames = 3U;
  config.max_hold_time_ms = 100;
  config.nominal_period_ms = 20;

  MotionClip clip(config, MakeJointMapper(), MakeProvider());
  clip.ResetToWindowStart();

  const MotionClipStepResult stale1 = clip.Advance(true);
  EXPECT_FALSE(stale1.request_passive);
  EXPECT_EQ(stale1.stale_frame_count, 1U);
  EXPECT_EQ(stale1.stale_hold_time_ms, 20);
  EXPECT_EQ(stale1.command.source_frame_index, 0U);

  const MotionClipStepResult stale4 = [&clip]() {
    clip.Advance(true);
    clip.Advance(true);
    return clip.Advance(true);
  }();
  EXPECT_TRUE(stale4.request_passive);
  EXPECT_NE(stale4.reason.find("stale frame count exceeded limit"), std::string::npos);
  EXPECT_EQ(stale4.command.source_frame_index, 0U);
}

TEST(MotionClipTest, RequestsPassiveAfterHoldTimeout) {
  MotionClipConfig config;
  config.loop = true;
  config.playback_rate = 1.0;
  config.max_frame_jump = 1U;
  config.stale_frame_behavior = StaleFrameBehavior::Hold;
  config.max_consecutive_stale_frames = 100U;
  config.max_hold_time_ms = 100;
  config.nominal_period_ms = 20;

  MotionClip clip(config, MakeJointMapper(), MakeProvider());
  clip.ResetToWindowStart();

  MotionClipStepResult last_result;
  for (int index = 0; index < 6; ++index) {
    last_result = clip.Advance(true);
  }

  EXPECT_TRUE(last_result.request_passive);
  EXPECT_NE(last_result.reason.find("stale hold time exceeded limit"), std::string::npos);
  EXPECT_EQ(last_result.stale_hold_time_ms, 120);
}

TEST(MotionClipTest, RejectsEntryWhenJointErrorExceedsThreshold) {
  MotionClipConfig config;
  config.clip_entry_joint_error_rad = 0.1;
  MotionClip clip(config, MakeJointMapper(), MakeProvider());

  std::vector<double> measured_policy_positions = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                                                   70.0, 80.0, 90.0, 100.0, 110.0, 120.0};
  measured_policy_positions[4] = 50.2;  // left_hip_pitch_joint differs by 0.2 rad

  const MotionClipEntryCheckResult entry_result = clip.CheckEntry(measured_policy_positions);
  EXPECT_FALSE(entry_result.accepted);
  EXPECT_EQ(entry_result.worst_joint_name, "left_hip_pitch_joint");
  EXPECT_GT(entry_result.max_joint_error_rad, 0.1);
  EXPECT_NE(entry_result.reason.find("refusing BEYOND_MIMIC_ARMED without interpolation"),
            std::string::npos);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
