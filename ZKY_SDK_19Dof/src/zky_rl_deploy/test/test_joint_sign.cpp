#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "zky_rl_deploy/core/joint_mapper.hpp"

namespace zky_rl_deploy {
namespace {

using JointOrder = JointMapper::JointOrder;
using SignConvention = JointMapper::SignConvention;
using SignVerification = JointMapper::SignVerification;

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

SignConvention MakeSignsWithNegativeRightKnee(const JointOrder& policy_order) {
  SignConvention signs;
  for (const std::string& joint_name : policy_order) {
    signs.emplace(joint_name, 1);
  }
  signs["right_knee_joint"] = -1;
  return signs;
}

SignVerification MakeVerificationWithUnverifiedRightKnee(const JointOrder& policy_order) {
  SignVerification verified;
  for (const std::string& joint_name : policy_order) {
    verified.emplace(joint_name, true);
  }
  verified["right_knee_joint"] = false;
  return verified;
}

TEST(JointMapperTest, AppliesNegativeSignToCommandAndFeedback) {
  // sign_convention 的职责是把“策略正方向”与“硬件真实正方向”对齐：
  // 若某关节硬件正向与训练坐标相反，命令和反馈两条路径都必须同时翻转。
  const JointOrder policy_order = MakePolicyOrder();
  JointMapper mapper(policy_order,
                     MakeHardwareOrder(),
                     policy_order,
                     MakeSignsWithNegativeRightKnee(policy_order),
                     MakeVerificationWithUnverifiedRightKnee(policy_order));

  JointMapper::JointValues policy_command(policy_order.size(), 0.0);
  policy_command[7] = 0.5;  // right_knee_joint in policy_order

  const JointMapper::JointValues hardware_command = mapper.PolicyToHardwareCommand(policy_command);
  EXPECT_DOUBLE_EQ(hardware_command[9], -0.5);  // right_knee_joint in hardware_order

  JointMapper::JointValues hardware_feedback(policy_order.size(), 0.0);
  hardware_feedback[9] = -0.25;

  const JointMapper::JointValues policy_feedback =
      mapper.HardwareToPolicyFeedback(hardware_feedback);
  EXPECT_DOUBLE_EQ(policy_feedback[7], 0.25);
}

TEST(JointMapperTest, RefusesActiveFunctionWhenAnySignIsUnverified) {
  const JointOrder policy_order = MakePolicyOrder();
  JointMapper mapper(policy_order,
                     MakeHardwareOrder(),
                     policy_order,
                     MakeSignsWithNegativeRightKnee(policy_order),
                     MakeVerificationWithUnverifiedRightKnee(policy_order));

  EXPECT_FALSE(mapper.AllSignsVerifiedForActiveFunction());
  EXPECT_EQ(mapper.GetUnverifiedSignJoints(), std::vector<std::string>({"right_knee_joint"}));
  EXPECT_THROW(mapper.ThrowIfSignsUnverifiedForActiveFunction(), std::logic_error);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
