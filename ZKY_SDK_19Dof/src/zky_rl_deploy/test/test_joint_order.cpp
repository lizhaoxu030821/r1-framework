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

TEST(JointMapperTest, ReordersPolicyOrderIntoHardwareOrderAndBack) {
  // 该测试覆盖 baseline 中最关键的左右腿重排：
  // 策略顺序左右腿交错，而硬件顺序按整条左腿后整条右腿排列，不能靠数组位置默认相等。
  const JointOrder policy_order = MakePolicyOrder();
  const JointOrder hardware_order = MakeHardwareOrder();
  JointMapper mapper(
      policy_order, hardware_order, policy_order, MakeAllPositiveSigns(policy_order),
      MakeAllVerified(policy_order));

  const JointMapper::JointValues policy_values = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,
                                                  7.0,  8.0,  9.0,  10.0, 11.0, 12.0};
  const JointMapper::JointValues expected_hardware_values = {1.0,  3.0,  5.0,  7.0,
                                                             9.0,  11.0, 2.0,  4.0,
                                                             6.0,  8.0,  10.0, 12.0};

  const JointMapper::JointValues hardware_values = mapper.PolicyToHardwareCommand(policy_values);
  EXPECT_EQ(hardware_values, expected_hardware_values);

  const JointMapper::JointValues policy_roundtrip =
      mapper.HardwareToPolicyFeedback(hardware_values);
  EXPECT_EQ(policy_roundtrip, policy_values);
}

TEST(JointMapperTest, RejectsMissingJointAtConstruction) {
  JointOrder hardware_order = MakeHardwareOrder();
  hardware_order.pop_back();

  EXPECT_THROW(
      JointMapper(MakePolicyOrder(), hardware_order, MakePolicyOrder(),
                  MakeAllPositiveSigns(MakePolicyOrder()), MakeAllVerified(MakePolicyOrder())),
      std::invalid_argument);
}

TEST(JointMapperTest, RejectsDuplicatedJointAtConstruction) {
  JointOrder duplicated_policy_order = MakePolicyOrder();
  duplicated_policy_order[1] = duplicated_policy_order[0];

  EXPECT_THROW(
      JointMapper(duplicated_policy_order, MakeHardwareOrder(), MakePolicyOrder(),
                  MakeAllPositiveSigns(MakePolicyOrder()), MakeAllVerified(MakePolicyOrder())),
      std::invalid_argument);
}

TEST(JointMapperTest, RejectsWrongVectorLengthAtRuntime) {
  const JointOrder policy_order = MakePolicyOrder();
  JointMapper mapper(
      policy_order, MakeHardwareOrder(), policy_order, MakeAllPositiveSigns(policy_order),
      MakeAllVerified(policy_order));

  EXPECT_THROW(
      mapper.PolicyToHardwareCommand({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}),
      std::invalid_argument);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
