#include <gtest/gtest.h>

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

JointOrder MakeCustomNpzOrder() {
  // 这里故意让 NPZ 顺序不同于策略顺序，验证 motion reference 的确是按名字重排，
  // 而不是“因为当前样例碰巧一样”才通过。
  return {"left_knee_joint",       "right_knee_joint",       "left_hip_yaw_joint",
          "right_hip_yaw_joint",   "left_hip_roll_joint",    "right_hip_roll_joint",
          "left_hip_pitch_joint",  "right_hip_pitch_joint",  "left_ankle_pitch_joint",
          "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"};
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

TEST(JointMapperTest, ReordersNpzReferenceIntoPolicyOrder) {
  const JointOrder policy_order = MakePolicyOrder();
  JointMapper mapper(
      policy_order, MakeHardwareOrder(), MakeCustomNpzOrder(),
      MakeAllPositiveSigns(policy_order), MakeAllVerified(policy_order));

  const JointMapper::JointValues npz_values = {70.0, 80.0, 10.0, 20.0, 30.0, 40.0,
                                               50.0, 60.0, 90.0, 100.0, 110.0, 120.0};
  const JointMapper::JointValues expected_policy_values = {10.0, 20.0, 30.0, 40.0,
                                                           50.0, 60.0, 70.0, 80.0,
                                                           90.0, 100.0, 110.0, 120.0};

  EXPECT_EQ(mapper.NpzToPolicyReference(npz_values), expected_policy_values);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
