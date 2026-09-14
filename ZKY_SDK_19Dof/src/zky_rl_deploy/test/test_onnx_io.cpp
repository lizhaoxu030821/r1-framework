#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "zky_rl_deploy/core/policy_runtime.hpp"

namespace zky_rl_deploy {
namespace {

PolicyRuntime::RawMetadataMap MakeReferenceRawMetadata() {
  return {
      {"joint_names",
       "left_hip_yaw_joint,right_hip_yaw_joint,left_hip_roll_joint,right_hip_roll_joint,"
       "left_hip_pitch_joint,right_hip_pitch_joint,left_knee_joint,right_knee_joint,"
       "left_ankle_pitch_joint,right_ankle_pitch_joint,left_ankle_roll_joint,right_ankle_roll_joint"},
      {"action_scale", "[0.038,0.038,0.042,0.042,0.083,0.083,0.083,0.083,0.025,0.025,0.025,0.025]"},
      {"default_joint_pos", "[0,0,0,0,-0.2,0.2,0.5,-0.5,0.3,-0.3,0,0]"},
      {"joint_stiffness", "[200,200,300,300,300,300,300,300,200,200,200,200]"},
      {"joint_damping", "[0.037,0.037,0.123,0.123,0.120,0.120,0.120,0.120,0.031,0.031,0.031,0.031]"}};
}

PolicyMetadataAuditCopy MakeMatchingAuditCopy() {
  PolicyMetadataAuditCopy audit_copy;
  audit_copy.joint_names = std::vector<std::string>{
      "left_hip_yaw_joint",   "right_hip_yaw_joint",   "left_hip_roll_joint",
      "right_hip_roll_joint", "left_hip_pitch_joint", "right_hip_pitch_joint",
      "left_knee_joint",      "right_knee_joint",      "left_ankle_pitch_joint",
      "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"};
  audit_copy.action_scale =
      std::vector<double>{0.038, 0.038, 0.042, 0.042, 0.083, 0.083,
                          0.083, 0.083, 0.025, 0.025, 0.025, 0.025};
  audit_copy.default_joint_pos =
      std::vector<double>{0.0, 0.0, 0.0, 0.0, -0.2, 0.2, 0.5, -0.5, 0.3, -0.3, 0.0, 0.0};
  audit_copy.joint_stiffness =
      std::vector<double>{200.0, 200.0, 300.0, 300.0, 300.0, 300.0,
                          300.0, 300.0, 200.0, 200.0, 200.0, 200.0};
  audit_copy.joint_damping =
      std::vector<double>{0.037, 0.037, 0.123, 0.123, 0.120, 0.120,
                          0.120, 0.120, 0.031, 0.031, 0.031, 0.031};
  return audit_copy;
}

TEST(PolicyRuntimeTest, ParsesAuthoritativeMetadataFromRawMap) {
  const PolicyMetadata metadata = PolicyRuntime::BuildAuthoritativeMetadataFromRawMap(
      "ref/policy_walk1subject1_260327.onnx", MakeReferenceRawMetadata());

  EXPECT_EQ(metadata.onnx_path, "ref/policy_walk1subject1_260327.onnx");
  EXPECT_EQ(metadata.joint_names.size(), PolicyRuntime::kExpectedJointCount);
  EXPECT_EQ(metadata.action_scale.size(), PolicyRuntime::kExpectedJointCount);
  EXPECT_EQ(metadata.default_joint_pos.size(), PolicyRuntime::kExpectedJointCount);
  EXPECT_EQ(metadata.joint_stiffness.size(), PolicyRuntime::kExpectedJointCount);
  EXPECT_EQ(metadata.joint_damping.size(), PolicyRuntime::kExpectedJointCount);
  EXPECT_EQ(metadata.joint_names.front(), "left_hip_yaw_joint");
  EXPECT_DOUBLE_EQ(metadata.default_joint_pos[4], -0.2);
}

TEST(PolicyRuntimeTest, RejectsWrongMetadataDimension) {
  PolicyRuntime::RawMetadataMap raw_metadata = MakeReferenceRawMetadata();
  raw_metadata["action_scale"] = "[0.038,0.038,0.042,0.042,0.083,0.083,0.083,0.083,0.025,0.025,0.025]";

  EXPECT_THROW(
      PolicyRuntime::BuildAuthoritativeMetadataFromRawMap(
          "ref/policy_walk1subject1_260327.onnx", raw_metadata),
      std::invalid_argument);
}

TEST(PolicyRuntimeTest, RejectsYamlAuditMismatch) {
  const PolicyMetadata metadata = PolicyRuntime::BuildAuthoritativeMetadataFromRawMap(
      "ref/policy_walk1subject1_260327.onnx", MakeReferenceRawMetadata());
  PolicyMetadataAuditCopy audit_copy = MakeMatchingAuditCopy();
  (*audit_copy.default_joint_pos)[4] = -0.25;

  EXPECT_THROW(PolicyRuntime::ValidateAuditCopyMatches(metadata, audit_copy),
               std::invalid_argument);
}

TEST(PolicyRuntimeTest, AcceptsMatchingYamlAuditCopy) {
  const PolicyMetadata metadata = PolicyRuntime::BuildAuthoritativeMetadataFromRawMap(
      "ref/policy_walk1subject1_260327.onnx", MakeReferenceRawMetadata());

  EXPECT_NO_THROW(PolicyRuntime::ValidateAuditCopyMatches(metadata, MakeMatchingAuditCopy()));
}

TEST(PolicyRuntimeTest, LoadsReferenceOnnxMetadataWhenRuntimeIsAvailable) {
  if (!PolicyRuntime::OnnxRuntimeEnabledAtBuild()) {
    GTEST_SKIP() << PolicyRuntime::OnnxRuntimeBuildSummary();
  }

  const std::filesystem::path onnx_path =
      std::filesystem::path(ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR) / ".." / ".." / "ref" /
      "policy_walk1subject1_260327.onnx";

  PolicyRuntimeConfig config;
  config.onnx_path = std::filesystem::weakly_canonical(onnx_path).string();
  config.metadata_audit_copy = MakeMatchingAuditCopy();

  PolicyRuntime runtime;
  const PolicyMetadata metadata = runtime.LoadMetadata(config);

  EXPECT_EQ(metadata.joint_names.size(), PolicyRuntime::kExpectedJointCount);
  EXPECT_EQ(metadata.joint_names.front(), "left_hip_yaw_joint");
  EXPECT_EQ(metadata.joint_names.back(), "right_ankle_roll_joint");
  EXPECT_DOUBLE_EQ(metadata.action_scale[0], 0.038);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
