#include <gtest/gtest.h>

#include <string>

#include "zky_rl_deploy/core/json_motion_frame_provider.hpp"
#include "zky_rl_deploy/core/stored_npz_archive.hpp"

namespace zky_rl_deploy {
namespace {

std::string MotionNpzPath() {
  return std::string(ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR) +
         "/../../resources/motion/zky/npz/walk1_subject1.npz";
}

std::string MotionJsonPath() {
  return std::string(ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR) +
         "/../../resources/motion/zky/json/walk1_subject1.json";
}

TEST(Stage0AssetsTest, ReadsJointShapesFromStoredNpzArchive) {
  const StoredNpzArchive archive = StoredNpzArchive::Open(MotionNpzPath());
  const NpyArrayInfo& joint_pos = archive.GetArrayInfo("joint_pos.npy");
  const NpyArrayInfo& joint_vel = archive.GetArrayInfo("joint_vel.npy");

  ASSERT_EQ(joint_pos.shape.size(), 2U);
  ASSERT_EQ(joint_vel.shape.size(), 2U);
  EXPECT_EQ(joint_pos.shape[1], 12U);
  EXPECT_EQ(joint_vel.shape[1], 12U);
  EXPECT_EQ(joint_pos.shape[0], joint_vel.shape[0]);
  EXPECT_EQ(joint_pos.dtype_descr, "<f4");
  EXPECT_FALSE(joint_pos.fortran_order);
}

TEST(Stage0AssetsTest, LoadsJsonFallbackFramesAndReferenceOrientation) {
  const JsonMotionFrameProvider provider(MotionJsonPath());
  ASSERT_GT(provider.FrameCount(), 0U);

  const MotionFrame frame = provider.GetFrame(0U);
  EXPECT_EQ(frame.joint_pos_npz_order.size(), 12U);
  EXPECT_EQ(frame.joint_vel_npz_order.size(), 12U);
  EXPECT_EQ(provider.ReferencePelvisOrientationWxyz(0U).size(), 4U);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
