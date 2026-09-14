#include <gtest/gtest.h>

#include <string>

#include "zky_rl_deploy/core/stage_request_resolver.hpp"

namespace zky_rl_deploy {
namespace {

std::string BringupStagesPath() {
  return std::string(ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR) + "/config/bringup_stages.yaml";
}

StageManager::VerificationSnapshot MakeVerifiedSnapshot() {
  StageManager::VerificationSnapshot verification;
  verification.t265_verified = true;
  verification.joint_sign_verified = true;
  verification.npz_order_verified = true;
  return verification;
}

void Promote(StageManager* manager,
             const std::string& stage_name,
             bool safe_state = true) {
  const StageManager::StageTransitionResult transition =
      manager->RequestStage({stage_name, MakeVerifiedSnapshot(), safe_state});
  ASSERT_TRUE(transition.accepted) << transition.Summary();
}

TEST(StageRequestResolverTest, ExpandsLongJumpToImmediateNextStage) {
  const StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());

  EXPECT_EQ(ResolveNextRequestedStageName(manager, "stage4_suspended_single_leg_static"),
            "stage1_ethercat_zero_output");
}

TEST(StageRequestResolverTest, AdvancesOneStageAtATimeAfterProgress) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  Promote(&manager, "stage1_ethercat_zero_output");

  EXPECT_EQ(ResolveNextRequestedStageName(manager, "stage4_suspended_single_leg_static"),
            "stage2_passive_validation");
}

TEST(StageRequestResolverTest, ReturnsDesiredStageWhenAlreadyAdjacent) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  Promote(&manager, "stage1_ethercat_zero_output");
  Promote(&manager, "stage2_passive_validation");

  EXPECT_EQ(ResolveNextRequestedStageName(manager, "stage3_single_joint"),
            "stage3_single_joint");
}

TEST(StageRequestResolverTest, ReturnsDesiredStageWhenAlreadyThere) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  Promote(&manager, "stage1_ethercat_zero_output");
  Promote(&manager, "stage2_passive_validation");
  Promote(&manager, "stage3_single_joint");
  Promote(&manager, "stage4_suspended_single_leg_static");

  EXPECT_EQ(ResolveNextRequestedStageName(manager, "stage4_suspended_single_leg_static"),
            "stage4_suspended_single_leg_static");
}

TEST(StageRequestResolverTest, LeavesUnknownStageUntouched) {
  const StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());

  EXPECT_EQ(ResolveNextRequestedStageName(manager, "stage42_unknown"),
            "stage42_unknown");
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
