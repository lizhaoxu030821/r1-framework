#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "zky_rl_deploy/core/stage_manager.hpp"

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

StageManager::ClipWindow MakeClipWindow(std::size_t start_frame, std::size_t end_frame) {
  StageManager::ClipWindow clip_window;
  clip_window.start_frame = start_frame;
  clip_window.end_frame = end_frame;
  clip_window.stride = 1U;
  return clip_window;
}

void PromoteSequentially(StageManager* manager,
                         const std::vector<std::string>& stage_names,
                         const StageManager::VerificationSnapshot& verification) {
  for (const std::string& stage_name : stage_names) {
    const StageManager::StageTransitionResult transition = manager->RequestStage(
        StageManager::StageRequest{stage_name, verification, true});
    ASSERT_TRUE(transition.accepted) << transition.Summary();
  }
}

TEST(StageManagerTest, LoadsYamlAndAllowsNormalSequentialStages) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());

  EXPECT_EQ(manager.current_stage().name, "stage0_software");
  EXPECT_DOUBLE_EQ(manager.current_stage().output_limits.action_scale, 0.0);

  const StageManager::VerificationSnapshot verified = MakeVerifiedSnapshot();
  PromoteSequentially(&manager,
                      {"stage1_ethercat_zero_output",
                       "stage2_passive_validation",
                       "stage3_single_joint"},
                      verified);

  EXPECT_EQ(manager.current_stage().name, "stage3_single_joint");
  EXPECT_TRUE(manager.current_stage().output_limits.allow_motor_output);
  EXPECT_EQ(manager.current_stage().output_limits.max_enabled_joints, 1U);
  EXPECT_DOUBLE_EQ(manager.current_stage().output_limits.kp_scale, 0.2);
  EXPECT_DOUBLE_EQ(manager.current_stage().output_limits.kd_scale, 0.2);
}

TEST(StageManagerTest, RejectsJumpStageRequest) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());

  const StageManager::StageTransitionResult transition = manager.RequestStage(
      StageManager::StageRequest{"stage3_single_joint", MakeVerifiedSnapshot(), true});

  EXPECT_FALSE(transition.accepted);
  EXPECT_EQ(manager.current_stage().name, "stage0_software");
  ASSERT_FALSE(transition.blocking_reasons.empty());
  EXPECT_NE(transition.blocking_reasons.front().find("jump stage is forbidden"), std::string::npos);
}

TEST(StageManagerTest, RejectsBeyondMimicActiveBeforeTwoHundredCleanShadowCycles) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  const StageManager::VerificationSnapshot verified = MakeVerifiedSnapshot();

  PromoteSequentially(&manager,
                      {"stage1_ethercat_zero_output",
                       "stage2_passive_validation",
                       "stage3_single_joint",
                       "stage4_suspended_single_leg_static",
                       "stage5_suspended_double_leg_static",
                       "stage6_shadow",
                       "stage7_suspended_beyond_mimic"},
                      verified);

  const StageManager::ClipWindow clip_window = MakeClipWindow(100U, 160U);
  for (std::size_t cycle = 0U; cycle < 199U; ++cycle) {
    const StageManager::ShadowCycleResult shadow_cycle =
        manager.ReportShadowCycle(StageManager::ShadowCycleInput{clip_window, false});
    EXPECT_EQ(shadow_cycle.shadow_cycles_without_fault, cycle + 1U);
  }

  const StageManager::ActiveEligibility eligibility =
      manager.EvaluateBeyondMimicActiveEligibility(StageManager::ActiveRequest{verified});

  EXPECT_FALSE(eligibility.allowed);
  EXPECT_EQ(eligibility.required_shadow_cycles_without_fault, 200U);
  EXPECT_EQ(eligibility.shadow_cycles_without_fault, 199U);
}

TEST(StageManagerTest, OutputLevelChangeResetsShadowCounter) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  const StageManager::VerificationSnapshot verified = MakeVerifiedSnapshot();

  PromoteSequentially(&manager,
                      {"stage1_ethercat_zero_output",
                       "stage2_passive_validation",
                       "stage3_single_joint",
                       "stage4_suspended_single_leg_static",
                       "stage5_suspended_double_leg_static",
                       "stage6_shadow"},
                      verified);

  const StageManager::ClipWindow clip_window = MakeClipWindow(100U, 160U);
  EXPECT_EQ(manager.ReportShadowCycle(StageManager::ShadowCycleInput{clip_window, false})
                .shadow_cycles_without_fault,
            1U);
  EXPECT_EQ(manager.ReportShadowCycle(StageManager::ShadowCycleInput{clip_window, false})
                .shadow_cycles_without_fault,
            2U);

  const StageManager::StageTransitionResult transition = manager.RequestStage(
      StageManager::StageRequest{"stage7_suspended_beyond_mimic", verified, true});

  EXPECT_TRUE(transition.accepted);
  EXPECT_TRUE(transition.shadow_counter_reset);
  EXPECT_EQ(transition.shadow_cycles_without_fault, 0U);
  EXPECT_NE(transition.shadow_counter_reset_reason.find("stage output level changed"),
            std::string::npos);
}

TEST(StageManagerTest, ShadowFaultResetsCounterToZero) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  const StageManager::VerificationSnapshot verified = MakeVerifiedSnapshot();

  PromoteSequentially(&manager,
                      {"stage1_ethercat_zero_output",
                       "stage2_passive_validation",
                       "stage3_single_joint",
                       "stage4_suspended_single_leg_static",
                       "stage5_suspended_double_leg_static",
                       "stage6_shadow"},
                      verified);

  const StageManager::ClipWindow clip_window = MakeClipWindow(100U, 160U);
  EXPECT_EQ(manager.ReportShadowCycle(StageManager::ShadowCycleInput{clip_window, false})
                .shadow_cycles_without_fault,
            1U);
  EXPECT_EQ(manager.ReportShadowCycle(StageManager::ShadowCycleInput{clip_window, false})
                .shadow_cycles_without_fault,
            2U);

  const StageManager::ShadowCycleResult fault_result =
      manager.ReportShadowCycle(StageManager::ShadowCycleInput{clip_window, true});

  EXPECT_TRUE(fault_result.counter_reset);
  EXPECT_EQ(fault_result.shadow_cycles_without_fault, 0U);
  EXPECT_NE(fault_result.reset_reason.find("shadow fault detected"), std::string::npos);
}

TEST(StageManagerTest, ChangingClipWindowResetsShadowCounter) {
  StageManager manager = StageManager::LoadFromYaml(BringupStagesPath());
  const StageManager::VerificationSnapshot verified = MakeVerifiedSnapshot();

  PromoteSequentially(&manager,
                      {"stage1_ethercat_zero_output",
                       "stage2_passive_validation",
                       "stage3_single_joint",
                       "stage4_suspended_single_leg_static",
                       "stage5_suspended_double_leg_static",
                       "stage6_shadow"},
                      verified);

  EXPECT_EQ(manager.ReportShadowCycle(
                StageManager::ShadowCycleInput{MakeClipWindow(100U, 160U), false})
                .shadow_cycles_without_fault,
            1U);
  EXPECT_EQ(manager.ReportShadowCycle(
                StageManager::ShadowCycleInput{MakeClipWindow(100U, 160U), false})
                .shadow_cycles_without_fault,
            2U);

  const StageManager::ShadowCycleResult clip_change_result =
      manager.ReportShadowCycle(
          StageManager::ShadowCycleInput{MakeClipWindow(120U, 180U), false});

  EXPECT_TRUE(clip_change_result.counter_reset);
  EXPECT_EQ(clip_change_result.shadow_cycles_without_fault, 0U);
  EXPECT_NE(clip_change_result.reset_reason.find("clip window changed"), std::string::npos);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
