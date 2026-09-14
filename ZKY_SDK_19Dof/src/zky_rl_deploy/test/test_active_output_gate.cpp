#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/active_output_gate.hpp"

namespace zky_rl_deploy {
namespace {

DeployContext MakeActiveDeployContext() {
  DeployContext context;
  context.dry_run_only = false;
  context.shadow_mode = false;
  context.enable_motor_output = true;
  context.publish_shadow = true;
  return context;
}

StageManager::StageDefinition MakeStageDefinition(bool allow_motor_output,
                                                  const std::string& name = "stage7_suspended_beyond_mimic") {
  StageManager::StageDefinition stage;
  stage.name = name;
  stage.output_limits.allow_motor_output = allow_motor_output;
  return stage;
}

StageManager::VerificationSnapshot MakeVerificationSnapshot(bool joint_sign_verified = true) {
  StageManager::VerificationSnapshot verification;
  verification.t265_verified = true;
  verification.joint_sign_verified = joint_sign_verified;
  verification.npz_order_verified = true;
  return verification;
}

SensorSync::Evaluation MakeSensorEvaluation(bool allowed = true) {
  SensorSync::Evaluation evaluation;
  evaluation.output_gate.requested_output_mode = SensorSync::OutputMode::kActiveHardware;
  evaluation.output_gate.allowed = allowed;
  if (!allowed) {
    evaluation.output_gate.blocking_reasons.push_back(
        "T265 extrinsic is unverified; verified=false calibration can stay in dry_run/shadow but cannot enter active_hardware");
  }
  return evaluation;
}

StageManager::ActiveEligibility MakeBeyondMimicEligibility(bool allowed = true) {
  StageManager::ActiveEligibility eligibility;
  eligibility.allowed = allowed;
  eligibility.stage = MakeStageDefinition(true);
  eligibility.shadow_cycles_without_fault = allowed ? 200U : 199U;
  eligibility.required_shadow_cycles_without_fault = 200U;
  if (!allowed) {
    eligibility.blocking_reasons.push_back(
        "shadow cycles without fault are insufficient: 199 < 200");
  }
  return eligibility;
}

bool ContainsReason(const std::vector<std::string>& reasons, const std::string& needle) {
  return std::any_of(reasons.begin(), reasons.end(), [&](const std::string& reason) {
    return reason.find(needle) != std::string::npos;
  });
}

TEST(ActiveOutputGateTest, DisableFlagAlwaysBlocksRealMotorParams) {
  ActiveOutputGateContext context;
  context.deploy_context = MakeActiveDeployContext();
  context.deploy_context.enable_motor_output = false;
  context.current_stage = MakeStageDefinition(true);
  context.verification = MakeVerificationSnapshot();
  context.sensor_evaluation = MakeSensorEvaluation(true);
  context.fsm_state = FsmState::kStandHold;

  const ActiveOutputGateDecision decision = EvaluateActiveOutputGate(context);

  EXPECT_FALSE(decision.allow_active_hardware_output);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "enable_motor_output=false"));
}

TEST(ActiveOutputGateTest, UnverifiedT265AlwaysBlocksRealMotorParams) {
  ActiveOutputGateContext context;
  context.deploy_context = MakeActiveDeployContext();
  context.current_stage = MakeStageDefinition(true);
  context.verification = MakeVerificationSnapshot();
  context.sensor_evaluation = MakeSensorEvaluation(false);
  context.fsm_state = FsmState::kStandHold;

  const ActiveOutputGateDecision decision = EvaluateActiveOutputGate(context);

  EXPECT_FALSE(decision.allow_active_hardware_output);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "T265 extrinsic is unverified"));
}

TEST(ActiveOutputGateTest, JointSignVerificationBlocksRealMotorParams) {
  ActiveOutputGateContext context;
  context.deploy_context = MakeActiveDeployContext();
  context.current_stage = MakeStageDefinition(true);
  context.verification = MakeVerificationSnapshot(false);
  context.sensor_evaluation = MakeSensorEvaluation(true);
  context.fsm_state = FsmState::kStandHold;

  const ActiveOutputGateDecision decision = EvaluateActiveOutputGate(context);

  EXPECT_FALSE(decision.allow_active_hardware_output);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "joint sign verification is incomplete"));
}

TEST(ActiveOutputGateTest, StageWithoutMotorOutputBlocksRealMotorParams) {
  ActiveOutputGateContext context;
  context.deploy_context = MakeActiveDeployContext();
  context.current_stage = MakeStageDefinition(false, "stage6_shadow");
  context.verification = MakeVerificationSnapshot();
  context.sensor_evaluation = MakeSensorEvaluation(true);
  context.fsm_state = FsmState::kStandHold;

  const ActiveOutputGateDecision decision = EvaluateActiveOutputGate(context);

  EXPECT_FALSE(decision.allow_active_hardware_output);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "current stage forbids motor output"));
}

TEST(ActiveOutputGateTest, CleanBeyondMimicCaseAllowsRealMotorParams) {
  ActiveOutputGateContext context;
  context.deploy_context = MakeActiveDeployContext();
  context.current_stage = MakeStageDefinition(true);
  context.verification = MakeVerificationSnapshot();
  context.sensor_evaluation = MakeSensorEvaluation(true);
  context.fsm_state = FsmState::kBeyondMimicActive;
  context.beyond_mimic_eligibility = MakeBeyondMimicEligibility(true);

  const ActiveOutputGateDecision decision = EvaluateActiveOutputGate(context);

  EXPECT_EQ(decision.requested_output_mode, SensorSync::OutputMode::kActiveHardware);
  EXPECT_TRUE(decision.active_hardware_requested);
  EXPECT_TRUE(decision.allow_active_hardware_output);
  EXPECT_TRUE(decision.blocking_reasons.empty());
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
