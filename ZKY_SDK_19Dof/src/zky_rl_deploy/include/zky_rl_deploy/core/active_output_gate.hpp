#pragma once

#include <string>
#include <vector>

#include "zky_rl_deploy/core/deploy_context.hpp"
#include "zky_rl_deploy/core/fsm.hpp"
#include "zky_rl_deploy/core/sensor_sync.hpp"
#include "zky_rl_deploy/core/stage_manager.hpp"

namespace zky_rl_deploy {

struct ActiveOutputGateContext {
  DeployContext deploy_context;
  StageManager::StageDefinition current_stage;
  StageManager::VerificationSnapshot verification;
  SensorSync::Evaluation sensor_evaluation;
  FsmState fsm_state{FsmState::kDisabled};
  StageManager::ActiveEligibility beyond_mimic_eligibility;
};

struct ActiveOutputGateDecision {
  SensorSync::OutputMode requested_output_mode{SensorSync::OutputMode::kDryRun};
  bool active_hardware_requested{false};
  bool allow_active_hardware_output{false};
  std::vector<std::string> blocking_reasons;

  std::string Summary() const;
};

SensorSync::OutputMode DetermineRequestedOutputMode(const DeployContext& deploy_context);
bool IsBeyondMimicState(FsmState fsm_state);
ActiveOutputGateDecision EvaluateActiveOutputGate(const ActiveOutputGateContext& context);

}  // namespace zky_rl_deploy
