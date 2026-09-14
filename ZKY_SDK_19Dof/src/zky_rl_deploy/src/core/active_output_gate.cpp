#include "zky_rl_deploy/core/active_output_gate.hpp"

#include <algorithm>
#include <sstream>

namespace zky_rl_deploy {
namespace {

void AppendUniqueReason(std::vector<std::string>* reasons, const std::string& reason) {
  if (std::find(reasons->begin(), reasons->end(), reason) == reasons->end()) {
    reasons->push_back(reason);
  }
}

std::string JoinReasons(const std::vector<std::string>& reasons) {
  if (reasons.empty()) {
    return "none";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < reasons.size(); ++index) {
    if (index != 0U) {
      stream << "; ";
    }
    stream << reasons[index];
  }
  return stream.str();
}

}  // namespace

SensorSync::OutputMode DetermineRequestedOutputMode(const DeployContext& deploy_context) {
  if (!deploy_context.dry_run_only && !deploy_context.shadow_mode &&
      deploy_context.enable_motor_output) {
    return SensorSync::OutputMode::kActiveHardware;
  }
  if (deploy_context.shadow_mode) {
    return SensorSync::OutputMode::kShadow;
  }
  return SensorSync::OutputMode::kDryRun;
}

bool IsBeyondMimicState(FsmState fsm_state) {
  return fsm_state == FsmState::kBeyondMimicArmed ||
         fsm_state == FsmState::kBeyondMimicActive;
}

ActiveOutputGateDecision EvaluateActiveOutputGate(const ActiveOutputGateContext& context) {
  ActiveOutputGateDecision decision;
  decision.requested_output_mode = DetermineRequestedOutputMode(context.deploy_context);
  decision.active_hardware_requested =
      decision.requested_output_mode == SensorSync::OutputMode::kActiveHardware;

  if (!context.deploy_context.enable_motor_output) {
    AppendUniqueReason(&decision.blocking_reasons,
                       "enable_motor_output=false keeps /motor_params publishing disabled");
  }
  if (context.deploy_context.dry_run_only) {
    AppendUniqueReason(&decision.blocking_reasons,
                       "dry_run_only=true keeps the deploy loop out of active hardware mode");
  }
  if (context.deploy_context.shadow_mode) {
    AppendUniqueReason(&decision.blocking_reasons,
                       "shadow_mode=true keeps commands on the shadow route");
  }
  if (!context.current_stage.output_limits.allow_motor_output) {
    AppendUniqueReason(
        &decision.blocking_reasons,
        "current stage forbids motor output: " + context.current_stage.name);
  }
  if (!context.verification.joint_sign_verified) {
    AppendUniqueReason(
        &decision.blocking_reasons,
        "joint sign verification is incomplete; real /motor_params must stay disabled");
  }

  if (decision.active_hardware_requested && !context.sensor_evaluation.output_gate.allowed) {
    if (context.sensor_evaluation.output_gate.blocking_reasons.empty()) {
      AppendUniqueReason(
          &decision.blocking_reasons,
          "SensorSync blocked active hardware output without reporting a detailed reason");
    } else {
      for (const std::string& reason : context.sensor_evaluation.output_gate.blocking_reasons) {
        AppendUniqueReason(&decision.blocking_reasons, reason);
      }
    }
  }

  if (IsBeyondMimicState(context.fsm_state) && !context.beyond_mimic_eligibility.allowed) {
    for (const std::string& reason : context.beyond_mimic_eligibility.blocking_reasons) {
      AppendUniqueReason(&decision.blocking_reasons, reason);
    }
  }

  decision.allow_active_hardware_output =
      decision.active_hardware_requested && decision.blocking_reasons.empty();
  return decision;
}

std::string ActiveOutputGateDecision::Summary() const {
  std::ostringstream stream;
  stream << "requested_output_mode=" << SensorSync::ToString(requested_output_mode)
         << ", active_hardware_requested="
         << (active_hardware_requested ? "true" : "false")
         << ", allow_active_hardware_output="
         << (allow_active_hardware_output ? "true" : "false")
         << ", blocking_reasons=" << JoinReasons(blocking_reasons);
  return stream.str();
}

}  // namespace zky_rl_deploy
