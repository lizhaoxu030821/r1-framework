#include "zky_rl_deploy/core/fsm.hpp"

#include <sstream>

namespace zky_rl_deploy {
namespace {

std::string JoinWarnings(const std::vector<std::string>& warnings) {
  if (warnings.empty()) {
    return "none";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < warnings.size(); ++index) {
    if (index != 0U) {
      stream << "; ";
    }
    stream << warnings[index];
  }
  return stream.str();
}

}  // namespace

Fsm::Fsm(FsmState initial_state) : state_(initial_state) {}

FsmTransitionResult Fsm::HandleJoystickEvents(const JoystickEvents& events,
                                              const FsmUpdateContext& context) {
  if (events.request_zero_output_estop) {
    return BuildTransitionResult(FsmState::kZeroOutputEstop, events, "home_estop");
  }

  if (state_ == FsmState::kZeroOutputEstop) {
    if (events.request_passive) {
      return BuildTransitionResult(
          FsmState::kPrecharge, events, "start_enable_recover_from_estop");
    }
    return BuildNoTransitionResult(events, "zero_output_estop_latched");
  }

  switch (state_) {
    case FsmState::kDisabled:
      if (events.request_passive) {
        return BuildTransitionResult(FsmState::kPassive, events, "start_enable_to_passive");
      }
      if (events.request_stand_init) {
        return BuildTransitionResult(
            FsmState::kStandInit, events, "back_reset_to_initial_position");
      }
      return BuildNoTransitionResult(events, "disabled_waiting");

    case FsmState::kPrecharge:
      if (events.request_passive) {
        if (context.precharge_checks_complete) {
          return BuildTransitionResult(
              FsmState::kPassive, events, "precharge_start_enable_to_passive");
        }
        return BuildNoTransitionResult(events, "precharge_checks_not_complete");
      }
      return BuildNoTransitionResult(events, "precharge_accepts_only_start_or_home");

    case FsmState::kPassive:
      if (events.request_passive) {
        return BuildNoTransitionResult(events, "already_passive");
      }
      if (events.request_stand_init) {
        return BuildTransitionResult(
            FsmState::kStandInit, events, "back_reset_to_initial_position");
      }
      return BuildNoTransitionResult(events, "passive_waiting");

    case FsmState::kStandInit:
      if (events.request_passive) {
        return BuildTransitionResult(FsmState::kPassive, events, "start_enable_to_passive");
      }
      if (context.stand_init_complete) {
        return BuildTransitionResult(FsmState::kStandHold, events, "stand_init_complete");
      }
      return BuildNoTransitionResult(events, "stand_init_running");

    case FsmState::kStandHold:
      if (events.request_passive) {
        return BuildTransitionResult(FsmState::kPassive, events, "start_enable_to_passive");
      }
      if (events.request_stand_init) {
        return BuildTransitionResult(
            FsmState::kStandInit, events, "back_reset_to_initial_position");
      }
      if (!events.function_request.has_value()) {
        return BuildNoTransitionResult(events, "stand_hold_waiting");
      }

      if (*events.function_request != FunctionRequest::kBeyondMimic) {
        FsmTransitionResult result =
            BuildNoTransitionResult(events, "reserved_function_request_warning_only");
        result.warnings.push_back(
            std::string("function request is not implemented: ") +
            JoystickMapper::ToString(*events.function_request));
        return result;
      }

      return BuildTransitionResult(FsmState::kBeyondMimicArmed, events, "enter_beyond_mimic_armed");

    case FsmState::kBeyondMimicArmed:
      if (events.request_passive) {
        return BuildTransitionResult(FsmState::kPassive, events, "start_enable_to_passive");
      }
      if (context.beyond_mimic_ready_for_active) {
        return BuildTransitionResult(
            FsmState::kBeyondMimicActive, events, "beyond_mimic_ready_for_active");
      }
      return BuildNoTransitionResult(events, "beyond_mimic_armed_shadow_checks");

    case FsmState::kBeyondMimicActive:
      if (events.request_passive) {
        return BuildTransitionResult(FsmState::kPassive, events, "start_enable_to_passive");
      }
      return BuildNoTransitionResult(events, "beyond_mimic_active_running");

    case FsmState::kZeroOutputEstop:
      break;
  }

  return BuildNoTransitionResult(events, "unhandled_state");
}

const char* Fsm::ToString(FsmState state) {
  switch (state) {
    case FsmState::kZeroOutputEstop:
      return "ZERO_OUTPUT_ESTOP";
    case FsmState::kDisabled:
      return "DISABLED";
    case FsmState::kPrecharge:
      return "PRECHARGE";
    case FsmState::kPassive:
      return "PASSIVE";
    case FsmState::kStandInit:
      return "STAND_INIT";
    case FsmState::kStandHold:
      return "STAND_HOLD";
    case FsmState::kBeyondMimicArmed:
      return "BEYOND_MIMIC_ARMED";
    case FsmState::kBeyondMimicActive:
      return "BEYOND_MIMIC_ACTIVE";
  }

  return "UNKNOWN";
}

std::string FsmTransitionResult::Summary() const {
  std::ostringstream stream;
  stream << "previous=" << Fsm::ToString(previous_state)
         << ", current=" << Fsm::ToString(current_state)
         << ", changed=" << (state_changed ? "true" : "false")
         << ", reason=" << reason
         << ", warnings=" << JoinWarnings(warnings);
  return stream.str();
}

FsmTransitionResult Fsm::BuildNoTransitionResult(const JoystickEvents& events,
                                                 const char* reason) const {
  FsmTransitionResult result;
  result.previous_state = state_;
  result.current_state = state_;
  result.state_changed = false;
  result.reason = reason;
  result.warnings = events.warnings;
  return result;
}

FsmTransitionResult Fsm::BuildTransitionResult(FsmState next_state,
                                               const JoystickEvents& events,
                                               const char* reason) {
  FsmTransitionResult result;
  result.previous_state = state_;
  result.current_state = next_state;
  result.state_changed = (state_ != next_state);
  result.reason = reason;
  result.warnings = events.warnings;
  state_ = next_state;

  return result;
}

}  // namespace zky_rl_deploy
