#pragma once

#include <string>
#include <vector>

#include "zky_rl_deploy/core/joystick_mapper.hpp"

namespace zky_rl_deploy {

enum class FsmState {
  kZeroOutputEstop,
  kDisabled,
  kPrecharge,
  kPassive,
  kStandInit,
  kStandHold,
  kBeyondMimicArmed,
  kBeyondMimicActive,
};

struct FsmUpdateContext {
  bool precharge_checks_complete{false};
  bool stand_init_complete{false};
  bool beyond_mimic_ready_for_active{false};
};

struct FsmTransitionResult {
  FsmState previous_state{FsmState::kDisabled};
  FsmState current_state{FsmState::kDisabled};
  bool state_changed{false};
  std::string reason;
  std::vector<std::string> warnings;

  std::string Summary() const;
};

class Fsm {
 public:
  explicit Fsm(FsmState initial_state = FsmState::kDisabled);

  FsmTransitionResult HandleJoystickEvents(const JoystickEvents& events,
                                           const FsmUpdateContext& context = {});

  FsmState state() const { return state_; }

  static const char* ToString(FsmState state);

 private:
  FsmTransitionResult BuildNoTransitionResult(const JoystickEvents& events,
                                              const char* reason) const;
  FsmTransitionResult BuildTransitionResult(FsmState next_state,
                                            const JoystickEvents& events,
                                            const char* reason);

  FsmState state_;
};

}  // namespace zky_rl_deploy
