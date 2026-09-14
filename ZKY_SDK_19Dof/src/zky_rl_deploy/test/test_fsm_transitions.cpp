#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <vector>

#include "zky_rl_deploy/core/fsm.hpp"
#include "zky_rl_deploy/core/joystick_mapper.hpp"

namespace zky_rl_deploy {
namespace {

using Milliseconds = std::chrono::milliseconds;

JoystickEvents UpdateMapper(JoystickMapper* mapper,
                            const JoystickMapper::TimePoint& now,
                            const JoystickState& state,
                            const Fsm& fsm) {
  return mapper->Update(now, state, fsm.state() == FsmState::kZeroOutputEstop);
}

TEST(FsmTest, HomeForcesZeroOutputEstopFromAnyState) {
  const std::array<FsmState, 8> all_states = {
      FsmState::kZeroOutputEstop, FsmState::kDisabled,    FsmState::kPrecharge,
      FsmState::kPassive,         FsmState::kStandInit,   FsmState::kStandHold,
      FsmState::kBeyondMimicArmed, FsmState::kBeyondMimicActive};

  for (FsmState initial_state : all_states) {
    JoystickMapper mapper;
    Fsm fsm(initial_state);

    JoystickState state;
    state.home_pressed = true;

    const JoystickEvents events =
        UpdateMapper(&mapper, JoystickMapper::TimePoint(Milliseconds(0)), state, fsm);
    const FsmTransitionResult result = fsm.HandleJoystickEvents(events);

    EXPECT_EQ(fsm.state(), FsmState::kZeroOutputEstop);
    EXPECT_EQ(result.current_state, FsmState::kZeroOutputEstop);
  }
}

TEST(FsmTest, BackIsIgnoredInsideZeroOutputEstopButStartRecoversToPrecharge) {
  JoystickMapper mapper;
  Fsm fsm(FsmState::kZeroOutputEstop);

  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(0)), JoystickState{}, fsm);

  JoystickState start_only;
  start_only.start_pressed = true;
  JoystickEvents events =
      UpdateMapper(&mapper, JoystickMapper::TimePoint(Milliseconds(10)), start_only, fsm);
  EXPECT_TRUE(events.request_passive);
  EXPECT_FALSE(events.request_stand_init);
  EXPECT_EQ(fsm.HandleJoystickEvents(events).current_state, FsmState::kPrecharge);

  fsm = Fsm(FsmState::kZeroOutputEstop);
  mapper.Reset();
  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(0)), JoystickState{}, fsm);

  JoystickState released;
  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(20)), released, fsm);

  JoystickState back_only;
  back_only.back_pressed = true;
  events = UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(30)), back_only, fsm);
  EXPECT_FALSE(events.request_passive);
  EXPECT_FALSE(events.request_stand_init);
  EXPECT_EQ(fsm.HandleJoystickEvents(events).current_state, FsmState::kZeroOutputEstop);
}

TEST(FsmTest, PrechargeCannotJumpDirectlyToStandInit) {
  JoystickMapper mapper;
  Fsm fsm(FsmState::kPrecharge);

  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(0)), JoystickState{}, fsm);

  JoystickState back_only;
  back_only.back_pressed = true;
  const JoystickEvents events =
      UpdateMapper(&mapper, JoystickMapper::TimePoint(Milliseconds(10)), back_only, fsm);
  EXPECT_TRUE(events.request_stand_init);

  FsmUpdateContext context;
  context.precharge_checks_complete = true;
  const FsmTransitionResult result = fsm.HandleJoystickEvents(events, context);
  EXPECT_EQ(result.current_state, FsmState::kPrecharge);
  EXPECT_EQ(fsm.state(), FsmState::kPrecharge);
}

TEST(FsmTest, AWithoutRbDoesNotRequestFunctionSwitch) {
  JoystickMapper mapper;
  Fsm fsm(FsmState::kStandHold);

  JoystickState neutral;
  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(0)), neutral, fsm);

  JoystickState a_only;
  a_only.a_pressed = true;
  const JoystickEvents events =
      UpdateMapper(&mapper, JoystickMapper::TimePoint(Milliseconds(10)), a_only, fsm);
  EXPECT_FALSE(events.function_request.has_value());
}

TEST(FsmTest, RbPlusARequestsBeyondMimicFunctionSwitch) {
  JoystickMapper mapper;
  Fsm fsm(FsmState::kStandHold);

  JoystickState neutral;
  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(0)), neutral, fsm);

  JoystickState guarded_combo;
  guarded_combo.rb_pressed = true;
  guarded_combo.a_pressed = true;
  const JoystickEvents events =
      UpdateMapper(&mapper, JoystickMapper::TimePoint(Milliseconds(10)), guarded_combo, fsm);
  ASSERT_TRUE(events.function_request.has_value());
  EXPECT_EQ(*events.function_request, FunctionRequest::kBeyondMimic);

  const FsmTransitionResult result = fsm.HandleJoystickEvents(events);
  EXPECT_EQ(result.current_state, FsmState::kBeyondMimicArmed);
  EXPECT_EQ(fsm.state(), FsmState::kBeyondMimicArmed);
}

TEST(FsmTest, ReleasingRbDoesNotForcePassiveAfterFunctionEntry) {
  JoystickMapper mapper;
  Fsm fsm(FsmState::kBeyondMimicActive);

  JoystickState held;
  held.rb_pressed = true;
  (void)UpdateMapper(
      &mapper, JoystickMapper::TimePoint(Milliseconds(0)), held, fsm);

  JoystickState released;
  const JoystickEvents events =
      UpdateMapper(&mapper, JoystickMapper::TimePoint(Milliseconds(10)), released, fsm);
  EXPECT_FALSE(events.rb_held);
  EXPECT_TRUE(events.rb_released);

  const FsmTransitionResult result = fsm.HandleJoystickEvents(events);
  EXPECT_EQ(result.current_state, FsmState::kBeyondMimicActive);
  EXPECT_EQ(fsm.state(), FsmState::kBeyondMimicActive);
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
