#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/motor_command.hpp"
#include "zky_rl_deploy/core/safety_gate.hpp"

namespace zky_rl_deploy {
namespace {

using Milliseconds = SafetyGate::Milliseconds;

SafetyGate::Config MakeConfig() {
  SafetyGate::Config config;
  config.joint_lower_limits = std::vector<double>(kMotorCommandJointCount, -1.5);
  config.joint_upper_limits = std::vector<double>(kMotorCommandJointCount, 1.5);
  config.max_target_step_rad = 0.5;
  config.max_joint_error_rad = 0.4;
  config.max_control_loop_jitter = Milliseconds(5);
  config.max_consecutive_overruns = 3U;
  config.max_stale_frame_count = 3U;
  config.max_stale_hold_time = Milliseconds(100);
  config.require_t265_verified_for_active = true;
  return config;
}

MotorCommandBuffer MakeNominalActiveCommand() {
  MotorCommandBuffer commands;
  commands.reserve(kMotorCommandJointCount);
  for (std::size_t index = 0; index < kMotorCommandJointCount; ++index) {
    const double base = 0.01 * static_cast<double>(index + 1U);
    commands.push_back(
        MotorCommand{20.0 + base, 1.0 + base, base, 0.0, 0.0, MotorMode::PositionPd});
  }
  return commands;
}

SafetyGate::RuntimeContext MakeNominalContext() {
  SafetyGate::RuntimeContext context;
  context.measured_joint_position = std::vector<double>(kMotorCommandJointCount, 0.0);
  context.previous_target_joint_position =
      std::vector<double>(kMotorCommandJointCount, 0.0);
  context.t265_verified = true;
  context.allow_active_hardware_output = false;
  context.control_loop_jitter = Milliseconds(2);
  context.consecutive_overruns = 0U;
  context.stale_frame_count = 0U;
  context.stale_hold_time = Milliseconds(0);
  return context;
}

bool ContainsReason(const std::vector<std::string>& reasons, const std::string& needle) {
  return std::any_of(reasons.begin(), reasons.end(), [&](const std::string& reason) {
    return reason.find(needle) != std::string::npos;
  });
}

TEST(MotorCommandTest, MotorModeEnumValuesMatchBaseline) {
  EXPECT_EQ(static_cast<int>(MotorMode::Disable), 0);
  EXPECT_EQ(static_cast<int>(MotorMode::Passive), 1);
  EXPECT_EQ(static_cast<int>(MotorMode::PositionPd), 2);
  EXPECT_EQ(static_cast<int>(MotorMode::Reserved), 99);
}

TEST(SafetyGateTest, RejectsNanOrInfActiveCommand) {
  SafetyGate gate(MakeConfig());
  MotorCommandBuffer commands = MakeNominalActiveCommand();
  commands[3].joint_position = std::numeric_limits<double>::quiet_NaN();
  commands[5].joint_torque = std::numeric_limits<double>::infinity();

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(commands, MakeNominalContext());

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_EQ(decision.mux_output.route, CommandMuxRoute::kZeroOutput);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "joint_position contains NaN/Inf"));
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "joint_torque contains NaN/Inf"));
}

TEST(SafetyGateTest, ActiveHardwareRejectRefreshesZeroMotorParams) {
  SafetyGate gate(MakeConfig());
  MotorCommandBuffer commands = MakeNominalActiveCommand();
  commands[1].joint_position = 2.0;
  SafetyGate::RuntimeContext context = MakeNominalContext();
  context.allow_active_hardware_output = true;

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(commands, context);

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_EQ(decision.mux_output.route, CommandMuxRoute::kZeroOutput);
  EXPECT_TRUE(decision.mux_output.publish_motor_params);
  EXPECT_TRUE(decision.mux_output.publish_zero_motor_params_on_active_hardware);
  EXPECT_FALSE(decision.mux_output.publish_shadow_motor_params);
  EXPECT_EQ(decision.mux_output.gated_motor_params,
            PackMotorParams(BuildDisableCommandBuffer()));
}

TEST(SafetyGateTest, ShadowRejectKeepsRealMotorParamsClosed) {
  SafetyGate gate(MakeConfig());
  MotorCommandBuffer commands = MakeNominalActiveCommand();
  commands[1].joint_position = 2.0;

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(commands, MakeNominalContext());

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_EQ(decision.mux_output.route, CommandMuxRoute::kZeroOutput);
  EXPECT_FALSE(decision.mux_output.publish_motor_params);
  EXPECT_FALSE(decision.mux_output.publish_zero_motor_params_on_active_hardware);
}

TEST(SafetyGateTest, RejectsActiveCommandWhenJitterExceedsLimit) {
  SafetyGate gate(MakeConfig());
  SafetyGate::RuntimeContext context = MakeNominalContext();
  context.control_loop_jitter = Milliseconds(8);

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(MakeNominalActiveCommand(), context);

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_EQ(decision.mux_output.route, CommandMuxRoute::kZeroOutput);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "control loop jitter exceeded"));
}

TEST(SafetyGateTest, RejectsDimensionMismatch) {
  SafetyGate gate(MakeConfig());
  MotorCommandBuffer commands = MakeNominalActiveCommand();
  commands.pop_back();

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(commands, MakeNominalContext());

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "candidate_joint_commands dimension mismatch"));
}

TEST(SafetyGateTest, RejectsJointLimitTargetJumpAndTrackingError) {
  SafetyGate gate(MakeConfig());
  MotorCommandBuffer commands = MakeNominalActiveCommand();
  SafetyGate::RuntimeContext context = MakeNominalContext();

  commands[1].joint_position = 2.0;
  commands[2].joint_position = 0.8;
  commands[4].joint_position = 0.6;
  context.previous_target_joint_position[2] = 0.0;
  context.measured_joint_position[4] = 0.0;

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(commands, context);

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "joint_position exceeds limit"));
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "single-step target jump exceeded"));
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "joint tracking error exceeded"));
}

TEST(MotorCommandTest, PacksMotorParamsInKpKdPosVelTauModeOrderAcross72Values) {
  MotorCommandBuffer commands;
  commands.reserve(kMotorCommandJointCount);
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    const double base = 100.0 * static_cast<double>(joint_index);
    const MotorMode mode = (joint_index % 3U == 0U)
                               ? MotorMode::Disable
                               : (joint_index % 3U == 1U ? MotorMode::Passive
                                                         : MotorMode::PositionPd);
    commands.push_back(MotorCommand{
        base + 1.0, base + 2.0, base + 3.0, base + 4.0, base + 5.0, mode});
  }

  const std::vector<double> packed = PackMotorParams(commands);

  ASSERT_EQ(packed.size(), kMotorParamsLength);
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    const std::size_t offset = joint_index * kMotorParamsFieldsPerJoint;
    const double base = 100.0 * static_cast<double>(joint_index);
    EXPECT_DOUBLE_EQ(packed[offset + 0U], base + 1.0);
    EXPECT_DOUBLE_EQ(packed[offset + 1U], base + 2.0);
    EXPECT_DOUBLE_EQ(packed[offset + 2U], base + 3.0);
    EXPECT_DOUBLE_EQ(packed[offset + 3U], base + 4.0);
    EXPECT_DOUBLE_EQ(packed[offset + 4U], base + 5.0);

    const double expected_mode =
        (joint_index % 3U == 0U) ? 0.0 : ((joint_index % 3U == 1U) ? 1.0 : 2.0);
    EXPECT_DOUBLE_EQ(packed[offset + 5U], expected_mode);
  }
}

TEST(SafetyGateTest, RejectsActiveCommandWhenT265IsUnverified) {
  SafetyGate gate(MakeConfig());
  SafetyGate::RuntimeContext context = MakeNominalContext();
  context.t265_verified = false;

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(MakeNominalActiveCommand(), context);

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "T265 calibration verified=false"));
}

TEST(SafetyGateTest, RejectsActiveCommandWhenOverrunOrStaleExceedsLimit) {
  SafetyGate gate(MakeConfig());
  SafetyGate::RuntimeContext context = MakeNominalContext();
  context.consecutive_overruns = 4U;
  context.stale_frame_count = 4U;
  context.stale_hold_time = Milliseconds(120);

  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(MakeNominalActiveCommand(), context);

  EXPECT_FALSE(decision.active_command_accepted);
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "control loop overrun count exceeded"));
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "stale frame count exceeded"));
  EXPECT_TRUE(ContainsReason(decision.blocking_reasons, "stale hold time exceeded"));
}

TEST(SafetyGateTest, AcceptsCleanCommandAndKeepsShadowOnlyRouteWhenUpperLayerDoesNotArmHardware) {
  SafetyGate gate(MakeConfig());
  const MotorCommandBuffer commands = MakeNominalActiveCommand();
  const SafetyGate::Decision decision =
      gate.EvaluateActiveCommand(commands, MakeNominalContext());

  EXPECT_TRUE(decision.active_command_accepted);
  EXPECT_TRUE(decision.blocking_reasons.empty());
  EXPECT_EQ(decision.mux_output.route, CommandMuxRoute::kShadowOnly);
  EXPECT_FALSE(decision.mux_output.publish_motor_params);
  EXPECT_TRUE(decision.mux_output.publish_shadow_motor_params);
  ASSERT_TRUE(decision.mux_output.shadow_motor_params.has_value());
  EXPECT_EQ(decision.mux_output.gated_motor_params.size(), kMotorParamsLength);
  EXPECT_EQ(decision.mux_output.shadow_motor_params->size(), kMotorParamsLength);

  const std::vector<double>& gated = decision.mux_output.gated_motor_params;
  EXPECT_TRUE(std::all_of(gated.begin(), gated.end(), [](double value) { return value == 0.0; }));
}

TEST(SafetyGateTest, AcceptsCleanCommandAndBuildsActiveRouteWhenUpperLayerArmsHardware) {
  SafetyGate gate(MakeConfig());
  SafetyGate::RuntimeContext context = MakeNominalContext();
  context.allow_active_hardware_output = true;
  const MotorCommandBuffer commands = MakeNominalActiveCommand();

  const SafetyGate::Decision decision = gate.EvaluateActiveCommand(commands, context);

  EXPECT_TRUE(decision.active_command_accepted);
  EXPECT_TRUE(decision.blocking_reasons.empty());
  EXPECT_EQ(decision.mux_output.route, CommandMuxRoute::kActiveHardware);
  EXPECT_TRUE(decision.mux_output.publish_motor_params);
  EXPECT_FALSE(decision.mux_output.publish_shadow_motor_params);
  EXPECT_EQ(decision.mux_output.gated_joint_commands.size(), kMotorCommandJointCount);
  EXPECT_EQ(decision.mux_output.gated_motor_params, PackMotorParams(commands));
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
