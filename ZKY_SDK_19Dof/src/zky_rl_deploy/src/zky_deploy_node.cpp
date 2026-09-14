#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <yaml-cpp/yaml.h>

#include "zky_rl_deploy/core/deploy_context.hpp"
#include "zky_rl_deploy/core/control_loop_timing.hpp"
#include "zky_rl_deploy/core/deploy_cycle_logger.hpp"
#include "zky_rl_deploy/core/fsm.hpp"
#include "zky_rl_deploy/core/json_motion_frame_provider.hpp"
#include "zky_rl_deploy/core/joint_mapper.hpp"
#include "zky_rl_deploy/core/motion_clip.hpp"
#include "zky_rl_deploy/core/motor_command.hpp"
#include "zky_rl_deploy/core/obs_builder.hpp"
#include "zky_rl_deploy/core/policy_action_runner.hpp"
#include "zky_rl_deploy/core/policy_runtime.hpp"
#include "zky_rl_deploy/core/active_output_gate.hpp"
#include "zky_rl_deploy/core/safety_gate.hpp"
#include "zky_rl_deploy/core/sensor_sync.hpp"
#include "zky_rl_deploy/core/stage_manager.hpp"
#include "zky_rl_deploy/core/stage_request_resolver.hpp"
#include "zky_rl_deploy/core/stand_trajectory.hpp"
#include "zky_rl_deploy/ros/parameter_loader.hpp"
#include "zky_rl_deploy/ros/ros_adapter.hpp"

namespace zky_rl_deploy {
namespace {

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

std::vector<double> ReorderPolicyValuesToHardwareOrderWithoutSign(
    const std::vector<double>& policy_values,
    const JointMapper::JointOrder& policy_order,
    const JointMapper::JointOrder& hardware_order) {
  if (policy_values.size() != policy_order.size()) {
    throw std::invalid_argument("policy_values size must match policy_order size");
  }

  std::unordered_map<std::string, std::size_t> policy_indices;
  policy_indices.reserve(policy_order.size());
  for (std::size_t index = 0; index < policy_order.size(); ++index) {
    policy_indices.emplace(policy_order[index], index);
  }

  std::vector<double> hardware_values(hardware_order.size(), 0.0);
  for (std::size_t hardware_index = 0; hardware_index < hardware_order.size(); ++hardware_index) {
    const auto policy_it = policy_indices.find(hardware_order[hardware_index]);
    if (policy_it == policy_indices.end()) {
      throw std::invalid_argument("hardware_order contains joint missing from policy_order");
    }
    hardware_values[hardware_index] = policy_values[policy_it->second];
  }
  return hardware_values;
}

std::vector<double> ScalePolicyTargetAroundDefault(
    const std::vector<double>& default_joint_pos_policy_order,
    const std::vector<double>& target_joint_pos_policy_order,
    double action_scale) {
  if (default_joint_pos_policy_order.size() != target_joint_pos_policy_order.size()) {
    throw std::invalid_argument("default_joint_pos and target_joint_pos must have identical sizes");
  }

  std::vector<double> scaled(target_joint_pos_policy_order.size(), 0.0);
  for (std::size_t index = 0; index < target_joint_pos_policy_order.size(); ++index) {
    scaled[index] = default_joint_pos_policy_order[index] +
                    action_scale *
                        (target_joint_pos_policy_order[index] -
                         default_joint_pos_policy_order[index]);
  }
  return scaled;
}

MotorCommandBuffer BuildPdShadowCommand(
    const std::vector<double>& target_joint_pos_hardware_order,
    const std::vector<double>& kp_hardware_order,
    const std::vector<double>& kd_hardware_order,
    const StageManager::StageDefinition& stage) {
  if (target_joint_pos_hardware_order.size() != kMotorCommandJointCount ||
      kp_hardware_order.size() != kMotorCommandJointCount ||
      kd_hardware_order.size() != kMotorCommandJointCount) {
    throw std::invalid_argument("shadow command inputs must be 12-dimensional");
  }

  MotorCommandBuffer commands = BuildDisableCommandBuffer();
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    if (joint_index >= stage.output_limits.max_enabled_joints) {
      continue;
    }

    commands[joint_index].kp = stage.output_limits.kp_scale * kp_hardware_order[joint_index];
    commands[joint_index].kd = stage.output_limits.kd_scale * kd_hardware_order[joint_index];
    commands[joint_index].joint_position = target_joint_pos_hardware_order[joint_index];
    commands[joint_index].joint_velocity = 0.0;
    commands[joint_index].joint_torque = 0.0;
    commands[joint_index].mode = MotorMode::PositionPd;
  }
  return commands;
}

std::pair<double, double> QuaternionXyzwToRollPitch(const std::array<double, 4>& quaternion_xyzw) {
  constexpr double kHalfPi = 1.57079632679489661923;

  const double x = quaternion_xyzw[0];
  const double y = quaternion_xyzw[1];
  const double z = quaternion_xyzw[2];
  const double w = quaternion_xyzw[3];

  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  const double roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = 2.0 * (w * y - z * x);
  const double pitch =
      std::abs(sinp) >= 1.0 ? std::copysign(kHalfPi, sinp) : std::asin(sinp);
  return {roll, pitch};
}

PolicyActionIoContract LoadPolicyActionIoContract(const std::string& package_root) {
  const YAML::Node policy_config =
      YAML::LoadFile((std::filesystem::path(package_root) / "config" / "policy_walk1subject1.yaml")
                         .string());
  const YAML::Node io_contract_node = policy_config["io_contract"];
  PolicyActionIoContract io_contract;
  io_contract.obs_input_name = io_contract_node["obs_input_name"].as<std::string>();
  io_contract.obs_input_dim = io_contract_node["obs_input_dim"].as<std::size_t>();
  io_contract.time_step_input_name = io_contract_node["time_step_input_name"].as<std::string>();
  io_contract.time_step_input_dim = io_contract_node["time_step_input_dim"].as<std::size_t>();
  io_contract.action_output_name = io_contract_node["action_output_name"].as<std::string>();
  io_contract.action_output_dim = io_contract_node["action_output_dim"].as<std::size_t>();
  io_contract.time_step_unit_s = io_contract_node["time_step_unit_s"].as<double>();
  return io_contract;
}

std::vector<double> ExtractJointPositionVector(const MotorCommandBuffer& commands) {
  std::vector<double> target_joint_position(kMotorCommandJointCount, 0.0);
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    target_joint_position[joint_index] = commands[joint_index].joint_position;
  }
  return target_joint_position;
}

std::vector<int> ExtractJointModes(const MotorCommandBuffer& commands) {
  std::vector<int> joint_modes(kMotorCommandJointCount, 0);
  for (std::size_t joint_index = 0; joint_index < kMotorCommandJointCount; ++joint_index) {
    joint_modes[joint_index] = static_cast<int>(commands[joint_index].mode);
  }
  return joint_modes;
}

std::string ActivePolicyTraceName(FsmState state, const std::string& candidate_source) {
  if (state == FsmState::kBeyondMimicArmed || state == FsmState::kBeyondMimicActive) {
    return "beyond_mimic/" + candidate_source;
  }
  if (state == FsmState::kStandInit) {
    return "stand_init";
  }
  if (state == FsmState::kStandHold) {
    return "stand_hold";
  }
  return "passive_shadow";
}

std::string CommandMuxRouteName(CommandMuxRoute route) {
  return ToString(route);
}

struct TimingDebugInfo {
  double joint_state_age_ms{-1.0};
  double t265_odom_age_ms{-1.0};
  double t265_imu_age_ms{-1.0};
  double joint_state_t265_odom_dt_ms{-1.0};
  double joint_state_t265_imu_dt_ms{-1.0};
  double t265_odom_t265_imu_dt_ms{-1.0};
};

double DurationToMilliseconds(const SensorSync::Duration& duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

template <typename SnapshotT>
double ComputeAgeMilliseconds(const std::optional<SnapshotT>& snapshot,
                              const SensorSync::TimePoint& now) {
  if (!snapshot.has_value()) {
    return -1.0;
  }
  return DurationToMilliseconds(now - snapshot->stamp);
}

template <typename SnapshotA, typename SnapshotB>
double ComputeAbsDtMilliseconds(const std::optional<SnapshotA>& a,
                                const std::optional<SnapshotB>& b) {
  if (!a.has_value() || !b.has_value()) {
    return -1.0;
  }
  return std::fabs(DurationToMilliseconds(a->stamp - b->stamp));
}

std::string FormatTimingValue(double value_ms) {
  if (value_ms < 0.0) {
    return "n/a";
  }

  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << value_ms;
  return stream.str();
}

TimingDebugInfo BuildTimingDebugInfo(const ros_adapter::InputSnapshot& snapshot,
                                     const SensorSync::TimePoint& now) {
  TimingDebugInfo info;
  info.joint_state_age_ms = ComputeAgeMilliseconds(snapshot.joint_feedback_policy_order, now);
  info.t265_odom_age_ms = ComputeAgeMilliseconds(snapshot.t265_odom, now);
  info.t265_imu_age_ms = ComputeAgeMilliseconds(snapshot.t265_imu, now);
  info.joint_state_t265_odom_dt_ms =
      ComputeAbsDtMilliseconds(snapshot.joint_feedback_policy_order, snapshot.t265_odom);
  info.joint_state_t265_imu_dt_ms =
      ComputeAbsDtMilliseconds(snapshot.joint_feedback_policy_order, snapshot.t265_imu);
  info.t265_odom_t265_imu_dt_ms = ComputeAbsDtMilliseconds(snapshot.t265_odom, snapshot.t265_imu);
  return info;
}

std::string BuildTimingSummary(const TimingDebugInfo& info) {
  std::ostringstream stream;
  stream << "ages_ms[joint=" << FormatTimingValue(info.joint_state_age_ms)
         << ",odom=" << FormatTimingValue(info.t265_odom_age_ms)
         << ",imu=" << FormatTimingValue(info.t265_imu_age_ms)
         << "], dt_ms[joint_odom=" << FormatTimingValue(info.joint_state_t265_odom_dt_ms)
         << ",joint_imu=" << FormatTimingValue(info.joint_state_t265_imu_dt_ms)
         << ",odom_imu=" << FormatTimingValue(info.t265_odom_t265_imu_dt_ms) << "]";
  return stream.str();
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  ros::init(argc, argv, "zky_deploy_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  zky_rl_deploy::ros_adapter::ParameterLoader loader;
  zky_rl_deploy::DeployContext context = loader.Load(private_nh);
  zky_rl_deploy::ros_adapter::RuntimeConfig runtime_config = loader.LoadRuntimeConfig(private_nh);

  try {
    zky_rl_deploy::JointMapper joint_mapper(runtime_config.robot_joints.policy_order,
                                            runtime_config.robot_joints.hardware_order,
                                            runtime_config.robot_joints.npz_order,
                                            runtime_config.robot_joints.sign_convention,
                                            runtime_config.robot_joints.sign_verification);
    zky_rl_deploy::ros_adapter::RuntimeRosAdapter ros_adapter(
        nh, private_nh, runtime_config, joint_mapper);

    zky_rl_deploy::PolicyRuntime policy_runtime;
    const zky_rl_deploy::PolicyMetadata metadata =
        policy_runtime.LoadMetadata(runtime_config.policy.runtime_config);
    if (metadata.joint_names != runtime_config.robot_joints.policy_order) {
      throw std::invalid_argument(
          "authoritative ONNX joint_names do not match robot_joints.yaml policy_order");
    }

    const std::vector<double> kp_hardware_order =
        zky_rl_deploy::ReorderPolicyValuesToHardwareOrderWithoutSign(
            metadata.joint_stiffness,
            metadata.joint_names,
            runtime_config.robot_joints.hardware_order);
    const std::vector<double> kd_hardware_order =
        zky_rl_deploy::ReorderPolicyValuesToHardwareOrderWithoutSign(
            metadata.joint_damping,
            metadata.joint_names,
            runtime_config.robot_joints.hardware_order);

    auto motion_frame_provider =
        std::make_shared<zky_rl_deploy::JsonMotionFrameProvider>(
            runtime_config.motion_clip.json_fallback_path);
    zky_rl_deploy::MotionClip motion_clip(
        runtime_config.motion_clip.runtime_config, joint_mapper, motion_frame_provider);
    zky_rl_deploy::ObsBuilder obs_builder(metadata.default_joint_pos);

    double stand_duration_s = 5.0;
    double stand_joint_velocity_limit = 0.3;
    private_nh.param("stand_duration_s", stand_duration_s, 5.0);
    private_nh.param("stand_joint_velocity_limit_rad_s", stand_joint_velocity_limit, 0.3);
    std::vector<double> stand_pose_policy_order = metadata.default_joint_pos;
    if (runtime_config.stand_pose_policy_override.has_value()) {
      stand_pose_policy_order = *runtime_config.stand_pose_policy_override;
    }

    zky_rl_deploy::StandTrajectory::Config stand_config;
    stand_config.nominal_duration =
        std::chrono::duration_cast<zky_rl_deploy::StandTrajectory::Duration>(
            std::chrono::duration<double>(stand_duration_s));
    stand_config.max_joint_velocity_rad_s =
        std::vector<double>(zky_rl_deploy::StandTrajectory::kJointCount,
                            stand_joint_velocity_limit);
    stand_config.max_joint_error_rad = runtime_config.safety_gate.max_joint_error_rad;
    stand_config.max_abs_roll_rad = 0.35;
    stand_config.max_abs_pitch_rad = 0.35;
    zky_rl_deploy::StandTrajectory stand_trajectory(
        stand_config, joint_mapper, stand_pose_policy_order);

    zky_rl_deploy::SensorSync sensor_sync(runtime_config.sensor_sync);
    zky_rl_deploy::SafetyGate safety_gate(runtime_config.safety_gate);
    zky_rl_deploy::StageManager stage_manager =
        zky_rl_deploy::StageManager::LoadFromYaml(runtime_config.bringup_stages_path);
    zky_rl_deploy::JoystickMapper joystick_mapper;
    zky_rl_deploy::Fsm fsm(zky_rl_deploy::FsmState::kPrecharge);
    zky_rl_deploy::DeployCycleCsvLogger cycle_logger(runtime_config.workspace_root);
    const zky_rl_deploy::PolicyActionIoContract action_io_contract =
        zky_rl_deploy::LoadPolicyActionIoContract(runtime_config.package_root);
    zky_rl_deploy::PolicyActionRunner action_runner(metadata.onnx_path, action_io_contract);

    ROS_INFO("zky_deploy_node deploy loop started.");
    ROS_INFO_STREAM("Loaded deploy context: " << zky_rl_deploy::BuildStartupSummary(context));
    ROS_INFO_STREAM("Public namespace: " << nh.getNamespace());
    ROS_INFO_STREAM("Policy metadata loaded from: " << metadata.onnx_path);
    ROS_INFO_STREAM("Deploy cycle csv logger: " << cycle_logger.file_path().string());
    for (const std::string& warning : runtime_config.startup_warnings) {
      ROS_WARN_STREAM(warning);
    }
    for (const std::string& warning : motion_frame_provider->warnings()) {
      ROS_WARN_STREAM(warning);
    }
    for (const std::string& blocker : action_runner.blockers()) {
      ROS_WARN_STREAM(blocker);
    }

    ros::AsyncSpinner spinner(4);
    spinner.start();

    std::atomic<bool> stop_requested{false};
    std::thread control_thread([&]() {
      const auto nominal_period = runtime_config.nominal_control_period;
      // ROS callback 只负责更新最近一次输入缓存，控制线程统一取快照并串行调度核心模块。
      // 这样可以避免多个 subscriber callback 各自做状态迁移或命令计算时互相打断，主控制链的时序才可追踪。
      // 这里明确不用 ros::Timer：Timer 仍依赖 ROS callback 队列调度，容易把控制周期和其他回调竞争混在一起；
      // baseline 要求的是 steady_clock + sleep_until 的绝对周期线程，便于稳定记录 dt/jitter/overrun。
      zky_rl_deploy::ControlLoopTimingTracker timing_tracker(nominal_period);
      zky_rl_deploy::FsmState previous_fsm_state = fsm.state();
      std::optional<std::vector<double>> previous_target_joint_position;
      bool motion_clip_started = false;
      bool stand_started = false;
      std::size_t shadow_cycles_without_fault = 0U;
      std::string last_stage_summary = "stage_manager_idle";
      std::size_t loop_index = 0U;

      while (ros::ok() && !stop_requested.load()) {
        std::this_thread::sleep_until(timing_tracker.next_loop_start());
        const auto loop_start = zky_rl_deploy::StandTrajectory::Clock::now();
        zky_rl_deploy::ControlLoopTimingTracker::LoopMetrics metrics =
            timing_tracker.BeginIteration(loop_start);

        const zky_rl_deploy::ros_adapter::InputSnapshot snapshot = ros_adapter.GetSnapshot();
        const auto timing_debug = zky_rl_deploy::BuildTimingDebugInfo(snapshot, loop_start);
        if (snapshot.joint_feedback_policy_order.has_value()) {
          sensor_sync.UpdateJointFeedback(*snapshot.joint_feedback_policy_order);
        }
        if (snapshot.t265_odom.has_value()) {
          sensor_sync.UpdateT265Odom(*snapshot.t265_odom);
        }
        if (snapshot.t265_imu.has_value()) {
          sensor_sync.UpdateT265Imu(*snapshot.t265_imu);
        }

        const zky_rl_deploy::SensorSync::OutputMode requested_output_mode =
            zky_rl_deploy::DetermineRequestedOutputMode(context);
        const zky_rl_deploy::SensorSync::Evaluation sensor_evaluation =
            sensor_sync.Evaluate(loop_start, requested_output_mode);

        const zky_rl_deploy::StageManager::VerificationSnapshot verification_snapshot = {
            sensor_sync.t265_extrinsic_verified(),
            joint_mapper.AllSignsVerifiedForActiveFunction(),
            runtime_config.robot_joints.npz_order_verified};
        const std::string desired_stage_name = ros_adapter.PollDesiredStageName();
        const std::string requested_stage_name =
            zky_rl_deploy::ResolveNextRequestedStageName(stage_manager, desired_stage_name);
        if (requested_stage_name != stage_manager.current_stage().name) {
          const auto transition = stage_manager.RequestStage(
              {requested_stage_name,
               verification_snapshot,
               fsm.state() == zky_rl_deploy::FsmState::kPassive ||
                   fsm.state() == zky_rl_deploy::FsmState::kStandHold});
          std::ostringstream stream;
          stream << transition.Summary()
                 << ", desired_stage=" << desired_stage_name;
          last_stage_summary = stream.str();
        } else {
          std::ostringstream stream;
          stream << "stage=" << stage_manager.current_stage().name
                 << ", desired_stage=" << desired_stage_name
                 << ", shadow_cycles=" << stage_manager.shadow_cycles_without_fault();
          last_stage_summary = stream.str();
        }
        const zky_rl_deploy::StageManager::ActiveEligibility beyond_mimic_active_eligibility =
            stage_manager.EvaluateBeyondMimicActiveEligibility({verification_snapshot});

        zky_rl_deploy::JoystickEvents joystick_events =
            joystick_mapper.Update(loop_start,
                                   snapshot.joystick_state,
                                   fsm.state() == zky_rl_deploy::FsmState::kZeroOutputEstop);

        bool stand_completed = false;
        std::optional<zky_rl_deploy::StandTrajectory::SampleResult> stand_sample;
        if (fsm.state() == zky_rl_deploy::FsmState::kStandInit && stand_started &&
            snapshot.joint_feedback_received) {
          zky_rl_deploy::StandTrajectory::RuntimeSafetyContext stand_context;
          stand_context.measured_joint_position_hardware_order = snapshot.joint_pos_hardware_order;
          if (snapshot.t265_odom.has_value()) {
            const auto roll_pitch = zky_rl_deploy::QuaternionXyzwToRollPitch(
                snapshot.t265_odom->pelvis_orientation_xyzw);
            stand_context.body_roll_rad = roll_pitch.first;
            stand_context.body_pitch_rad = roll_pitch.second;
          }
          stand_sample = stand_trajectory.Sample(loop_start, stand_context);
          stand_completed = stand_sample->completed && !stand_sample->request_safe_exit;
          if (stand_sample->request_safe_exit) {
            joystick_events.request_passive = true;
          }
        }

        zky_rl_deploy::FsmUpdateContext fsm_context;
        fsm_context.precharge_checks_complete =
            sensor_evaluation.has_complete_snapshot && snapshot.joy_received;
        fsm_context.stand_init_complete = stand_completed;
        fsm_context.beyond_mimic_ready_for_active =
            requested_output_mode == zky_rl_deploy::SensorSync::OutputMode::kActiveHardware &&
            sensor_evaluation.output_gate.allowed &&
            beyond_mimic_active_eligibility.allowed;
        const zky_rl_deploy::FsmTransitionResult fsm_transition =
            fsm.HandleJoystickEvents(joystick_events, fsm_context);

        std::optional<zky_rl_deploy::MotionClipStepResult> seeded_motion_step;
        if (fsm.state() != previous_fsm_state) {
          if (fsm.state() == zky_rl_deploy::FsmState::kStandInit &&
              snapshot.joint_feedback_received) {
            stand_trajectory.Start(loop_start, snapshot.joint_pos_hardware_order);
            stand_started = true;
          } else if (fsm.state() != zky_rl_deploy::FsmState::kStandInit) {
            stand_started = false;
          }

          if (fsm.state() == zky_rl_deploy::FsmState::kBeyondMimicArmed &&
              snapshot.joint_feedback_policy_order.has_value()) {
            const auto entry_check = motion_clip.CheckEntry(
                snapshot.joint_feedback_policy_order->joint_pos_policy_order);
            if (entry_check.accepted) {
              seeded_motion_step = motion_clip.ResetToWindowStart();
              motion_clip_started = true;
            } else {
              motion_clip_started = false;
              ROS_WARN_STREAM("Motion clip entry rejected: " << entry_check.reason);
            }
          } else if (fsm.state() != zky_rl_deploy::FsmState::kBeyondMimicArmed &&
                     fsm.state() != zky_rl_deploy::FsmState::kBeyondMimicActive) {
            motion_clip_started = false;
          }
        }
        previous_fsm_state = fsm.state();

        std::optional<zky_rl_deploy::MotionClipStepResult> motion_step = seeded_motion_step;
        if (!motion_step.has_value() &&
            fsm.state() == zky_rl_deploy::FsmState::kBeyondMimicArmed &&
            !motion_clip_started &&
            snapshot.joint_feedback_policy_order.has_value()) {
          const auto entry_check = motion_clip.CheckEntry(
              snapshot.joint_feedback_policy_order->joint_pos_policy_order);
          if (entry_check.accepted) {
            motion_step = motion_clip.ResetToWindowStart();
            motion_clip_started = true;
          }
        }
        std::vector<std::string> shadow_fault_reasons;
        std::array<double, 4> reference_pelvis_orientation_wxyz = {1.0, 0.0, 0.0, 0.0};
        if ((fsm.state() == zky_rl_deploy::FsmState::kBeyondMimicArmed ||
             fsm.state() == zky_rl_deploy::FsmState::kBeyondMimicActive) &&
            motion_clip_started && !motion_step.has_value()) {
          motion_step = motion_clip.Advance(false);
          if (motion_step->request_passive) {
            shadow_fault_reasons.push_back(motion_step->reason);
          }
        }
        if (motion_step.has_value()) {
          reference_pelvis_orientation_wxyz =
              motion_frame_provider->ReferencePelvisOrientationWxyz(
                  motion_step->command.source_frame_index);
        }

        std::optional<zky_rl_deploy::ObservationBuildResult> observation;
        std::vector<double> action_for_log(zky_rl_deploy::kMotorCommandJointCount, 0.0);
        bool action_valid_for_log = false;
        std::string action_source_for_log = "unavailable";
        double inference_time_ms = 0.0;
        if (motion_step.has_value() && sensor_evaluation.latest_complete_snapshot.has_value()) {
          zky_rl_deploy::ObservationInput observation_input;
          observation_input.motion_command = motion_step->command;
          observation_input.current_pelvis_orientation_xyzw =
              sensor_evaluation.latest_complete_snapshot->t265_odom.pelvis_orientation_xyzw;
          observation_input.reference_pelvis_orientation_wxyz = reference_pelvis_orientation_wxyz;
          observation_input.base_ang_vel_body =
              sensor_evaluation.latest_complete_snapshot->t265_imu.base_ang_vel_body_rad_s;
          observation_input.measured_joint_pos_policy_order =
              sensor_evaluation.latest_complete_snapshot->joint_feedback.joint_pos_policy_order;
          observation_input.measured_joint_vel_policy_order =
              sensor_evaluation.latest_complete_snapshot->joint_feedback.joint_vel_policy_order;
          try {
            observation = obs_builder.BuildObservation(observation_input);
            if (action_runner.available()) {
              const auto inference_start = zky_rl_deploy::StandTrajectory::Clock::now();
              action_for_log = action_runner.Run(
                  observation->observation,
                  static_cast<float>(action_io_contract.time_step_unit_s));
              inference_time_ms =
                  std::chrono::duration<double, std::milli>(
                      zky_rl_deploy::StandTrajectory::Clock::now() - inference_start)
                      .count();
              action_valid_for_log = true;
              action_source_for_log = "onnx_shadow";
            } else {
              action_source_for_log = "onnx_unavailable";
            }
          } catch (const std::exception& exception) {
            shadow_fault_reasons.push_back(
                std::string("ObsBuilder failure: ") + exception.what());
          }
        }

        if (!sensor_evaluation.passive_reasons.empty()) {
          shadow_fault_reasons.insert(shadow_fault_reasons.end(),
                                      sensor_evaluation.passive_reasons.begin(),
                                      sensor_evaluation.passive_reasons.end());
        }

        if (motion_step.has_value()) {
          const auto shadow_cycle = stage_manager.ReportShadowCycle(
              {zky_rl_deploy::StageManager::ClipWindow{
                   runtime_config.motion_clip.runtime_config.start_frame,
                   runtime_config.motion_clip.runtime_config.end_frame.value_or(
                       motion_frame_provider->FrameCount() - 1U),
                   runtime_config.motion_clip.runtime_config.clip_stride},
               !shadow_fault_reasons.empty()});
          shadow_cycles_without_fault = shadow_cycle.shadow_cycles_without_fault;
        } else {
          shadow_cycles_without_fault = stage_manager.shadow_cycles_without_fault();
        }

        zky_rl_deploy::MotorCommandBuffer candidate_shadow_commands =
            zky_rl_deploy::BuildDisableCommandBuffer();
        bool candidate_shadow_available = false;
        std::string candidate_source = "zero_output";
        if (fsm.state() == zky_rl_deploy::FsmState::kStandInit && stand_sample.has_value()) {
          candidate_shadow_commands = zky_rl_deploy::BuildPdShadowCommand(
              stand_sample->joint_position_hardware_order,
              kp_hardware_order,
              kd_hardware_order,
              stage_manager.current_stage());
          candidate_shadow_available = true;
          candidate_source = "stand_init_shadow";
        } else if (fsm.state() == zky_rl_deploy::FsmState::kStandHold) {
          candidate_shadow_commands = zky_rl_deploy::BuildPdShadowCommand(
              stand_trajectory.stand_pose_hardware_order(),
              kp_hardware_order,
              kd_hardware_order,
              stage_manager.current_stage());
          candidate_shadow_available = true;
          candidate_source = "stand_hold_shadow";
        } else if (motion_step.has_value()) {
          const std::vector<double> scaled_policy_target =
              zky_rl_deploy::ScalePolicyTargetAroundDefault(
                  metadata.default_joint_pos,
                  motion_step->command.joint_pos_policy_order,
                  stage_manager.current_stage().output_limits.action_scale);
          const std::vector<double> target_hardware =
              joint_mapper.PolicyToHardwareCommand(scaled_policy_target);
          candidate_shadow_commands = zky_rl_deploy::BuildPdShadowCommand(
              target_hardware,
              kp_hardware_order,
              kd_hardware_order,
              stage_manager.current_stage());
          candidate_shadow_available = true;
          candidate_source = "motion_clip_shadow_reference";
        }

        zky_rl_deploy::SafetyGate::Decision gate_decision;
        const zky_rl_deploy::ActiveOutputGateDecision active_output_gate =
            zky_rl_deploy::EvaluateActiveOutputGate(
                {context,
                 stage_manager.current_stage(),
                 verification_snapshot,
                 sensor_evaluation,
                 fsm.state(),
                 beyond_mimic_active_eligibility});
        if (candidate_shadow_available && snapshot.joint_feedback_received) {
          zky_rl_deploy::SafetyGate::RuntimeContext gate_context;
          gate_context.measured_joint_position = snapshot.joint_pos_hardware_order;
          gate_context.previous_target_joint_position =
              previous_target_joint_position.value_or(std::vector<double>{});
          gate_context.t265_verified = sensor_sync.t265_extrinsic_verified();
          gate_context.allow_active_hardware_output =
              active_output_gate.allow_active_hardware_output;
          gate_context.control_loop_jitter =
              std::chrono::duration_cast<zky_rl_deploy::SafetyGate::Milliseconds>(
                  std::chrono::duration<double, std::milli>(metrics.jitter_ms));
          gate_context.consecutive_overruns = metrics.consecutive_overruns;
          gate_context.stale_frame_count = motion_step.has_value() ? motion_step->stale_frame_count : 0U;
          gate_context.stale_hold_time =
              motion_step.has_value()
                  ? zky_rl_deploy::SafetyGate::Milliseconds(motion_step->stale_hold_time_ms)
                  : zky_rl_deploy::SafetyGate::Milliseconds(0);
          gate_decision =
              safety_gate.EvaluateActiveCommand(candidate_shadow_commands, gate_context);
        } else {
          gate_decision.active_command_accepted = false;
          gate_decision.blocking_reasons.push_back(
              "no candidate command or no joint feedback snapshot");
          gate_decision.mux_output = active_output_gate.active_hardware_requested
                                         ? zky_rl_deploy::CommandMux::BuildZeroHardwareOutput(
                                               gate_decision.blocking_reasons)
                                         : zky_rl_deploy::CommandMux::BuildZeroOutput(
                                               gate_decision.blocking_reasons);
        }

        if (active_output_gate.active_hardware_requested &&
            !gate_decision.mux_output.publish_motor_params) {
          std::vector<std::string> zero_refresh_reasons = gate_decision.blocking_reasons;
          if (zero_refresh_reasons.empty()) {
            zero_refresh_reasons = active_output_gate.blocking_reasons;
          }
          if (zero_refresh_reasons.empty()) {
            zero_refresh_reasons.push_back(
                "active hardware requested but no active command route was selected; refreshing zero output");
          }
          gate_decision.active_command_accepted = false;
          gate_decision.blocking_reasons = zero_refresh_reasons;
          gate_decision.mux_output =
              zky_rl_deploy::CommandMux::BuildZeroHardwareOutput(zero_refresh_reasons);
        }

        if (candidate_shadow_available) {
          std::vector<double> current_target_joint_position(
              zky_rl_deploy::kMotorCommandJointCount, 0.0);
          for (std::size_t joint_index = 0;
               joint_index < zky_rl_deploy::kMotorCommandJointCount;
               ++joint_index) {
            current_target_joint_position[joint_index] =
                candidate_shadow_commands[joint_index].joint_position;
          }
          previous_target_joint_position = current_target_joint_position;
        }

        if (context.publish_shadow) {
          std::vector<double> shadow_motor_params =
              zky_rl_deploy::PackMotorParams(candidate_shadow_commands);
          // shadow 输出与真实硬件输出必须分离：
          // /zky/shadow_motor_params 用来观察“如果放行，控制器想发什么”，而真实 /motor_params 仍保持关闭，
          // 这样我们才能在不碰硬件的情况下调试映射、观测和安全门，不会把 shadow 诊断误当成真实下发。
          ros_adapter.PublishShadowMotorParams(shadow_motor_params);
        }
        if (gate_decision.mux_output.publish_motor_params) {
          ros_adapter.PublishMotorParams(gate_decision.mux_output.gated_motor_params);
        }

        // 当前这一轮是否真正错过下一次计划启动时刻，只能在主循环尾部根据 finish time 判断。
        // SafetyGate 在本轮里使用的是“上一轮结束后已经累计的 overrun 次数”，避免把正常 wakeup jitter 误判成 overrun。
        metrics.consecutive_overruns =
            timing_tracker.FinishIteration(zky_rl_deploy::StandTrajectory::Clock::now());

        std::ostringstream control_mode_stream;
        control_mode_stream << "fsm=" << zky_rl_deploy::Fsm::ToString(fsm.state())
                            << ", stage=" << stage_manager.current_stage().name
                            << ", dry_run_only=" << context.dry_run_only
                            << ", shadow_mode=" << context.shadow_mode
                            << ", requested_output_mode="
                            << zky_rl_deploy::SensorSync::ToString(requested_output_mode)
                            << ", dt_ms=" << metrics.dt_ms
                            << ", jitter_ms=" << metrics.jitter_ms
                            << ", overrun_count=" << metrics.consecutive_overruns
                            << ", shadow_cycles_without_fault=" << shadow_cycles_without_fault;
        ros_adapter.PublishControlMode(control_mode_stream.str());

        const std::string current_policy_trace =
            zky_rl_deploy::ActivePolicyTraceName(fsm.state(), candidate_source);
        ros_adapter.PublishCurrentPolicyTrace(current_policy_trace);
        ros_adapter.PublishFsmRequestTrace(zky_rl_deploy::Fsm::ToString(fsm.state()));
        ros_adapter.PublishJoyButtonsSnapshot(snapshot.joystick_state);

        std::ostringstream safety_status_stream;
        safety_status_stream << "precharge_checks_complete="
                             << (sensor_evaluation.has_complete_snapshot && snapshot.joy_received)
                             << ", timing=" << zky_rl_deploy::BuildTimingSummary(timing_debug)
                             << ", sensor_sync=" << sensor_evaluation.Summary()
                             << ", joy=" << snapshot.joy_summary
                             << ", stage_manager=" << last_stage_summary
                             << ", fsm_transition=" << fsm_transition.Summary()
                             << ", adapter_warnings=" << zky_rl_deploy::JoinReasons(snapshot.warnings)
                             << ", startup_warnings="
                             << zky_rl_deploy::JoinReasons(runtime_config.startup_warnings);
        ros_adapter.PublishSafetyStatus(safety_status_stream.str());

        std::ostringstream output_gate_stream;
        const char* motor_params_payload =
            gate_decision.mux_output.publish_zero_motor_params_on_active_hardware
                ? "zero_output"
                : (gate_decision.mux_output.publish_motor_params ? "active_command" : "none");
        output_gate_stream << "candidate_source=" << candidate_source
                           << ", active_output_gate=" << active_output_gate.Summary()
                           << ", safety_gate=" << gate_decision.Summary()
                           << ", sensor_output_gate="
                           << (sensor_evaluation.output_gate.allowed ? "allowed" : "blocked")
                           << ", sensor_output_gate_reasons="
                           << zky_rl_deploy::JoinReasons(
                                  sensor_evaluation.output_gate.blocking_reasons)
                           << ", publish_shadow=" << (context.publish_shadow ? "true" : "false")
                           << ", motor_params_payload=" << motor_params_payload
                           << ", real_motor_params_published="
                           << (gate_decision.mux_output.publish_motor_params ? "true" : "false")
                           << ", real_motor_params_active_command_published="
                           << (gate_decision.mux_output.publish_motor_params &&
                                       !gate_decision.mux_output
                                            .publish_zero_motor_params_on_active_hardware
                                   ? "true"
                                   : "false");
        ros_adapter.PublishOutputGate(output_gate_stream.str());

        ROS_INFO_STREAM_THROTTLE(
            1.0,
            "[DEPLOY] fsm=" << zky_rl_deploy::Fsm::ToString(fsm.state())
                            << ", stage=" << stage_manager.current_stage().name
                            << ", candidate=" << candidate_source
                            << ", route="
                            << zky_rl_deploy::CommandMuxRouteName(gate_decision.mux_output.route)
                            << ", motion_clip_started=" << (motion_clip_started ? 1 : 0)
                            << ", shadow=" << shadow_cycles_without_fault
                            << ", sensor=" << zky_rl_deploy::SensorSync::ToString(
                                                   sensor_evaluation.severity)
                            << ", sensor_gate="
                            << (sensor_evaluation.output_gate.allowed ? "allowed" : "blocked")
                            << ", active_allow="
                            << (active_output_gate.allow_active_hardware_output ? "true" : "false")
                            << ", active_cmd="
                            << (gate_decision.active_command_accepted ? "true" : "false")
                            << ", timing=" << zky_rl_deploy::BuildTimingSummary(timing_debug)
                            << ", blockers="
                            << zky_rl_deploy::JoinReasons(gate_decision.blocking_reasons));
        if (!sensor_evaluation.passive_reasons.empty()) {
          ROS_WARN_STREAM_THROTTLE(
              1.0,
              "[DEPLOY] passive sensor fault: "
                  << zky_rl_deploy::JoinReasons(sensor_evaluation.passive_reasons)
                  << ", timing=" << zky_rl_deploy::BuildTimingSummary(timing_debug));
        }

        std::ostringstream debug_stream;
        debug_stream << "policy_metadata_loaded=true"
                     << ", onnx_path=" << metadata.onnx_path
                     << ", motion_clip_json=" << motion_frame_provider->source_path()
                     << ", motion_clip_started=" << motion_clip_started
                     << ", motion_frame="
                     << (motion_step.has_value()
                             ? std::to_string(motion_step->command.source_frame_index)
                             : std::string("none"))
                     << ", obs_built=" << observation.has_value()
                     << ", obs_dim="
                     << (observation.has_value()
                             ? std::to_string(observation->observation.size())
                             : std::string("0"))
                     << ", action_logged=" << (action_valid_for_log ? "true" : "false")
                     << ", action_source=" << action_source_for_log
                     << ", note=final command route is controlled by the motion clip reference, active output gate, FSM and SafetyGate.";
        ros_adapter.PublishBeyondMimicDebug(debug_stream.str());

        zky_rl_deploy::DeployCycleLogRecord log_record;
        log_record.ros_time_s = ros::Time::now().toSec();
        log_record.loop_index = loop_index++;
        log_record.fsm_state = zky_rl_deploy::Fsm::ToString(fsm.state());
        log_record.fsm_request_trace = zky_rl_deploy::Fsm::ToString(fsm.state());
        log_record.current_policy_trace = current_policy_trace;
        log_record.stage_name = stage_manager.current_stage().name;
        log_record.candidate_source = candidate_source;
        log_record.joy_received = snapshot.joy_received;
        log_record.joy_semantic_mapping_ready = snapshot.joy_semantic_mapping_ready;
        log_record.joy_start_pressed = snapshot.joystick_state.start_pressed;
        log_record.joy_back_pressed = snapshot.joystick_state.back_pressed;
        log_record.joy_home_pressed = snapshot.joystick_state.home_pressed;
        log_record.joy_rb_pressed = snapshot.joystick_state.rb_pressed;
        log_record.joy_a_pressed = snapshot.joystick_state.a_pressed;
        log_record.joy_b_pressed = snapshot.joystick_state.b_pressed;
        log_record.joy_x_pressed = snapshot.joystick_state.x_pressed;
        log_record.joy_y_pressed = snapshot.joystick_state.y_pressed;
        log_record.joy_summary = snapshot.joy_summary;
        log_record.beyond_mimic_action_valid = action_valid_for_log;
        log_record.beyond_mimic_action_source = action_source_for_log;
        log_record.inference_time_ms = inference_time_ms;
        log_record.beyond_mimic_action = action_for_log;
        log_record.target_joint_position =
            zky_rl_deploy::ExtractJointPositionVector(candidate_shadow_commands);
        log_record.limited_motor_params_route =
            zky_rl_deploy::CommandMuxRouteName(gate_decision.mux_output.route);
        log_record.publish_motor_params = gate_decision.mux_output.publish_motor_params;
        log_record.publish_shadow_motor_params =
            gate_decision.mux_output.publish_shadow_motor_params;
        if (gate_decision.mux_output.shadow_motor_params.has_value() &&
            gate_decision.mux_output.shadow_joint_commands.has_value()) {
          log_record.limited_motor_params = *gate_decision.mux_output.shadow_motor_params;
          log_record.joint_modes =
              zky_rl_deploy::ExtractJointModes(*gate_decision.mux_output.shadow_joint_commands);
        } else {
          log_record.limited_motor_params = gate_decision.mux_output.gated_motor_params;
          log_record.joint_modes = zky_rl_deploy::ExtractJointModes(
              gate_decision.mux_output.gated_joint_commands);
        }
        log_record.joint_feedback_received = snapshot.joint_feedback_received;
        log_record.joint_feedback_position = snapshot.joint_pos_hardware_order;
        log_record.joint_feedback_velocity = snapshot.joint_vel_hardware_order;
        log_record.t265_odom_received = snapshot.t265_odom.has_value();
        log_record.t265_imu_received = snapshot.t265_imu.has_value();
        if (snapshot.t265_odom.has_value()) {
          log_record.t265_orientation_xyzw = snapshot.t265_odom->pelvis_orientation_xyzw;
          log_record.t265_odom_position = snapshot.t265_odom->position_flu_m;
        }
        if (snapshot.t265_imu.has_value()) {
          log_record.t265_base_ang_vel_body = snapshot.t265_imu->base_ang_vel_body_rad_s;
        }
        log_record.joint_state_age_ms = timing_debug.joint_state_age_ms;
        log_record.t265_odom_age_ms = timing_debug.t265_odom_age_ms;
        log_record.t265_imu_age_ms = timing_debug.t265_imu_age_ms;
        log_record.joint_state_t265_odom_dt_ms = timing_debug.joint_state_t265_odom_dt_ms;
        log_record.joint_state_t265_imu_dt_ms = timing_debug.joint_state_t265_imu_dt_ms;
        log_record.t265_odom_t265_imu_dt_ms = timing_debug.t265_odom_t265_imu_dt_ms;
        log_record.dt_ms = metrics.dt_ms;
        log_record.jitter_ms = metrics.jitter_ms;
        log_record.overrun_count = metrics.consecutive_overruns;
        log_record.stale_frame_count = motion_step.has_value() ? motion_step->stale_frame_count : 0U;
        log_record.stale_hold_time_ms = motion_step.has_value() ? motion_step->stale_hold_time_ms : 0;
        log_record.max_frame_jump_observed =
            motion_step.has_value() ? motion_step->max_frame_jump_observed : 0U;
        log_record.shadow_cycles_without_fault = shadow_cycles_without_fault;
        log_record.safety_trigger_reason =
            gate_decision.blocking_reasons.empty()
                ? "none"
                : zky_rl_deploy::JoinReasons(gate_decision.blocking_reasons);
        // 新增这些日志字段只是为了把安全门控和 shadow 输出过程完整落盘，方便和 motor_control CSV 对账；
        // 它们不参与任何输出放行条件，也不会改变 dry_run/shadow 边界。
        cycle_logger.Append(log_record);
      }
    });

    ros::waitForShutdown();
    stop_requested.store(true);
    if (control_thread.joinable()) {
      control_thread.join();
    }
    spinner.stop();
  } catch (const std::exception& exception) {
    ROS_FATAL_STREAM("zky_deploy_node failed to initialize deploy loop: "
                     << exception.what());
    return 1;
  }

  return 0;
}
