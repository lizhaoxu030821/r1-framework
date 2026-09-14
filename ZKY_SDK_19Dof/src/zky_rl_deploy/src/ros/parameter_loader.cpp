#include "zky_rl_deploy/ros/parameter_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace zky_rl_deploy {
namespace ros_adapter {
namespace {

std::filesystem::path PackageRootPath() {
#ifdef ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR
  return std::filesystem::path(ZKY_RL_DEPLOY_PACKAGE_SOURCE_DIR);
#else
  return std::filesystem::current_path();
#endif
}

std::filesystem::path WorkspaceRootPath() {
  const std::filesystem::path package_root = PackageRootPath();
  if (package_root.has_parent_path() && package_root.parent_path().has_parent_path()) {
    return package_root.parent_path().parent_path();
  }
  return std::filesystem::current_path();
}

std::filesystem::path ResolveWorkspaceRelativePath(const std::string& raw_path) {
  const std::filesystem::path candidate(raw_path);
  if (candidate.is_absolute()) {
    return candidate;
  }
  return WorkspaceRootPath() / candidate;
}

template <typename T>
std::vector<T> ReadRequiredSequence(const YAML::Node& node, const char* key) {
  const YAML::Node child = node[key];
  if (!child || !child.IsSequence()) {
    throw std::invalid_argument(std::string("missing required sequence: ") + key);
  }
  return child.as<std::vector<T>>();
}

template <typename T, std::size_t Size>
std::array<T, Size> ReadRequiredArray(const YAML::Node& node, const char* key) {
  const std::vector<T> values = ReadRequiredSequence<T>(node, key);
  if (values.size() != Size) {
    std::ostringstream stream;
    stream << "sequence " << key << " must have exactly " << Size << " entries";
    throw std::invalid_argument(stream.str());
  }

  std::array<T, Size> result{};
  std::copy(values.begin(), values.end(), result.begin());
  return result;
}

template <typename T>
T ReadOptionalScalar(const YAML::Node& node, const char* key, const T& default_value) {
  const YAML::Node child = node[key];
  if (!child) {
    return default_value;
  }
  return child.as<T>();
}

int ReadOptionalIndex(const YAML::Node& node, const char* key, int default_value) {
  const YAML::Node child = node[key];
  if (!child || child.IsNull()) {
    return default_value;
  }
  return child.as<int>();
}

JointMapper::SignConvention ReadSignConvention(const YAML::Node& node) {
  JointMapper::SignConvention sign_convention;
  for (const auto& entry : node) {
    sign_convention.emplace(entry.first.as<std::string>(), entry.second.as<int>());
  }
  return sign_convention;
}

JointMapper::SignVerification ReadSignVerification(const YAML::Node& node) {
  JointMapper::SignVerification sign_verification;
  for (const auto& entry : node) {
    sign_verification.emplace(entry.first.as<std::string>(), entry.second.as<bool>());
  }
  return sign_verification;
}

PolicyMetadataAuditCopy ReadMetadataAuditCopy(const YAML::Node& node) {
  PolicyMetadataAuditCopy audit_copy;
  audit_copy.reject_startup_if_mismatch =
      ReadOptionalScalar<bool>(node, "reject_startup_if_mismatch", true);

  if (const YAML::Node child = node["joint_names"]) {
    audit_copy.joint_names = child.as<std::vector<std::string>>();
  }
  if (const YAML::Node child = node["action_scale"]) {
    audit_copy.action_scale = child.as<std::vector<double>>();
  }
  if (const YAML::Node child = node["default_joint_pos"]) {
    audit_copy.default_joint_pos = child.as<std::vector<double>>();
  }
  if (const YAML::Node child = node["joint_stiffness"]) {
    audit_copy.joint_stiffness = child.as<std::vector<double>>();
  }
  if (const YAML::Node child = node["joint_damping"]) {
    audit_copy.joint_damping = child.as<std::vector<double>>();
  }
  return audit_copy;
}

}  // namespace

DeployContext ParameterLoader::Load(const ::ros::NodeHandle& private_nh) const {
  DeployContext context;
  context.parameter_namespace = private_nh.getNamespace();
  context.config_namespace = context.parameter_namespace;
  context.config_root = (PackageRootPath() / "config").string();

  // 骨架阶段先固定通过私有命名空间读参，后续再逐步扩展到策略、传感器和安全分层配置。
  private_nh.param<std::string>(
      "config_namespace", context.config_namespace, context.parameter_namespace);
  private_nh.param<std::string>("config_root", context.config_root, context.config_root);
  private_nh.param("dry_run_only", context.dry_run_only, true);
  private_nh.param("shadow_mode", context.shadow_mode, true);
  private_nh.param("enable_motor_output", context.enable_motor_output, false);
  private_nh.param("publish_shadow", context.publish_shadow, true);

  return context;
}

RuntimeConfig ParameterLoader::LoadRuntimeConfig(const ::ros::NodeHandle& private_nh) const {
  RuntimeConfig config;
  config.package_root = PackageRootPath().string();
  config.workspace_root = WorkspaceRootPath().string();

  const std::filesystem::path package_root = PackageRootPath();
  std::string config_root_path = (package_root / "config").string();
  // launch 可显式改写配置目录，但仍要求目录内文件名遵守 baseline 约定，
  // 这样既能切换配置集，又不会把“随手缺文件”的风险带进控制主链。
  private_nh.param<std::string>("config_root", config_root_path, config_root_path);
  const std::filesystem::path config_root(config_root_path);

  const YAML::Node joystick_config = YAML::LoadFile((config_root / "joystick_beitong.yaml").string());
  config.topics.joy_topic = ReadOptionalScalar<std::string>(joystick_config, "ros_topic", "/joy");
  config.joy_mapping.raw_mapping_confirmed =
      joystick_config["mapping_status"]["raw_indices_confirmed"].as<bool>();
  config.joy_mapping.start_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["START"], "index", -1);
  config.joy_mapping.back_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["BACK"], "index", -1);
  config.joy_mapping.home_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["HOME"], "index", -1);
  config.joy_mapping.rb_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["RB"], "index", -1);
  config.joy_mapping.a_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["A"], "index", -1);
  config.joy_mapping.x_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["X"], "index", -1);
  config.joy_mapping.b_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["B"], "index", -1);
  config.joy_mapping.y_button_index =
      ReadOptionalIndex(joystick_config["buttons"]["Y"], "index", -1);
  const YAML::Node robot_joints_config = YAML::LoadFile((config_root / "robot_joints.yaml").string());
  config.robot_joints.policy_order =
      ReadRequiredSequence<std::string>(robot_joints_config, "policy_order");
  config.robot_joints.hardware_order =
      ReadRequiredSequence<std::string>(robot_joints_config, "hardware_order");
  config.robot_joints.npz_order =
      ReadRequiredSequence<std::string>(robot_joints_config, "npz_order");
  config.robot_joints.sign_convention =
      ReadSignConvention(robot_joints_config["sign_convention"]);
  config.robot_joints.sign_verification =
      ReadSignVerification(robot_joints_config["sign_verification"]["joints"]);
  config.robot_joints.npz_order_verified =
      config.robot_joints.npz_order.size() == 12U;

  const YAML::Node policy_config = YAML::LoadFile((config_root / "policy_walk1subject1.yaml").string());
  const YAML::Node policy_node = policy_config["policy"];
  config.policy.runtime_config.onnx_path =
      ResolveWorkspaceRelativePath(policy_node["path"].as<std::string>()).string();
  config.policy.nominal_control_period_ms =
      ReadOptionalScalar<int>(policy_node, "nominal_control_period_ms", 20);
  config.nominal_control_period =
      std::chrono::milliseconds(config.policy.nominal_control_period_ms);
  config.policy.runtime_config.metadata_audit_copy =
      ReadMetadataAuditCopy(policy_config["metadata_audit_copy"]);
  if (config.policy.runtime_config.metadata_audit_copy.default_joint_pos.has_value()) {
    config.policy.audit_default_joint_pos =
        *config.policy.runtime_config.metadata_audit_copy.default_joint_pos;
  }
  if (config.policy.runtime_config.metadata_audit_copy.joint_stiffness.has_value()) {
    config.policy.audit_joint_stiffness =
        *config.policy.runtime_config.metadata_audit_copy.joint_stiffness;
  }
  if (config.policy.runtime_config.metadata_audit_copy.joint_damping.has_value()) {
    config.policy.audit_joint_damping =
        *config.policy.runtime_config.metadata_audit_copy.joint_damping;
  }

  const YAML::Node motion_clip_config =
      YAML::LoadFile((config_root / "motion_clip_walk1subject1.yaml").string());
  const YAML::Node motion_clip_node = motion_clip_config["motion_clip"];
  config.motion_clip.npz_path =
      ResolveWorkspaceRelativePath(motion_clip_node["path"].as<std::string>()).string();
  const std::filesystem::path npz_path(config.motion_clip.npz_path);
  config.motion_clip.json_fallback_path =
      (npz_path.parent_path().parent_path() / "json" / (npz_path.stem().string() + ".json")).string();
  config.motion_clip.runtime_config.start_frame =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "start_frame", 0U);
  if (const YAML::Node end_frame = motion_clip_node["end_frame"]; end_frame && !end_frame.IsNull()) {
    config.motion_clip.runtime_config.end_frame = end_frame.as<std::size_t>();
  }
  config.motion_clip.runtime_config.clip_stride =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "clip_stride", 1U);
  config.motion_clip.runtime_config.loop =
      ReadOptionalScalar<bool>(motion_clip_node, "loop", true);
  config.motion_clip.runtime_config.playback_rate =
      ReadOptionalScalar<double>(motion_clip_node, "playback_rate", 1.0);
  config.motion_clip.runtime_config.max_frame_jump =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "max_frame_jump", 1U);
  config.motion_clip.runtime_config.max_consecutive_stale_frames = ReadOptionalScalar<std::size_t>(
      motion_clip_node, "max_consecutive_stale_frames", 3U);
  config.motion_clip.runtime_config.max_hold_time_ms =
      ReadOptionalScalar<int>(motion_clip_node, "max_hold_time_ms", 100);
  config.motion_clip.runtime_config.clip_entry_joint_error_rad =
      motion_clip_node["entry"]["clip_entry_joint_error_rad"].as<double>();
  config.motion_clip.runtime_config.nominal_period_ms =
      config.policy.nominal_control_period_ms;

  const std::string stale_frame_behavior =
      ReadOptionalScalar<std::string>(motion_clip_node, "stale_frame_behavior", "hold");
  if (stale_frame_behavior == "hold") {
    config.motion_clip.runtime_config.stale_frame_behavior = StaleFrameBehavior::Hold;
  } else if (stale_frame_behavior == "catchup") {
    config.motion_clip.runtime_config.stale_frame_behavior = StaleFrameBehavior::Catchup;
  } else if (stale_frame_behavior == "passive") {
    config.motion_clip.runtime_config.stale_frame_behavior = StaleFrameBehavior::Passive;
  } else {
    throw std::invalid_argument("unsupported stale_frame_behavior in motion_clip_walk1subject1.yaml");
  }

  const std::filesystem::path stand_pose_path = config_root / "stand_pose.yaml";
  if (std::filesystem::exists(stand_pose_path)) {
    const YAML::Node stand_pose_config = YAML::LoadFile(stand_pose_path.string());
    if (const YAML::Node override = stand_pose_config["stand_pose"]["policy_order_override"];
        override && !override.IsNull()) {
      config.stand_pose_policy_override = override.as<std::vector<double>>();
    }
  }

  const YAML::Node sensor_calibration_config =
      YAML::LoadFile((config_root / "sensor_calibration.yaml").string());
  const YAML::Node safety_limits_config =
      YAML::LoadFile((config_root / "safety_limits.yaml").string());
  config.sensor_sync.joint_state_timeout = std::chrono::milliseconds(
      safety_limits_config["sensor_sync"]["joint_state_timeout_ms"].as<int>());
  config.sensor_sync.t265_odom_timeout = std::chrono::milliseconds(
      safety_limits_config["sensor_sync"]["t265_odom_timeout_ms"].as<int>());
  config.sensor_sync.t265_imu_timeout = std::chrono::milliseconds(
      safety_limits_config["sensor_sync"]["t265_imu_timeout_ms"].as<int>());
  config.sensor_sync.sensor_sync_warn_dt = std::chrono::milliseconds(
      safety_limits_config["sensor_sync"]["sensor_sync_warn_dt_ms"].as<int>());
  config.sensor_sync.sensor_sync_passive_dt = std::chrono::milliseconds(
      safety_limits_config["sensor_sync"]["sensor_sync_passive_dt_ms"].as<int>());
  config.sensor_sync.t265_extrinsic_verified =
      sensor_calibration_config["t265_to_pelvis"]["verified"].as<bool>();
  config.sensor_sync.allow_dry_run_with_unverified_extrinsic =
      sensor_calibration_config["gating"]["allow_dry_run_with_unverified_extrinsic"].as<bool>();
  config.sensor_sync.allow_shadow_with_unverified_extrinsic =
      sensor_calibration_config["gating"]["allow_shadow_with_unverified_extrinsic"].as<bool>();
  config.sensor_sync.allow_motor_output_with_unverified_extrinsic =
      sensor_calibration_config["gating"]["allow_motor_output_with_unverified_extrinsic"].as<bool>();
  config.t265_extrinsic.rotation_quat_xyzw =
      ReadRequiredArray<double, 4>(sensor_calibration_config["t265_to_pelvis"],
                                   "rotation_quat_xyzw");
  config.t265_extrinsic.translation_xyz_m =
      ReadRequiredArray<double, 3>(sensor_calibration_config["t265_to_pelvis"],
                                   "translation_xyz_m");

  config.safety_gate.joint_lower_limits =
      std::vector<double>(12U, -3.14159265358979323846);
  config.safety_gate.joint_upper_limits =
      std::vector<double>(12U, 3.14159265358979323846);
  config.safety_gate.max_target_step_rad = 0.5;
  config.safety_gate.max_joint_error_rad = 0.35;
  config.safety_gate.max_control_loop_jitter = std::chrono::milliseconds(
      safety_limits_config["control_loop"]["max_jitter_ms"].as<int>());
  config.safety_gate.max_consecutive_overruns =
      safety_limits_config["control_loop"]["max_consecutive_overruns"].as<std::size_t>();
  config.safety_gate.max_stale_frame_count =
      safety_limits_config["beyond_mimic"]["max_consecutive_stale_frames"].as<std::size_t>();
  config.safety_gate.max_stale_hold_time = std::chrono::milliseconds(
      safety_limits_config["beyond_mimic"]["max_hold_time_ms"].as<int>());
  config.safety_gate.require_t265_verified_for_active =
      safety_limits_config["observation"]["reject_unverified_t265_extrinsic"].as<bool>();

  config.topics.joint_state_topic = "/joint_states";
  config.topics.t265_odom_topic = "/camera/odom/sample";
  config.topics.t265_imu_topic = "/camera/imu";
  config.bringup_stages_path = (config_root / "bringup_stages.yaml").string();

  private_nh.param<std::string>("desired_stage_name", config.desired_stage_name, "stage0_software");

  if (!config.joy_mapping.raw_mapping_confirmed) {
    config.startup_warnings.push_back(
        "joystick_beitong.yaml raw indices are unconfirmed; /joy callback stays in semantic shadow mode only.");
  }
  if (!config.sensor_sync.t265_extrinsic_verified) {
    config.startup_warnings.push_back(
        "sensor_calibration.yaml keeps T265 verified=false; node will remain dry_run/shadow only.");
  }
  config.startup_warnings.push_back(
      "joint limits are not yet provided in robot_joints.yaml; SafetyGate uses placeholder +/-pi rad bounds.");
  config.startup_warnings.push_back(
      "dry_run/shadow node currently uses standard String/Float64MultiArray diagnostics; later phases should define custom /zky messages.");

  return config;
}

}  // namespace ros_adapter
}  // namespace zky_rl_deploy
