#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include "zky_rl_deploy/core/json_motion_frame_provider.hpp"
#include "zky_rl_deploy/core/joint_mapper.hpp"
#include "zky_rl_deploy/core/motion_clip.hpp"
#include "zky_rl_deploy/core/obs_builder.hpp"
#include "zky_rl_deploy/core/policy_runtime.hpp"
#include "zky_rl_deploy/core/stored_npz_archive.hpp"

namespace zky_rl_deploy {
namespace {

struct PolicyIoContract {
  std::string obs_input_name{"obs"};
  std::size_t obs_input_dim{69U};
  std::string time_step_input_name{"time_step"};
  std::size_t time_step_input_dim{1U};
  std::string action_output_name{"actions"};
  std::size_t action_output_dim{12U};
  double time_step_unit_s{0.02};
  bool last_action_init_zeros_required{true};
};

struct Stage0Config {
  std::string policy_yaml_path;
  std::string motion_clip_yaml_path;
  std::string robot_joints_yaml_path;
  std::string sensor_calibration_yaml_path;
  std::size_t cycles{10000U};
};

struct MinMaxTracker {
  double min{std::numeric_limits<double>::infinity()};
  double max{-std::numeric_limits<double>::infinity()};

  void Update(double value) {
    min = std::min(min, value);
    max = std::max(max, value);
  }

  bool valid() const { return std::isfinite(min) && std::isfinite(max); }

  std::string Summary() const {
    std::ostringstream stream;
    if (!valid()) {
      stream << "n/a";
    } else {
      stream << "[" << min << ", " << max << "]";
    }
    return stream.str();
  }
};

struct Stage0Report {
  std::size_t executed_cycles{0U};
  std::size_t obs_dim{0U};
  std::size_t action_dim{0U};
  MinMaxTracker action_range;
  MinMaxTracker target_joint_pos_range;
  MinMaxTracker motion_ref_ori_b_range;
  std::vector<std::size_t> joint_pos_npz_shape;
  std::vector<std::size_t> joint_vel_npz_shape;
  bool npz_joint_dimension_is_12{false};
  bool last_action_init_is_zeros{false};
  bool stale_frame_exceeded_requests_passive{false};
  bool no_nan_inf_in_10000_cycles{true};
  bool used_mock_action_runner{false};
  bool framework_checks_passed{false};
  bool checklist_ready{false};
  std::string motion_data_source_note;
  std::string jitter_note;
  std::vector<std::string> runtime_blockers;
  std::vector<std::string> checklist_blockers;

  std::string Summary() const {
    std::ostringstream stream;
    stream << std::boolalpha;
    stream << "Stage 0 Dry Run Summary\n";
    stream << "obs shape: [1, " << obs_dim << "]\n";
    stream << "action shape: [1, " << action_dim << "]\n";
    stream << "action range: " << action_range.Summary() << "\n";
    stream << "target joint pos range: " << target_joint_pos_range.Summary() << "\n";
    stream << "motion_ref_ori_b range: " << motion_ref_ori_b_range.Summary() << "\n";
    stream << "joint_pos npz shape: [";
    for (std::size_t index = 0; index < joint_pos_npz_shape.size(); ++index) {
      if (index != 0U) {
        stream << ", ";
      }
      stream << joint_pos_npz_shape[index];
    }
    stream << "]\n";
    stream << "joint_vel npz shape: [";
    for (std::size_t index = 0; index < joint_vel_npz_shape.size(); ++index) {
      if (index != 0U) {
        stream << ", ";
      }
      stream << joint_vel_npz_shape[index];
    }
    stream << "]\n";
    stream << "npz_order second dimension == 12: " << npz_joint_dimension_is_12 << "\n";
    stream << "last_action_init=zeros: " << last_action_init_is_zeros << "\n";
    stream << "stale frame exceeded -> PASSIVE: " << stale_frame_exceeded_requests_passive << "\n";
    stream << "10000 cycles no NaN/Inf: " << no_nan_inf_in_10000_cycles << "\n";
    stream << "used mock action runner: " << used_mock_action_runner << "\n";
    stream << "framework checks passed: " << framework_checks_passed << "\n";
    stream << "checklist ready: " << checklist_ready << "\n";
    stream << "motion data source: " << motion_data_source_note << "\n";
    stream << "jitter: " << jitter_note << "\n";
    if (!runtime_blockers.empty()) {
      stream << "runtime blockers:\n";
      for (const std::string& blocker : runtime_blockers) {
        stream << "  - " << blocker << "\n";
      }
    }
    if (!checklist_blockers.empty()) {
      stream << "checklist blockers:\n";
      for (const std::string& blocker : checklist_blockers) {
        stream << "  - " << blocker << "\n";
      }
    }
    return stream.str();
  }
};

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
  const std::filesystem::path path(raw_path);
  if (path.is_absolute()) {
    return path;
  }
  return WorkspaceRootPath() / path;
}

template <typename T>
T ReadRequiredScalar(const YAML::Node& node, const char* key) {
  const YAML::Node child = node[key];
  if (!child) {
    throw std::invalid_argument(std::string("missing required scalar: ") + key);
  }
  return child.as<T>();
}

template <typename T>
T ReadOptionalScalar(const YAML::Node& node, const char* key, const T& default_value) {
  const YAML::Node child = node[key];
  if (!child) {
    return default_value;
  }
  return child.as<T>();
}

template <typename T>
std::vector<T> ReadRequiredSequence(const YAML::Node& node, const char* key) {
  const YAML::Node child = node[key];
  if (!child || !child.IsSequence()) {
    throw std::invalid_argument(std::string("missing required sequence: ") + key);
  }
  return child.as<std::vector<T>>();
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

std::array<double, 4> WxyzToRosXyzw(const std::array<double, 4>& quaternion_wxyz) {
  return {quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]};
}

bool AllFinite(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

bool AllFinite(const std::array<double, 6>& values) {
  return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

bool AllZeros(const std::vector<double>& values, double tolerance = 1e-12) {
  return std::all_of(values.begin(), values.end(), [&](double value) {
    return std::abs(value) <= tolerance;
  });
}

std::vector<double> AddBoundedSinusoid(const std::vector<double>& reference,
                                       std::size_t cycle_index,
                                       double amplitude) {
  std::vector<double> values = reference;
  for (std::size_t index = 0; index < values.size(); ++index) {
    const double phase = 0.01 * static_cast<double>(cycle_index) + 0.13 * static_cast<double>(index);
    values[index] += amplitude * std::sin(phase);
  }
  return values;
}

std::array<double, 3> SimulatedBaseAngularVelocity(std::size_t cycle_index) {
  const double t = 0.01 * static_cast<double>(cycle_index);
  return {0.3 * std::sin(t), 0.25 * std::cos(0.5 * t), 0.2 * std::sin(0.25 * t)};
}

std::vector<double> BuildMockAction(const MotionClipCommand& motion_command,
                                    const PolicyMetadata& metadata) {
  std::vector<double> action(metadata.default_joint_pos.size(), 0.0);
  for (std::size_t index = 0; index < action.size(); ++index) {
    const double scale = metadata.action_scale[index];
    if (scale <= 0.0) {
      continue;
    }
    const double raw_action =
        (motion_command.joint_pos_policy_order[index] - metadata.default_joint_pos[index]) / scale;
    action[index] = std::clamp(raw_action, -1.0, 1.0);
  }
  return action;
}

struct LoadedPolicyConfig {
  PolicyRuntimeConfig runtime_config;
  PolicyIoContract io_contract;
  bool yaml_declares_last_action_init_zeros{false};
};

LoadedPolicyConfig LoadPolicyConfig(const std::string& policy_yaml_path) {
  const YAML::Node root = YAML::LoadFile(policy_yaml_path);
  const YAML::Node policy_node = root["policy"];
  const YAML::Node io_contract_node = root["io_contract"];

  LoadedPolicyConfig loaded;
  loaded.runtime_config.onnx_path =
      ResolveWorkspaceRelativePath(ReadRequiredScalar<std::string>(policy_node, "path")).string();
  loaded.runtime_config.metadata_audit_copy =
      ReadMetadataAuditCopy(root["metadata_audit_copy"]);
  loaded.yaml_declares_last_action_init_zeros =
      ReadRequiredScalar<std::string>(policy_node, "last_action_init") == "zeros";
  loaded.io_contract.obs_input_name =
      ReadRequiredScalar<std::string>(io_contract_node, "obs_input_name");
  loaded.io_contract.obs_input_dim =
      ReadRequiredScalar<std::size_t>(io_contract_node, "obs_input_dim");
  loaded.io_contract.time_step_input_name =
      ReadRequiredScalar<std::string>(io_contract_node, "time_step_input_name");
  loaded.io_contract.time_step_input_dim =
      ReadRequiredScalar<std::size_t>(io_contract_node, "time_step_input_dim");
  loaded.io_contract.action_output_name =
      ReadRequiredScalar<std::string>(io_contract_node, "action_output_name");
  loaded.io_contract.action_output_dim =
      ReadRequiredScalar<std::size_t>(io_contract_node, "action_output_dim");
  loaded.io_contract.time_step_unit_s =
      ReadRequiredScalar<double>(io_contract_node, "time_step_unit_s");
  loaded.io_contract.last_action_init_zeros_required =
      ReadOptionalScalar<bool>(root["runtime_constraints"],
                               "require_last_action_zero_bootstrap",
                               true);
  return loaded;
}

PolicyMetadata BuildMetadataFromAuditCopy(const PolicyRuntimeConfig& runtime_config) {
  const PolicyMetadataAuditCopy& audit_copy = runtime_config.metadata_audit_copy;
  if (!audit_copy.joint_names.has_value() || !audit_copy.action_scale.has_value() ||
      !audit_copy.default_joint_pos.has_value() || !audit_copy.joint_stiffness.has_value() ||
      !audit_copy.joint_damping.has_value()) {
    throw std::invalid_argument(
        "metadata audit copy is incomplete; cannot build offline fallback metadata");
  }

  PolicyMetadata metadata;
  metadata.onnx_path = runtime_config.onnx_path;
  metadata.joint_names = *audit_copy.joint_names;
  metadata.action_scale = *audit_copy.action_scale;
  metadata.default_joint_pos = *audit_copy.default_joint_pos;
  metadata.joint_stiffness = *audit_copy.joint_stiffness;
  metadata.joint_damping = *audit_copy.joint_damping;

  if (metadata.joint_names.size() != PolicyRuntime::kExpectedJointCount ||
      metadata.action_scale.size() != PolicyRuntime::kExpectedJointCount ||
      metadata.default_joint_pos.size() != PolicyRuntime::kExpectedJointCount ||
      metadata.joint_stiffness.size() != PolicyRuntime::kExpectedJointCount ||
      metadata.joint_damping.size() != PolicyRuntime::kExpectedJointCount) {
    throw std::invalid_argument("metadata audit copy does not contain 12-dimensional policy fields");
  }
  return metadata;
}

JointMapper LoadJointMapper(const std::string& robot_joints_yaml_path) {
  const YAML::Node root = YAML::LoadFile(robot_joints_yaml_path);
  const JointMapper::JointOrder policy_order =
      ReadRequiredSequence<std::string>(root, "policy_order");
  const JointMapper::JointOrder hardware_order =
      ReadRequiredSequence<std::string>(root, "hardware_order");
  const JointMapper::JointOrder npz_order =
      ReadRequiredSequence<std::string>(root, "npz_order");
  return JointMapper(policy_order,
                     hardware_order,
                     npz_order,
                     ReadSignConvention(root["sign_convention"]),
                     ReadSignVerification(root["sign_verification"]["joints"]));
}

MotionClipConfig LoadMotionClipConfig(const std::string& motion_clip_yaml_path,
                                      std::string* npz_path,
                                      std::string* json_path) {
  const YAML::Node root = YAML::LoadFile(motion_clip_yaml_path);
  const YAML::Node motion_clip_node = root["motion_clip"];
  *npz_path =
      ResolveWorkspaceRelativePath(ReadRequiredScalar<std::string>(motion_clip_node, "path"))
          .string();
  const std::filesystem::path npz_filesystem_path(*npz_path);
  *json_path =
      (npz_filesystem_path.parent_path().parent_path() / "json" /
       (npz_filesystem_path.stem().string() + ".json"))
          .string();

  MotionClipConfig config;
  config.start_frame =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "start_frame", 0U);
  if (const YAML::Node end_frame = motion_clip_node["end_frame"]; end_frame && !end_frame.IsNull()) {
    config.end_frame = end_frame.as<std::size_t>();
  }
  config.clip_stride =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "clip_stride", 1U);
  config.loop = ReadOptionalScalar<bool>(motion_clip_node, "loop", true);
  config.playback_rate =
      ReadOptionalScalar<double>(motion_clip_node, "playback_rate", 1.0);
  config.max_frame_jump =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "max_frame_jump", 1U);
  config.max_consecutive_stale_frames =
      ReadOptionalScalar<std::size_t>(motion_clip_node, "max_consecutive_stale_frames", 3U);
  config.max_hold_time_ms =
      ReadOptionalScalar<int>(motion_clip_node, "max_hold_time_ms", 100);
  config.clip_entry_joint_error_rad =
      ReadRequiredScalar<double>(motion_clip_node["entry"], "clip_entry_joint_error_rad");
  config.nominal_period_ms = 20;
  const std::string stale_frame_behavior =
      ReadOptionalScalar<std::string>(motion_clip_node, "stale_frame_behavior", "hold");
  if (stale_frame_behavior == "hold") {
    config.stale_frame_behavior = StaleFrameBehavior::Hold;
  } else if (stale_frame_behavior == "catchup") {
    config.stale_frame_behavior = StaleFrameBehavior::Catchup;
  } else if (stale_frame_behavior == "passive") {
    config.stale_frame_behavior = StaleFrameBehavior::Passive;
  } else {
    throw std::invalid_argument("unsupported stale_frame_behavior in motion clip config");
  }
  return config;
}

void LoadSensorCalibrationChecklist(const std::string& sensor_calibration_yaml_path,
                                    Stage0Report* report) {
  const YAML::Node root = YAML::LoadFile(sensor_calibration_yaml_path);
  const bool t265_verified =
      ReadRequiredScalar<bool>(root["t265_to_pelvis"], "verified");
  if (!t265_verified) {
    report->checklist_blockers.push_back(
        "sensor_calibration.yaml t265_to_pelvis.verified=false；Stage 0 完整 checklist 仍未满足。");
  }
}

class OfflineOnnxActionRunner {
 public:
  OfflineOnnxActionRunner(const std::string& onnx_path, PolicyIoContract io_contract)
      : io_contract_(std::move(io_contract)) {
#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
    try {
      Ort::SessionOptions session_options;
      session_options.SetIntraOpNumThreads(1);
      session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
      session_ = std::make_unique<Ort::Session>(GetOrtEnv(), onnx_path.c_str(), session_options);
      available_ = true;
    } catch (const std::exception& exception) {
      blockers_.push_back(std::string("failed to create ONNX Runtime session: ") + exception.what());
      available_ = false;
    }
#else
    (void)onnx_path;
    blockers_.push_back(PolicyRuntime::OnnxRuntimeBuildSummary());
    available_ = false;
#endif
  }

  bool available() const { return available_; }
  const std::vector<std::string>& blockers() const { return blockers_; }

  std::vector<double> Run(const std::vector<double>& observation, float time_step_scalar) const {
    if (!available_) {
      throw std::runtime_error("ONNX action runner is not available");
    }
    if (observation.size() != io_contract_.obs_input_dim) {
      throw std::invalid_argument("observation size does not match policy io contract");
    }

#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
    std::vector<float> observation_f32(observation.size(), 0.0F);
    std::transform(observation.begin(),
                   observation.end(),
                   observation_f32.begin(),
                   [](double value) { return static_cast<float>(value); });
    std::array<float, 1> time_step = {time_step_scalar};

    const std::array<std::int64_t, 2> obs_shape = {
        1, static_cast<std::int64_t>(io_contract_.obs_input_dim)};
    const std::array<std::int64_t, 2> time_step_shape = {
        1, static_cast<std::int64_t>(io_contract_.time_step_input_dim)};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value obs_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        observation_f32.data(),
        observation_f32.size(),
        obs_shape.data(),
        obs_shape.size());
    Ort::Value time_step_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        time_step.data(),
        time_step.size(),
        time_step_shape.data(),
        time_step_shape.size());

    std::array<const char*, 2> input_names = {
        io_contract_.obs_input_name.c_str(), io_contract_.time_step_input_name.c_str()};
    std::array<const char*, 1> output_names = {io_contract_.action_output_name.c_str()};
    std::array<Ort::Value, 2> input_tensors = {std::move(obs_tensor), std::move(time_step_tensor)};

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                        input_names.data(),
                                        input_tensors.data(),
                                        input_tensors.size(),
                                        output_names.data(),
                                        output_names.size());
    if (output_tensors.size() != 1U) {
      throw std::runtime_error("policy inference did not return exactly one actions tensor");
    }

    const float* action_data = output_tensors.front().GetTensorData<float>();
    const auto tensor_shape =
        output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    if (tensor_shape.size() != 2U || tensor_shape[0] != 1LL ||
        tensor_shape[1] != static_cast<std::int64_t>(io_contract_.action_output_dim)) {
      throw std::runtime_error("policy actions tensor shape does not match io_contract");
    }

    std::vector<double> action(io_contract_.action_output_dim, 0.0);
    for (std::size_t index = 0; index < action.size(); ++index) {
      action[index] = static_cast<double>(action_data[index]);
    }
    return action;
#else
    (void)observation;
    (void)time_step_scalar;
    throw std::runtime_error("ONNX Runtime support is disabled");
#endif
  }

 private:
#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
  static Ort::Env& GetOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zky_rl_deploy_stage0_dry_run");
    return env;
  }

  std::unique_ptr<Ort::Session> session_;
#endif
  PolicyIoContract io_contract_;
  std::vector<std::string> blockers_;
  bool available_{false};
};

Stage0Report RunStage0DryRun(const Stage0Config& config) {
  Stage0Report report;
  report.jitter_note =
      "n/a（该工具是离线纯软件 dry_run，不运行 steady_clock + sleep_until 控制线程）";

  const LoadedPolicyConfig loaded_policy = LoadPolicyConfig(config.policy_yaml_path);
  PolicyRuntime policy_runtime;
  PolicyMetadata metadata;
  try {
    metadata = policy_runtime.LoadMetadata(loaded_policy.runtime_config);
  } catch (const std::exception& exception) {
    report.runtime_blockers.push_back(
        std::string("failed to load ONNX metadata; falling back to YAML audit copy: ") +
        exception.what());
    metadata = BuildMetadataFromAuditCopy(loaded_policy.runtime_config);
  }
  const JointMapper joint_mapper = LoadJointMapper(config.robot_joints_yaml_path);
  std::string motion_npz_path;
  std::string motion_json_path;
  MotionClipConfig motion_clip_config =
      LoadMotionClipConfig(config.motion_clip_yaml_path, &motion_npz_path, &motion_json_path);
  LoadSensorCalibrationChecklist(config.sensor_calibration_yaml_path, &report);

  const StoredNpzArchive npz_archive = StoredNpzArchive::Open(motion_npz_path);
  report.joint_pos_npz_shape = npz_archive.GetArrayInfo("joint_pos.npy").shape;
  report.joint_vel_npz_shape = npz_archive.GetArrayInfo("joint_vel.npy").shape;
  report.npz_joint_dimension_is_12 =
      report.joint_pos_npz_shape.size() == 2U && report.joint_vel_npz_shape.size() == 2U &&
      report.joint_pos_npz_shape[1] == 12U && report.joint_vel_npz_shape[1] == 12U;
  if (!report.npz_joint_dimension_is_12) {
    report.runtime_blockers.push_back(
        "walk1_subject1.npz 的 joint_pos/joint_vel 第二维不是 12，无法满足 Stage 0 契约。");
  }

  auto frame_provider = std::make_shared<JsonMotionFrameProvider>(motion_json_path);
  report.motion_data_source_note =
      "实际帧回放使用 JSON fallback；真实 NPZ 仍已通过 stored header 读取验证 joint_pos/joint_vel shape。";
  MotionClip motion_clip(motion_clip_config, joint_mapper, frame_provider);
  ObsBuilder obs_builder(metadata.default_joint_pos);
  report.last_action_init_is_zeros =
      loaded_policy.yaml_declares_last_action_init_zeros &&
      loaded_policy.io_contract.last_action_init_zeros_required &&
      AllZeros(obs_builder.last_action());

  OfflineOnnxActionRunner action_runner(metadata.onnx_path, loaded_policy.io_contract);
  if (!action_runner.available()) {
    report.used_mock_action_runner = true;
    report.runtime_blockers.insert(report.runtime_blockers.end(),
                                   action_runner.blockers().begin(),
                                   action_runner.blockers().end());
  }

  MotionClipStepResult motion_step = motion_clip.ResetToWindowStart();
  std::array<double, 4> previous_current_orientation_xyzw =
      WxyzToRosXyzw(frame_provider->ReferencePelvisOrientationWxyz(
          motion_step.command.source_frame_index));

  for (std::size_t cycle = 0U; cycle < config.cycles; ++cycle) {
    if (cycle > 0U) {
      motion_step = motion_clip.Advance(false);
    }

    ObservationInput observation_input;
    observation_input.motion_command = motion_step.command;
    observation_input.reference_pelvis_orientation_wxyz =
        frame_provider->ReferencePelvisOrientationWxyz(motion_step.command.source_frame_index);
    observation_input.current_pelvis_orientation_xyzw = previous_current_orientation_xyzw;
    observation_input.base_ang_vel_body = SimulatedBaseAngularVelocity(cycle);
    observation_input.measured_joint_pos_policy_order =
        AddBoundedSinusoid(motion_step.command.joint_pos_policy_order, cycle, 0.005);
    observation_input.measured_joint_vel_policy_order =
        AddBoundedSinusoid(motion_step.command.joint_vel_policy_order, cycle, 0.01);

    const ObservationBuildResult observation = obs_builder.BuildObservation(observation_input);
    report.obs_dim = observation.observation.size();
    if (cycle == 0U && !AllZeros(observation.last_action_used)) {
      report.last_action_init_is_zeros = false;
    }

    if (!AllFinite(observation.observation) || !AllFinite(observation.motion_ref_ori_b)) {
      report.no_nan_inf_in_10000_cycles = false;
      report.runtime_blockers.push_back("observation or motion_ref_ori_b contains NaN/Inf");
      break;
    }
    for (double value : observation.motion_ref_ori_b) {
      report.motion_ref_ori_b_range.Update(value);
    }

    std::vector<double> action;
    if (action_runner.available()) {
      action = action_runner.Run(
          observation.observation,
          static_cast<float>(cycle * loaded_policy.io_contract.time_step_unit_s));
    } else {
      action = BuildMockAction(motion_step.command, metadata);
    }
    report.action_dim = action.size();

    if (!AllFinite(action)) {
      report.no_nan_inf_in_10000_cycles = false;
      report.runtime_blockers.push_back("action contains NaN/Inf");
      break;
    }
    for (double value : action) {
      report.action_range.Update(value);
    }

    std::vector<double> target_joint_pos(metadata.default_joint_pos.size(), 0.0);
    for (std::size_t joint_index = 0; joint_index < target_joint_pos.size(); ++joint_index) {
      target_joint_pos[joint_index] =
          metadata.default_joint_pos[joint_index] + action[joint_index] * metadata.action_scale[joint_index];
      report.target_joint_pos_range.Update(target_joint_pos[joint_index]);
    }
    if (!AllFinite(target_joint_pos)) {
      report.no_nan_inf_in_10000_cycles = false;
      report.runtime_blockers.push_back("target joint position contains NaN/Inf");
      break;
    }

    obs_builder.UpdateLastAction(action);
    previous_current_orientation_xyzw =
        WxyzToRosXyzw(observation_input.reference_pelvis_orientation_wxyz);
    ++report.executed_cycles;
  }

  MotionClip stale_clip(motion_clip_config, joint_mapper, frame_provider);
  (void)stale_clip.ResetToWindowStart();
  for (std::size_t stale_cycle = 0U;
       stale_cycle <= motion_clip_config.max_consecutive_stale_frames + 1U;
       ++stale_cycle) {
    const MotionClipStepResult stale_step = stale_clip.Advance(true);
    if (stale_step.request_passive) {
      report.stale_frame_exceeded_requests_passive = true;
      break;
    }
  }
  if (!report.stale_frame_exceeded_requests_passive) {
    report.runtime_blockers.push_back("stale frame exceeded did not request PASSIVE");
  }
  if (report.obs_dim != ObsBuilder::kObservationDim) {
    report.runtime_blockers.push_back("observation dimension is not 69");
  }
  if (report.action_dim != PolicyRuntime::kExpectedJointCount) {
    report.runtime_blockers.push_back("action dimension is not 12");
  }

  report.framework_checks_passed =
      report.executed_cycles == config.cycles && report.obs_dim == ObsBuilder::kObservationDim &&
      report.action_dim == PolicyRuntime::kExpectedJointCount &&
      report.no_nan_inf_in_10000_cycles &&
      report.npz_joint_dimension_is_12 && report.last_action_init_is_zeros &&
      report.stale_frame_exceeded_requests_passive;
  report.checklist_ready = report.framework_checks_passed &&
                           report.runtime_blockers.empty() &&
                           report.checklist_blockers.empty();
  return report;
}

Stage0Config BuildDefaultConfig() {
  const std::filesystem::path config_root = PackageRootPath() / "config";
  Stage0Config config;
  config.policy_yaml_path = (config_root / "policy_walk1subject1.yaml").string();
  config.motion_clip_yaml_path = (config_root / "motion_clip_walk1subject1.yaml").string();
  config.robot_joints_yaml_path = (config_root / "robot_joints.yaml").string();
  config.sensor_calibration_yaml_path = (config_root / "sensor_calibration.yaml").string();
  return config;
}

}  // namespace
}  // namespace zky_rl_deploy

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  try {
    // 该工具明确不上硬件、也不依赖 ROS 回调；
    // 它只验证 Stage 0 所需的数据链、ONNX I/O、motion clip/obs 拼接和 stale 行为。
    const auto config = zky_rl_deploy::BuildDefaultConfig();
    const auto report = zky_rl_deploy::RunStage0DryRun(config);
    std::cout << report.Summary();
    return report.framework_checks_passed ? 0 : 2;
  } catch (const std::exception& exception) {
    std::cerr << "Stage 0 dry_run failed: " << exception.what() << "\n";
    return 1;
  }
}
