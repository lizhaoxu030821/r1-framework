#include "zky_rl_deploy/core/policy_runtime.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <stdexcept>

#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace zky_rl_deploy {
namespace {

std::string Trim(std::string value) {
  const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

std::string StripOptionalBrackets(std::string value) {
  value = Trim(std::move(value));
  if (value.size() >= 2U && value.front() == '[' && value.back() == ']') {
    return Trim(value.substr(1U, value.size() - 2U));
  }
  return value;
}

std::vector<std::string> SplitCommaSeparatedStrings(const std::string& raw_value,
                                                    const std::string& field_name) {
  const std::string normalized = StripOptionalBrackets(raw_value);
  std::vector<std::string> tokens;
  std::stringstream stream(normalized);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = Trim(std::move(token));
    if (token.empty()) {
      throw std::invalid_argument(field_name + " contains an empty item.");
    }
    tokens.push_back(token);
  }

  if (tokens.empty()) {
    throw std::invalid_argument(field_name + " must not be empty.");
  }
  return tokens;
}

std::vector<double> SplitCommaSeparatedDoubles(const std::string& raw_value,
                                               const std::string& field_name) {
  const std::vector<std::string> tokens = SplitCommaSeparatedStrings(raw_value, field_name);
  std::vector<double> values;
  values.reserve(tokens.size());
  for (const std::string& token : tokens) {
    try {
      std::size_t parsed_size = 0U;
      const double value = std::stod(token, &parsed_size);
      if (parsed_size != token.size()) {
        throw std::invalid_argument(field_name + " contains a non-numeric token: " + token);
      }
      values.push_back(value);
    } catch (const std::exception& exception) {
      throw std::invalid_argument(field_name + " contains an invalid numeric token: " + token +
                                  " (" + exception.what() + ")");
    }
  }
  return values;
}

const std::string& RequireMetadataField(const PolicyRuntime::RawMetadataMap& raw_metadata_map,
                                        const std::string& field_name) {
  const auto field_it = raw_metadata_map.find(field_name);
  if (field_it == raw_metadata_map.end()) {
    throw std::invalid_argument("Missing required ONNX metadata field: " + field_name);
  }
  return field_it->second;
}

template <typename T>
void ValidateExpectedSize(const std::vector<T>& values, const std::string& field_name) {
  if (values.size() != PolicyRuntime::kExpectedJointCount) {
    std::ostringstream stream;
    stream << field_name << " must contain exactly "
           << PolicyRuntime::kExpectedJointCount << " entries, got " << values.size();
    throw std::invalid_argument(stream.str());
  }
}

template <typename T>
void ThrowMismatch(const std::string& field_name,
                   const std::vector<T>& authoritative_values,
                   const std::vector<T>& audit_values) {
  if (authoritative_values.size() != audit_values.size()) {
    std::ostringstream stream;
    stream << "YAML audit copy for " << field_name << " has size " << audit_values.size()
           << ", but ONNX metadata has size " << authoritative_values.size();
    throw std::invalid_argument(stream.str());
  }

  for (std::size_t index = 0; index < authoritative_values.size(); ++index) {
    if (authoritative_values[index] != audit_values[index]) {
      std::ostringstream stream;
      stream << "YAML audit copy for " << field_name << " mismatches ONNX metadata at index "
             << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
Ort::Env& GetOrtEnv() {
  static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "zky_rl_deploy_policy_runtime");
  return env;
}

PolicyRuntime::RawMetadataMap ReadRawMetadataMapFromOnnxFile(const std::string& onnx_path) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

  Ort::Session session(GetOrtEnv(), onnx_path.c_str(), session_options);
  Ort::ModelMetadata model_metadata = session.GetModelMetadata();
  Ort::AllocatorWithDefaultOptions allocator;

  PolicyRuntime::RawMetadataMap raw_metadata_map;
  const std::vector<std::string> required_fields = {"joint_names",
                                                    "action_scale",
                                                    "default_joint_pos",
                                                    "joint_stiffness",
                                                    "joint_damping"};
  raw_metadata_map.reserve(required_fields.size());
  for (const std::string& field_name : required_fields) {
    auto metadata_value =
        model_metadata.LookupCustomMetadataMapAllocated(field_name.c_str(), allocator);
    if (!metadata_value) {
      throw std::invalid_argument("ONNX metadata field is missing: " + field_name);
    }
    raw_metadata_map.emplace(field_name, metadata_value.get());
  }
  return raw_metadata_map;
}
#endif

}  // namespace

bool PolicyRuntime::OnnxRuntimeEnabledAtBuild() {
#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
  return true;
#else
  return false;
#endif
}

std::string PolicyRuntime::OnnxRuntimeBuildSummary() {
#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
  return "ONNX Runtime support is enabled for PolicyRuntime.";
#else
  return "ONNX Runtime support is disabled. Enable ZKY_RL_DEPLOY_ENABLE_ONNXRUNTIME and make "
         "sure onnxruntime headers/libraries are discoverable to load model metadata.";
#endif
}

PolicyMetadata PolicyRuntime::LoadMetadata(const PolicyRuntimeConfig& config) const {
  ValidateOnnxPath(config.onnx_path);

#if ZKY_RL_DEPLOY_HAS_ONNXRUNTIME
  const PolicyMetadata authoritative_metadata =
      BuildAuthoritativeMetadataFromRawMap(config.onnx_path, ReadRawMetadataMapFromOnnxFile(config.onnx_path));
  ValidateAuditCopyMatches(authoritative_metadata, config.metadata_audit_copy);
  return authoritative_metadata;
#else
  (void)config;
  throw std::runtime_error(OnnxRuntimeBuildSummary());
#endif
}

PolicyMetadata PolicyRuntime::BuildAuthoritativeMetadataFromRawMap(
    const std::string& onnx_path, const RawMetadataMap& raw_metadata_map) {
  PolicyMetadata metadata;
  metadata.onnx_path = onnx_path;
  metadata.joint_names =
      SplitCommaSeparatedStrings(RequireMetadataField(raw_metadata_map, "joint_names"),
                                 "joint_names");
  metadata.action_scale =
      SplitCommaSeparatedDoubles(RequireMetadataField(raw_metadata_map, "action_scale"),
                                 "action_scale");
  metadata.default_joint_pos =
      SplitCommaSeparatedDoubles(RequireMetadataField(raw_metadata_map, "default_joint_pos"),
                                 "default_joint_pos");
  metadata.joint_stiffness =
      SplitCommaSeparatedDoubles(RequireMetadataField(raw_metadata_map, "joint_stiffness"),
                                 "joint_stiffness");
  metadata.joint_damping =
      SplitCommaSeparatedDoubles(RequireMetadataField(raw_metadata_map, "joint_damping"),
                                 "joint_damping");

  // baseline 要求这些 metadata 向量都必须严格是 12 维。
  // 一旦维度不对，就说明模型、配置或解析过程至少有一处不可信，必须在启动阶段直接拦下。
  ValidateExpectedSize(metadata.joint_names, "joint_names");
  ValidateExpectedSize(metadata.action_scale, "action_scale");
  ValidateExpectedSize(metadata.default_joint_pos, "default_joint_pos");
  ValidateExpectedSize(metadata.joint_stiffness, "joint_stiffness");
  ValidateExpectedSize(metadata.joint_damping, "joint_damping");

  return metadata;
}

void PolicyRuntime::ValidateAuditCopyMatches(const PolicyMetadata& authoritative_metadata,
                                             const PolicyMetadataAuditCopy& audit_copy) {
  if (!audit_copy.reject_startup_if_mismatch) {
    return;
  }

  // YAML 副本只能用于人工审阅，绝不能把 ONNX metadata 静默覆盖掉。
  // 如果这里允许“不一致也继续跑”，就可能在不知情的情况下把关节顺序、默认角或 PD 增益切到错误版本，
  // 直接破坏策略观测和输出语义，因此必须拒绝启动。
  if (audit_copy.joint_names.has_value()) {
    ThrowMismatch("joint_names", authoritative_metadata.joint_names, *audit_copy.joint_names);
  }
  if (audit_copy.action_scale.has_value()) {
    ThrowMismatch("action_scale", authoritative_metadata.action_scale, *audit_copy.action_scale);
  }
  if (audit_copy.default_joint_pos.has_value()) {
    ThrowMismatch("default_joint_pos",
                  authoritative_metadata.default_joint_pos,
                  *audit_copy.default_joint_pos);
  }
  if (audit_copy.joint_stiffness.has_value()) {
    ThrowMismatch("joint_stiffness",
                  authoritative_metadata.joint_stiffness,
                  *audit_copy.joint_stiffness);
  }
  if (audit_copy.joint_damping.has_value()) {
    ThrowMismatch("joint_damping",
                  authoritative_metadata.joint_damping,
                  *audit_copy.joint_damping);
  }
}

void PolicyRuntime::ValidateOnnxPath(const std::string& onnx_path) {
  if (onnx_path.empty()) {
    throw std::invalid_argument("ONNX path must not be empty.");
  }

  if (!std::filesystem::exists(onnx_path)) {
    throw std::invalid_argument("ONNX path does not exist: " + onnx_path);
  }
}

}  // namespace zky_rl_deploy
