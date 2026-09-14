#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace zky_rl_deploy {

// 该结构表示运行时真正信任的策略 metadata。
// baseline 明确要求 ONNX metadata 才是权威来源，因此这里保存的是“从 ONNX 读出来并通过校验”的结果，
// 不是 YAML 想让它变成什么样就变成什么样。
struct PolicyMetadata {
  std::string onnx_path;
  std::vector<std::string> joint_names;
  std::vector<double> action_scale;
  std::vector<double> default_joint_pos;
  std::vector<double> joint_stiffness;
  std::vector<double> joint_damping;
};

// 该结构对应 YAML 中用于人工审阅的 metadata 副本。
// YAML 的职责是让人看懂和做离线审查，而不是覆盖 ONNX；因此这些字段是“审计副本”，不是运行时真值。
struct PolicyMetadataAuditCopy {
  std::optional<std::vector<std::string>> joint_names;
  std::optional<std::vector<double>> action_scale;
  std::optional<std::vector<double>> default_joint_pos;
  std::optional<std::vector<double>> joint_stiffness;
  std::optional<std::vector<double>> joint_damping;
  bool reject_startup_if_mismatch{true};
};

struct PolicyRuntimeConfig {
  std::string onnx_path;
  PolicyMetadataAuditCopy metadata_audit_copy;
};

class PolicyRuntime {
 public:
  using RawMetadataMap = std::unordered_map<std::string, std::string>;
  static constexpr std::size_t kExpectedJointCount = 12U;

  // 这两个接口用于向上层明确当前构建是否包含真实 ONNX Runtime 能力。
  // 若依赖缺失，必须给出清晰错误，而不是把“无法读取 metadata”伪装成空结果继续运行。
  static bool OnnxRuntimeEnabledAtBuild();
  static std::string OnnxRuntimeBuildSummary();

  PolicyMetadata LoadMetadata(const PolicyRuntimeConfig& config) const;

  // 纯 C++ 解析接口，允许单元测试直接注入 metadata 文本，不依赖真实 ONNX Runtime 或外部模型文件。
  static PolicyMetadata BuildAuthoritativeMetadataFromRawMap(
      const std::string& onnx_path, const RawMetadataMap& raw_metadata_map);

  // YAML 审计副本只能“证明自己和 ONNX 一致”，不能覆盖 ONNX。
  // 一旦出现差异，必须拒绝启动，避免策略顺序、PD 增益或默认角静默漂移到未知状态。
  static void ValidateAuditCopyMatches(const PolicyMetadata& authoritative_metadata,
                                       const PolicyMetadataAuditCopy& audit_copy);

 private:
  static void ValidateOnnxPath(const std::string& onnx_path);
};

}  // namespace zky_rl_deploy
