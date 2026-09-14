#include "zky_rl_deploy/core/stage_manager.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "zky_rl_deploy/core/motor_command.hpp"

namespace zky_rl_deploy {
namespace {

constexpr double kDoubleTolerance = 1e-12;

bool AlmostEqual(double lhs, double rhs) {
  return std::fabs(lhs - rhs) <= kDoubleTolerance;
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

bool OutputLimitsChanged(const StageManager::StageOutputLimits& lhs,
                         const StageManager::StageOutputLimits& rhs) {
  return lhs.allow_motor_output != rhs.allow_motor_output ||
         lhs.max_enabled_joints != rhs.max_enabled_joints ||
         !AlmostEqual(lhs.action_scale, rhs.action_scale) ||
         !AlmostEqual(lhs.kp_scale, rhs.kp_scale) ||
         !AlmostEqual(lhs.kd_scale, rhs.kd_scale);
}

bool OutputLevelMorePermissive(const StageManager::StageOutputLimits& current,
                               const StageManager::StageOutputLimits& target) {
  return (!current.allow_motor_output && target.allow_motor_output) ||
         target.max_enabled_joints > current.max_enabled_joints ||
         target.action_scale > current.action_scale + kDoubleTolerance ||
         target.kp_scale > current.kp_scale + kDoubleTolerance ||
         target.kd_scale > current.kd_scale + kDoubleTolerance;
}

std::size_t ParseStageOrdinal(const std::string& stage_name) {
  constexpr const char* kPrefix = "stage";
  if (stage_name.rfind(kPrefix, 0U) != 0U) {
    throw std::invalid_argument("stage name must start with 'stage': " + stage_name);
  }

  std::size_t cursor = std::char_traits<char>::length(kPrefix);
  if (cursor >= stage_name.size() || !std::isdigit(stage_name[cursor])) {
    throw std::invalid_argument("stage name must contain numeric ordinal: " + stage_name);
  }

  std::size_t ordinal = 0U;
  while (cursor < stage_name.size() && std::isdigit(stage_name[cursor])) {
    ordinal = ordinal * 10U + static_cast<std::size_t>(stage_name[cursor] - '0');
    ++cursor;
  }
  return ordinal;
}

template <typename T>
T ReadRequiredScalar(const YAML::Node& node, const char* key) {
  const YAML::Node value = node[key];
  if (!value) {
    throw std::invalid_argument(std::string("missing required key: ") + key);
  }
  return value.as<T>();
}

template <typename T>
T ReadOptionalScalar(const YAML::Node& node, const char* key, const T& default_value) {
  const YAML::Node value = node[key];
  if (!value) {
    return default_value;
  }
  return value.as<T>();
}

StageManager::StageRules ParseStageRules(const YAML::Node& root) {
  StageManager::StageRules rules;
  const YAML::Node node = root["stage_rules"];
  if (!node) {
    return rules;
  }

  rules.monotonic_escalation_only =
      ReadOptionalScalar<bool>(node, "monotonic_escalation_only", true);
  rules.require_stage_manager_and_safety_supervisor_ack =
      ReadOptionalScalar<bool>(node, "require_stage_manager_and_safety_supervisor_ack", true);
  rules.require_passive_or_stand_hold_before_scale_up =
      ReadOptionalScalar<bool>(node, "require_passive_or_stand_hold_before_scale_up", true);
  rules.min_shadow_cycles_before_active_function = ReadOptionalScalar<std::size_t>(
      node, "min_shadow_cycles_before_active_function", 200U);
  return rules;
}

StageManager::StageDefinition ParseStageDefinition(const std::string& stage_name,
                                                   const YAML::Node& stage_node) {
  if (!stage_node.IsMap()) {
    throw std::invalid_argument("stage definition must be a YAML map: " + stage_name);
  }

  StageManager::StageDefinition stage;
  stage.name = stage_name;
  stage.ordinal = ParseStageOrdinal(stage_name);
  stage.allow_passive_request =
      ReadOptionalScalar<bool>(stage_node, "allow_passive_request", false);
  stage.requirements.require_t265_verified =
      ReadOptionalScalar<bool>(stage_node, "require_t265_verified", false);
  stage.requirements.require_joint_sign_verified =
      ReadOptionalScalar<bool>(stage_node, "require_joint_sign_verified", false);
  stage.requirements.require_npz_order_verified =
      ReadOptionalScalar<bool>(stage_node, "require_npz_order_verified", false);
  stage.requirements.min_shadow_cycles_without_fault =
      ReadOptionalScalar<std::size_t>(stage_node, "min_shadow_cycles_without_fault", 0U);
  stage.requirements.require_shadow_cycles_without_fault =
      ReadOptionalScalar<std::size_t>(stage_node, "require_shadow_cycles_without_fault", 0U);
  if (const YAML::Node max_run_time_s = stage_node["max_run_time_s"]) {
    stage.requirements.max_run_time_s = max_run_time_s.as<double>();
  }

  stage.output_limits.allow_motor_output =
      ReadRequiredScalar<bool>(stage_node, "allow_motor_output");
  stage.output_limits.max_enabled_joints = ReadOptionalScalar<std::size_t>(
      stage_node, "max_enabled_joints", kMotorCommandJointCount);
  stage.output_limits.action_scale = ReadRequiredScalar<double>(stage_node, "action_scale");
  stage.output_limits.kp_scale = ReadOptionalScalar<double>(stage_node, "kp_scale", 0.0);
  stage.output_limits.kd_scale = ReadOptionalScalar<double>(stage_node, "kd_scale", 0.0);

  if (!std::isfinite(stage.output_limits.action_scale) ||
      !std::isfinite(stage.output_limits.kp_scale) ||
      !std::isfinite(stage.output_limits.kd_scale)) {
    throw std::invalid_argument("stage scales must be finite: " + stage_name);
  }
  if (stage.output_limits.action_scale < 0.0 || stage.output_limits.kp_scale < 0.0 ||
      stage.output_limits.kd_scale < 0.0) {
    throw std::invalid_argument("stage scales must be >= 0: " + stage_name);
  }
  if (stage.output_limits.max_enabled_joints == 0U ||
      stage.output_limits.max_enabled_joints > kMotorCommandJointCount) {
    throw std::invalid_argument("max_enabled_joints must be within [1, 12]: " + stage_name);
  }

  return stage;
}

std::vector<StageManager::StageDefinition> ParseOrderedStages(const YAML::Node& root) {
  const YAML::Node stages_node = root["stages"];
  if (!stages_node || !stages_node.IsMap()) {
    throw std::invalid_argument("bringup_stages.yaml must contain a stages map");
  }

  std::vector<StageManager::StageDefinition> stages;
  stages.reserve(stages_node.size());
  for (const auto& stage_entry : stages_node) {
    const std::string stage_name = stage_entry.first.as<std::string>();
    stages.push_back(ParseStageDefinition(stage_name, stage_entry.second));
  }

  std::sort(stages.begin(), stages.end(), [](const auto& lhs, const auto& rhs) {
    if (lhs.ordinal != rhs.ordinal) {
      return lhs.ordinal < rhs.ordinal;
    }
    return lhs.name < rhs.name;
  });

  std::unordered_set<std::size_t> ordinals;
  std::unordered_set<std::string> names;
  for (const auto& stage : stages) {
    if (!ordinals.insert(stage.ordinal).second) {
      throw std::invalid_argument("duplicate stage ordinal detected in bringup_stages.yaml");
    }
    if (!names.insert(stage.name).second) {
      throw std::invalid_argument("duplicate stage name detected in bringup_stages.yaml");
    }
  }

  return stages;
}

const StageManager::StageDefinition* FindStageByName(
    const std::vector<StageManager::StageDefinition>& stages,
    const std::string& stage_name) {
  const auto stage_it = std::find_if(stages.begin(), stages.end(), [&](const auto& stage) {
    return stage.name == stage_name;
  });
  return stage_it == stages.end() ? nullptr : &(*stage_it);
}

}  // namespace

StageManager StageManager::LoadFromYaml(const std::string& yaml_path) {
  try {
    const YAML::Node root = YAML::LoadFile(yaml_path);
    return StageManager(ParseStageRules(root), ParseOrderedStages(root));
  } catch (const YAML::Exception& exception) {
    throw std::runtime_error("failed to load bringup_stages.yaml: " + std::string(exception.what()));
  }
}

StageManager::StageManager(StageRules stage_rules, std::vector<StageDefinition> ordered_stages)
    : stage_rules_(std::move(stage_rules)), ordered_stages_(std::move(ordered_stages)) {
  if (ordered_stages_.empty()) {
    throw std::invalid_argument("StageManager requires at least one stage");
  }
}

StageManager::StageTransitionResult StageManager::RequestStage(const StageRequest& request) {
  StageTransitionResult result;
  result.requested_stage_name = request.target_stage_name;
  result.previous_stage_name = current_stage().name;
  result.current_stage_name = current_stage().name;
  result.applied_stage = current_stage();
  result.shadow_cycles_without_fault = shadow_cycles_without_fault_;

  const StageDefinition* target_stage = FindStageByName(ordered_stages_, request.target_stage_name);
  if (target_stage == nullptr) {
    result.blocking_reasons.push_back("unknown stage: " + request.target_stage_name);
    return result;
  }

  if (stage_rules_.monotonic_escalation_only &&
      target_stage->ordinal > current_stage().ordinal + 1U) {
    std::ostringstream stream;
    stream << "jump stage is forbidden: current=" << current_stage().name
           << ", requested=" << target_stage->name;
    result.blocking_reasons.push_back(stream.str());
  }

  if (OutputLevelMorePermissive(current_stage().output_limits, target_stage->output_limits) &&
      stage_rules_.require_passive_or_stand_hold_before_scale_up &&
      !request.scale_up_safe_state) {
    result.blocking_reasons.push_back(
        "scale-up requires PASSIVE or STAND_HOLD acknowledgement before applying a more permissive stage.");
  }

  const std::vector<std::string> requirement_reasons =
      ValidateRequirements(*target_stage, request.verification);
  result.blocking_reasons.insert(result.blocking_reasons.end(),
                                 requirement_reasons.begin(),
                                 requirement_reasons.end());
  if (!result.blocking_reasons.empty()) {
    return result;
  }

  const StageDefinition previous_stage = current_stage();
  const std::size_t new_stage_index = static_cast<std::size_t>(
      std::distance(ordered_stages_.begin(),
                    std::find_if(ordered_stages_.begin(),
                                 ordered_stages_.end(),
                                 [&](const auto& stage) { return stage.name == target_stage->name; })));
  current_stage_index_ = new_stage_index;
  result.accepted = true;
  result.current_stage_name = current_stage().name;
  result.applied_stage = current_stage();

  if (OutputLimitsChanged(previous_stage.output_limits, current_stage().output_limits)) {
    // 输出倍率和启用关节数只能走 StageManager 统一联锁，不能被参数服务器绕过；
    // 否则日志里看到的是“同一个 stage”，实际硬件却已经被人偷偷放大输出，shadow 证据就失真了。
    std::ostringstream stream;
    stream << "stage output level changed from " << previous_stage.name << " to "
           << current_stage().name;
    ResetShadowCounter(stream.str());
    result.shadow_counter_reset = true;
    result.shadow_counter_reset_reason = last_shadow_reset_reason_;
  }

  result.shadow_cycles_without_fault = shadow_cycles_without_fault_;
  return result;
}

StageManager::ShadowCycleResult StageManager::ReportShadowCycle(const ShadowCycleInput& input) {
  ValidateClipWindow(input.clip_window);

  ShadowCycleResult result;
  std::vector<std::string> reset_reasons;

  if (last_clip_window_.has_value()) {
    const bool clip_window_changed =
        last_clip_window_->start_frame != input.clip_window.start_frame ||
        last_clip_window_->end_frame != input.clip_window.end_frame ||
        last_clip_window_->stride != input.clip_window.stride;
    if (clip_window_changed) {
      // clip 窗口一变，原来累计的 200 个 shadow 周期就不再代表“当前 reference 接入点”；
      // 如果不重新累计，系统会把旧窗口的平滑性证明错误复用于新窗口，等于绕过了重新验窗。
      reset_reasons.push_back("clip window changed from " + last_clip_window_->Summary() + " to " +
                              input.clip_window.Summary());
    }
  }

  if (input.fault_detected) {
    reset_reasons.push_back(
        "shadow fault detected; counters must restart from zero after any safety violation");
  }

  last_clip_window_ = input.clip_window;

  if (!reset_reasons.empty()) {
    ResetShadowCounter(JoinReasons(reset_reasons));
    result.counter_reset = true;
    result.reset_reason = last_shadow_reset_reason_;
  } else {
    ++shadow_cycles_without_fault_;
  }

  result.shadow_cycles_without_fault = shadow_cycles_without_fault_;
  result.required_cycles_for_current_stage = RequiredShadowCyclesForCurrentStage();
  result.current_stage_shadow_requirement_satisfied =
      result.required_cycles_for_current_stage == 0U ||
      shadow_cycles_without_fault_ >= result.required_cycles_for_current_stage;
  return result;
}

StageManager::ActiveEligibility StageManager::EvaluateBeyondMimicActiveEligibility(
    const ActiveRequest& request) const {
  ActiveEligibility result;
  result.stage = current_stage();
  result.shadow_cycles_without_fault = shadow_cycles_without_fault_;
  result.required_shadow_cycles_without_fault = RequiredShadowCyclesForActive(current_stage());

  if (result.required_shadow_cycles_without_fault == 0U) {
    result.blocking_reasons.push_back(
        "current stage does not declare BEYOND_MIMIC_ACTIVE eligibility");
  }

  if (!current_stage().output_limits.allow_motor_output) {
    result.blocking_reasons.push_back("current stage forbids motor output");
  }

  const std::vector<std::string> requirement_reasons =
      ValidateRequirements(current_stage(), request.verification);
  result.blocking_reasons.insert(result.blocking_reasons.end(),
                                 requirement_reasons.begin(),
                                 requirement_reasons.end());

  if (shadow_cycles_without_fault_ < result.required_shadow_cycles_without_fault) {
    std::ostringstream stream;
    stream << "shadow cycles without fault are insufficient: " << shadow_cycles_without_fault_
           << " < " << result.required_shadow_cycles_without_fault;
    result.blocking_reasons.push_back(stream.str());
  }

  result.allowed = result.blocking_reasons.empty();
  return result;
}

void StageManager::ValidateClipWindow(const ClipWindow& clip_window) {
  if (clip_window.end_frame < clip_window.start_frame) {
    throw std::invalid_argument("clip window end_frame must be >= start_frame");
  }
  if (clip_window.stride == 0U) {
    throw std::invalid_argument("clip window stride must be > 0");
  }
}

std::size_t StageManager::RequiredShadowCyclesForCurrentStage() const {
  const StageRequirements& requirements = current_stage().requirements;
  if (requirements.min_shadow_cycles_without_fault == 0U &&
      requirements.require_shadow_cycles_without_fault == 0U) {
    return 0U;
  }

  return std::max(stage_rules_.min_shadow_cycles_before_active_function,
                  std::max(requirements.min_shadow_cycles_without_fault,
                           requirements.require_shadow_cycles_without_fault));
}

std::size_t StageManager::RequiredShadowCyclesForActive(const StageDefinition& stage) const {
  if (stage.requirements.require_shadow_cycles_without_fault == 0U) {
    return 0U;
  }
  return std::max(stage_rules_.min_shadow_cycles_before_active_function,
                  stage.requirements.require_shadow_cycles_without_fault);
}

std::vector<std::string> StageManager::ValidateRequirements(
    const StageDefinition& stage,
    const VerificationSnapshot& verification) const {
  std::vector<std::string> reasons;
  if (stage.requirements.require_t265_verified && !verification.t265_verified) {
    reasons.push_back("T265 verification is required by " + stage.name);
  }
  if (stage.requirements.require_joint_sign_verified && !verification.joint_sign_verified) {
    reasons.push_back("joint sign verification is required by " + stage.name);
  }
  if (stage.requirements.require_npz_order_verified && !verification.npz_order_verified) {
    reasons.push_back("npz order verification is required by " + stage.name);
  }
  return reasons;
}

void StageManager::ResetShadowCounter(const std::string& reason) {
  shadow_cycles_without_fault_ = 0U;
  last_shadow_reset_reason_ = reason;
}

std::string StageManager::ClipWindow::Summary() const {
  std::ostringstream stream;
  stream << "start=" << start_frame << ", end=" << end_frame << ", stride=" << stride;
  return stream.str();
}

std::string StageManager::StageTransitionResult::Summary() const {
  std::ostringstream stream;
  stream << "accepted=" << (accepted ? "true" : "false")
         << ", requested_stage=" << requested_stage_name
         << ", previous_stage=" << previous_stage_name
         << ", current_stage=" << current_stage_name
         << ", shadow_counter_reset=" << (shadow_counter_reset ? "true" : "false")
         << ", shadow_counter_reset_reason="
         << (shadow_counter_reset_reason.empty() ? "none" : shadow_counter_reset_reason)
         << ", shadow_cycles_without_fault=" << shadow_cycles_without_fault
         << ", blocking_reasons=" << JoinReasons(blocking_reasons);
  return stream.str();
}

std::string StageManager::ShadowCycleResult::Summary() const {
  std::ostringstream stream;
  stream << "counter_reset=" << (counter_reset ? "true" : "false")
         << ", reset_reason=" << (reset_reason.empty() ? "none" : reset_reason)
         << ", shadow_cycles_without_fault=" << shadow_cycles_without_fault
         << ", required_cycles_for_current_stage=" << required_cycles_for_current_stage
         << ", current_stage_shadow_requirement_satisfied="
         << (current_stage_shadow_requirement_satisfied ? "true" : "false");
  return stream.str();
}

std::string StageManager::ActiveEligibility::Summary() const {
  std::ostringstream stream;
  stream << "allowed=" << (allowed ? "true" : "false")
         << ", stage=" << stage.name
         << ", shadow_cycles_without_fault=" << shadow_cycles_without_fault
         << ", required_shadow_cycles_without_fault=" << required_shadow_cycles_without_fault
         << ", blocking_reasons=" << JoinReasons(blocking_reasons);
  return stream.str();
}

}  // namespace zky_rl_deploy
