#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace zky_rl_deploy {

class StageManager {
 public:
  struct StageRules {
    bool monotonic_escalation_only{true};
    bool require_stage_manager_and_safety_supervisor_ack{true};
    bool require_passive_or_stand_hold_before_scale_up{true};
    std::size_t min_shadow_cycles_before_active_function{200U};
  };

  struct StageRequirements {
    bool require_t265_verified{false};
    bool require_joint_sign_verified{false};
    bool require_npz_order_verified{false};
    std::size_t min_shadow_cycles_without_fault{0U};
    std::size_t require_shadow_cycles_without_fault{0U};
    std::optional<double> max_run_time_s;
  };

  struct StageOutputLimits {
    bool allow_motor_output{false};
    std::size_t max_enabled_joints{12U};
    double action_scale{0.0};
    double kp_scale{0.0};
    double kd_scale{0.0};
  };

  struct StageDefinition {
    std::string name;
    std::size_t ordinal{0U};
    bool allow_passive_request{false};
    StageRequirements requirements;
    StageOutputLimits output_limits;
  };

  struct VerificationSnapshot {
    bool t265_verified{false};
    bool joint_sign_verified{false};
    bool npz_order_verified{false};
  };

  struct ClipWindow {
    std::size_t start_frame{0U};
    std::size_t end_frame{0U};
    std::size_t stride{1U};

    std::string Summary() const;
  };

  struct StageRequest {
    std::string target_stage_name;
    VerificationSnapshot verification;
    bool scale_up_safe_state{true};
  };

  struct ActiveRequest {
    VerificationSnapshot verification;
  };

  struct StageTransitionResult {
    bool accepted{false};
    std::string requested_stage_name;
    std::string previous_stage_name;
    std::string current_stage_name;
    std::vector<std::string> blocking_reasons;
    bool shadow_counter_reset{false};
    std::string shadow_counter_reset_reason;
    std::size_t shadow_cycles_without_fault{0U};
    StageDefinition applied_stage;

    std::string Summary() const;
  };

  struct ShadowCycleInput {
    ClipWindow clip_window;
    bool fault_detected{false};
  };

  struct ShadowCycleResult {
    bool counter_reset{false};
    std::string reset_reason;
    std::size_t shadow_cycles_without_fault{0U};
    std::size_t required_cycles_for_current_stage{0U};
    bool current_stage_shadow_requirement_satisfied{true};

    std::string Summary() const;
  };

  struct ActiveEligibility {
    bool allowed{false};
    std::vector<std::string> blocking_reasons;
    std::size_t shadow_cycles_without_fault{0U};
    std::size_t required_shadow_cycles_without_fault{0U};
    StageDefinition stage;

    std::string Summary() const;
  };

  static StageManager LoadFromYaml(const std::string& yaml_path);

  const StageRules& stage_rules() const { return stage_rules_; }
  const StageDefinition& current_stage() const { return ordered_stages_[current_stage_index_]; }
  const std::vector<StageDefinition>& ordered_stages() const { return ordered_stages_; }
  std::size_t shadow_cycles_without_fault() const { return shadow_cycles_without_fault_; }
  const std::optional<ClipWindow>& last_clip_window() const { return last_clip_window_; }
  const std::string& last_shadow_reset_reason() const { return last_shadow_reset_reason_; }

  StageTransitionResult RequestStage(const StageRequest& request);
  ShadowCycleResult ReportShadowCycle(const ShadowCycleInput& input);
  ActiveEligibility EvaluateBeyondMimicActiveEligibility(
      const ActiveRequest& request) const;

 private:
  StageManager(StageRules stage_rules, std::vector<StageDefinition> ordered_stages);

  static void ValidateClipWindow(const ClipWindow& clip_window);

  std::size_t RequiredShadowCyclesForCurrentStage() const;
  std::size_t RequiredShadowCyclesForActive(const StageDefinition& stage) const;
  std::vector<std::string> ValidateRequirements(const StageDefinition& stage,
                                                const VerificationSnapshot& verification) const;
  void ResetShadowCounter(const std::string& reason);

  StageRules stage_rules_;
  std::vector<StageDefinition> ordered_stages_;
  std::size_t current_stage_index_{0U};
  std::size_t shadow_cycles_without_fault_{0U};
  std::optional<ClipWindow> last_clip_window_;
  std::string last_shadow_reset_reason_;
};

}  // namespace zky_rl_deploy
