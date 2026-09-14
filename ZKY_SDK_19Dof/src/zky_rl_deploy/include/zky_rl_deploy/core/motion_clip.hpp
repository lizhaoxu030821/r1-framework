#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "zky_rl_deploy/core/joint_mapper.hpp"

namespace zky_rl_deploy {

struct MotionFrame {
  std::vector<double> joint_pos_npz_order;
  std::vector<double> joint_vel_npz_order;
};

// 当前先把 NPZ 读取能力抽象成 provider 接口。
// 这样 MotionClip 可以先把窗口、安全入口和 stale 行为定义清楚，
// 后续无论数据来自 NPZ、离线缓存还是别的格式，都不会把核心状态机逻辑绑死在某个第三方解析库上。
class MotionFrameProvider {
 public:
  virtual ~MotionFrameProvider() = default;

  virtual std::size_t FrameCount() const = 0;
  virtual MotionFrame GetFrame(std::size_t frame_index) const = 0;
};

enum class StaleFrameBehavior {
  Hold,
  Catchup,
  Passive,
};

struct MotionClipConfig {
  std::size_t start_frame{0U};
  std::optional<std::size_t> end_frame;
  std::size_t clip_stride{1U};
  bool loop{true};
  double playback_rate{1.0};
  std::size_t max_frame_jump{1U};
  StaleFrameBehavior stale_frame_behavior{StaleFrameBehavior::Hold};
  std::size_t max_consecutive_stale_frames{3U};
  int max_hold_time_ms{100};
  double clip_entry_joint_error_rad{0.1};
  int nominal_period_ms{20};
};

struct MotionClipCommand {
  std::size_t source_frame_index{0U};
  std::vector<double> joint_pos_policy_order;
  std::vector<double> joint_vel_policy_order;
};

struct MotionClipEntryCheckResult {
  bool accepted{false};
  std::string reason;
  double max_joint_error_rad{0.0};
  std::string worst_joint_name;
  MotionClipCommand window_start_command;
};

struct MotionClipStepResult {
  MotionClipCommand command;
  std::size_t frame_jump{0U};
  std::size_t stale_frame_count{0U};
  int stale_hold_time_ms{0};
  std::size_t max_frame_jump_observed{0U};
  bool looped{false};
  bool request_passive{false};
  std::string reason;
};

class MotionClip {
 public:
  MotionClip(MotionClipConfig config,
             JointMapper joint_mapper,
             std::shared_ptr<const MotionFrameProvider> frame_provider);

  MotionClipEntryCheckResult CheckEntry(
      const std::vector<double>& current_measured_joint_pos_policy_order) const;

  MotionClipStepResult ResetToWindowStart();
  MotionClipStepResult Advance(bool source_frame_stale);

 private:
  MotionClipCommand BuildCommandForWindowOffset(std::size_t window_offset) const;
  std::size_t ResolveWindowFrameIndex(std::size_t window_offset) const;
  MotionClipStepResult BuildStepResultFromCurrentCommand(std::size_t frame_jump,
                                                         bool looped,
                                                         bool request_passive,
                                                         const std::string& reason) const;
  void ValidateConfig();

  MotionClipConfig config_;
  JointMapper joint_mapper_;
  std::shared_ptr<const MotionFrameProvider> frame_provider_;
  std::size_t resolved_end_frame_{0U};
  std::size_t window_frame_count_{0U};

  std::size_t current_window_offset_{0U};
  double phase_accumulator_{0.0};
  std::size_t stale_frame_count_{0U};
  int stale_hold_time_ms_{0};
  std::size_t max_frame_jump_observed_{0U};
  bool has_started_{false};
  MotionClipCommand last_command_;
};

}  // namespace zky_rl_deploy
