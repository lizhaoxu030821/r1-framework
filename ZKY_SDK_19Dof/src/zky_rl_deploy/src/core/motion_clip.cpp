#include "zky_rl_deploy/core/motion_clip.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace zky_rl_deploy {

MotionClip::MotionClip(MotionClipConfig config,
                       JointMapper joint_mapper,
                       std::shared_ptr<const MotionFrameProvider> frame_provider)
    : config_(std::move(config)),
      joint_mapper_(std::move(joint_mapper)),
      frame_provider_(std::move(frame_provider)) {
  ValidateConfig();
}

MotionClipEntryCheckResult MotionClip::CheckEntry(
    const std::vector<double>& current_measured_joint_pos_policy_order) const {
  MotionClipEntryCheckResult result;
  result.window_start_command = BuildCommandForWindowOffset(0U);

  if (current_measured_joint_pos_policy_order.size() != joint_mapper_.policy_order().size()) {
    std::ostringstream stream;
    stream << "entry joint vector size mismatch: expected "
           << joint_mapper_.policy_order().size() << ", got "
           << current_measured_joint_pos_policy_order.size();
    result.reason = stream.str();
    return result;
  }

  double max_joint_error = 0.0;
  std::string worst_joint_name;
  const auto& policy_order = joint_mapper_.policy_order();
  for (std::size_t joint_index = 0; joint_index < policy_order.size(); ++joint_index) {
    const double joint_error = std::abs(
      result.window_start_command.joint_pos_policy_order[joint_index] -
      current_measured_joint_pos_policy_order[joint_index]);
    if (joint_error > max_joint_error) {
      max_joint_error = joint_error;
      worst_joint_name = policy_order[joint_index];
    }
  }

  result.max_joint_error_rad = max_joint_error;
  result.worst_joint_name = worst_joint_name;

  if (max_joint_error > config_.clip_entry_joint_error_rad) {
    // 这里显式拒绝“先插值到窗口第一帧再接管”的做法。
    // 因为 motion clip 的窗口起点可能本身就是非稳态动作，相当于在已使能状态下强行把机器人拖向一个动态姿态，
    // 这会绕过 shadow 阶段对平滑衔接窗口的筛选要求，因此必须直接拒绝进入 armed。
    std::ostringstream stream;
    stream << "clip entry joint error exceeds threshold at " << worst_joint_name << ": "
           << max_joint_error << " rad > " << config_.clip_entry_joint_error_rad
           << " rad; refusing BEYOND_MIMIC_ARMED without interpolation";
      result.reason = stream.str();
      return result;
  }

  result.accepted = true;
  result.reason = "entry joint error within threshold";
  return result;
}

MotionClipStepResult MotionClip::ResetToWindowStart() {
  current_window_offset_ = 0U;
  phase_accumulator_ = 0.0;
  stale_frame_count_ = 0U;
  stale_hold_time_ms_ = 0;
  max_frame_jump_observed_ = 0U;
  has_started_ = true;
  last_command_ = BuildCommandForWindowOffset(current_window_offset_);
  return BuildStepResultFromCurrentCommand(0U, false, false, "reset_to_window_start");
}

MotionClipStepResult MotionClip::Advance(bool source_frame_stale) {
  if (!has_started_) {
    return ResetToWindowStart();
  }

  if (source_frame_stale) {
    if (config_.stale_frame_behavior != StaleFrameBehavior::Hold) {
      MotionClipStepResult result = BuildStepResultFromCurrentCommand(
          0U, false, true, "unsupported stale_frame_behavior requires PASSIVE");
      result.stale_frame_count = stale_frame_count_;
      result.stale_hold_time_ms = stale_hold_time_ms_;
      return result;
    }

    ++stale_frame_count_;
    stale_hold_time_ms_ += config_.nominal_period_ms;

    std::string reason = "holding last motion frame due to stale source frame";
    bool request_passive = false;
    if (stale_frame_count_ > config_.max_consecutive_stale_frames) {
      request_passive = true;
      std::ostringstream stream;
      stream << "stale frame count exceeded limit: " << stale_frame_count_ << " > "
             << config_.max_consecutive_stale_frames;
      reason = stream.str();
    } else if (stale_hold_time_ms_ > config_.max_hold_time_ms) {
      // stale hold 超时必须退出到 PASSIVE，因为继续保持过期 reference 相当于在盲飞。
      // 一旦真实状态已经偏离，而上层还在复用旧动作帧，就会让 shadow 结果和实际闭环行为逐渐脱节。
      request_passive = true;
      std::ostringstream stream;
      stream << "stale hold time exceeded limit: " << stale_hold_time_ms_ << " ms > "
             << config_.max_hold_time_ms << " ms";
      reason = stream.str();
    }

    return BuildStepResultFromCurrentCommand(0U, false, request_passive, reason);
  }

  stale_frame_count_ = 0U;
  stale_hold_time_ms_ = 0;

  phase_accumulator_ += config_.playback_rate;
  const std::size_t requested_frame_jump = static_cast<std::size_t>(std::floor(phase_accumulator_));
  phase_accumulator_ -= static_cast<double>(requested_frame_jump);

  if (requested_frame_jump > config_.max_frame_jump) {
    max_frame_jump_observed_ = std::max(max_frame_jump_observed_, requested_frame_jump);
    std::ostringstream stream;
    stream << "requested frame jump exceeded max_frame_jump: " << requested_frame_jump << " > "
           << config_.max_frame_jump;
    return BuildStepResultFromCurrentCommand(0U, false, true, stream.str());
  }

  bool looped = false;
  if (requested_frame_jump > 0U) {
    max_frame_jump_observed_ = std::max(max_frame_jump_observed_, requested_frame_jump);

    if (config_.loop) {
      const std::size_t next_offset =
          (current_window_offset_ + requested_frame_jump) % window_frame_count_;
      looped = current_window_offset_ + requested_frame_jump >= window_frame_count_;
      current_window_offset_ = next_offset;
    } else {
      const std::size_t unclamped_offset = current_window_offset_ + requested_frame_jump;
      current_window_offset_ = std::min(unclamped_offset, window_frame_count_ - 1U);
      looped = false;
    }
  }

  last_command_ = BuildCommandForWindowOffset(current_window_offset_);
  return BuildStepResultFromCurrentCommand(
      requested_frame_jump, looped, false, "advanced_motion_clip");
}

MotionClipCommand MotionClip::BuildCommandForWindowOffset(std::size_t window_offset) const {
  const std::size_t frame_index = ResolveWindowFrameIndex(window_offset);
  const MotionFrame frame = frame_provider_->GetFrame(frame_index);

  MotionClipCommand command;
  command.source_frame_index = frame_index;
  command.joint_pos_policy_order =
      joint_mapper_.NpzToPolicyReference(frame.joint_pos_npz_order);
  command.joint_vel_policy_order =
      joint_mapper_.NpzToPolicyReference(frame.joint_vel_npz_order);
  return command;
}

std::size_t MotionClip::ResolveWindowFrameIndex(std::size_t window_offset) const {
  if (window_offset >= window_frame_count_) {
    throw std::out_of_range("window_offset exceeds configured motion clip window.");
  }
  return config_.start_frame + window_offset * config_.clip_stride;
}

MotionClipStepResult MotionClip::BuildStepResultFromCurrentCommand(
    std::size_t frame_jump, bool looped, bool request_passive, const std::string& reason) const {
  MotionClipStepResult result;
  result.command = last_command_;
  result.frame_jump = frame_jump;
  result.stale_frame_count = stale_frame_count_;
  result.stale_hold_time_ms = stale_hold_time_ms_;
  result.max_frame_jump_observed = max_frame_jump_observed_;
  result.looped = looped;
  result.request_passive = request_passive;
  result.reason = reason;
  return result;
}

void MotionClip::ValidateConfig() {
  if (!frame_provider_) {
    throw std::invalid_argument("MotionFrameProvider must not be null.");
  }
  if (frame_provider_->FrameCount() == 0U) {
    throw std::invalid_argument("MotionFrameProvider must contain at least one frame.");
  }
  if (config_.clip_stride == 0U) {
    throw std::invalid_argument("clip_stride must be greater than zero.");
  }
  if (config_.playback_rate <= 0.0) {
    throw std::invalid_argument("playback_rate must be greater than zero.");
  }
  if (config_.max_frame_jump == 0U) {
    throw std::invalid_argument("max_frame_jump must be greater than zero.");
  }
  if (config_.max_hold_time_ms < 0) {
    throw std::invalid_argument("max_hold_time_ms must be non-negative.");
  }
  if (config_.nominal_period_ms <= 0) {
    throw std::invalid_argument("nominal_period_ms must be greater than zero.");
  }
  if (config_.start_frame >= frame_provider_->FrameCount()) {
    throw std::invalid_argument("start_frame exceeds provider frame count.");
  }

  resolved_end_frame_ = config_.end_frame.value_or(frame_provider_->FrameCount() - 1U);
  if (resolved_end_frame_ >= frame_provider_->FrameCount()) {
    throw std::invalid_argument("end_frame exceeds provider frame count.");
  }
  if (resolved_end_frame_ < config_.start_frame) {
    throw std::invalid_argument("end_frame must be greater than or equal to start_frame.");
  }

  window_frame_count_ =
      ((resolved_end_frame_ - config_.start_frame) / config_.clip_stride) + 1U;
  if (window_frame_count_ == 0U) {
    throw std::invalid_argument("motion clip window must contain at least one frame.");
  }
}

}  // namespace zky_rl_deploy
