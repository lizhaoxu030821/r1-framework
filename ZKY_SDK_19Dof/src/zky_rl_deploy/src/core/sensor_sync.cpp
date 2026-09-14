#include "zky_rl_deploy/core/sensor_sync.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace zky_rl_deploy {
namespace {

using Milliseconds = std::chrono::milliseconds;

template <typename Container>
void ValidateFiniteValues(const Container& values, const char* field_name) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!std::isfinite(values[index])) {
      std::ostringstream stream;
      stream << field_name << " contains non-finite value at index " << index;
      throw std::invalid_argument(stream.str());
    }
  }
}

SensorSync::Duration AbsDuration(SensorSync::Duration duration) {
  return duration >= SensorSync::Duration::zero() ? duration : -duration;
}

Milliseconds CeilToMilliseconds(SensorSync::Duration duration) {
  const Milliseconds truncated = std::chrono::duration_cast<Milliseconds>(duration);
  if (truncated < duration) {
    return truncated + Milliseconds(1);
  }
  return truncated;
}

std::string FormatMilliseconds(SensorSync::Duration duration) {
  const Milliseconds milliseconds = CeilToMilliseconds(AbsDuration(duration));
  return std::to_string(milliseconds.count()) + " ms";
}

void AddSnapshotMissingReason(const char* label, std::vector<std::string>* passive_reasons) {
  passive_reasons->push_back(std::string("missing latest ") + label + " snapshot");
}

void AppendTimeoutReason(const SensorSync::TimePoint& now,
                         const SensorSync::TimePoint& stamp,
                         const Milliseconds& timeout,
                         const char* label,
                         std::vector<std::string>* passive_reasons) {
  const SensorSync::Duration age = now - stamp;
  if (age < SensorSync::Duration::zero()) {
    passive_reasons->push_back(
        std::string(label) +
        " stamp is newer than the control clock; all snapshots must share one monotonic time base.");
    return;
  }

  if (age > timeout) {
    std::ostringstream stream;
    stream << label << " age " << FormatMilliseconds(age)
           << " exceeds timeout " << timeout.count() << " ms";
    passive_reasons->push_back(stream.str());
  }
}

void AppendDtReason(const SensorSync::Duration& dt,
                    const char* label,
                    const SensorSync::Config& config,
                    std::vector<std::string>* warn_reasons,
                    std::vector<std::string>* passive_reasons) {
  if (dt > config.sensor_sync_passive_dt) {
    std::ostringstream stream;
    stream << label << " dt " << FormatMilliseconds(dt)
           << " exceeds passive threshold " << config.sensor_sync_passive_dt.count() << " ms";
    passive_reasons->push_back(stream.str());
    return;
  }

  if (dt > config.sensor_sync_warn_dt) {
    std::ostringstream stream;
    stream << label << " dt " << FormatMilliseconds(dt)
           << " exceeds warn threshold " << config.sensor_sync_warn_dt.count() << " ms";
    warn_reasons->push_back(stream.str());
  }
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

std::string BuildUnverifiedExtrinsicReason(SensorSync::OutputMode output_mode) {
  std::ostringstream stream;
  stream << "T265 extrinsic is unverified; verified=false calibration can stay in dry_run/shadow "
         << "but cannot enter " << SensorSync::ToString(output_mode);
  return stream.str();
}

}  // namespace

SensorSync::SensorSync(Config config) : config_(std::move(config)) { ValidateConfig(config_); }

void SensorSync::UpdateJointFeedback(JointFeedbackSnapshot snapshot) {
  ValidateJointFeedbackSnapshot(snapshot);
  latest_joint_feedback_ = std::move(snapshot);
}

void SensorSync::UpdateT265Odom(T265OdomSnapshot snapshot) {
  ValidateT265OdomSnapshot(snapshot);
  latest_t265_odom_ = std::move(snapshot);
}

void SensorSync::UpdateT265Imu(T265ImuSnapshot snapshot) {
  ValidateT265ImuSnapshot(snapshot);
  latest_t265_imu_ = std::move(snapshot);
}

bool SensorSync::HasCompleteSnapshot() const {
  return latest_joint_feedback_.has_value() && latest_t265_odom_.has_value() &&
         latest_t265_imu_.has_value();
}

SensorSync::Evaluation SensorSync::Evaluate(TimePoint now,
                                            OutputMode requested_output_mode) const {
  Evaluation evaluation;
  evaluation.output_gate.requested_output_mode = requested_output_mode;

  if (!latest_joint_feedback_) {
    AddSnapshotMissingReason("joint_state", &evaluation.passive_reasons);
  }
  if (!latest_t265_odom_) {
    AddSnapshotMissingReason("t265 odom", &evaluation.passive_reasons);
  }
  if (!latest_t265_imu_) {
    AddSnapshotMissingReason("t265 imu", &evaluation.passive_reasons);
  }

  if (latest_joint_feedback_) {
    AppendTimeoutReason(now,
                        latest_joint_feedback_->stamp,
                        config_.joint_state_timeout,
                        "joint_state",
                        &evaluation.passive_reasons);
  }
  if (latest_t265_odom_) {
    AppendTimeoutReason(now,
                        latest_t265_odom_->stamp,
                        config_.t265_odom_timeout,
                        "t265 odom",
                        &evaluation.passive_reasons);
  }
  if (latest_t265_imu_) {
    AppendTimeoutReason(now,
                        latest_t265_imu_->stamp,
                        config_.t265_imu_timeout,
                        "t265 imu",
                        &evaluation.passive_reasons);
  }

  if (HasCompleteSnapshot()) {
    evaluation.has_complete_snapshot = true;
    evaluation.latest_complete_snapshot = LatestSnapshotBundle{
        *latest_joint_feedback_, *latest_t265_odom_, *latest_t265_imu_};

    // 观测里会把关节、姿态和角速度当成“同一时刻”的状态一起拼接。
    // 如果这些时间戳不同步，策略拿到的就会变成“旧姿态 + 新角速度 + 中间时刻关节”的混合快照，
    // 直接破坏训练时的观测语义，所以这里必须把 pairwise dt 显式检查并在超限时阻断 active。
    AppendDtReason(AbsDuration(latest_joint_feedback_->stamp - latest_t265_odom_->stamp),
                   "joint_state<->t265_odom",
                   config_,
                   &evaluation.warn_reasons,
                   &evaluation.passive_reasons);
    AppendDtReason(AbsDuration(latest_joint_feedback_->stamp - latest_t265_imu_->stamp),
                   "joint_state<->t265_imu",
                   config_,
                   &evaluation.warn_reasons,
                   &evaluation.passive_reasons);
    AppendDtReason(AbsDuration(latest_t265_odom_->stamp - latest_t265_imu_->stamp),
                   "t265_odom<->t265_imu",
                   config_,
                   &evaluation.warn_reasons,
                   &evaluation.passive_reasons);
  }

  if (!evaluation.passive_reasons.empty()) {
    evaluation.severity = Severity::kPassive;
  } else if (!evaluation.warn_reasons.empty()) {
    evaluation.severity = Severity::kWarn;
  } else {
    evaluation.severity = Severity::kOk;
  }

  evaluation.output_gate.allowed = true;
  if (requested_output_mode == OutputMode::kActiveHardware &&
      !evaluation.passive_reasons.empty()) {
    evaluation.output_gate.allowed = false;
    evaluation.output_gate.blocking_reasons = evaluation.passive_reasons;
  }

  if (!AllowsOutputMode(requested_output_mode)) {
    evaluation.output_gate.allowed = false;
    evaluation.output_gate.blocking_reasons.push_back(
        BuildUnverifiedExtrinsicReason(requested_output_mode));
  }

  return evaluation;
}

bool SensorSync::AllowsOutputMode(OutputMode output_mode) const {
  if (config_.t265_extrinsic_verified) {
    return true;
  }

  switch (output_mode) {
    case OutputMode::kDryRun:
      return config_.allow_dry_run_with_unverified_extrinsic;
    case OutputMode::kShadow:
      return config_.allow_shadow_with_unverified_extrinsic;
    case OutputMode::kActiveHardware:
      return config_.allow_motor_output_with_unverified_extrinsic;
  }

  return false;
}

bool SensorSync::AllowsActiveHardwareOutputWithCurrentCalibration() const {
  return AllowsOutputMode(OutputMode::kActiveHardware);
}

void SensorSync::ThrowIfT265ExtrinsicUnverifiedForActiveHardwareOutput() const {
  if (AllowsActiveHardwareOutputWithCurrentCalibration()) {
    return;
  }

  // T265 的平移方向已经对齐，并不意味着姿态外参也已经被三轴验证。
  // 在 verified=false 时继续放开真实硬件输出，会把未确认的 pelvis 姿态和角速度送进策略，
  // 这相当于让机器人在“影子观测”上做真动作，因此只能停留在 dry_run/shadow。
  throw std::logic_error(BuildUnverifiedExtrinsicReason(OutputMode::kActiveHardware));
}

const char* SensorSync::ToString(Severity severity) {
  switch (severity) {
    case Severity::kOk:
      return "ok";
    case Severity::kWarn:
      return "warn";
    case Severity::kPassive:
      return "passive";
  }

  return "unknown";
}

const char* SensorSync::ToString(OutputMode output_mode) {
  switch (output_mode) {
    case OutputMode::kDryRun:
      return "dry_run";
    case OutputMode::kShadow:
      return "shadow";
    case OutputMode::kActiveHardware:
      return "active_hardware";
  }

  return "unknown";
}

std::string SensorSync::Evaluation::Summary() const {
  std::ostringstream stream;
  stream << "severity=" << SensorSync::ToString(severity)
         << ", warn_reasons=" << JoinReasons(warn_reasons)
         << ", passive_reasons=" << JoinReasons(passive_reasons)
         << ", requested_output=" << SensorSync::ToString(output_gate.requested_output_mode)
         << ", output_allowed=" << (output_gate.allowed ? "true" : "false")
         << ", output_blocking_reasons=" << JoinReasons(output_gate.blocking_reasons);
  return stream.str();
}

void SensorSync::ValidateConfig(const Config& config) {
  if (config.joint_state_timeout <= Milliseconds::zero()) {
    throw std::invalid_argument("joint_state_timeout must be > 0 ms");
  }
  if (config.t265_odom_timeout <= Milliseconds::zero()) {
    throw std::invalid_argument("t265_odom_timeout must be > 0 ms");
  }
  if (config.t265_imu_timeout <= Milliseconds::zero()) {
    throw std::invalid_argument("t265_imu_timeout must be > 0 ms");
  }
  if (config.sensor_sync_warn_dt <= Milliseconds::zero()) {
    throw std::invalid_argument("sensor_sync_warn_dt must be > 0 ms");
  }
  if (config.sensor_sync_passive_dt < config.sensor_sync_warn_dt) {
    throw std::invalid_argument(
        "sensor_sync_passive_dt must be >= sensor_sync_warn_dt");
  }
}

void SensorSync::ValidateJointFeedbackSnapshot(const JointFeedbackSnapshot& snapshot) {
  if (snapshot.joint_pos_policy_order.empty()) {
    throw std::invalid_argument("joint_pos_policy_order must not be empty");
  }
  if (snapshot.joint_pos_policy_order.size() != snapshot.joint_vel_policy_order.size()) {
    throw std::invalid_argument(
        "joint_pos_policy_order and joint_vel_policy_order must have identical sizes");
  }

  ValidateFiniteValues(snapshot.joint_pos_policy_order, "joint_pos_policy_order");
  ValidateFiniteValues(snapshot.joint_vel_policy_order, "joint_vel_policy_order");
}

void SensorSync::ValidateT265OdomSnapshot(const T265OdomSnapshot& snapshot) {
  ValidateFiniteValues(snapshot.position_flu_m, "position_flu_m");
  ValidateFiniteValues(snapshot.pelvis_orientation_xyzw, "pelvis_orientation_xyzw");
}

void SensorSync::ValidateT265ImuSnapshot(const T265ImuSnapshot& snapshot) {
  ValidateFiniteValues(snapshot.base_ang_vel_body_rad_s, "base_ang_vel_body_rad_s");
}

}  // namespace zky_rl_deploy
