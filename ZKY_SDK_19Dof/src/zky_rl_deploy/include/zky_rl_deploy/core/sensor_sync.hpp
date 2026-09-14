#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace zky_rl_deploy {

class SensorSync {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;
  using Duration = Clock::duration;

  enum class Severity { kOk, kWarn, kPassive };
  enum class OutputMode { kDryRun, kShadow, kActiveHardware };

  struct Config {
    std::chrono::milliseconds joint_state_timeout{20};
    std::chrono::milliseconds t265_odom_timeout{40};
    std::chrono::milliseconds t265_imu_timeout{20};
    std::chrono::milliseconds sensor_sync_warn_dt{20};
    std::chrono::milliseconds sensor_sync_passive_dt{40};
    bool t265_extrinsic_verified{false};
    bool allow_dry_run_with_unverified_extrinsic{true};
    bool allow_shadow_with_unverified_extrinsic{true};
    bool allow_motor_output_with_unverified_extrinsic{false};
  };

  struct JointFeedbackSnapshot {
    TimePoint stamp{};
    std::vector<double> joint_pos_policy_order;
    std::vector<double> joint_vel_policy_order;
  };

  struct T265OdomSnapshot {
    TimePoint stamp{};
    std::array<double, 3> position_flu_m{};
    // /camera/odom/sample 的 position 方向和机器人 FLU 一致，只能说明平移轴方向已经对齐，
    // 不代表四元数就已经是 pelvis 姿态。相机安装角、IMU 定义和固连误差仍可能引入一个固定旋转，
    // 所以这里只有在 Stage 0 验证过外参后，才能把姿态当成 pelvis frame 的输入继续往下游传。
    std::array<double, 4> pelvis_orientation_xyzw{};
  };

  struct T265ImuSnapshot {
    TimePoint stamp{};
    // 这里保存的是已经解释为 pelvis/body frame 的 base_ang_vel，单位 rad/s；
    // 轴向必须和训练仿真一致，不能把还停留在 T265 本体坐标系里的原始角速度直接塞进观测。
    std::array<double, 3> base_ang_vel_body_rad_s{};
  };

  struct LatestSnapshotBundle {
    JointFeedbackSnapshot joint_feedback;
    T265OdomSnapshot t265_odom;
    T265ImuSnapshot t265_imu;
  };

  struct OutputGateDecision {
    OutputMode requested_output_mode{OutputMode::kDryRun};
    bool allowed{true};
    std::vector<std::string> blocking_reasons;
  };

  struct Evaluation {
    Severity severity{Severity::kOk};
    std::vector<std::string> warn_reasons;
    std::vector<std::string> passive_reasons;
    bool has_complete_snapshot{false};
    std::optional<LatestSnapshotBundle> latest_complete_snapshot;
    OutputGateDecision output_gate;

    std::string Summary() const;
  };

  explicit SensorSync(Config config);

  void UpdateJointFeedback(JointFeedbackSnapshot snapshot);
  void UpdateT265Odom(T265OdomSnapshot snapshot);
  void UpdateT265Imu(T265ImuSnapshot snapshot);

  bool HasCompleteSnapshot() const;

  Evaluation Evaluate(TimePoint now, OutputMode requested_output_mode) const;

  bool AllowsOutputMode(OutputMode output_mode) const;
  bool AllowsActiveHardwareOutputWithCurrentCalibration() const;
  void ThrowIfT265ExtrinsicUnverifiedForActiveHardwareOutput() const;

  const Config& config() const { return config_; }
  bool t265_extrinsic_verified() const { return config_.t265_extrinsic_verified; }
  void SetT265ExtrinsicVerified(bool verified) { config_.t265_extrinsic_verified = verified; }

  const std::optional<JointFeedbackSnapshot>& latest_joint_feedback() const {
    return latest_joint_feedback_;
  }
  const std::optional<T265OdomSnapshot>& latest_t265_odom() const { return latest_t265_odom_; }
  const std::optional<T265ImuSnapshot>& latest_t265_imu() const { return latest_t265_imu_; }

  static const char* ToString(Severity severity);
  static const char* ToString(OutputMode output_mode);

 private:
  static void ValidateConfig(const Config& config);
  static void ValidateJointFeedbackSnapshot(const JointFeedbackSnapshot& snapshot);
  static void ValidateT265OdomSnapshot(const T265OdomSnapshot& snapshot);
  static void ValidateT265ImuSnapshot(const T265ImuSnapshot& snapshot);

  Config config_;
  std::optional<JointFeedbackSnapshot> latest_joint_feedback_;
  std::optional<T265OdomSnapshot> latest_t265_odom_;
  std::optional<T265ImuSnapshot> latest_t265_imu_;
};

}  // namespace zky_rl_deploy
