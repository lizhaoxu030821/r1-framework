#pragma once

#include <array>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <nav_msgs/Odometry.h>
#include <ros/node_handle.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>

#include "zky_rl_deploy/core/joystick_mapper.hpp"
#include "zky_rl_deploy/core/joint_mapper.hpp"
#include "zky_rl_deploy/core/motion_clip.hpp"
#include "zky_rl_deploy/core/policy_runtime.hpp"
#include "zky_rl_deploy/core/safety_gate.hpp"
#include "zky_rl_deploy/core/sensor_sync.hpp"
#include "zky_rl_deploy/core/t265_extrinsics.hpp"

namespace zky_rl_deploy {

namespace ros_adapter {

struct TopicConfig {
  std::string joy_topic{"/joy"};
  std::string joint_state_topic{"/joint_states"};
  std::string t265_odom_topic{"/camera/odom/sample"};
  std::string t265_imu_topic{"/camera/imu"};
  std::string motor_params_topic{"/motor_params"};
  std::string safety_status_topic{"/zky/safety_status"};
  std::string control_mode_topic{"/zky/control_mode"};
  std::string output_gate_topic{"/zky/output_gate"};
  std::string beyond_mimic_debug_topic{"/zky/beyond_mimic_debug"};
  std::string shadow_motor_params_topic{"/zky/shadow_motor_params"};
  std::string current_policy_topic{"/current_policy"};
  std::string fsm_request_topic{"/fsm_request"};
  std::string joy_buttons_snapshot_topic{"/joy_buttons_snapshot"};
};

struct JoySemanticMapping {
  bool raw_mapping_confirmed{false};
  int start_button_index{-1};
  int back_button_index{-1};
  int home_button_index{-1};
  int rb_button_index{-1};
  int a_button_index{-1};
  int x_button_index{-1};
  int b_button_index{-1};
  int y_button_index{-1};
};

struct RobotJointsConfig {
  JointMapper::JointOrder policy_order;
  JointMapper::JointOrder hardware_order;
  JointMapper::JointOrder npz_order;
  JointMapper::SignConvention sign_convention;
  JointMapper::SignVerification sign_verification;
  bool npz_order_verified{false};
};

struct PolicyConfig {
  PolicyRuntimeConfig runtime_config;
  int nominal_control_period_ms{20};
  std::vector<double> audit_default_joint_pos;
  std::vector<double> audit_joint_stiffness;
  std::vector<double> audit_joint_damping;
};

struct MotionClipFileConfig {
  MotionClipConfig runtime_config;
  std::string npz_path;
  std::string json_fallback_path;
};

struct RuntimeConfig {
  std::string package_root;
  std::string workspace_root;
  TopicConfig topics;
  JoySemanticMapping joy_mapping;
  RobotJointsConfig robot_joints;
  PolicyConfig policy;
  MotionClipFileConfig motion_clip;
  std::optional<std::vector<double>> stand_pose_policy_override;
  SensorSync::Config sensor_sync;
  SafetyGate::Config safety_gate;
  T265ExtrinsicConfig t265_extrinsic;
  std::string bringup_stages_path;
  std::string desired_stage_name{"stage0_software"};
  std::chrono::milliseconds nominal_control_period{20};
  std::vector<std::string> startup_warnings;
};

struct InputSnapshot {
  bool joy_received{false};
  bool joy_semantic_mapping_ready{false};
  JoystickState joystick_state;
  std::string joy_summary;

  bool joint_feedback_received{false};
  std::vector<double> joint_pos_hardware_order;
  std::vector<double> joint_vel_hardware_order;
  std::optional<SensorSync::JointFeedbackSnapshot> joint_feedback_policy_order;
  std::optional<SensorSync::T265OdomSnapshot> t265_odom;
  std::optional<SensorSync::T265ImuSnapshot> t265_imu;
  std::vector<std::string> warnings;
};

class RuntimeRosAdapter {
 public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  RuntimeRosAdapter(const ::ros::NodeHandle& nh,
                    const ::ros::NodeHandle& private_nh,
                    const RuntimeConfig& config,
                    const JointMapper& joint_mapper);

  InputSnapshot GetSnapshot() const;
  std::string PollDesiredStageName() const;

  void PublishSafetyStatus(const std::string& summary) const;
  void PublishControlMode(const std::string& summary) const;
  void PublishOutputGate(const std::string& summary) const;
  void PublishBeyondMimicDebug(const std::string& summary) const;
  void PublishMotorParams(const std::vector<double>& motor_params) const;
  void PublishShadowMotorParams(const std::vector<double>& shadow_motor_params) const;
  void PublishCurrentPolicyTrace(const std::string& current_policy) const;
  void PublishFsmRequestTrace(const std::string& fsm_request) const;
  void PublishJoyButtonsSnapshot(const JoystickState& joystick_state) const;

 private:
  void JoyCallback(const sensor_msgs::Joy::ConstPtr& message);
  void JointStateCallback(const sensor_msgs::JointState::ConstPtr& message);
  void T265OdomCallback(const nav_msgs::Odometry::ConstPtr& message);
  void T265ImuCallback(const sensor_msgs::Imu::ConstPtr& message);

  JoystickState BuildJoystickStateFromRawMessage(const sensor_msgs::Joy& message,
                                                 bool* semantic_mapping_ready,
                                                 std::string* summary) const;

  ::ros::NodeHandle nh_;
  ::ros::NodeHandle private_nh_;
  TopicConfig topics_;
  JoySemanticMapping joy_mapping_;
  JointMapper::JointOrder hardware_order_;
  const JointMapper& joint_mapper_;
  T265ExtrinsicConfig t265_extrinsic_;

  ::ros::Subscriber joy_subscriber_;
  ::ros::Subscriber joint_state_subscriber_;
  ::ros::Subscriber t265_odom_subscriber_;
  ::ros::Subscriber t265_imu_subscriber_;

  ::ros::Publisher safety_status_publisher_;
  ::ros::Publisher control_mode_publisher_;
  ::ros::Publisher output_gate_publisher_;
  ::ros::Publisher beyond_mimic_debug_publisher_;
  ::ros::Publisher motor_params_publisher_;
  ::ros::Publisher shadow_motor_params_publisher_;
  ::ros::Publisher current_policy_publisher_;
  ::ros::Publisher fsm_request_publisher_;
  ::ros::Publisher joy_buttons_snapshot_publisher_;

  mutable std::mutex mutex_;
  bool joy_received_{false};
  bool joy_semantic_mapping_ready_{false};
  JoystickState joystick_state_{};
  std::string joy_summary_{"no /joy message received yet"};

  bool joint_feedback_received_{false};
  std::vector<double> joint_pos_hardware_order_;
  std::vector<double> joint_vel_hardware_order_;
  std::optional<SensorSync::JointFeedbackSnapshot> joint_feedback_policy_order_;
  std::optional<SensorSync::T265OdomSnapshot> t265_odom_;
  std::optional<SensorSync::T265ImuSnapshot> t265_imu_;
  std::vector<std::string> warnings_;
};

}  // namespace ros_adapter
}  // namespace zky_rl_deploy
