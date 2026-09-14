#include "zky_rl_deploy/ros/ros_adapter.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace zky_rl_deploy {
namespace ros_adapter {
namespace {

template <typename T>
bool IndexInRange(int index, const std::vector<T>& values) {
  return index >= 0 && static_cast<std::size_t>(index) < values.size();
}

std::string JoinWarnings(const std::vector<std::string>& warnings) {
  if (warnings.empty()) {
    return "none";
  }

  std::ostringstream stream;
  for (std::size_t index = 0; index < warnings.size(); ++index) {
    if (index != 0U) {
      stream << "; ";
    }
    stream << warnings[index];
  }
  return stream.str();
}
bool ReadButtonOrDefault(const sensor_msgs::Joy& message, int button_index, bool default_value) {
  return IndexInRange(button_index, message.buttons)
             ? (message.buttons[button_index] != 0)
             : default_value;
}

void AppendWarningOnce(std::vector<std::string>* warnings, const std::string& warning) {
  if (std::find(warnings->begin(), warnings->end(), warning) == warnings->end()) {
    warnings->push_back(warning);
  }
}

}  // namespace

RuntimeRosAdapter::RuntimeRosAdapter(const ::ros::NodeHandle& nh,
                                     const ::ros::NodeHandle& private_nh,
                                     const RuntimeConfig& config,
                                     const JointMapper& joint_mapper)
    : nh_(nh),
      private_nh_(private_nh),
      topics_(config.topics),
      joy_mapping_(config.joy_mapping),
      hardware_order_(config.robot_joints.hardware_order),
      joint_mapper_(joint_mapper),
      t265_extrinsic_(config.t265_extrinsic),
      joint_pos_hardware_order_(hardware_order_.size(), 0.0),
      joint_vel_hardware_order_(hardware_order_.size(), 0.0) {
  joy_subscriber_ = nh_.subscribe(topics_.joy_topic, 1, &RuntimeRosAdapter::JoyCallback, this);
  joint_state_subscriber_ =
      nh_.subscribe(topics_.joint_state_topic, 1, &RuntimeRosAdapter::JointStateCallback, this);
  t265_odom_subscriber_ =
      nh_.subscribe(topics_.t265_odom_topic, 1, &RuntimeRosAdapter::T265OdomCallback, this);
  t265_imu_subscriber_ =
      nh_.subscribe(topics_.t265_imu_topic, 1, &RuntimeRosAdapter::T265ImuCallback, this);

  // 当前仓库还没有专门的诊断 message 定义，因此先用 std_msgs/String 发布结构化摘要；
  // 后续若要做上位机可视化和日志解析，再收敛成自定义消息类型，不影响当前 dry_run/shadow 主循环联调。
  safety_status_publisher_ =
      nh_.advertise<std_msgs::String>(topics_.safety_status_topic, 1, true);
  control_mode_publisher_ =
      nh_.advertise<std_msgs::String>(topics_.control_mode_topic, 1, true);
  output_gate_publisher_ =
      nh_.advertise<std_msgs::String>(topics_.output_gate_topic, 1, true);
  beyond_mimic_debug_publisher_ =
      nh_.advertise<std_msgs::String>(topics_.beyond_mimic_debug_topic, 1, true);
  motor_params_publisher_ =
      nh_.advertise<std_msgs::Float64MultiArray>(topics_.motor_params_topic, 1);
  shadow_motor_params_publisher_ = nh_.advertise<std_msgs::Float64MultiArray>(
      topics_.shadow_motor_params_topic, 1, true);
  current_policy_publisher_ =
      nh_.advertise<std_msgs::String>(topics_.current_policy_topic, 1, true);
  fsm_request_publisher_ =
      nh_.advertise<std_msgs::String>(topics_.fsm_request_topic, 1, true);
  joy_buttons_snapshot_publisher_ = nh_.advertise<std_msgs::Int32MultiArray>(
      topics_.joy_buttons_snapshot_topic, 1, true);
}

InputSnapshot RuntimeRosAdapter::GetSnapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);

  InputSnapshot snapshot;
  snapshot.joy_received = joy_received_;
  snapshot.joy_semantic_mapping_ready = joy_semantic_mapping_ready_;
  snapshot.joystick_state = joystick_state_;
  snapshot.joy_summary = joy_summary_;

  snapshot.joint_feedback_received = joint_feedback_received_;
  snapshot.joint_pos_hardware_order = joint_pos_hardware_order_;
  snapshot.joint_vel_hardware_order = joint_vel_hardware_order_;
  snapshot.joint_feedback_policy_order = joint_feedback_policy_order_;
  snapshot.t265_odom = t265_odom_;
  snapshot.t265_imu = t265_imu_;
  snapshot.warnings = warnings_;
  return snapshot;
}

std::string RuntimeRosAdapter::PollDesiredStageName() const {
  std::string desired_stage_name;
  private_nh_.param<std::string>("desired_stage_name", desired_stage_name, "stage0_software");
  return desired_stage_name;
}

void RuntimeRosAdapter::PublishSafetyStatus(const std::string& summary) const {
  std_msgs::String message;
  message.data = summary;
  safety_status_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishControlMode(const std::string& summary) const {
  std_msgs::String message;
  message.data = summary;
  control_mode_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishOutputGate(const std::string& summary) const {
  std_msgs::String message;
  message.data = summary;
  output_gate_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishBeyondMimicDebug(const std::string& summary) const {
  std_msgs::String message;
  message.data = summary;
  beyond_mimic_debug_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishMotorParams(const std::vector<double>& motor_params) const {
  std_msgs::Float64MultiArray message;
  message.data = motor_params;
  motor_params_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishShadowMotorParams(
    const std::vector<double>& shadow_motor_params) const {
  std_msgs::Float64MultiArray message;
  message.data = shadow_motor_params;
  shadow_motor_params_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishCurrentPolicyTrace(const std::string& current_policy) const {
  std_msgs::String message;
  message.data = current_policy;
  current_policy_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishFsmRequestTrace(const std::string& fsm_request) const {
  std_msgs::String message;
  message.data = fsm_request;
  fsm_request_publisher_.publish(message);
}

void RuntimeRosAdapter::PublishJoyButtonsSnapshot(const JoystickState& joystick_state) const {
  std_msgs::Int32MultiArray message;
  message.data = {
      joystick_state.a_pressed ? 1 : 0,
      joystick_state.b_pressed ? 1 : 0,
      joystick_state.x_pressed ? 1 : 0,
      joystick_state.y_pressed ? 1 : 0,
      0,
      joystick_state.rb_pressed ? 1 : 0,
      joystick_state.back_pressed ? 1 : 0,
      joystick_state.start_pressed ? 1 : 0,
      joystick_state.home_pressed ? 1 : 0,
      0,
      0};
  joy_buttons_snapshot_publisher_.publish(message);
}

void RuntimeRosAdapter::JoyCallback(const sensor_msgs::Joy::ConstPtr& message) {
  bool semantic_mapping_ready = false;
  std::string summary;
  const JoystickState joystick_state =
      BuildJoystickStateFromRawMessage(*message, &semantic_mapping_ready, &summary);

  std::lock_guard<std::mutex> lock(mutex_);
  joy_received_ = true;
  joy_semantic_mapping_ready_ = semantic_mapping_ready;
  joystick_state_ = joystick_state;
  joy_summary_ = std::move(summary);
}

void RuntimeRosAdapter::JointStateCallback(const sensor_msgs::JointState::ConstPtr& message) {
  const TimePoint callback_time = Clock::now();
  std::vector<double> joint_pos_hardware_order(hardware_order_.size(), 0.0);
  std::vector<double> joint_vel_hardware_order(hardware_order_.size(), 0.0);

  if (message->position.size() != hardware_order_.size()) {
    std::lock_guard<std::mutex> lock(mutex_);
    AppendWarningOnce(&warnings_, "joint_state position vector does not match hardware joint count");
    return;
  }

  if (message->name.size() == hardware_order_.size()) {
    bool exact_hardware_order_match = true;
    for (std::size_t index = 0; index < hardware_order_.size(); ++index) {
      if (message->name[index] != hardware_order_[index]) {
        exact_hardware_order_match = false;
        break;
      }
    }

    if (exact_hardware_order_match) {
      joint_pos_hardware_order = message->position;
      if (message->velocity.size() == message->position.size()) {
        joint_vel_hardware_order = message->velocity;
      }
    } else {
    std::unordered_map<std::string, std::size_t> name_to_index;
    name_to_index.reserve(message->name.size());
    for (std::size_t index = 0; index < message->name.size(); ++index) {
      name_to_index.emplace(message->name[index], index);
    }

    for (std::size_t joint_index = 0; joint_index < hardware_order_.size(); ++joint_index) {
      const auto source_it = name_to_index.find(hardware_order_[joint_index]);
      if (source_it == name_to_index.end()) {
        std::lock_guard<std::mutex> lock(mutex_);
        AppendWarningOnce(&warnings_,
                          "joint_state is missing required joint: " +
                              hardware_order_[joint_index]);
        return;
      }
      joint_pos_hardware_order[joint_index] = message->position[source_it->second];
      if (message->velocity.size() == message->position.size()) {
        joint_vel_hardware_order[joint_index] = message->velocity[source_it->second];
      }
    }
    }
  } else {
    // 当前底层 joint_states 顺序与 hardware_order 已知一致；
    // 这里保留顺序回退只是为了 dry_run/shadow 联调不被“临时无 name 字段”完全卡死，但会继续保留 warning。
    joint_pos_hardware_order = message->position;
    if (message->velocity.size() == message->position.size()) {
      joint_vel_hardware_order = message->velocity;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    AppendWarningOnce(
        &warnings_,
        "joint_state names are missing or incomplete; falling back to configured hardware_order.");
  }

  SensorSync::JointFeedbackSnapshot joint_feedback_snapshot;
  joint_feedback_snapshot.stamp = callback_time;
  joint_feedback_snapshot.joint_pos_policy_order =
      joint_mapper_.HardwareToPolicyFeedback(joint_pos_hardware_order);
  joint_feedback_snapshot.joint_vel_policy_order =
      joint_mapper_.HardwareToPolicyFeedback(joint_vel_hardware_order);

  std::lock_guard<std::mutex> lock(mutex_);
  joint_feedback_received_ = true;
  joint_pos_hardware_order_ = std::move(joint_pos_hardware_order);
  joint_vel_hardware_order_ = std::move(joint_vel_hardware_order);
  joint_feedback_policy_order_ = std::move(joint_feedback_snapshot);
}

void RuntimeRosAdapter::T265OdomCallback(const nav_msgs::Odometry::ConstPtr& message) {
  SensorSync::T265OdomSnapshot snapshot;
  snapshot.stamp = Clock::now();
  const T265Extrinsics::Vector3 raw_position_flu_m = {message->pose.pose.position.x,
                                                      message->pose.pose.position.y,
                                                      message->pose.pose.position.z};
  const T265Extrinsics::QuaternionXyzw raw_orientation_xyzw = {
      message->pose.pose.orientation.x,
      message->pose.pose.orientation.y,
      message->pose.pose.orientation.z,
      message->pose.pose.orientation.w};
  try {
    snapshot.pelvis_orientation_xyzw =
        T265Extrinsics::TransformPoseFrameOrientationToPelvis(raw_orientation_xyzw,
                                                              t265_extrinsic_);
    snapshot.position_flu_m = T265Extrinsics::TransformPoseFramePositionToPelvis(
        raw_position_flu_m, snapshot.pelvis_orientation_xyzw, t265_extrinsic_);
  } catch (const std::invalid_argument& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    AppendWarningOnce(&warnings_, std::string("failed to transform T265 odom into pelvis frame: ") +
                                     error.what());
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  t265_odom_ = std::move(snapshot);
}

void RuntimeRosAdapter::T265ImuCallback(const sensor_msgs::Imu::ConstPtr& message) {
  SensorSync::T265ImuSnapshot snapshot;
  snapshot.stamp = Clock::now();
  const T265Extrinsics::Vector3 raw_angular_velocity_rad_s = {message->angular_velocity.x,
                                                              message->angular_velocity.y,
                                                              message->angular_velocity.z};
  try {
    snapshot.base_ang_vel_body_rad_s =
        T265Extrinsics::TransformImuOpticalAngularVelocityToPelvis(raw_angular_velocity_rad_s,
                                                                   t265_extrinsic_);
  } catch (const std::invalid_argument& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    AppendWarningOnce(&warnings_, std::string("failed to transform T265 imu into pelvis frame: ") +
                                     error.what());
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  t265_imu_ = std::move(snapshot);
}

JoystickState RuntimeRosAdapter::BuildJoystickStateFromRawMessage(
    const sensor_msgs::Joy& message, bool* semantic_mapping_ready, std::string* summary) const {
  JoystickState joystick_state;
  std::vector<std::string> warnings;

  if (!joy_mapping_.raw_mapping_confirmed) {
    *semantic_mapping_ready = false;
    *summary = "raw /joy semantic mapping is not confirmed in joystick_beitong.yaml; semantic buttons stay false";
    return joystick_state;
  }

  joystick_state.start_pressed =
      ReadButtonOrDefault(message, joy_mapping_.start_button_index, false);
  joystick_state.back_pressed =
      ReadButtonOrDefault(message, joy_mapping_.back_button_index, false);
  joystick_state.home_pressed =
      ReadButtonOrDefault(message, joy_mapping_.home_button_index, false);
  joystick_state.rb_pressed =
      ReadButtonOrDefault(message, joy_mapping_.rb_button_index, false);
  joystick_state.a_pressed = ReadButtonOrDefault(message, joy_mapping_.a_button_index, false);
  joystick_state.x_pressed = ReadButtonOrDefault(message, joy_mapping_.x_button_index, false);
  joystick_state.b_pressed = ReadButtonOrDefault(message, joy_mapping_.b_button_index, false);
  joystick_state.y_pressed = ReadButtonOrDefault(message, joy_mapping_.y_button_index, false);
  *semantic_mapping_ready = true;
  std::ostringstream stream;
  stream << "semantic joy: START=" << joystick_state.start_pressed
         << ", BACK=" << joystick_state.back_pressed
         << ", HOME=" << joystick_state.home_pressed
         << ", RB=" << joystick_state.rb_pressed
         << ", warnings=" << JoinWarnings(warnings);
  *summary = stream.str();
  return joystick_state;
}

}  // namespace ros_adapter
}  // namespace zky_rl_deploy
