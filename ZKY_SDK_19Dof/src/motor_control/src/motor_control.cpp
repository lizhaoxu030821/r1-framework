#include "butterworth_filter.hpp"
#include "transmit.h"
#include "kinematics.h"
#include <Eigen/Dense>
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <array>
#include <filesystem>
#include <mutex>
#include <cmath>    // 用于fabs绝对值计算
#include <algorithm> // 用于std::min/max比较
#include <sys/stat.h> // 用于stat函数


namespace motor_control {
  constexpr int NUM_LEGS = 2;
  constexpr int NUM_MOTORS_PERLEG = ZKY_LEG_MOTORS_PER_SLAVE;  // 每条腿6个电机
  constexpr int LOWER_BODY_MOTORS = NUM_LEGS * NUM_MOTORS_PERLEG;
  constexpr int UPPER_BODY_MOTORS = ZKY_UPPER_MOTORS_ON_SLAVE;
  constexpr int TOTAL_MOTORS = ZKY_TOTAL_MOTORS;
  constexpr int LEFT_LEG_OFFSET = 0;
  constexpr int RIGHT_LEG_OFFSET = NUM_MOTORS_PERLEG;
  constexpr int UPPER_BODY_OFFSET = LOWER_BODY_MOTORS;
  constexpr int PARAMS_PER_MOTOR = 6;
  constexpr int NUM_ETHERCAT_SLAVES = ZKY_SLAVE_NUMBER;
  constexpr std::array<int, NUM_ETHERCAT_SLAVES> SLAVE_MOTOR_OFFSETS = {
    LEFT_LEG_OFFSET, RIGHT_LEG_OFFSET, UPPER_BODY_OFFSET
  };
  constexpr std::array<int, NUM_ETHERCAT_SLAVES> SLAVE_MOTOR_COUNTS = {
    NUM_MOTORS_PERLEG, NUM_MOTORS_PERLEG, UPPER_BODY_MOTORS
  };
  static_assert(TOTAL_MOTORS == LOWER_BODY_MOTORS + UPPER_BODY_MOTORS,
                "19DOF motor count must be 12 lower-body + 7 upper-body/waist");
  
  // 关节名称定义（19 DOF，顺序必须与 /motor_params 和 EtherCAT 路由表一致）
  const std::vector<std::string> JOINT_NAMES = {
    "left_hip_yaw_joint",         // 0  从站0 CAN1 id1
    "left_hip_roll_joint",        // 1  从站0 CAN1 id2
    "left_hip_pitch_joint",       // 2  从站0 CAN1 id3
    "left_knee_joint",            // 3  从站0 CAN2 id4
    "left_ankle_pitch_joint",     // 4  关节角经并联踝运动学转换到从站0 motor5
    "left_ankle_roll_joint",      // 5  关节角经并联踝运动学转换到从站0 motor6
    "right_hip_yaw_joint",        // 6  从站1 CAN1 id1
    "right_hip_roll_joint",       // 7  从站1 CAN1 id2
    "right_hip_pitch_joint",      // 8  从站1 CAN1 id3
    "right_knee_joint",           // 9  从站1 CAN2 id4
    "right_ankle_pitch_joint",    // 10 关节角经并联踝运动学转换到从站1 motor5
    "right_ankle_roll_joint",     // 11 关节角经并联踝运动学转换到从站1 motor6
    "left_shoulder_pitch_joint",  // 12 从站2 CAN1 id1
    "left_shoulder_roll_joint",   // 13 从站2 CAN1 id2
    "left_elbow_joint",           // 14 从站2 CAN1 id3
    "right_shoulder_pitch_joint", // 15 从站2 CAN2 id4
    "right_shoulder_roll_joint",  // 16 从站2 CAN2 id5（按右臂命名修正原需求文字中的 left_* 笔误）
    "right_elbow_joint",          // 17 从站2 CAN2 id6（按右臂命名修正原需求文字中的 left_* 笔误）
    "body_joint"                  // 18 从站2 CAN2 id7，腰部旋转自由度
  };
  // const std::vector<std::string> JOINT_NAMES = {
  //   "L_joint1",
  //   "L_joint2",
  //   "L_joint3",
  //   "L_joint4",
  //   "L_joint5",
  //   "L_joint6",
  //   "R_joint1",
  //   "R_joint2",
  //   "R_joint3",
  //   "R_joint4",
  //   "R_joint5",
  //   "R_joint6"
  // };

  inline size_t toEthercatIndex(size_t joint_index)
  {
    // 当前全局关节顺序已经按 EtherCAT 反馈数组排列：
    // 0~5 左腿、6~11 右腿、12~18 上肢/腰部，因此 ROS 侧索引可直接透传。
    return joint_index;
  }

  // 创建运动学转换对象（左右腿各一个）
  ParallelNumCal kinematic_left;
  ParallelNumCal kinematic_right;

  inline float clampValue(float value, float lower, float upper)
  {
    return std::max(lower, std::min(value, upper));
  }

  struct MotorParams {
      float kp, kd, pos, vel, tau, mode;
  };
  // 关节数据结构体（包含位置、速度、力矩信息）
  struct JointData {
    double position;  // 关节角度（弧度）
    double velocity;  // 关节角速度（弧度/秒）
    double torque;    // 关节输出力矩（牛·米）
  };


  //每个关节位置限制
  // const std::array<std::pair<float, float>, TOTAL_MOTORS> POSITION_LIMITS = {{
  //   {-10.0f, 10.0f},      // L_joint1
  //   {-10.0f, 10.0f},     // L_joint2
  //   {-10.0f, 10.0f},     // L_joint3
  //   {-10.0f, 10.0f},     // L_joint4
  //   {-10.0f, 10.0f},       // L_joint5
  //   {-10.0f, 10.0f},       // L_joint6
  //   {-10.0f, 10.0f},      // R_joint1
  //   {-10.0f, 10.0f},     // R_joint2
  //   {-10.0f, 10.0f},     // R_joint3
  //   {-10.0f, 10.0f},     // R_joint4
  //   {-10.0f, 10.0f},      // R_joint5
  //   {-10.0f, 10.0f}    // R_joint6
  // }};
  // 关节位置限制（rad）- 根据实际机械限制调整
  // 注意：限制过紧会频繁触发安全控制，影响行走
  const std::array<std::pair<float, float>, TOTAL_MOTORS> POSITION_LIMITS = {{
    {-1.0f, 1.1f},      // L_joint1 (left_hip_yaw)
    {-0.35f, 0.91f},    // L_joint2 (left_hip_roll) - 放宽下限
    {-1.0f, 0.45f},     // L_joint3 (left_hip_pitch)
    {-0.79f, 1.77f},    // L_joint4 (left_knee)
    {-0.32f, 0.85f},    // L_joint5 (left_ankle_pitch)
    {-0.9f, 0.5f},      // L_joint6 (left_ankle_roll)
    {-1.0f, 1.1f},      // R_joint1 (right_hip_yaw)
    {-0.91f, 0.35f},    // R_joint2 (right_hip_roll) - 放宽上限 0.22->0.35
    {-0.45f, 1.0f},     // R_joint3 (right_hip_pitch)
    {-1.81f, 0.65f},    // R_joint4 (right_knee)
    {-0.89f, 0.38f},    // R_joint5 (right_ankle_pitch)
    {-0.5f, 0.9f},      // R_joint6 (right_ankle_roll)
    {-1.57f, 1.57f},    // left_shoulder_pitch_joint，URDF 19DOF 初始限位
    {-1.57f, 1.57f},    // left_shoulder_roll_joint，URDF 19DOF 初始限位
    {-1.57f, 1.57f},    // left_elbow_joint，URDF 19DOF 初始限位
    {-1.57f, 1.57f},    // right_shoulder_pitch_joint，URDF 19DOF 初始限位
    {-1.57f, 1.57f},    // right_shoulder_roll_joint，URDF 19DOF 初始限位
    {-1.57f, 1.57f},    // right_elbow_joint，URDF 19DOF 初始限位
    {-1.57f, 1.57f}     // body_joint，腰部旋转自由度 URDF 初始限位
  }};


  // 巴特沃斯滤波器参数
  const double dt = 0.001; //采样时间
  const double cutoff_pos = 8.0; // 截止频率
  const double cutoff_vel = 10.0; // 截止频率
  const double cutoff_tau = 10.0; // 截止频率
  const double cutoff_ankle = 12.0; // 截止频率
  const int order = 3; // 滤波器阶数
  const double fs = 1000; // 采样频率

  //每个关节力矩限制，覆盖19个电机
  // const std::array<float, TOTAL_MOTORS> TORQUE_LIMITS = {
  //   10.0f, 20.0f, 20.0f, 10.0f, 10.0f, 10.0f,
  //   10.0f, 20.0f, 20.0f, 10.0f, 10.0f, 10.0f
  // };
  const std::array<float, TOTAL_MOTORS> TORQUE_LIMITS = {
    30.0f, 50.0f, 100.0f, 100.0f, 20.0f, 20.0f,
    30.0f, 50.0f, 100.0f, 100.0f, 20.0f, 20.0f,
    20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 30.0f
  };
  std::array<MotorParams, TOTAL_MOTORS> motorParamsArray;
  std::mutex motor_params_mutex;  //互斥量，保护对电机参数的访问
  std::mutex state_trace_mutex;   // 保护 FSM/按键快照日志字段
  //定义全局变量储存IMU角速度\线加速度\四元数
  std::vector<double> angular_velocity(3, 0.0);
  std::vector<double> linear_acceleration(3, 0.0);
  // geometry_msgs::Vector3 angular_velocity;
  // geometry_msgs::Vector3 linear_acceleration;
  std::vector<double> orientation{0.0, 0.0, 0.0, 1.0};
  //定义全局变量储存足端传感器返回值
  Eigen::VectorXd raw_ankleAngles(2);
  bool contact_l = true;
  bool contact_r = true;
  // double contact_l = 0.0;
  // double contact_r = 0.0;
  std::vector<float> current_pos(TOTAL_MOTORS);
  std::vector<float> current_vel(TOTAL_MOTORS);
  std::vector<float> current_tau(TOTAL_MOTORS);
  double raw_laser_h = 0.0;
  double filtered_laser_h = 0.0;
  std::vector<double> odom_camera_pos(3, 0.0);
  double odom_pose_cov = 0.1;
  std::string ethercat_interface = "enp45s0";
  std::string current_policy_name = "unknown";
  std::string fsm_request_name = "invalid";
  std::vector<int> joy_buttons_snapshot(11, 0);

  constexpr int BTN_A = 0;
  constexpr int BTN_B = 1;
  constexpr int BTN_X = 2;
  constexpr int BTN_Y = 3;
  constexpr int BTN_L1 = 4;
  constexpr int BTN_R1 = 5;
  constexpr int BTN_BACK = 6;
  constexpr int BTN_START = 7;
  constexpr int BTN_GUIDE = 8;

  constexpr std::array<float, TOTAL_MOTORS> MAX_TARGET_STEP = {
    0.005f, 0.005f, 0.005f, 0.005f, 0.003f, 0.003f,
    0.005f, 0.005f, 0.005f, 0.005f, 0.003f, 0.003f,
    0.004f, 0.004f, 0.004f, 0.004f, 0.004f, 0.004f, 0.003f
  };

  constexpr std::array<float, TOTAL_MOTORS> MAX_TRACKING_ERROR = {
    0.12f, 0.12f, 0.12f, 0.12f, 0.08f, 0.08f,
    0.12f, 0.12f, 0.12f, 0.12f, 0.08f, 0.08f,
    0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.10f, 0.08f
  };

  std::array<float, TOTAL_MOTORS> last_safe_pos_cmd{};
  std::array<float, TOTAL_MOTORS> last_requested_pos_cmd{};
  std::array<uint8_t, TOTAL_MOTORS> pos_step_limited{};
  std::array<uint8_t, TOTAL_MOTORS> tracking_error_limited{};
  bool last_safe_pos_initialized = false;

  int getButtonValue(const std::vector<int>& buttons, size_t idx)
  {
    return idx < buttons.size() ? buttons[idx] : 0;
  }

  void LogMotorDateRecvState(const std::vector<float>& motor_positions)
  {
    if (motor_positions.size() < TOTAL_MOTORS) {
      return;
    }

    ROS_INFO("[DEBUG] ===== Final motorDate_recv state =====");
    for (int slave = 0; slave < NUM_ETHERCAT_SLAVES; ++slave) {
      const int offset = SLAVE_MOTOR_OFFSETS[slave];
      const int count = SLAVE_MOTOR_COUNTS[slave];
      std::ostringstream line;
      line << "[DEBUG] Slave " << slave << " motors (" << offset << "-"
           << (offset + count - 1) << "):";
      for (int motor = 0; motor < count; ++motor) {
        line << " " << std::fixed << std::setprecision(3)
             << motor_positions[offset + motor];
      }
      ROS_INFO("%s", line.str().c_str());
    }
  }

  void paramsCallback(const std_msgs::Float64MultiArray::ConstPtr& msg)
  {
    const size_t expected_19dof = TOTAL_MOTORS * PARAMS_PER_MOTOR;
    const size_t expected_lower_body = LOWER_BODY_MOTORS * PARAMS_PER_MOTOR;
    if (msg->data.size() != expected_19dof &&
        msg->data.size() != expected_lower_body)
    {
      ROS_ERROR("Invalid motor parameters count, expected %zu (19DOF) or %zu (lower body only), got %zu",
                expected_19dof, expected_lower_body, msg->data.size());
      return;
    }
    const bool has_upper_body_command = (msg->data.size() == expected_19dof);
    if (!has_upper_body_command) {
      ROS_WARN_THROTTLE(2.0,
                        "/motor_params is lower-body-only (%zu values); upper-body and body_joint commands stay zero.",
                        msg->data.size());
    }

    std::vector<double> left_feedback_motor_pos(NUM_MOTORS_PERLEG, 0.0);
    std::vector<double> right_feedback_motor_pos(NUM_MOTORS_PERLEG, 0.0);
    {
      std::lock_guard<std::mutex> lock(motor_params_mutex);
      for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i)
      {
        left_feedback_motor_pos[i] = current_pos[LEFT_LEG_OFFSET + i];
        right_feedback_motor_pos[i] = current_pos[RIGHT_LEG_OFFSET + i];
      }
    }

    // 处理左腿（关节0-5 -> 电机0-5）
    std::vector<double> left_joint_pos(6), left_joint_vel(6);
    for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i)
    {
      const size_t baseIndex = i * PARAMS_PER_MOTOR;
      left_joint_pos[i] = msg->data[baseIndex + 2];  // pos
      left_joint_vel[i] = msg->data[baseIndex + 3];  // vel
    }
    
    // 左腿：关节空间 -> 电机空间（位置和速度IK）
    std::vector<double> left_motor_pos = kinematic_left.from_joint6_to_motor(left_joint_pos);
    std::vector<double> left_motor_vel = kinematic_left.from_joint6_vel_to_motor_vel(left_joint_pos, left_joint_vel);
    
    // 处理右腿（关节6-11 -> 电机6-11）
    std::vector<double> right_joint_pos(6), right_joint_vel(6);
    for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i)
    {
      const size_t baseIndex = (RIGHT_LEG_OFFSET + i) * PARAMS_PER_MOTOR;
      right_joint_pos[i] = msg->data[baseIndex + 2];  // pos
      right_joint_vel[i] = msg->data[baseIndex + 3];  // vel
    }
    
    // 右腿：关节空间 -> 电机空间（位置和速度IK）
    std::vector<double> right_motor_pos = kinematic_right.from_joint6_to_motor(right_joint_pos);
    std::vector<double> right_motor_vel = kinematic_right.from_joint6_vel_to_motor_vel(right_joint_pos, right_joint_vel);
    
    // ===== 踝关节雅可比力矩变换 =====
    // 仅对上层显式下发的踝关节关节空间力矩做 J^{-T} 映射；
    // 不在这里重复使用 kp/kd/pos 计算 PD 力矩，避免与电机内置 PD 叠加。
    
    // 左腿踝关节力矩变换
    std::vector<double> left_motor_tau(6, 0.0);
    {
        std::vector<double> joint_tau(6, 0.0);
        joint_tau[4] = msg->data[4 * PARAMS_PER_MOTOR + 4];
        joint_tau[5] = msg->data[5 * PARAMS_PER_MOTOR + 4];

        if (std::fabs(joint_tau[4]) > 1e-6 || std::fabs(joint_tau[5]) > 1e-6)
        {
            left_motor_tau =
                kinematic_left.from_joint6_tau_to_motor_tau(left_feedback_motor_pos, joint_tau);
        }
    }
    
    // 右腿踝关节力矩变换
    std::vector<double> right_motor_tau(6, 0.0);
    {
        std::vector<double> joint_tau(6, 0.0);
        joint_tau[4] = msg->data[(RIGHT_LEG_OFFSET + 4) * PARAMS_PER_MOTOR + 4];
        joint_tau[5] = msg->data[(RIGHT_LEG_OFFSET + 5) * PARAMS_PER_MOTOR + 4];

        if (std::fabs(joint_tau[4]) > 1e-6 || std::fabs(joint_tau[5]) > 1e-6)
        {
            right_motor_tau =
                kinematic_right.from_joint6_tau_to_motor_tau(right_feedback_motor_pos, joint_tau);
        }
    }
    
    // 将转换后的电机空间命令保存到 motorParamsArray。
    // 非踝关节保持原始 tau 透传；踝关节将关节空间 tau 映射到电机空间。
    std::array<MotorParams, TOTAL_MOTORS> updated_params{};
    for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i)
    {
      const size_t baseIndex = i * PARAMS_PER_MOTOR;
      const size_t motorIndex = LEFT_LEG_OFFSET + i;
      updated_params[motorIndex].kp   = msg->data[baseIndex];
      updated_params[motorIndex].kd   = msg->data[baseIndex + 1];
      updated_params[motorIndex].pos  = left_motor_pos[i];
      updated_params[motorIndex].vel  = left_motor_vel[i];
      updated_params[motorIndex].tau  = (i >= 4)
          ? static_cast<float>(left_motor_tau[i])
          : msg->data[baseIndex + 4];
      updated_params[motorIndex].mode = msg->data[baseIndex + 5];
    }
    
    for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i)
    {
      const size_t motorIndex = RIGHT_LEG_OFFSET + i;
      const size_t baseIndex = motorIndex * PARAMS_PER_MOTOR;
      updated_params[motorIndex].kp   = msg->data[baseIndex];
      updated_params[motorIndex].kd   = msg->data[baseIndex + 1];
      updated_params[motorIndex].pos  = right_motor_pos[i];
      updated_params[motorIndex].vel  = right_motor_vel[i];
      updated_params[motorIndex].tau  = (i >= 4)
          ? static_cast<float>(right_motor_tau[i])
          : msg->data[baseIndex + 4];
      updated_params[motorIndex].mode = msg->data[baseIndex + 5];
    }

    if (has_upper_body_command)
    {
      // 上肢6电机和腰部1电机为串联自由度，不需要并联踝 IK/Jacobian；
      // 按 /motor_params 中 12~18 的关节空间命令直接透传到同索引电机。
      for (size_t i = UPPER_BODY_OFFSET; i < TOTAL_MOTORS; ++i)
      {
        const size_t baseIndex = i * PARAMS_PER_MOTOR;
        updated_params[i].kp   = msg->data[baseIndex];
        updated_params[i].kd   = msg->data[baseIndex + 1];
        updated_params[i].pos  = msg->data[baseIndex + 2];
        updated_params[i].vel  = msg->data[baseIndex + 3];
        updated_params[i].tau  = msg->data[baseIndex + 4];
        updated_params[i].mode = msg->data[baseIndex + 5];
      }
    }

    {
      std::lock_guard<std::mutex> lock(motor_params_mutex);
      motorParamsArray = updated_params;
    }
    
    static size_t ankle_jac_dbg_cnt = 0;
    if (ankle_jac_dbg_cnt++ % 500 == 0) {
        ROS_DEBUG("[Ankle Jac] L_tau_motor=[%.2f, %.2f] R_tau_motor=[%.2f, %.2f]",
                  left_motor_tau[4], left_motor_tau[5],
                  right_motor_tau[4], right_motor_tau[5]);
    }
  }

  //IMU回调函数
  void IMUCallback(const sensor_msgs::Imu::ConstPtr& IMU_msgs){
    angular_velocity[0] = IMU_msgs -> angular_velocity.x;
    angular_velocity[1] = IMU_msgs -> angular_velocity.y;
    angular_velocity[2] = IMU_msgs -> angular_velocity.z;
    linear_acceleration[0] = IMU_msgs -> linear_acceleration.x;
    linear_acceleration[1] = IMU_msgs -> linear_acceleration.y;
    linear_acceleration[2] = IMU_msgs -> linear_acceleration.z;
    // orientation[0] = IMU_msgs -> orientation.x;
    // orientation[1] = IMU_msgs -> orientation.y;
    // orientation[2] = IMU_msgs -> orientation.z;
    // orientation[3] = IMU_msgs -> orientation.w;
    // std::cout << "orientation: [" << orientation[0] << ", " << orientation[1] << ", " << orientation[2] << ", " << orientation[3] << "]" << std::endl;
    // 实时打印出角速度和线加速度 
    // ROS_INFO("Angular Velocity: x=%.3f, y=%.3f, z=%.3f", angular_velocity.x, angular_velocity.y, angular_velocity.z); 
    // ROS_INFO("Linear Acceleration: x=%.3f, y=%.3f, z=%.3f", linear_acceleration.x, linear_acceleration.y, linear_acceleration.z);
    // ROS_INFO("Get IMU Callback");

  }

  //文件名生成函数
  std::string getTimestamp(){
    std::time_t now = std::time(nullptr);
    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(buffer);
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
    odom_camera_pos[0] = msg->pose.pose.position.x;
    odom_camera_pos[1] = msg->pose.pose.position.y;
    odom_camera_pos[2] = msg->pose.pose.position.z;
    odom_pose_cov = msg->pose.covariance[0];
    // std::cout << "Get Odom Callback" << std::endl;
    orientation[0] = msg -> pose.pose.orientation.x;
    orientation[1] = msg -> pose.pose.orientation.y;
    orientation[2] = msg -> pose.pose.orientation.z;
    orientation[3] = msg -> pose.pose.orientation.w;
  }

  void currentPolicyCallback(const std_msgs::String::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(state_trace_mutex);
    current_policy_name = msg->data;
  }

  void fsmRequestCallback(const std_msgs::String::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(state_trace_mutex);
    fsm_request_name = msg->data;
  }

  void joyButtonsSnapshotCallback(const std_msgs::Int32MultiArray::ConstPtr& msg)
  {
    std::lock_guard<std::mutex> lock(state_trace_mutex);
    joy_buttons_snapshot.assign(msg->data.begin(), msg->data.end());
  }


  void rosThread() {
      ros::NodeHandle nh;
      ros::Subscriber motor_params_sub = nh.subscribe("/motor_params", 10, paramsCallback);
      // ros::Subscriber IMU_sub = nh.subscribe("imu/data", 10, IMUCallback);
      ros::Subscriber IMU_sub = nh.subscribe("/camera/imu", 10, IMUCallback);
      // ros::Subscriber laser_sub = nh.subscribe("laser", 10, laserCallback);
      ros::Subscriber odom_sub = nh.subscribe("/camera/odom/sample", 10, odomCallback);
      ros::Subscriber current_policy_sub = nh.subscribe("/current_policy", 10, currentPolicyCallback);
      ros::Subscriber fsm_request_sub = nh.subscribe("/fsm_request", 10, fsmRequestCallback);
      ros::Subscriber joy_buttons_sub = nh.subscribe("/joy_buttons_snapshot", 10, joyButtonsSnapshotCallback);
        
      ros::Publisher motorCallback_pub = nh.advertise<std_msgs::Float64MultiArray>("motor_callback", 1);
      ros::Publisher joint_state_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 10);
      
      // RViz 可视化：发布 IMU、Odometry 和 TF 变换（带 _rviz 后缀）
      ros::Publisher imu_rviz_pub = nh.advertise<sensor_msgs::Imu>("imu_rviz", 10);
      ros::Publisher odom_rviz_pub = nh.advertise<nav_msgs::Odometry>("odom_rviz", 10);
      tf2_ros::TransformBroadcaster odom_broadcaster;

      auto start_time = std::chrono::high_resolution_clock::now();
      std::string filename = "motor_data_" + getTimestamp() + ".csv";
      // 使用相对路径创建文件路径
      std::filesystem::path current_file_path = __FILE__;
      std::filesystem::path project_root = current_file_path.parent_path().parent_path().parent_path().parent_path();
      std::filesystem::path record_data_path = project_root / "record_data" / "motor_data_record";
      std::filesystem::path full_path = record_data_path / filename;
      std::cout << "File will be saved at: " << full_path << std::endl;
      //创建目录（如果不存在）
      std::filesystem::create_directories(record_data_path);
      //打开文件并写入表头
      std::ofstream MotorRecordFile(full_path);
      //检查文件是否成功打开
      if (!MotorRecordFile.is_open()){
        ROS_ERROR("Failed to open file: %s", filename.c_str());
        return;
      }
      MotorRecordFile << "Time," << "world_time"
                      << ",fsm_request,current_policy"
                      << ",btn_a,btn_b,btn_x,btn_y,btn_l1,btn_r1,btn_back,btn_start,btn_guide";
      for (int i = 1; i <= TOTAL_MOTORS; ++i) {
        MotorRecordFile << ",M" << i << "_pos,M" << i << "_vel,M" << i << "_tau,M" << i << "_pos_filter,M" << i << "_vel_filter,M" << i << "_tau_filter";
        MotorRecordFile << ",M" << i << "_pos_target,M" << i << "_vel_target,M" << i << "_tau_target,M" << i << "_mode";
        MotorRecordFile << ",M" << i << "_pos_target_raw,M" << i << "_step_limited,M" << i << "_tracking_limited";
      }
      MotorRecordFile << ",AnkleAngle_l," << "AnkleAngle_r," << "Ankle_filtered_l," << "Ankle_filtered_r," << "contact_l," << "contact_r,"
                        << "orientation.x," << "orientation.y," << "orientation.z," << "orientation.w,"
                        << "angular_velocity_x," << "angular_velocity_y," << "angular_velocity_z,"
                        << "linear_acceleration_x," << "linear_acceleration_y," << "linear_acceleration_z," 
                        << "raw_laser_h," << "filtered_laser_h," 
                        << "odom_x," << "odom_y," << "odom_z"<< "\n";


      tssfa_se::ButterworthFilter filter_pos(dt, cutoff_pos, order);
      tssfa_se::ButterworthFilter filter_vel(dt, cutoff_vel, order);
      tssfa_se::ButterworthFilter filter_tau(dt, cutoff_tau, order);
      tssfa_se::ButterworthFilter filter_ankle(dt, cutoff_ankle, order);


      Eigen::VectorXd raw_pos(TOTAL_MOTORS); // 当前位置数组
      Eigen::VectorXd raw_vel(TOTAL_MOTORS); // 滤波前速度
      Eigen::VectorXd raw_tau(TOTAL_MOTORS); // 滤波前力矩
      Eigen::VectorXd filtered_pos(TOTAL_MOTORS); // 滤波后位置
      Eigen::VectorXd filtered_vel(TOTAL_MOTORS); // 滤波后速度
      Eigen::VectorXd filtered_tau(TOTAL_MOTORS); // 滤波后力矩
      Eigen::VectorXd filtered_ankle(2); // 滤波后脚踝

      ros::Rate loop_rate(500);
      while (ros::ok()) {
          std_msgs::Float64MultiArray motorCallback_msg;
          motorCallback_msg.data.resize(3 * TOTAL_MOTORS + 1); //(3个数据（位置，速度，力矩)*总电机数+时间）
          for (int i = 0; i < TOTAL_MOTORS; ++i) {
            // raw_pos[i] = motors[i].Get_Position();
            // raw_vel[i] = motors[i].Get_Velocity();
            // raw_tau[i] = motors[i].Get_tau();
            raw_pos[i] = current_pos[i];
            raw_vel[i] = current_vel[i];
            raw_tau[i] = current_tau[i];
          }
          // 应用butterworth滤波器
          filtered_pos = filter_pos.applyButterworth(raw_pos);
          filtered_vel = filter_vel.applyButterworth(raw_vel);
          filtered_tau = filter_tau.applyButterworth(raw_tau);
          filtered_ankle = filter_ankle.applyButterworth(raw_ankleAngles);
          auto ros_now = ros::Time::now();
          auto now = std::chrono::system_clock::now();
          auto now_seconds = std::chrono::system_clock::to_time_t(now);
          auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);
          auto us = now_us.time_since_epoch().count() % 1000000;

          std::tm* local_time = std::localtime(&now_seconds);  //将时间转换成本地时间结构体
          std::ostringstream oss;
          oss << std::put_time(local_time, "%H%M%S") << "." << std::setw(6) << std::setfill('0') << us;
          std::string time_str = oss.str();
          std::string current_policy_snapshot;
          std::string fsm_request_snapshot;
          std::vector<int> joy_buttons_snapshot_local;
          {
            std::lock_guard<std::mutex> trace_lock(state_trace_mutex);
            current_policy_snapshot = current_policy_name;
            fsm_request_snapshot = fsm_request_name;
            joy_buttons_snapshot_local = joy_buttons_snapshot;
          }
          //写入时间戳
          MotorRecordFile << ros_now << "," << time_str << ","
                          << fsm_request_snapshot << "," << current_policy_snapshot << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_A) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_B) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_X) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_Y) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_L1) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_R1) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_BACK) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_START) << ","
                          << getButtonValue(joy_buttons_snapshot_local, BTN_GUIDE) << ",";
          std::array<MotorParams, TOTAL_MOTORS> motor_params_snapshot{};
          std::array<float, TOTAL_MOTORS> last_requested_pos_cmd_snapshot{};
          std::array<uint8_t, TOTAL_MOTORS> pos_step_limited_snapshot{};
          std::array<uint8_t, TOTAL_MOTORS> tracking_error_limited_snapshot{};
          {
              std::lock_guard<std::mutex> lock(motor_params_mutex);
              motor_params_snapshot = motorParamsArray;
              last_requested_pos_cmd_snapshot = last_requested_pos_cmd;
              pos_step_limited_snapshot = pos_step_limited;
              tracking_error_limited_snapshot = tracking_error_limited;
          }
          for (int i = 0; i < TOTAL_MOTORS; ++i) {
            motorCallback_msg.data[i * 3] = raw_pos[i];
            motorCallback_msg.data[i * 3 + 1] = raw_vel[i];
            motorCallback_msg.data[i * 3 + 2] = raw_tau[i];

            MotorRecordFile << raw_pos[i] << ","
                << raw_vel[i] << "," << raw_tau[i] << ","
                << filtered_pos[i] << "," << filtered_vel[i] << ","
                << filtered_tau[i] << "," << motor_params_snapshot[i].pos << ","
                << motor_params_snapshot[i].vel << "," << motor_params_snapshot[i].tau << ","
                << motor_params_snapshot[i].mode << ","
                << last_requested_pos_cmd_snapshot[i] << ","
                << static_cast<int>(pos_step_limited_snapshot[i]) << ","
                << static_cast<int>(tracking_error_limited_snapshot[i]) << ",";
          }
          auto pub_now = std::chrono::high_resolution_clock::now();
          // 计算从 start_time 到现在的时间差（以秒为单位）
          auto time_since_start = std::chrono::duration_cast<std::chrono::duration<double>>(pub_now - start_time).count();
          motorCallback_msg.data[3 * TOTAL_MOTORS] = time_since_start;
          MotorRecordFile << raw_ankleAngles[0] << "," << raw_ankleAngles[1] << ","
                            << filtered_ankle[0] << "," << filtered_ankle[1] << ","
                            << contact_l << "," << contact_r << ","
                            << orientation[0] << "," << orientation[1] << "," << orientation[2]
                            << "," << orientation[3] << ","
                            << angular_velocity[0] << "," << angular_velocity[1] << ","
                            << angular_velocity[2] << ","
                            << linear_acceleration[0] << "," << linear_acceleration[1] << ","
                            << linear_acceleration[2] << ","
                            << raw_laser_h  << "," << filtered_laser_h << ","
                            << odom_camera_pos[0] << "," << odom_camera_pos[1] << ","
                            << odom_camera_pos[2] << '\n';
        
          motorCallback_pub.publish(motorCallback_msg);
          
          // 发布 JointState 消息
          sensor_msgs::JointState joint_state_msg;
          joint_state_msg.header.stamp = ros::Time::now();
          joint_state_msg.name = JOINT_NAMES;
          joint_state_msg.position.resize(TOTAL_MOTORS);
          joint_state_msg.velocity.resize(TOTAL_MOTORS);
          joint_state_msg.effort.resize(TOTAL_MOTORS);
          joint_state_msg.header.frame_id = "base_link";
          
          // 左腿：电机空间 -> 关节空间
          std::vector<double> left_motor_pos(6), left_motor_vel(6);
          for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i) {
            left_motor_pos[i] = raw_pos[i];
            left_motor_vel[i] = raw_vel[i];
          }
          std::vector<double> left_joint_pos = kinematic_left.from_motor6_to_joint(left_motor_pos);
          std::vector<double> left_joint_vel = kinematic_left.from_motor6_vel_to_joint_vel(left_motor_pos, left_motor_vel);
          
          // 右腿：电机空间 -> 关节空间
          std::vector<double> right_motor_pos(6), right_motor_vel(6);
          for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i) {
            right_motor_pos[i] = raw_pos[RIGHT_LEG_OFFSET + i];
            right_motor_vel[i] = raw_vel[RIGHT_LEG_OFFSET + i];
          }
          std::vector<double> right_joint_pos = kinematic_right.from_motor6_to_joint(right_motor_pos);
          std::vector<double> right_joint_vel = kinematic_right.from_motor6_vel_to_joint_vel(right_motor_pos, right_motor_vel);
          
          // 填充关节状态消息
          for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i) {
            joint_state_msg.position[i] = left_joint_pos[i];
            joint_state_msg.velocity[i] = left_joint_vel[i];
            joint_state_msg.effort[i] = raw_tau[i];  // 力矩保持原值
          }
          for (size_t i = 0; i < NUM_MOTORS_PERLEG; ++i) {
            const size_t joint_index = RIGHT_LEG_OFFSET + i;
            joint_state_msg.position[joint_index] = right_joint_pos[i];
            joint_state_msg.velocity[joint_index] = right_joint_vel[i];
            joint_state_msg.effort[joint_index] = raw_tau[joint_index];  // 力矩保持原值
          }
          for (size_t i = UPPER_BODY_OFFSET; i < TOTAL_MOTORS; ++i) {
            // 上肢和腰部没有并联机构，反馈电机角度即 URDF 关节角度。
            joint_state_msg.position[i] = raw_pos[i];
            joint_state_msg.velocity[i] = raw_vel[i];
            joint_state_msg.effort[i] = raw_tau[i];
          }
          
          joint_state_pub.publish(joint_state_msg);
          
          // ========== RViz 可视化：发布 IMU、Odometry 和 TF 变换 (带 _rviz 后缀) ==========
          ros::Time current_time = ros::Time::now();
          
          // 1. 发布 IMU 数据到 imu_rviz 话题
          sensor_msgs::Imu imu_rviz_msg;
          imu_rviz_msg.header.stamp = current_time;
          imu_rviz_msg.header.frame_id = "base_link";
          
          // 线性加速度
          imu_rviz_msg.linear_acceleration.x = linear_acceleration[0];
          imu_rviz_msg.linear_acceleration.y = linear_acceleration[1];
          imu_rviz_msg.linear_acceleration.z = linear_acceleration[2];
          
          // 角速度
          imu_rviz_msg.angular_velocity.x = angular_velocity[0];
          imu_rviz_msg.angular_velocity.y = angular_velocity[1];
          imu_rviz_msg.angular_velocity.z = angular_velocity[2];
          
          // 姿态（四元数归一化处理）
          double quat_norm = std::sqrt(orientation[0] * orientation[0] + 
                                       orientation[1] * orientation[1] + 
                                       orientation[2] * orientation[2] + 
                                       orientation[3] * orientation[3]);
          if (std::abs(quat_norm - 1.0) > 0.01 && quat_norm > 0) {
            // 归一化四元数
            imu_rviz_msg.orientation.x = orientation[0] / quat_norm;
            imu_rviz_msg.orientation.y = orientation[1] / quat_norm;
            imu_rviz_msg.orientation.z = orientation[2] / quat_norm;
            imu_rviz_msg.orientation.w = orientation[3] / quat_norm;
          } else {
            // 已经是归一化的，直接使用
            imu_rviz_msg.orientation.x = orientation[0];
            imu_rviz_msg.orientation.y = orientation[1];
            imu_rviz_msg.orientation.z = orientation[2];
            imu_rviz_msg.orientation.w = orientation[3];
          }
          
          imu_rviz_pub.publish(imu_rviz_msg);
          
          // 2. 发布 Odometry 数据到 odom_rviz 话题
          nav_msgs::Odometry odom_rviz_msg;
          odom_rviz_msg.header.stamp = current_time;
          odom_rviz_msg.header.frame_id = "odom";
          odom_rviz_msg.child_frame_id = "base_link";
          
          // 位置（来自相机里程计）
          odom_rviz_msg.pose.pose.position.x = odom_camera_pos[0];
          odom_rviz_msg.pose.pose.position.y = odom_camera_pos[1];
          odom_rviz_msg.pose.pose.position.z = odom_camera_pos[2];
          
          // 姿态（使用 IMU 的归一化四元数）
          odom_rviz_msg.pose.pose.orientation = imu_rviz_msg.orientation;
          
          // 线速度（设为0）
          odom_rviz_msg.twist.twist.linear.x = 0.0;
          odom_rviz_msg.twist.twist.linear.y = 0.0;
          odom_rviz_msg.twist.twist.linear.z = 0.0;
          
          // 角速度（来自 IMU）
          odom_rviz_msg.twist.twist.angular.x = angular_velocity[0];
          odom_rviz_msg.twist.twist.angular.y = angular_velocity[1];
          odom_rviz_msg.twist.twist.angular.z = angular_velocity[2];
          
          odom_rviz_pub.publish(odom_rviz_msg);
          
          // 3. 发布 TF 变换 (odom -> base_link)
          geometry_msgs::TransformStamped odom_trans;
          odom_trans.header.stamp = current_time;
          odom_trans.header.frame_id = "odom";
          odom_trans.child_frame_id = "base_link";
          
          // 位置
          odom_trans.transform.translation.x = odom_rviz_msg.pose.pose.position.x;
          odom_trans.transform.translation.y = odom_rviz_msg.pose.pose.position.y;
          odom_trans.transform.translation.z = odom_rviz_msg.pose.pose.position.z;
          
          // 姿态
          odom_trans.transform.rotation = odom_rviz_msg.pose.pose.orientation;
          
          odom_broadcaster.sendTransform(odom_trans);
          // ========== RViz 可视化结束 ==========
          
          // std::vector<double> back_pos = kinematics.from_m4_to_q(0, 0.9237422943115234, -0.9229803085327148, 1.429579734802246,
          //                                                          0, -0.9477758407592773,  1.0229263305664062, -1.3994436264038086);

          // std::cout << "back_pos: " ;
          // for (int i = 0; i < 2 * NUM_MOTORS_PERLEG; ++i) {
          //   std::cout << back_pos[i] << "  ";
          // }
          // std::cout << std::endl;

          ros::spinOnce();
          // std::this_thread::sleep_for(std::chrono::microseconds(1000));  // 为了减小子线程运行频率
          loop_rate.sleep();
        }
      MotorRecordFile.close();
  }

  void controlMotors()
  {
    ROS_INFO("=== Initializing EtherCAT interface [%s] ===", ethercat_interface.c_str());
    int slave_count = EtherCAT_Init(const_cast<char*>(ethercat_interface.c_str()));
    if (slave_count <= 0)
    {
      ROS_FATAL("Failed to initialize EtherCAT interface on %s", ethercat_interface.c_str());
      return;
    }
    ROS_INFO("Detected %d EtherCAT slaves", slave_count);

  std::array<YKSMotorData, TOTAL_MOTORS> zero_cmd{};
  EtherCAT_Send_Command(zero_cmd.data());

  {
    std::lock_guard<std::mutex> lock(motor_params_mutex);
    for (auto& params : motorParamsArray)
    {
      params = MotorParams{ 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    }
  }

  // ===== 关键修复: 在开始读取数据前给slave充分的准备时间 =====
  ROS_INFO("=== Waiting for slaves to stabilize before main loop ===");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // 200ms延迟
  
  // 预热: 发送命令并接收数据多次，让slave完全就绪
  ROS_INFO("=== Warming up EtherCAT communication (send + receive cycles) ===");
  for (int warmup = 0; warmup < 50; warmup++)
  {
    EtherCAT_Send_Command(zero_cmd.data());  // 发送零命令
    std::this_thread::sleep_for(std::chrono::microseconds(500));  // 0.5ms
    EtherCAT_Get_State();  // 接收反馈数据
    std::this_thread::sleep_for(std::chrono::microseconds(500));  // 0.5ms
  }
  
  ROS_INFO("=== Warmup complete, checking all slaves data ===");
  // 检查所有slave数据是否正常
  bool all_slaves_ok = true;
  int healthy_slaves = 0;
  
  for (int slave = 0; slave < NUM_ETHERCAT_SLAVES; slave++)
  {
    bool slave_ready = false;
    int motor_start = SLAVE_MOTOR_OFFSETS[slave];
    int motor_count = SLAVE_MOTOR_COUNTS[slave];
    int motor_end = motor_start + motor_count;
    int non_zero_count = 0;
    
    // 检查该slave的所有电机数据，统计非零电机数量
    for (int i = motor_start; i < motor_end; i++)
    {
      if (std::fabs(motorDate_recv[i].pos_) > 0.001 ||
          std::fabs(motorDate_recv[i].vel_) > 0.001 ||
          std::fabs(motorDate_recv[i].tau_) > 0.001)
      {
        non_zero_count++;
      }
    }
    
    slave_ready = (non_zero_count > 0);
    
    // 报告该slave的状态
    if (slave_ready)
    {
      ROS_INFO("✓ Slave %d (motors %d-%d) data OK [%d/%d motors responding]", 
               slave, motor_start, motor_end - 1, non_zero_count, motor_count);
      healthy_slaves++;
    }
    else
    {
      ROS_WARN("⚠ Slave %d (motors %d-%d) data ALL ZEROS after warmup!", 
               slave, motor_start, motor_end - 1);
      ROS_WARN("   Please check slave %d hardware connection and power supply!", slave);
      all_slaves_ok = false;
    }
  }
  
  // 总结
  if (all_slaves_ok)
  {
    ROS_INFO("✓✓✓ All %d slaves ready - starting control loop ✓✓✓", NUM_ETHERCAT_SLAVES);
  }
  else
  {
    ROS_WARN("⚠⚠⚠ Only %d/%d slaves healthy - continuing anyway, but expect problems! ⚠⚠⚠", 
             healthy_slaves, NUM_ETHERCAT_SLAVES);
  }

  std::vector<float> warmup_motor_positions(TOTAL_MOTORS, 0.0f);
  for (int i = 0; i < TOTAL_MOTORS; ++i)
  {
    warmup_motor_positions[i] = static_cast<float>(motorDate_recv[i].pos_);
  }
  LogMotorDateRecvState(warmup_motor_positions);
  
  ROS_INFO("=== Starting motor control main loop ===");
  ros::Rate control_rate(500); // 1 kHz 控制循环
  const auto startup_trace_begin = std::chrono::steady_clock::now();
  auto next_startup_trace = startup_trace_begin;
  while (ros::ok())
  {
    EtherCAT_Get_State();

      std::array<YKSMotorData, TOTAL_MOTORS> command{};
      std::array<MotorParams, TOTAL_MOTORS> requested_params{};
      std::array<float, TOTAL_MOTORS> next_last_requested_pos_cmd{};
      std::array<uint8_t, TOTAL_MOTORS> next_pos_step_limited{};
      std::array<uint8_t, TOTAL_MOTORS> next_tracking_error_limited{};
      std::array<float, TOTAL_MOTORS> next_last_safe_pos_cmd{};
      std::vector<float> next_current_pos(TOTAL_MOTORS, 0.0f);
      std::vector<float> next_current_vel(TOTAL_MOTORS, 0.0f);
      std::vector<float> next_current_tau(TOTAL_MOTORS, 0.0f);
      bool safe_pos_initialized = false;
      {
        std::lock_guard<std::mutex> lock(motor_params_mutex);
        requested_params = motorParamsArray;
        next_last_safe_pos_cmd = last_safe_pos_cmd;
        safe_pos_initialized = last_safe_pos_initialized;
      }

      for (size_t i = 0; i < TOTAL_MOTORS; ++i)
      {
        const size_t ec_index = toEthercatIndex(i);
        const auto& feedback = motorDate_recv[ec_index];

        next_current_pos[i] = static_cast<float>(feedback.pos_);
        next_current_vel[i] = static_cast<float>(feedback.vel_);
        next_current_tau[i] = static_cast<float>(feedback.tau_);

        const float requested_pos = requested_params[i].pos;
        next_last_requested_pos_cmd[i] = requested_pos;
        next_pos_step_limited[i] = 0;
        next_tracking_error_limited[i] = 0;

        float safe_pos = clampValue(requested_pos,
                                    POSITION_LIMITS[i].first,
                                    POSITION_LIMITS[i].second);
        float safe_tau = clampValue(requested_params[i].tau,
                                    -TORQUE_LIMITS[i], TORQUE_LIMITS[i]);
        float safe_vel = requested_params[i].vel;
        float safe_kp = requested_params[i].kp;
        float safe_kd = requested_params[i].kd;

        const float feedback_motor = next_current_pos[i];
        constexpr float SAFETY_KP = 300.0f;
        constexpr float SAFETY_KD = 3.0f;

        if (feedback_motor < POSITION_LIMITS[i].first)
        {
          safe_kp = SAFETY_KP;
          safe_kd = SAFETY_KD;
          safe_pos = POSITION_LIMITS[i].first;
          safe_vel = 0.0f;
          safe_tau = 0.0f;
          ROS_WARN_THROTTLE(1.0,
                            "Motor %zu position %.3f below lower limit %.3f, enabling safety control",
                            i + 1, feedback_motor, POSITION_LIMITS[i].first);
        }
        else if (feedback_motor > POSITION_LIMITS[i].second)
        {
          safe_kp = SAFETY_KP;
          safe_kd = SAFETY_KD;
          safe_pos = POSITION_LIMITS[i].second;
          safe_vel = 0.0f;
          safe_tau = 0.0f;
          ROS_WARN_THROTTLE(1.0,
                            "Motor %zu position %.3f above upper limit %.3f, enabling safety control",
                            i + 1, feedback_motor, POSITION_LIMITS[i].second);
        }

        const float tracking_lower = feedback_motor - MAX_TRACKING_ERROR[i];
        const float tracking_upper = feedback_motor + MAX_TRACKING_ERROR[i];
        const float tracking_clamped = clampValue(safe_pos, tracking_lower, tracking_upper);
        if (std::fabs(tracking_clamped - safe_pos) > 1e-6f)
        {
          next_tracking_error_limited[i] = 1;
          ROS_DEBUG_THROTTLE(0.5,
                             "Motor %zu target %.3f too far from feedback %.3f, clamp to tracking window [%.3f, %.3f]",
                             i + 1, safe_pos, feedback_motor, tracking_lower, tracking_upper);
          safe_pos = tracking_clamped;
        }

        if (!safe_pos_initialized)
        {
          next_last_safe_pos_cmd[i] = feedback_motor;
        }

        const float requested_delta = safe_pos - next_last_safe_pos_cmd[i];
        const float limited_delta = clampValue(requested_delta,
                                               -MAX_TARGET_STEP[i],
                                               MAX_TARGET_STEP[i]);
        if (std::fabs(limited_delta - requested_delta) > 1e-6f)
        {
          next_pos_step_limited[i] = 1;
          ROS_DEBUG_THROTTLE(0.5,
                             "Motor %zu target step %.4f rad too large, limited to %.4f rad",
                             i + 1, requested_delta, limited_delta);
        }
        safe_pos = next_last_safe_pos_cmd[i] + limited_delta;
        next_last_safe_pos_cmd[i] = safe_pos;

        command[ec_index].kp_ = safe_kp;
        command[ec_index].kd_ = safe_kd;
        command[ec_index].pos_des_ = safe_pos;
        command[ec_index].vel_des_ = safe_vel;
        command[ec_index].ff_ = safe_tau;
      }

      const auto startup_now = std::chrono::steady_clock::now();
      if (startup_now - startup_trace_begin <= std::chrono::seconds(20) &&
          startup_now >= next_startup_trace) {
        LogMotorDateRecvState(next_current_pos);
        next_startup_trace = startup_now + std::chrono::seconds(1);
      }

      {
        std::lock_guard<std::mutex> lock(motor_params_mutex);
        current_pos = next_current_pos;
        current_vel = next_current_vel;
        current_tau = next_current_tau;
        last_requested_pos_cmd = next_last_requested_pos_cmd;
        pos_step_limited = next_pos_step_limited;
        tracking_error_limited = next_tracking_error_limited;
        last_safe_pos_cmd = next_last_safe_pos_cmd;
        last_safe_pos_initialized = true;
      }

      EtherCAT_Send_Command(command.data());
      // std::this_thread::sleep_for(std::chrono::microseconds(1000));
      control_rate.sleep();
    }

    ROS_WARN("=== Stopping all motors ===");
    std::array<YKSMotorData, TOTAL_MOTORS> zero_stop{};
    EtherCAT_Send_Command(zero_stop.data());
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "motor_control");

  ros::NodeHandle nh("~");
  nh.param<std::string>("ethercat_interface", motor_control::ethercat_interface, std::string("enp45s0"));
  ros::Duration(0.1).sleep();

  ROS_INFO("=== EtherCAT Parameter Configuration ===");
  ROS_INFO("Using interface: %s", motor_control::ethercat_interface.c_str());
  ROS_INFO("==============================");

  std::thread ros_thread(motor_control::rosThread);
  motor_control::controlMotors();
  ros_thread.join();
  return 0;
}
