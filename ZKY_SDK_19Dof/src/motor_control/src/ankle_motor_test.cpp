#include "kinematics.h"
#include "transmit.h"
#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <thread>
#include <chrono>
#include <cmath>
#include <iostream>
#include <locale.h>

// 安全参数配置
namespace ankle_test {
    constexpr int NUM_LEGS = 2;
    constexpr int NUM_MOTORS_PERLEG = ZKY_LEG_MOTORS_PER_SLAVE;
    // 底层 EtherCAT_Send_Command 已扩展为 19 DOF，因此测试命令数组也必须是19个槽位。
    // 本测试只写左右脚踝 motor5/motor6，上肢和腰部电机保持零命令。
    constexpr int TOTAL_MOTORS = ZKY_TOTAL_MOTORS;
    
    // 测试参数 - 使用较小的角度幅值确保安全
    constexpr double ANKLE_ROLL_AMPLITUDE = 0.12;   // ±0.08弧度 (约±4.6度)
    constexpr double ANKLE_PITCH_AMPLITUDE = 0.12;  // ±0.08弧度 (约±4.6度)
    constexpr double TEST_FREQUENCY = 0.5;          // 0.5Hz - 慢速运动更安全
    constexpr double TEST_DURATION = 20.0;          // 测试持续20秒
    
    // 电机控制增益 - 使用较低的增益保证平稳运动
    constexpr float KP = 50.0f;  // 位置增益
    constexpr float KD = 1.0f;   // 速度增益
    
    // 初始化位置（零位）
    constexpr double ANKLE_ROLL_INIT = 0.0;
    constexpr double ANKLE_PITCH_INIT = 0.0;
    
    // EtherCAT接口配置
    std::string ethercat_interface = "enp100s0";
    
    // 创建运动学对象
    ParallelNumCal kinematic_left;
    ParallelNumCal kinematic_right;
}

// 测试函数：生成正弦波形的脚踝运动轨迹
void runAnkleMotorTest(const std::string& test_leg)
{
    using namespace ankle_test;
    
    ROS_INFO("========================================");
    ROS_INFO("开始脚踝电机测试");
    ROS_INFO("测试腿: %s", test_leg.c_str());
    ROS_INFO("Roll 幅值: ±%.2f 度", ANKLE_ROLL_AMPLITUDE * 180.0 / M_PI);
    ROS_INFO("Pitch 幅值: ±%.2f 度", ANKLE_PITCH_AMPLITUDE * 180.0 / M_PI);
    ROS_INFO("测试频率: %.2f Hz", TEST_FREQUENCY);
    ROS_INFO("测试时长: %.1f 秒", TEST_DURATION);
    ROS_INFO("========================================");
    
    // 初始化 EtherCAT
    ROS_INFO("正在初始化 EtherCAT 接口 [%s]...", ethercat_interface.c_str());
    int slave_count = EtherCAT_Init(const_cast<char*>(ethercat_interface.c_str()));
    if (slave_count <= 0)
    {
        ROS_FATAL("EtherCAT 初始化失败!");
        return;
    }
    ROS_INFO("检测到 %d 个 EtherCAT 从站", slave_count);
    
    // 发送零命令初始化
    std::array<YKSMotorData, TOTAL_MOTORS> zero_cmd{};
    EtherCAT_Send_Command(zero_cmd.data());
    ros::Duration(0.5).sleep();
    
    // 选择运动学对象
    ParallelNumCal* kinematic = (test_leg == "left") ? &kinematic_left : &kinematic_right;
    
    // 确定要控制的电机索引（左腿：4,5  右腿：10,11）
    int motor5_idx = (test_leg == "left") ? 4 : 10;
    int motor6_idx = (test_leg == "left") ? 5 : 11;
    
    ROS_INFO("控制的电机索引: 电机%d 和 电机%d", motor5_idx + 1, motor6_idx + 1);
    
    // 测试循环
    auto start_time = std::chrono::high_resolution_clock::now();
    ros::Rate loop_rate(1000);  // 1kHz 控制频率
    
    ROS_INFO("开始运动，请注意观察...");
    
    // 阶段1: 缓慢移动到初始位置（2秒）
    ROS_INFO("阶段1: 移动到初始位置...");
    double init_duration = 2.0;
    auto phase1_start = std::chrono::high_resolution_clock::now();
    
    while (ros::ok())
    {
        auto now = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(now - phase1_start).count();
        
        if (t > init_duration) break;
        
        // 获取电机状态
        EtherCAT_Get_State();
        
        // 插值到初始位置
        double alpha = std::min(1.0, t / init_duration);  // 0 -> 1
        double ankle_roll = ANKLE_ROLL_INIT * alpha;
        double ankle_pitch = ANKLE_PITCH_INIT * alpha;
        
        // 计算电机目标角度
        Eigen::Vector2d motor_angles = kinematic->EulerToMotorAngle(ankle_roll, ankle_pitch);
        
        // 准备电机命令
        std::array<YKSMotorData, TOTAL_MOTORS> command{};
        
        // 只控制测试腿的脚踝电机，其他电机保持零力矩
        command[motor5_idx].kp_ = KP;
        command[motor5_idx].kd_ = KD;
        command[motor5_idx].pos_des_ = motor_angles[0];
        command[motor5_idx].vel_des_ = 0.0;
        command[motor5_idx].ff_ = 0.0;
        
        command[motor6_idx].kp_ = KP;
        command[motor6_idx].kd_ = KD;
        command[motor6_idx].pos_des_ = motor_angles[1];
        command[motor6_idx].vel_des_ = 0.0;
        command[motor6_idx].ff_ = 0.0;
        
        EtherCAT_Send_Command(command.data());
        loop_rate.sleep();
    }
    
    ROS_INFO("到达初始位置");
    
    // 阶段2: 周期性正弦运动
    ROS_INFO("阶段2: 开始周期性运动测试...");
    auto phase2_start = std::chrono::high_resolution_clock::now();
    
    while (ros::ok())
    {
        auto now = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(now - phase2_start).count();
        
        // 检查是否超过测试时长
        if (t > TEST_DURATION)
        {
            ROS_INFO("测试完成!");
            break;
        }
        
        // 获取电机状态
        EtherCAT_Get_State();
        
        // 生成正弦轨迹
        double omega = 2.0 * M_PI * TEST_FREQUENCY;
        
        // Roll 和 Pitch 使用不同的相位，产生圆锥运动
        double ankle_roll = ANKLE_ROLL_INIT + ANKLE_ROLL_AMPLITUDE * std::sin(omega * t);
        double ankle_pitch = ANKLE_PITCH_INIT + ANKLE_PITCH_AMPLITUDE * std::sin(omega * t + M_PI / 2.0);
        
        // 计算角速度（用于前馈控制）
        double ankle_roll_vel = ANKLE_ROLL_AMPLITUDE * omega * std::cos(omega * t);
        double ankle_pitch_vel = ANKLE_PITCH_AMPLITUDE * omega * std::cos(omega * t + M_PI / 2.0);
        
        // 使用运动学函数计算电机目标角度
        Eigen::Vector2d motor_angles = kinematic->EulerToMotorAngle(ankle_roll, ankle_pitch);
        
        // 使用运动学函数计算电机目标角速度
        Eigen::Vector2d motor_vels = kinematic->AnkleVelToMotorVel(ankle_roll, ankle_pitch, 
                                                                     ankle_roll_vel, ankle_pitch_vel);
        
        // 准备电机命令
        std::array<YKSMotorData, TOTAL_MOTORS> command{};
        
        command[motor5_idx].kp_ = KP;
        command[motor5_idx].kd_ = KD;
        command[motor5_idx].pos_des_ = motor_angles[0];
        command[motor5_idx].vel_des_ = motor_vels[0];
        command[motor5_idx].ff_ = 0.0;
        
        command[motor6_idx].kp_ = KP;
        command[motor6_idx].kd_ = KD;
        command[motor6_idx].pos_des_ = motor_angles[1];
        command[motor6_idx].vel_des_ = motor_vels[1];
        command[motor6_idx].ff_ = 0.0;
        
        // 每秒打印一次状态
        if (static_cast<int>(t * 10) % 10 == 0)
        {
            // 读取实际电机反馈
            double actual_motor5 = motorDate_recv[motor5_idx].pos_;
            double actual_motor6 = motorDate_recv[motor6_idx].pos_;
            
            // 使用正运动学计算实际脚踝角度
            Eigen::Vector2d actual_motor_angles(actual_motor5, actual_motor6);
            Eigen::Vector2d actual_ankle_angles = kinematic->MotorAngleToEuler(actual_motor_angles);
            
            ROS_INFO("时间: %.1fs | 目标Roll: %.2f° Pitch: %.2f° | 实际Roll: %.2f° Pitch: %.2f°",
                     t,
                     ankle_roll * 180.0 / M_PI,
                     ankle_pitch * 180.0 / M_PI,
                     actual_ankle_angles[0] * 180.0 / M_PI,
                     actual_ankle_angles[1] * 180.0 / M_PI);
        }
        
        EtherCAT_Send_Command(command.data());
        loop_rate.sleep();
    }
    
    // 阶段3: 缓慢返回零位
    ROS_INFO("阶段3: 返回零位...");
    auto phase3_start = std::chrono::high_resolution_clock::now();
    double return_duration = 2.0;
    
    // 记录当前位置
    EtherCAT_Get_State();
    double current_motor5 = motorDate_recv[motor5_idx].pos_;
    double current_motor6 = motorDate_recv[motor6_idx].pos_;
    
    while (ros::ok())
    {
        auto now = std::chrono::high_resolution_clock::now();
        double t = std::chrono::duration<double>(now - phase3_start).count();
        
        if (t > return_duration) break;
        
        EtherCAT_Get_State();
        
        // 插值回零
        double alpha = 1.0 - std::min(1.0, t / return_duration);  // 1 -> 0
        
        std::array<YKSMotorData, TOTAL_MOTORS> command{};
        command[motor5_idx].kp_ = KP;
        command[motor5_idx].kd_ = KD;
        command[motor5_idx].pos_des_ = current_motor5 * alpha;
        command[motor5_idx].vel_des_ = 0.0;
        command[motor5_idx].ff_ = 0.0;
        
        command[motor6_idx].kp_ = KP;
        command[motor6_idx].kd_ = KD;
        command[motor6_idx].pos_des_ = current_motor6 * alpha;
        command[motor6_idx].vel_des_ = 0.0;
        command[motor6_idx].ff_ = 0.0;
        
        EtherCAT_Send_Command(command.data());
        loop_rate.sleep();
    }
    
    // 发送零命令停止
    ROS_INFO("停止所有电机");
    EtherCAT_Send_Command(zero_cmd.data());
    
    ROS_INFO("========================================");
    ROS_INFO("脚踝电机测试完成!");
    ROS_INFO("========================================");
}

int main(int argc, char** argv)
{
    //设置本地化以支持中文输出
    setlocale(LC_ALL, "");
    ros::init(argc, argv, "ankle_motor_test");
    ros::NodeHandle nh("~");
    
    // 从参数服务器获取配置
    nh.param<std::string>("ethercat_interface", ankle_test::ethercat_interface, 
                          std::string("enp45s0"));
    
    std::string test_leg;
    nh.param<std::string>("test_leg", test_leg, std::string("left"));
    
    if (test_leg != "left" && test_leg != "right")
    {
        ROS_ERROR("无效的 test_leg 参数: %s (应该是 'left' 或 'right')", test_leg.c_str());
        return -1;
    }
    
    ROS_INFO("========================================");
    ROS_INFO("脚踝电机测试程序");
    ROS_INFO("EtherCAT 接口: %s", ankle_test::ethercat_interface.c_str());
    ROS_INFO("测试腿: %s", test_leg.c_str());
    ROS_INFO("========================================");
    ROS_INFO("警告: 请确保机器人处于安全位置!");
    ROS_INFO("程序将在3秒后开始...");
    
    ros::Duration(3.0).sleep();
    
    // 运行测试
    runAnkleMotorTest(test_leg);
    
    return 0;
}

