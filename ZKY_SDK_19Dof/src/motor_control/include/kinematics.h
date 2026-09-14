// Planning_motion.h
#ifndef PLANNING_MOTION_H
#define PLANNING_MOTION_H

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <utility>

class Planning_motion {
public:
    // 构造函数
    Planning_motion();

    // 成员变量
    int swing_max_adjust_time;        // 调姿时间
    double joint2_init;               // 初始值，向后为正（弧度）
    double joint3_init;               // 初始值，向后为正（弧度）
    Eigen::Vector3d left_base_pos_inBody;      // 左腿在身体坐标系下的位置
    Eigen::Vector3d right_base_pos_inBody;     // 右腿在身体坐标系下的位置
    Eigen::Vector3d left_base_pos_inWorld;     // 左腿在世界坐标系下的位置
    Eigen::Vector3d right_base_pos_inWorld;    // 右腿在世界坐标系下的位置
    double landingvel;                // 着地速度
    double landinngtime;              // 着地时间

    // 旋转矩阵
    Eigen::Matrix4d Rotz(double theta);    // 绕Z轴旋转
    Eigen::Matrix4d Rotx(double theta);    // 绕X轴旋转
    Eigen::Matrix4d Roty(double theta);    // 绕Y轴旋转

    // 平移矩阵
    Eigen::Matrix4d Transx(double d);       // 沿X轴平移
    Eigen::Matrix4d Transy(double d);       //沿Y轴平移
    Eigen::Matrix4d Transz(double d);       // 沿Z轴平移

    // 角度转换
    double deg2rad(double deg);             // 角度转弧度

    // 正运动学
    Eigen::Vector3d Forward_kinematic(const std::string& whichleg, double theta1, double theta2, double theta3, double theta4);

    // 逆运动学
    std::vector<double> Inverse_kinematic(const std::string& whichleg, double px, double py, double pz);

    // 踝关节力矩
    double torque_ankle(double angle);

    // 期望质心参数
    std::pair<Eigen::Vector3d, Eigen::Vector3d> exp_com_parameter(const Eigen::Vector3d& base_init_pos, const Eigen::Vector3d& v, double dt);

    // 足部位置校正
    Eigen::Vector3d Foot_position_correction(double time_stance, const Eigen::Vector3d& cur_body_pos, const Eigen::Vector3d& cur_body_lin_vel, const Eigen::Vector3d& des_body_lin_vel, const Eigen::Vector3d& des_body_ang_vel);

    // 多项式转换函数
    double delta_star_trans(double alpha);
    double delta_star_dot_trans(double alpha, double alpha_dot);
    double delta_star_slash_trans(double alpha);
    double tau_alpha_trans(double alpha, double tau_delta_star);
    double alpha_trans(double delta_star);
    double alpha_dot_trans(double delta_star, double delta_star_dot);
    double tau_delta_star_trans(double delta_star, double tau_alpha);

    // q与m之间的转换
    std::vector<double> from_q_to_m(double q1, double q2, double q3, double q5, double q6, double q7);
    std::vector<double> from_m_to_q(double m1, double m2, double m3, double m4, double m5, double m6);
    std::vector<double> from_q_to_m_der(double q1, double q2, double q3, double q5, double q6, double q7);
    std::vector<double> from_m_to_q_der(double m1, double m2, double m3, double m4, double m5, double m6);

    std::vector<double> from_q4_to_m(double q1, double q2, double q3, double q4, double q5, double q6, double q7, double q8);
    std::vector<double> from_m4_to_q(double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8);
    std::vector<double> from_q4_to_m_der(double q1, double q2, double q3, double q4, double q5, double q6, double q7, double q8);
    std::vector<double> from_m4_to_q_der(double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8);
};

// 踝关节并联结构运动学求解类
// 基于坐标变换计算逆运动学（姿态角->电机角度），基于牛顿迭代法数值计算正运动学（电机角度->姿态角）
class ParallelNumCal {
public:
    // 构造函数
    ParallelNumCal();

    // 结构参数
    double a;       // 电机输出端转臂长度
    double h;       // 两电机间距与固定坐标系的原点的距离
    double d;       // 踝关节并联结构横杆长度
    double b;       // 横杆与踝关节旋转坐标系原点垂直z方向距离
    double H;       // 固定坐标系的原点与踝关节旋转坐标系原点距离
    double L1;      // 电机5的长杆长度
    double L2;      // 电机6的长杆长度
    double motor5_offset;      // 电机5的长杆长度
    double motor6_offset;      // 电机6的长杆长度

    /* ========= real ↔ internal ========= */
    inline Eigen::Vector2d toInternal(const Eigen::Vector2d& q_real) const;
    inline Eigen::Vector2d toReal(const Eigen::Vector2d& q_internal) const;

    Eigen::Matrix3d RotAnkle(double roll, double pitch);
    // 计算雅可比矩阵
    // 输入参数：p - 踝关节末端姿态角[roll, pitch]，q_real - 电机角度[电机5, 电机6]
    Eigen::Matrix2d JacobianCal(const Eigen::Vector2d& p);
    Eigen::Matrix2d JacobianCal(const Eigen::Vector2d& p, const Eigen::Vector2d& q_internal);
    


    // 逆运动学：通过踝关节末端pitch和roll方向的角度phi和psi计算两个电机输出端的旋转角度alpha
    // 输入参数：psi - roll角度，phi - pitch角度
    // 返回值：alpha - 电机角度[电机5角度，电机6角度]
    Eigen::Vector2d EulerToMotorAngle(double psi, double phi);

    // 正运动学：通过两个电机输出端的旋转角度alpha计算踝关节末端pitch和roll方向的角度phi和psi（牛顿迭代法）
    // 输入参数：q_ref - 电机参考角度[电机5角度，电机6角度]
    // 返回值：theta - 踝关节末端姿态角[roll角度，pitch角度]
    Eigen::Vector2d MotorAngleToEuler(const Eigen::Vector2d& q_ref);

    // 踝关节角速度转电机角速度（逆向速度运动学）
    // 输入参数：
    //   psi - 踝关节roll角度（弧度）
    //   phi - 踝关节pitch角度（弧度）
    //   psi_dot - 踝关节roll方向角速度（弧度/秒）
    //   phi_dot - 踝关节pitch方向角速度（弧度/秒）
    // 返回值：
    //   motor_vel - 电机角速度[电机5角速度, 电机6角速度]（弧度/秒）
    Eigen::Vector2d AnkleVelToMotorVel(double psi, double phi, double psi_dot, double phi_dot);

    // 电机角速度转踝关节角速度（正向速度运动学）
    // 输入参数：
    //   q - 电机角度[电机5角度, 电机6角度]（弧度）
    //   q_dot - 电机角速度[电机5角速度, 电机6角速度]（弧度/秒）
    // 返回值：
    //   ankle_vel - 踝关节角速度[roll角速度, pitch角速度]（弧度/秒）
    Eigen::Vector2d MotorVelToAnkleVel(const Eigen::Vector2d& q, const Eigen::Vector2d& q_dot);

    // ========== 单腿6电机整合转换函数 ==========
    
    // 电机位置转关节位置（6电机 -> 6关节）
    // 输入参数：motor_pos - 6个电机位置 [m1, m2, m3, m4, m5, m6]（弧度）
    //   m1~m4: 线性映射到关节1~4
    //   m5, m6: 踝关节并联电机，通过正运动学解算得到关节5和6（踝关节roll和pitch）
    // 返回值：joint_pos - 6个关节位置 [q1, q2, q3, q4, q5, q6]（弧度）
    std::vector<double> from_motor6_to_joint(double m1, double m2, double m3, double m4, double m5, double m6);
    
    // 关节位置转电机位置（6关节 -> 6电机）
    // 输入参数：joint_pos - 6个关节位置 [q1, q2, q3, q4, q5, q6]（弧度）
    //   q1~q4: 线性映射到电机1~4
    //   q5, q6: 踝关节roll和pitch，通过逆运动学解算得到电机5和6
    // 返回值：motor_pos - 6个电机位置 [m1, m2, m3, m4, m5, m6]（弧度）
    std::vector<double> from_joint6_to_motor(double q1, double q2, double q3, double q4, double q5, double q6);
    
    // 电机位置转关节位置（向量版本）
    std::vector<double> from_motor6_to_joint(const std::vector<double>& motor_pos);
    
    // 关节位置转电机位置（向量版本）
    std::vector<double> from_joint6_to_motor(const std::vector<double>& joint_pos);
    
    // 电机速度转关节速度（6电机速度 -> 6关节速度）
    // 输入参数：
    //   motor_pos - 当前6个电机位置（弧度）
    //   motor_vel - 6个电机速度（弧度/秒）
    // 返回值：joint_vel - 6个关节速度（弧度/秒）
    // 注意：前4个关节速度与电机速度线性关系，关节5、6速度通过雅可比矩阵逆计算
    std::vector<double> from_motor6_vel_to_joint_vel(const std::vector<double>& motor_pos, const std::vector<double>& motor_vel);
    
    // 关节速度转电机速度（6关节速度 -> 6电机速度）
    // 输入参数：
    //   joint_pos - 当前6个关节位置（弧度）
    //   joint_vel - 6个关节速度（弧度/秒）
    // 返回值：motor_vel - 6个电机速度（弧度/秒）
    // 注意：前4个电机速度与关节速度线性关系，电机5、6速度通过雅可比矩阵计算
    std::vector<double> from_joint6_vel_to_motor_vel(const std::vector<double>& joint_pos, const std::vector<double>& joint_vel);
    
    // 电机速度转关节速度的雅可比导数（用于计算速度映射的线性系数）
    // 返回6x6的映射关系，前4个是单位阵，后2个涉及踝关节并联机构的雅可比矩阵
    std::vector<double> from_motor6_to_joint_der(const std::vector<double>& motor_pos);
    
    // 关节速度转电机速度的雅可比导数
    std::vector<double> from_joint6_to_motor_der(const std::vector<double>& joint_pos);
    
    // ========== 单腿6电机力矩变换函数 ==========
    
    // 踝关节力矩转电机力矩（雅可比转置逆变换）
    // 利用虚功原理: dq_internal = J * dp_ankle => tau_ankle = J^T * tau_motor_internal
    //   => tau_motor_internal = J^{-T} * tau_ankle
    //   最后将 internal 力矩转换到 real 电机空间（motor6 轴反向）
    // 输入参数：
    //   tau_roll  - 踝关节 roll 方向力矩 (N·m)
    //   tau_pitch - 踝关节 pitch 方向力矩 (N·m)
    //   motor_pos - 当前踝关节电机角度 [电机5角度, 电机6角度]（弧度，用于计算当前构型的雅可比）
    // 返回值：
    //   motor_tau - 电机力矩 [电机5力矩, 电机6力矩] (N·m)
    Eigen::Vector2d AnkleTauToMotorTau(double tau_roll, double tau_pitch,
                                       const Eigen::Vector2d& motor_pos);
    
    // 关节力矩转电机力矩（6关节力矩 -> 6电机力矩）
    // 输入参数：
    //   motor_pos - 当前6个电机位置（弧度，用于计算踝关节当前构型的雅可比）
    //   joint_tau - 6个关节力矩 (N·m)
    //     joint_tau[0..3]: 直接映射到电机力矩（串联关节）
    //     joint_tau[4]: ankle_pitch 力矩
    //     joint_tau[5]: ankle_roll 力矩
    // 返回值：motor_tau - 6个电机力矩 (N·m)
    // 注意：前4个力矩直接映射，后2个通过雅可比 J^{-T} 变换
    std::vector<double> from_joint6_tau_to_motor_tau(const std::vector<double>& motor_pos,
                                                     const std::vector<double>& joint_tau);
};

#endif // PLANNING_MOTION_H     