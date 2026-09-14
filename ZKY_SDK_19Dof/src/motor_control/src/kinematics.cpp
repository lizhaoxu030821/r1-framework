// Planning_motion.cpp
#include "kinematics.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <algorithm>
inline double wrapToPi(double x)
{
    while (x > M_PI)  x -= 2*M_PI;
    while (x < -M_PI) x += 2*M_PI;
    return x;
}
// 构造函数实现
Planning_motion::Planning_motion()
    : swing_max_adjust_time(5000),
      joint2_init(10.0 * M_PI / 180.0),   // 将角度转换为弧度
      joint3_init(-20.0 * M_PI / 180.0),  // 将角度转换为弧度
      left_base_pos_inBody(-0.06199213683160286, 0.1, -0.8258536498088229),
      right_base_pos_inBody(-0.06199213683160286, -0.1, -0.8258536498088229),
      left_base_pos_inWorld(0, 0, 0),
      right_base_pos_inWorld(0, 0, 0),
      landingvel(0),
      landinngtime(0)
{
    // 可以在此初始化其他成员变量
}

// 旋转矩阵绕Z轴
Eigen::Matrix4d Planning_motion::Rotz(double theta) {
    Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
    R(0,0) = std::cos(theta);
    R(0,1) = -std::sin(theta);
    R(1,0) = std::sin(theta);
    R(1,1) = std::cos(theta);
    return R;
}

// 旋转矩阵绕X轴
Eigen::Matrix4d Planning_motion::Rotx(double theta) {
    Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
    R(1,1) = std::cos(theta);
    R(1,2) = -std::sin(theta);
    R(2,1) = std::sin(theta);
    R(2,2) = std::cos(theta);
    return R;
}

// 旋转矩阵绕Y轴
Eigen::Matrix4d Planning_motion::Roty(double theta) {
    Eigen::Matrix4d R = Eigen::Matrix4d::Identity();
    R(0,0) = std::cos(theta);
    R(0,2) = std::sin(theta);
    R(2,0) = -std::sin(theta);
    R(2,2) = std::cos(theta);
    return R;
}

// 平移矩阵沿X轴
Eigen::Matrix4d Planning_motion::Transx(double d) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0,3) = d;
    return T;
}

// 平移矩阵沿Y轴
Eigen::Matrix4d Planning_motion::Transy(double d) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(1,3) = d;
    return T;
}

// 平移矩阵沿Z轴
Eigen::Matrix4d Planning_motion::Transz(double d) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(2,3) = d;
    return T;
}

// 角度转弧度
double Planning_motion::deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

// 正运动学实现
Eigen::Vector3d Planning_motion::Forward_kinematic(const std::string& whichleg, double theta1, double theta2, double theta3, double theta4) {
    // 角度偏置
    theta2 += joint2_init;
    theta3 += joint3_init;

    // base 到第一关节的偏置（单位：米）
    double dx_0 = 7.62e-3;
    double dy_0 = (whichleg == "left") ? 120e-3 : -120e-3;
    double dz_0 = -116e-3;

    // 第一关节到第二关节的偏置和连杆长度
    double dy_1 = -0.0; // 取消偏置
    double l1 = 420e-3;

    // 第二关节到第三关节的偏置和连杆长度
    double dy_2 = 0.0; // 取消偏置
    double l2 = 350e-3;

    // 第三关节到第四关节的偏置和连杆长度
    double dy_3 = -0.0; // 取消偏置
    double l3 = 0.0;     // 无连杆

    // 计算各个变换矩阵
    Eigen::Matrix4d T1 = Transx(dx_0) * Transy(dy_0) * Transz(dz_0) * Rotx(theta1);
    Eigen::Matrix4d T2 = Roty(theta2) * Transy(dy_1) * Transz(-l1);
    Eigen::Matrix4d T3 = Roty(theta3) * Transy(dy_2) * Transz(-l2);
    Eigen::Matrix4d T4 = Roty(theta4) * Transy(dy_3) * Transz(-l3);

    // 末端执行器的变换矩阵
    Eigen::Matrix4d T_end = T1 * T2 * T3 * T4;

    // 提取脚的位置
    Eigen::Vector3d Position = T_end.block<3,1>(0,3);

    return Position;
}

// 逆运动学实现
std::vector<double> Planning_motion::Inverse_kinematic(const std::string& whichleg, double px, double py, double pz) {
    double dx_0 = 7.62e-3;
    double dy_0 = (whichleg == "left") ? 120e-3 : -120e-3;
    double dz_0 = -116e-3;
    double l1 = 420e-3;
    double l2 = 350e-3;

    // 计算theta1
    double theta1 = std::asin((py - dy_0) / std::sqrt(std::pow(py - dy_0, 2) + std::pow(pz - dz_0, 2)));

    // 计算连杆长度
    double l = std::sqrt(std::pow(px - dx_0, 2) + std::pow(py - dy_0, 2) + std::pow(pz - dz_0, 2));

    // 计算theta3
    double theta3 = -std::acos((std::pow(l1, 2) + std::pow(l2, 2) - std::pow(l, 2)) / (2 * l1 * l2)) + M_PI;

    // 计算theta2
    double theta2 = -std::asin((px - dx_0) / l) - std::acos((std::pow(l, 2) + std::pow(l1, 2) - std::pow(l2, 2)) / (2 * l * l1));

    // 返回关节角度，进行初始值的偏置
    return {theta1, theta2 + (-10.0 * M_PI / 180.0), theta3 + (20.0 * M_PI / 180.0)};
}

// 踝关节力矩计算
double Planning_motion::torque_ankle(double angle) {
    // 定义参数值，单位为弧度或米，刚度填入N/m
    double a = 4.27e-3;      // 小滑轮位置
    double r1 = 5e-3;        // 小滑轮半径
    double r2 = 21e-3;       // 脚踝滑轮半径
    double x0 = 64.5e-3;     // 初始弹簧绳端位置
    double x00 = 5e-3;       // 初始位置弹簧预压缩量
    double k = 17250.0;      // 弹簧刚度

    // 计算力矩，单位为N*m
    double torque = k * r2 * (x0 + x00 - std::sqrt(std::pow(std::sqrt(std::pow(x0, 2) + std::pow(a, 2) - std::pow(r1, 2)) - r2 * angle, 2) + std::pow(r1, 2) - std::pow(a, 2)));
    return torque;
}

// 期望质心参数计算
std::pair<Eigen::Vector3d, Eigen::Vector3d> Planning_motion::exp_com_parameter(const Eigen::Vector3d& base_init_pos, const Eigen::Vector3d& v, double dt) {
    // 计算质心位置和速度
    Eigen::Vector3d com_pos = base_init_pos + v * dt;
    Eigen::Vector3d com_vel = v;

    // 角度和角速度设为零
    Eigen::Vector3d angle = Eigen::Vector3d::Zero();
    Eigen::Vector3d angularvel = Eigen::Vector3d::Zero();

    return std::make_pair(com_pos, com_vel);
}

// 足部位置校正
Eigen::Vector3d Planning_motion::Foot_position_correction(double time_stance, const Eigen::Vector3d& cur_body_pos, const Eigen::Vector3d& cur_body_lin_vel, const Eigen::Vector3d& des_body_lin_vel, const Eigen::Vector3d& des_body_ang_vel) {
    // 定义参数
    double raibert_kp_x = 0.1;
    double raibert_kp_y = 0.1;
    double foot_delta_x_limit = 100e-3;
    double foot_delta_y_limit = 50e-3;

    // 计算相对位置偏移
    double rel_pos_x = raibert_kp_x * (des_body_lin_vel[0] - cur_body_lin_vel[0]);
    double rel_pos_y = raibert_kp_y * (des_body_lin_vel[1] - cur_body_lin_vel[1]);
    double rel_pos_z = 0.0;

    // 限制偏移量
    rel_pos_x = std::max(std::min(rel_pos_x, foot_delta_x_limit), -foot_delta_x_limit);
    rel_pos_y = std::max(std::min(rel_pos_y, foot_delta_y_limit), -foot_delta_y_limit);

    return Eigen::Vector3d(rel_pos_x, rel_pos_y, rel_pos_z);
}

// 多项式转换函数实现（示例，实际需要根据具体公式调整）
double Planning_motion::delta_star_trans(double alpha) {
    double delta_star = 0.0413 * std::pow(alpha, 5) - 0.4853 * std::pow(alpha, 4) + 1.2995 * std::pow(alpha, 3) - 1.1041 * std::pow(alpha, 2) - 1.8596 * alpha + 6.6924 - 0.349;
    return delta_star;
}

double Planning_motion::delta_star_dot_trans(double alpha, double alpha_dot) {
    double delta_star_dot = (0.2066 * std::pow(alpha, 4) - 1.9411 * std::pow(alpha, 3) + 3.8984 * std::pow(alpha, 2) - 2.2083 * alpha - 1.8596) * alpha_dot;
    return delta_star_dot;
}

double Planning_motion::delta_star_slash_trans(double alpha) {
    double delta_star_slash = (0.2066 * std::pow(alpha, 4) - 1.9411 * std::pow(alpha, 3) + 3.8984 * std::pow(alpha, 2) - 2.2083 * alpha - 1.8596);
    return delta_star_slash;
}

double Planning_motion::tau_alpha_trans(double alpha, double tau_delta_star) {
    double tau_alpha = (0.2066 * std::pow(alpha, 4) - 1.9411 * std::pow(alpha, 3) + 3.8984 * std::pow(alpha, 2) - 2.2083 * alpha - 1.8596) * tau_delta_star;
    return tau_alpha;
}

double Planning_motion::alpha_trans(double delta_star) {
    double delta = delta_star + 0.349;
    double alpha = 0.000887 * std::pow(delta,5) - 0.0310 * std::pow(delta,4) + 0.3940 * std::pow(delta,3) - 2.3218 * std::pow(delta,2) + 5.9207 * delta - 3.3977;
    return alpha;
}

double Planning_motion::alpha_dot_trans(double delta_star, double delta_star_dot) {
    double delta = delta_star + 0.349;
    double alpha_dot = (0.0044 * std::pow(delta,4) - 0.1241 * std::pow(delta,3) + 1.1819 * std::pow(delta,2) - 4.6435 * delta + 5.9207) * delta_star_dot;
    return alpha_dot;
}

double Planning_motion::tau_delta_star_trans(double delta_star, double tau_alpha) {
    double delta = delta_star + 0.349;
    double tau_delta_star = (0.0044 * std::pow(delta,4) - 0.1241 * std::pow(delta,3) + 1.1819 * std::pow(delta,2) - 4.6435 * delta + 5.9207) * tau_alpha;
    return tau_delta_star;
}

// q转m
std::vector<double> Planning_motion::from_q_to_m(double q1, double q2, double q3, double q5, double q6, double q7) {
    double m1 = q1;
    double m2 = -q2 + 0.286;
    double m3 = alpha_trans(q3 + 2.95) - 1.7607;
    double m4 = q5;
    double m5 = q6 - 0.286;
    double m6 = -alpha_trans(q7 + 2.95) + 1.7607;
    return {m1, m2, m3, m4, m5, m6};
}

// m转q
std::vector<double> Planning_motion::from_m_to_q(double m1, double m2, double m3, double m4, double m5, double m6) {
    double q1 = m1;
    double q2 = -(m2 - 0.286);
    double q3 = delta_star_trans(m3 + 1.7607) - 2.95;
    double q5 = m4;
    double q6 = -(-0.286 - m5);
    double q7 = delta_star_trans(-m6 + 1.7607) - 2.95;
    return {q1, q2, q3, q5, q6, q7};
}

// q转m的导数
std::vector<double> Planning_motion::from_q_to_m_der(double q1, double q2, double q3, double q5, double q6, double q7) {
    double m1_der = 1.0;
    double m2_der = -1.0;
    double m3_der = alpha_dot_trans(q3 + 2.95, 1.0);
    double m4_der = 1.0;
    double m5_der = 1.0;
    double m6_der = -alpha_dot_trans(q7 + 2.95, 1.0);
    return {m1_der, m2_der, m3_der, m4_der, m5_der, m6_der};
}

// m转q的导数
std::vector<double> Planning_motion::from_m_to_q_der(double m1, double m2, double m3, double m4, double m5, double m6) {
    double q1_der = 1.0;
    double q2_der = -1.0;
    double q3_der = delta_star_dot_trans(m3 + 1.7607, 1.0);
    double q5_der = 1.0;
    double q6_der = 1.0;
    double q7_der = -delta_star_dot_trans(-m6 + 1.7607, 1.0);
    return {q1_der, q2_der, q3_der, q5_der, q6_der, q7_der};
}

// q4转m
std::vector<double> Planning_motion::from_q4_to_m(double q1, double q2, double q3, double q4, double q5, double q6, double q7, double q8) {
    double m1 = -q1;
    double m2 = -q2 + 0.526;
    double m3 = -q3;
    double m4 = q4 + deg2rad(48);
    double m5 = q5 - 0.05;
    double m6 = q6 - 0.50;
    double m7 = q7 + 0.05;
    double m8 = -q8 - deg2rad(43);
    return {m1, m2, m3, m4, m5, m6, m7, m8};
}

// m转q
std::vector<double> Planning_motion::from_m4_to_q(double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8) {
    double q1 = -m1;
    double q2 = -m2 + 0.526;         //+0.456 -> +0.756 左大腿向后偏
    double q3 = -m3;                 //-0.1 -> -0.2 左小腿向前偏
    double q4 = m4 - deg2rad(48);    //-41 -> -70 左脚踝向下偏
    double q5 = m5 + 0.05 ;          //0 -> 0.1 右髋关节向里
    double q6 = m6 + 0.50;           //+0.06 -> +0.46 右大腿向后偏
    double q7 = m7 - 0.05;           //+0.15 -> 0.45 右小腿向后偏
    double q8 = -m8 - deg2rad(43);   //-43 -> -55 右脚踝向下偏
    return {q1, q2, q3, q4, q5, q6, q7, q8};
}

// q转m的导数
std::vector<double> Planning_motion::from_q4_to_m_der(double q1, double q2, double q3, double q4, double q5, double q6, double q7, double q8) {
    double m1_der = -1.0;
    double m2_der = -1.0;
    double m3_der = -1.0;
    double m4_der = 1.0;
    double m5_der = 1.0;
    double m6_der = 1.0;
    double m7_der = 1.0;
    double m8_der = -1.0;
    return {m1_der, m2_der, m3_der, m4_der, m5_der, m6_der, m7_der, m8_der};
}

// m转q的导数
std::vector<double> Planning_motion::from_m4_to_q_der(double m1, double m2, double m3, double m4, double m5, double m6, double m7, double m8) {
    double q1_der = -1.0;
    double q2_der = -1.0;
    double q3_der = -1.0;
    double q4_der = 1.0;
    double q5_der = 1.0;
    double q6_der = 1.0;
    double q7_der = 1.0;
    double q8_der = -1.0;
    return {q1_der, q2_der, q3_der, q4_der, q5_der, q6_der, q7_der, q8_der};
}

// ==================== ParallelNumCal 类实现 ====================

// 构造函数实现
ParallelNumCal::ParallelNumCal()
    : a(0.056),          // 电机输出端转臂长度
      h(0.044),          // 两电机间距与固定坐标系的原点的距离
    //   d(0.05),         // 踝关节并联结构横杆长度
      d(0.0635),         // 踝关节并联结构横杆长度      
      b(0.01544),        // 横杆与踝关节旋转坐标系原点垂直z方向距离
    //   H(0.19356),        // 固定坐标系的原点与踝关节旋转坐标系原点距离
      H(0.20894),        // 固定坐标系的原点与踝关节旋转坐标系原点距离
      L1(0.253),         // 电机5的长杆长度
      L2(0.165),          // 电机6的长杆长度
      motor5_offset(std::asin( b / a )),  // 电机5初始偏置角度
      motor6_offset(std::asin( b / a ))
    //   motor5_offset(10.0 * M_PI / 180.0),  // 电机5初始偏置角度
    //   motor6_offset(10.0 * M_PI / 180.0)
{                                                    
    // 初始化完成
}

inline Eigen::Vector2d
ParallelNumCal::toInternal(const Eigen::Vector2d& q_real) const
{
    return Eigen::Vector2d(
        q_real[0] - motor5_offset,
       -(q_real[1] - motor6_offset)   // m6 轴反向
    );
}

inline Eigen::Vector2d
ParallelNumCal::toReal(const Eigen::Vector2d& q_internal) const
{
    return Eigen::Vector2d(
        q_internal[0] + motor5_offset,
       -(q_internal[1] + motor6_offset)
    );
}


// 计算雅可比矩阵
Eigen::Matrix2d ParallelNumCal::JacobianCal(const Eigen::Vector2d& p, const Eigen::Vector2d& q_internal) {
    // p[0] = roll (psi), p[1] = pitch (phi)
    // q_internal[0] = 电机5角度, q_internal[1] = 电机6角度
    // 踝关节旋转坐标系下的连杆踝关节端坐标（在固定坐标系下）
    Eigen::Vector3d B1_0;
    B1_0 << -a * std::cos(p[1]) + 0.5 * d * std::sin(p[0]) * std::sin(p[1]) + b * std::cos(p[0]) * std::sin(p[1]),
            0.5 * d * std::cos(p[0]) - b * std::sin(p[0]),
            a * std::sin(p[1]) + 0.5 * d * std::sin(p[0]) * std::cos(p[1]) + b * std::cos(p[0]) * std::cos(p[1]) + H;
    
    Eigen::Vector3d B2_0;
    B2_0 << -a * std::cos(p[1]) - 0.5 * d * std::sin(p[0]) * std::sin(p[1]) + b * std::cos(p[0]) * std::sin(p[1]),
            -0.5 * d * std::cos(p[0]) - b * std::sin(p[0]),
            a * std::sin(p[1]) - 0.5 * d * std::sin(p[0]) * std::cos(p[1]) + b * std::cos(p[0]) * std::cos(p[1]) + H;
    
    // 固定坐标系下的长杆电机端坐标（即转臂末端坐标）
    Eigen::Vector3d A1_0;
    A1_0 << -a * std::cos(q_internal[0]),
            0.5 * d,
            -h + a * std::sin(q_internal[0]);
    
    Eigen::Vector3d A2_0;
    A2_0 << -a * std::cos(q_internal[1]),
            -0.5 * d,
            h + a * std::sin(q_internal[1]);
    
    // 连杆向量（从电机端到踝关节端）
    Eigen::Vector3d dx1 = B1_0 - A1_0;
    Eigen::Vector3d dx2 = B2_0 - A2_0;
    
    // 计算雅可比矩阵元素
    double J11_num = dx1[0] * (0.5 * d * std::cos(p[0]) * std::sin(p[1]) - b * std::sin(p[0]) * std::sin(p[1])) +
                     dx1[1] * (-0.5 * d * std::sin(p[0]) - b * std::cos(p[0])) +
                     dx1[2] * (0.5 * d * std::cos(p[0]) * std::cos(p[1]) - b * std::sin(p[0]) * std::cos(p[1]));
    double J11_den = dx1[0] * a * std::sin(q_internal[0]) + dx1[2] * a * std::cos(q_internal[0]);
    double J11 = J11_num / J11_den;
    
    double J12_num = dx1[0] * (a * std::sin(p[1]) + 0.5 * d * std::sin(p[0]) * std::cos(p[1]) + b * std::cos(p[0]) * std::cos(p[1])) +
                     dx1[2] * (a * std::cos(p[1]) - 0.5 * d * std::sin(p[0]) * std::sin(p[1]) - b * std::cos(p[0]) * std::sin(p[1]));
    double J12_den = dx1[0] * a * std::sin(q_internal[0]) + dx1[2] * a * std::cos(q_internal[0]);
    double J12 = J12_num / J12_den;
    
    double J21_num = dx2[0] * (-0.5 * d * std::cos(p[0]) * std::sin(p[1]) - b * std::sin(p[0]) * std::sin(p[1])) +
                     dx2[1] * (0.5 * d * std::sin(p[0]) - b * std::cos(p[0])) +
                     dx2[2] * (-0.5 * d * std::cos(p[0]) * std::cos(p[1]) - b * std::sin(p[0]) * std::cos(p[1]));
    double J21_den = dx2[0] * a * std::sin(q_internal[1]) + dx2[2] * a * std::cos(q_internal[1]);
    double J21 = J21_num / J21_den;
    
    double J22_num = dx2[0] * (a * std::sin(p[1]) - 0.5 * d * std::sin(p[0]) * std::cos(p[1]) + b * std::cos(p[0]) * std::cos(p[1])) +
                     dx2[2] * (a * std::cos(p[1]) + 0.5 * d * std::sin(p[0]) * std::sin(p[1]) - b * std::cos(p[0]) * std::sin(p[1]));
    // double J22_den = dx2[0] * a * std::sin(q_real[0]) + dx2[2] * a * std::cos(q_real[0]);
    double J22_den = dx2[0] * a * std::sin(q_internal[1]) + dx2[2] * a * std::cos(q_internal[1]);

    double J22 = J22_num / J22_den;
    
    Eigen::Matrix2d Jac;
    Jac << J11, J12,
           J21, J22;
    // std::cout << "Jac" << Jac << std::endl;

    return Jac;
}

// 逆运动学：踝关节姿态角 -> 电机角度
Eigen::Vector2d ParallelNumCal::EulerToMotorAngle(double psi, double phi) {
    // psi = roll角度, phi = pitch角度
    // phi += 0.25;

    // 踝关节旋转坐标系与固定坐标系的齐次变换矩阵
    Eigen::Matrix4d T_01;
    T_01 << std::cos(phi), std::sin(phi) * std::sin(psi), std::sin(phi) * std::cos(psi), 0,
            0, std::cos(psi), -std::sin(psi), 0,
            -std::sin(phi), std::cos(phi) * std::sin(psi), std::cos(phi) * std::cos(psi), H,
            0, 0, 0, 1;
    
    // 旋转坐标系下的长杆踝关节端坐标
    Eigen::Vector4d B1_1, B2_1;

    // B1_1 << -a, d / 2.0, b, 1;
    // B2_1 << -a, -d / 2.0, b, 1;

    double B_1_x = std::sqrt( a * a - b * b );
    B1_1 << -B_1_x, d / 2.0, -b, 1;
    B2_1 << -B_1_x, -d / 2.0, -b, 1;
    
    // 固定坐标系下的长杆踝关节端坐标
    Eigen::Vector4d B1_0 = T_01 * B1_1;
    Eigen::Vector4d B2_0 = T_01 * B2_1;
    
    // 相对于电机坐标系的踝关节端位置向量
    Eigen::Vector3d _B1_0 = B1_0.head<3>() - Eigen::Vector3d(0, d / 2.0, -h);  // 相对于电机5
    Eigen::Vector3d _B2_0 = B2_0.head<3>() - Eigen::Vector3d(0, -d / 2.0, h);  // 相对于电机6
    // std::cout << "B1_0 " << B1_0  << std::endl;
    // std::cout << "B2_0 " << B2_0  << std::endl;

    // 电机5角度计算（基于余弦定理和几何关系）
    double Den_1 = std::sqrt(_B1_0[0] * _B1_0[0] + _B1_0[2] * _B1_0[2]);
    // double angle_tmp_1 = std::atan(_B1_0[0] / _B1_0[2]);
    double angle_tmp_1 = std::atan2(_B1_0[0], _B1_0[2]);

    double alpha_1 = std::asin(-(L1 * L1 - _B1_0.squaredNorm() - a * a) / (2 * a * Den_1)) + angle_tmp_1;
    
    // 电机6角度计算（基于余弦定理和几何关系）
    double Den_2 = std::sqrt(_B2_0[0] * _B2_0[0] + _B2_0[2] * _B2_0[2]);
    // std::cout << "Den_2 " << Den_2  << std::endl;

    // double angle_tmp_2 = std::atan(_B2_0[0] / _B2_0[2]);
    double angle_tmp_2 = std::atan2(_B2_0[0], _B2_0[2]);
    // std::cout << "angle_tmp_2 " << angle_tmp_2  << std::endl;

    double alpha_2 = std::asin(-(L2 * L2 - _B2_0.squaredNorm() - a * a) / (2 * a * Den_2)) + angle_tmp_2;
    
    // std::cout << "alpha_1 " << alpha_1  << std::endl;
    // std::cout << "alpha_2 " << alpha_2  << std::endl;
    // Eigen::Vector2d alpha;
    // alpha << alpha_1, alpha_2;
    Eigen::Vector2d q_internal;
    q_internal << alpha_1, alpha_2;
    Eigen::Vector2d q_real = toReal(q_internal);
    // std::cout << "q_real[0] " << q_real[0]  << std::endl;
    // std::cout << "q_real[1] " << q_real[1]  << std::endl;


    return q_real;
}

// 正运动学：电机角度 -> 踝关节姿态角（牛顿迭代法）
Eigen::Vector2d ParallelNumCal::MotorAngleToEuler(const Eigen::Vector2d& q_ref) {
    int Numiter = 20;  // 最大迭代次数
    Eigen::Vector2d q_internal_target = toInternal(q_ref);
    // 牛顿迭代法求解
    int i = 0;
    // 初始姿态角设置（迭代初值）
    Eigen::Vector2d p(0.0, 0.0);  // [roll, pitch]
    // 初始误差设置
    Eigen::Vector2d q_err(1, 1);
    
    // 迭代求解直到收敛（误差小于1e-6）
    while (std::abs(q_err[0]) > 1e-6 || std::abs(q_err[1]) > 1e-6) {
        // 1. 计算当前末端姿态下的电机角度（逆运动学）
        Eigen::Vector2d q_internal_now = toInternal(EulerToMotorAngle(p[0], p[1])) ;
        
        // 2. 计算雅可比矩阵
        Eigen::Matrix2d Jac = JacobianCal(p, q_internal_now);
        
        // 3. 计算当前电机角度误差
        q_err = q_internal_now - q_internal_target;
        
        // 4. 牛顿迭代更新姿态角：p_new = p - J^(-1) * error
        // p = p - Jac.inverse() * q_err;
        // p = p - Jac.completeOrthogonalDecomposition().solve(q_err); 
        // std::cout << q_real << "  q_real" << std::endl;
        // std::cout << q_ref << "  q_ref" << std::endl;                                                              
        // std::cout << "!!!!!!!!!!!!!" << std::endl;
        // std::cout << q_err << "  q_err" << std::endl;
         // LM 阻尼
        Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        double lambda = 1e-4;

        Eigen::Vector2d dp =
            (Jac.transpose() * Jac + lambda * I)
                .ldlt()
                .solve(Jac.transpose() * q_err);

        // 限制步长
        double max_step = 5.0 * M_PI / 180.0;
        dp[0] = std::clamp(dp[0], -max_step, max_step);
        dp[1] = std::clamp(dp[1], -max_step, max_step);

        p -= dp;

        // 角度 wrap
        p[0] = wrapToPi(p[0]);
        p[1] = wrapToPi(p[1]);

        // 最大迭代步数
        if (i > Numiter) {
            std::cout << "迭代未完成" << std::endl;
            break;
        }
        i++;
    }
    return p;
}

// 踝关节角速度转电机角速度（逆向速度运动学）
Eigen::Vector2d ParallelNumCal::AnkleVelToMotorVel(double psi, double phi, double psi_dot, double phi_dot) {
    // 1. 先通过逆运动学计算当前踝关节姿态下对应的电机角度
    Eigen::Vector2d motor_angles = EulerToMotorAngle(psi, phi);
    
    // 2. 计算当前位置的雅可比矩阵
    Eigen::Vector2d p(psi, phi);
    Eigen::Vector2d q_internal =
        toInternal(EulerToMotorAngle(psi, phi));
    Eigen::Matrix2d Jac = JacobianCal(p, q_internal);
    
    // 3. 踝关节角速度向量 p_dot = [roll角速度, pitch角速度]
    Eigen::Vector2d ankle_vel(psi_dot, phi_dot);
    
    // 4. 通过雅可比矩阵计算电机角速度：q_dot = J * p_dot
    Eigen::Vector2d motor_vel = Jac * ankle_vel;
    // return toReal(motor_vel);  // [电机5角速度, 电机6角速度]

    Eigen::Vector2d motor_vel_real;
    motor_vel_real[0] = motor_vel[0];      // m5
    motor_vel_real[1] = -motor_vel[1];     // m6_real    
    return motor_vel_real;  // [电机5角速度, 电机6角速度]


}

// 电机角速度转踝关节角速度（正向速度运动学）
Eigen::Vector2d ParallelNumCal::MotorVelToAnkleVel(const Eigen::Vector2d& q, const Eigen::Vector2d& q_dot) {
   Eigen::Vector2d q_internal = toInternal(q);
    Eigen::Vector2d q_dot_internal(
        q_dot[0],
       -q_dot[1]
    );
    // 1. 先通过正运动学计算当前电机角度下对应的踝关节姿态
    Eigen::Vector2d ankle_angles = MotorAngleToEuler(q);
    
    // 2. 计算当前位置的雅可比矩阵
    Eigen::Matrix2d Jac = JacobianCal(ankle_angles, q_internal);
    
    // 3. 通过雅可比矩阵的逆计算踝关节角速度：p_dot = J^(-1) * q_dot
    Eigen::Vector2d ankle_vel = Jac.inverse() * q_dot_internal;
    
    return ankle_vel;  // [roll角速度, pitch角速度]
}

// ==================== 单腿6电机整合转换函数实现 ====================

// 电机位置转关节位置（6电机 -> 6关节）
std::vector<double> ParallelNumCal::from_motor6_to_joint(double m1, double m2, double m3, double m4, double m5, double m6) {
    std::vector<double> joint_pos(6);
    
    // 前4个关节：线性映射
    joint_pos[0] = m1;
    joint_pos[1] = m2;
    joint_pos[2] = m3;
    joint_pos[3] = m4;

    // 关节5和6（踝关节roll和pitch）：通过并联正运动学计算
    Eigen::Vector2d ankle_motors(m5, m6);
    Eigen::Vector2d ankle_joints = MotorAngleToEuler(ankle_motors);
    joint_pos[4] = ankle_joints[1];  // roll
    joint_pos[5] = ankle_joints[0];  // pitch

    // joint_pos[4] = (m5 +(-m6) ) / 2;  // roll
    // joint_pos[5] = -((-m6) - m5) / 2;  // pitch
    
    // std::cout << "joint_pos[4]: " << joint_pos[4] << ", joint_pos[5]: " << joint_pos[5] << std::endl;

    return joint_pos;
}

// 关节位置转电机位置（6关节 -> 6电机）
std::vector<double> ParallelNumCal::from_joint6_to_motor(double q1, double q2, double q3, double q4, double q5, double q6) {
    std::vector<double> motor_pos(6);
    
    // 前4个电机：线性映射
    motor_pos[0] = q1;
    motor_pos[1] = q2;
    motor_pos[2] = q3;
    motor_pos[3] = q4;
    
    // 电机5和6（踝关节并联电机）：通过并联逆运动学计算
    double ankle_roll = q6;
    double ankle_pitch = q5;
    Eigen::Vector2d ankle_motors = EulerToMotorAngle(ankle_roll, ankle_pitch);
    motor_pos[4] = ankle_motors[0];
    motor_pos[5] = ankle_motors[1];

    // motor_pos[4] = q5 + q6;
    // motor_pos[5] = q6 - q5;  
    
    return motor_pos;
}

// 电机位置转关节位置（向量版本）
std::vector<double> ParallelNumCal::from_motor6_to_joint(const std::vector<double>& motor_pos) {
    if (motor_pos.size() != 6) {
        throw std::runtime_error("from_motor6_to_joint: 输入必须是6个电机位置！");
    }
    return from_motor6_to_joint(motor_pos[0], motor_pos[1], motor_pos[2], 
                                 motor_pos[3], motor_pos[4], motor_pos[5]);
}

// 关节位置转电机位置（向量版本）
std::vector<double> ParallelNumCal::from_joint6_to_motor(const std::vector<double>& joint_pos) {
    if (joint_pos.size() != 6) {
        throw std::runtime_error("from_joint6_to_motor: 输入必须是6个关节位置！");
    }
    return from_joint6_to_motor(joint_pos[0], joint_pos[1], joint_pos[2], 
                                 joint_pos[3], joint_pos[4], joint_pos[5]);
}

// 电机速度转关节速度（6电机速度 -> 6关节速度）
std::vector<double> ParallelNumCal::from_motor6_vel_to_joint_vel(const std::vector<double>& motor_pos, const std::vector<double>& motor_vel) {
    if (motor_pos.size() != 6 || motor_vel.size() != 6) {
        throw std::runtime_error("from_motor6_vel_to_joint_vel: 输入必须是6个电机位置和速度！");
    }
    
    std::vector<double> joint_vel(6);
    
    // 前4个关节速度：线性映射（直接等于电机速度）
    joint_vel[0] = motor_vel[0];
    joint_vel[1] = motor_vel[1];
    joint_vel[2] = motor_vel[2];
    joint_vel[3] = motor_vel[3];
    
    // 关节5、6速度（踝关节roll和pitch速度）：通过雅可比矩阵逆计算
    Eigen::Vector2d ankle_motor_pos(motor_pos[4], motor_pos[5]);
    Eigen::Vector2d ankle_motor_vel(motor_vel[4], motor_vel[5]);
    Eigen::Vector2d ankle_joint_vel = MotorVelToAnkleVel(ankle_motor_pos, ankle_motor_vel);
    joint_vel[4] = ankle_joint_vel[1];  // roll速度
    joint_vel[5] = ankle_joint_vel[0];  // pitch速度
    
    return joint_vel;
}

// 关节速度转电机速度（6关节速度 -> 6电机速度）
std::vector<double> ParallelNumCal::from_joint6_vel_to_motor_vel(const std::vector<double>& joint_pos, const std::vector<double>& joint_vel) {
    if (joint_pos.size() != 6 || joint_vel.size() != 6) {
        throw std::runtime_error("from_joint6_vel_to_motor_vel: 输入必须是6个关节位置和速度！");
    }
    
    std::vector<double> motor_vel(6);
    
    // 前4个电机速度：线性映射（直接等于关节速度）
    motor_vel[0] = joint_vel[0];
    motor_vel[1] = joint_vel[1];
    motor_vel[2] = joint_vel[2];
    motor_vel[3] = joint_vel[3];
    
    // 电机5、6速度（踝关节并联电机速度）：通过雅可比矩阵计算
    double ankle_roll = joint_pos[5];
    double ankle_pitch = joint_pos[4];
    double ankle_roll_vel = joint_vel[5];
    double ankle_pitch_vel = joint_vel[4];
    Eigen::Vector2d ankle_motor_vel = AnkleVelToMotorVel(ankle_roll, ankle_pitch, 
                                                          ankle_roll_vel, ankle_pitch_vel);
    motor_vel[4] = ankle_motor_vel[0];
    motor_vel[5] = ankle_motor_vel[1];
    
    return motor_vel;
}

// 电机速度转关节速度的雅可比导数
std::vector<double> ParallelNumCal::from_motor6_to_joint_der(const std::vector<double>& motor_pos) {
    if (motor_pos.size() != 6) {
        throw std::runtime_error("from_motor6_to_joint_der: 输入必须是6个电机位置！");
    }
    
    // 返回6个导数值，前4个是1.0（线性映射），后2个是踝关节雅可比矩阵逆的对角线元素的简化表示
    std::vector<double> derivatives(6);
    
    // 前4个关节：单位映射
    derivatives[0] = 1.0;
    derivatives[1] = 1.0;
    derivatives[2] = 1.0;
    derivatives[3] = 1.0;
    
    // 关节5、6：通过雅可比矩阵逆得到导数关系
    // 这里返回简化的标量导数，实际是2x2雅可比矩阵逆的迹的平均值作为代表
    Eigen::Vector2d ankle_motor_pos(motor_pos[4], motor_pos[5]);
    Eigen::Vector2d ankle_joint_pos = MotorAngleToEuler(ankle_motor_pos);
    Eigen::Matrix2d Jac = JacobianCal(ankle_joint_pos, ankle_motor_pos);
    Eigen::Matrix2d Jac_inv = Jac.inverse();
    
    // 简化表示：返回雅可比矩阵逆的对角元素（近似线性导数）
    derivatives[4] = Jac_inv(0, 0);  // d(joint5)/d(motor5)的主导分量
    derivatives[5] = Jac_inv(1, 1);  // d(joint6)/d(motor6)的主导分量
    
    return derivatives;
}

// ==================== 单腿6电机力矩变换函数实现 ====================

Eigen::Vector2d ParallelNumCal::AnkleTauToMotorTau(
    double tau_roll, double tau_pitch,
    const Eigen::Vector2d& motor_pos)
{
    // FK: 电机角 -> 踝关节角 [psi(roll), phi(pitch)]
    Eigen::Vector2d ankle_pos = MotorAngleToEuler(motor_pos);

    // 当前构型的雅可比 (dq_internal / dp_ankle)
    Eigen::Vector2d q_int = toInternal(motor_pos);
    Eigen::Matrix2d J = JacobianCal(ankle_pos, q_int);

    double det = J.determinant();
    if (std::abs(det) < 1e-8) {
        return Eigen::Vector2d::Zero();
    }

    // p-space: [psi(roll), phi(pitch)]
    Eigen::Vector2d tau_ankle(tau_roll, tau_pitch);

    // tau_motor_internal = J^{-T} * tau_ankle
    Eigen::Vector2d tau_m_int = J.inverse().transpose() * tau_ankle;

    // internal -> real（motor6 轴反向）
    return Eigen::Vector2d(tau_m_int[0], -tau_m_int[1]);
}

std::vector<double> ParallelNumCal::from_joint6_tau_to_motor_tau(
    const std::vector<double>& motor_pos,
    const std::vector<double>& joint_tau)
{
    if (motor_pos.size() != 6 || joint_tau.size() != 6) {
        throw std::runtime_error("from_joint6_tau_to_motor_tau: 输入必须是6个电机位置和6个关节力矩！");
    }

    std::vector<double> motor_tau(6);

    // 前4个关节: 串联结构，力矩直接映射
    motor_tau[0] = joint_tau[0];
    motor_tau[1] = joint_tau[1];
    motor_tau[2] = joint_tau[2];
    motor_tau[3] = joint_tau[3];

    // 踝关节: 并联结构，通过 J^{-T} 变换
    // joint_tau[4] = ankle_pitch 力矩, joint_tau[5] = ankle_roll 力矩
    // AnkleTauToMotorTau 的 p-space 顺序为 (roll, pitch)
    Eigen::Vector2d ankle_motor_tau = AnkleTauToMotorTau(
        joint_tau[5],  // roll
        joint_tau[4],  // pitch
        Eigen::Vector2d(motor_pos[4], motor_pos[5]));

    motor_tau[4] = ankle_motor_tau[0];
    motor_tau[5] = ankle_motor_tau[1];

    return motor_tau;
}

// 关节速度转电机速度的雅可比导数
std::vector<double> ParallelNumCal::from_joint6_to_motor_der(const std::vector<double>& joint_pos) {
    if (joint_pos.size() != 6) {
        throw std::runtime_error("from_joint6_to_motor_der: 输入必须是6个关节位置！");
    }
    
    // 返回6个导数值，前4个是1.0（线性映射），后2个是踝关节雅可比矩阵的对角线元素的简化表示
    std::vector<double> derivatives(6);
    
    // 前4个电机：单位映射
    derivatives[0] = 1.0;
    derivatives[1] = 1.0;
    derivatives[2] = 1.0;
    derivatives[3] = 1.0;
    
    // 电机5、6：通过雅可比矩阵得到导数关系
    double ankle_roll = joint_pos[4];
    double ankle_pitch = joint_pos[5];
    Eigen::Vector2d ankle_motor_pos = EulerToMotorAngle(ankle_roll, ankle_pitch);
    Eigen::Vector2d ankle_joint_pos(ankle_roll, ankle_pitch);
    Eigen::Matrix2d Jac = JacobianCal(ankle_joint_pos, ankle_motor_pos);
    
    // 简化表示：返回雅可比矩阵的对角元素（近似线性导数）
    derivatives[4] = Jac(0, 0);  // d(motor5)/d(joint5)的主导分量
    derivatives[5] = Jac(1, 1);  // d(motor6)/d(joint6)的主导分量
    
    return derivatives;
}