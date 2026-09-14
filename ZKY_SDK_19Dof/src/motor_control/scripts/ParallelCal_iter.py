"""
踝关节并联结构正逆运动学
"""

import numpy as np
from numpy import sin as s
from numpy import cos as c

# 踝关节并联结构运动学求解类
# 基于坐标变换计算逆运动学（姿态角->电机角度），基于牛顿迭代法数值计算正运动学（电机角度->姿态角）
class ParallelNumCal:
    """
    踝关节并联结构正逆运动学求解器
    
    结构参数说明：
    - 两个电机(电机5和电机6)通过并联连杆机构驱动踝关节
    - 电机输出转臂连接长杆，长杆另一端连接踝关节横杆
    - 通过控制两个电机角度实现踝关节的pitch和roll方向运动
    """
    def __init__(self):
        # self.a = 0.02           # 电机输出端转臂长度
        # self.h = 0.04           # 两电机间距与固定坐标系的原点的距离
        # self.d = 0.081          # 踝关节并联结构横杆长度
        # self.b = 0.004          # 横杆与踝关节旋转坐标系原点垂直z方向距离
        # self.H = 0.174          # 固定坐标系的原点与踝关节旋转坐标系原点距离
        # self.L1 = 0.218         # 电机5的长杆长度
        # self.L2 = 0.138         # 电机6的长杆长度
        # pass
        self.a = 0.056           # 电机输出端转臂长度
        self.h = 0.044           # 两电机间距与固定坐标系的原点的距离
        self.d = 0.0645          # 踝关节并联结构横杆长度
        self.b = 0.01544          # 横杆与踝关节旋转坐标系原点垂直z方向距离
        self.H = 0.19356         # 固定坐标系的原点与踝关节旋转坐标系原点距离
        self.L1 = 0.253        # 电机5的长杆长度
        self.L2 = 0.165         # 电机6的长杆长度
        pass
    
    # 计算雅可比矩阵
    # 输入参数：p - 踝关节末端姿态角[roll, pitch]，q_real - 电机角度[电机5, 电机6]
    def JacobianCal(self, p, q_real):
        a = self.a
        d = self.d
        b = self.b
        B1_0 = np.array([-a*c(p[1]) + 0.5*d*s(p[0])*s(p[1]) + b*c(p[0])*s(p[1]),
                        0.5*d*c(p[0]) - b*s(p[0]),
                        a*s(p[1]) + 0.5*d*s(p[0])*c(p[1]) + b*c(p[0])*c(p[1]) + self.H])
        
        B2_0 = np.array([-a*c(p[1]) - 0.5*d*s(p[0])*s(p[1]) + b*c(p[0])*s(p[1]),
                        -0.5*d*c(p[0]) - b*s(p[0]),
                        a*s(p[1]) - 0.5*d*s(p[0])*c(p[1]) + b*c(p[0])*c(p[1]) + self.H])
        
        # 固定坐标系下的长杆电机端坐标（即转臂末端坐标）
        A1_0 = np.array([-a*c(q_real[0]), 0.5*d, -self.h + a*s(q_real[0])])  # 电机5转臂末端
        A2_0 = np.array([-a*c(q_real[1]), -0.5*d, self.h + a*s(q_real[1])])  # 电机6转臂末端

        # 连杆向量（从电机端到踝关节端）
        dx1 = B1_0 - A1_0  # 长杆1向量
        dx2 = B2_0 - A2_0  # 长杆2向量

        # 计算雅可比矩阵（用于牛顿迭代法求解正运动学）
        J11 = (dx1[0]*(0.5*d*c(p[0])*s(p[1])-b*s(p[0])*s(p[1])) + dx1[1]*(-0.5*d*s(p[0])-b*c(p[0])) + \
            dx1[2]*(0.5*d*c(p[0])*c(p[1])-b*s(p[0])*c(p[1])))/(dx1[0]*a*s(q_real[0])+dx1[2]*a*c(q_real[0]))
        J12 = (dx1[0]*(a*s(p[1])+0.5*d*s(p[0])*c(p[1])+b*c(p[0])*c(p[1])) + dx1[1]*0 + \
            dx1[2]*(a*c(p[1])-0.5*d*s(p[0])*s(p[1])-b*c(p[0])*s(p[1])))/(dx1[0]*a*s(q_real[0])+dx1[2]*a*c(q_real[0]))
        J21 = (dx2[0]*(-0.5*d*c(p[0])*s(p[1])-b*s(p[0])*s(p[1])) + dx2[1]*(0.5*d*s(p[0])-b*c(p[0])) + \
            dx2[2]*(-0.5*d*c(p[0])*c(p[1])-b*s(p[0])*c(p[1])))/(dx2[0]*a*s(q_real[1])+dx2[2]*a*c(q_real[1]))
        J22 = (dx2[0]*(a*s(p[1])-0.5*d*s(p[0])*c(p[1])+b*c(p[0])*c(p[1])) + dx2[1]*0 + \
            dx2[2]*(a*c(p[1])+0.5*d*s(p[0])*s(p[1])-b*c(p[0])*s(p[1])))/(dx2[0]*a*s(q_real[0])+dx2[2]*a*c(q_real[0]))
        
        Jac = np.array([[J11, J12], [J21, J22]])

        return Jac

    # 逆运动学：通过踝关节末端pitch和roll方向的角度phi和psi计算两个电机输出端的旋转角度alpha
    # 输入参数：psi - roll角度，phi - pitch角度
    # 返回值：alpha - 电机角度[电机5角度，电机6角度]
    def EulerToMotorAngle(self, psi, phi):
        # 踝关节旋转坐标系与固定坐标系的齐次变换矩阵
        T_01 = np.array([[np.cos(phi), np.sin(phi)*np.sin(psi), np.sin(phi)*np.cos(psi), 0],
                         [0, np.cos(psi), -np.sin(psi), 0],
                         [-np.sin(phi), np.cos(phi)*np.sin(psi), np.cos(phi)*np.cos(psi), self.H],
                         [0, 0, 0, 1]])
        
        
        # 旋转坐标系下的长杆踝关节端坐标
        B1_1= np.array([-self.a, self.d/2, self.b, 1])
        B2_1= np.array([-self.a, -self.d/2, self.b, 1])

        # 固定坐标系下的长杆踝关节端坐标
        B1_0 = T_01 @ B1_1
        B2_0 = T_01 @ B2_1

        # 相对于电机坐标系的踝关节端位置向量
        _B1_0 = B1_0[0:3] - np.array([0, self.d/2, -self.h])  # 相对于电机5
        _B2_0 = B2_0[0:3] - np.array([0, -self.d/2, self.h])  # 相对于电机6

        # 电机5角度计算（基于余弦定理和几何关系）
        Den_1 = np.sqrt( _B1_0[0]**2+ _B1_0[2]**2)  # 投影距离
        angle_tmp_1 = np.arctan(_B1_0[0]/_B1_0[2])  # 基准角度
        alpha_1 = np.arcsin(-(self.L1**2 - np.sum(_B1_0**2) - self.a**2)/(2*self.a*Den_1)) + angle_tmp_1

        # 电机6角度计算（基于余弦定理和几何关系）
        Den_2 = np.sqrt( _B2_0[0]**2+ _B2_0[2]**2)  # 投影距离
        angle_tmp_2 = np.arctan(_B2_0[0]/_B2_0[2])  # 基准角度
        alpha_2 = np.arcsin(-(self.L2**2 - np.sum(_B2_0**2) - self.a**2)/(2*self.a*Den_2)) + angle_tmp_2

        # 返回电机角度：[电机5角度， 电机6角度]
        alpha = np.array([alpha_1, alpha_2])

        # print(T_01)
        # print(B1_1, B2_1)
        # print(B1_0, B2_0)
        # print(_B1_0, _B2_0)
        # print(Den_1, angle_tmp_1, alpha_1)

        return alpha

    # 正运动学：通过两个电机输出端的旋转角度alpha计算踝关节末端pitch和roll方向的角度phi和psi（牛顿迭代法）
    # 输入参数：q_ref - 电机参考角度[电机5角度，电机6角度]
    # 返回值：theta - 踝关节末端姿态角[roll角度，pitch角度]
    def MotorAngleToEuler(self, q_ref):
        a = self.a
        d = self.d
        b = self.b
        Numiter = 20  # 最大迭代次数

        # 牛顿迭代法求解
        i = 0
        # 初始姿态角设置（迭代初值）
        p = np.array([0, 0])  # [roll, pitch]
        # 初始误差设置
        q_err = np.array([1, 1])  # 电机角度误差
        # 迭代求解直到收敛（误差小于1e-6）
        while np.abs(q_err[0]) > 1e-6 or np.abs(q_err[1]) > 1e-6:
            # 1. 计算当前末端姿态下的电机角度（逆运动学）
            q_real = self.EulerToMotorAngle(p[0], p[1])
            # print("q_real: ", q_real)
            # print("="*50)
            # print("theta_now: ", p)
            # 2. 计算雅可比矩阵
            Jac = self.JacobianCal(p, q_real)

            # 3. 计算当前电机角度误差
            q_err = q_real - q_ref

            # 4. 牛顿迭代更新姿态角：p_new = p - J^(-1) * error
            p = p - np.linalg.inv(Jac) @ q_err
            # print("Jac: ", Jac)
            # print("q_real: ", q_real)
            # print("q_err: ", q_err)
            # print("theta_next: ", p)
            # print(i)

            # 最大迭代步数
            if i > Numiter:
                break
            i+=1

            print("="*50)
        
        # 返回踝关节末端姿态角：[roll角度，pitch角度]
        theta = p
        return theta

    # 踝关节角速度转电机角速度（逆向速度运动学）
    # 输入参数：
    #   psi - 踝关节roll角度（弧度）
    #   phi - 踝关节pitch角度（弧度）
    #   psi_dot - 踝关节roll方向角速度（弧度/秒）
    #   phi_dot - 踝关节pitch方向角速度（弧度/秒）
    # 返回值：
    #   motor_vel - 电机角速度[电机5角速度, 电机6角速度]（弧度/秒）
    # 
    # 原理说明：
    # 对于并联机构，雅可比矩阵描述了末端速度与关节速度之间的映射关系：
    # q_dot = J * p_dot
    # 其中：q_dot = [电机5角速度, 电机6角速度]^T
    #       p_dot = [roll角速度, pitch角速度]^T
    #       J 是雅可比矩阵
    def AnkleVelToMotorVel(self, psi, phi, psi_dot, phi_dot):
        # 1. 先通过逆运动学计算当前踝关节姿态下对应的电机角度
        motor_angles = self.EulerToMotorAngle(psi, phi)
        
        # 2. 计算当前位置的雅可比矩阵
        # 踝关节姿态向量 p = [roll, pitch]
        p = np.array([psi, phi])
        Jac = self.JacobianCal(p, motor_angles)
        
        # 3. 踝关节角速度向量 p_dot = [roll角速度, pitch角速度]
        ankle_vel = np.array([psi_dot, phi_dot])
        
        # 4. 通过雅可比矩阵计算电机角速度：q_dot = J * p_dot
        motor_vel = Jac @ ankle_vel
        
        return motor_vel  # [电机5角速度, 电机6角速度]

    # 电机角速度转踝关节角速度（正向速度运动学）
    # 输入参数：
    #   q - 电机角度[电机5角度, 电机6角度]（弧度）
    #   q_dot - 电机角速度[电机5角速度, 电机6角速度]（弧度/秒）
    # 返回值：
    #   ankle_vel - 踝关节角速度[roll角速度, pitch角速度]（弧度/秒）
    # 
    # 原理说明：
    # 对于并联机构，已知电机角速度求末端角速度需要使用雅可比矩阵的逆：
    # p_dot = J^(-1) * q_dot
    # 其中：p_dot = [roll角速度, pitch角速度]^T
    #       q_dot = [电机5角速度, 电机6角速度]^T
    #       J^(-1) 是雅可比矩阵的逆
    def MotorVelToAnkleVel(self, q, q_dot):
        # 1. 先通过正运动学计算当前电机角度下对应的踝关节姿态
        ankle_angles = self.MotorAngleToEuler(q)
        
        # 2. 计算当前位置的雅可比矩阵
        Jac = self.JacobianCal(ankle_angles, q)
        
        # 3. 通过雅可比矩阵的逆计算踝关节角速度：p_dot = J^(-1) * q_dot
        ankle_vel = np.linalg.inv(Jac) @ q_dot
        
        return ankle_vel  # [roll角速度, pitch角速度]


if __name__ == "__main__":
    # 创建踝关节并联结构运动学求解器实例
    ParaCal = ParallelNumCal()
    
    print("="*50)
    print("测试1：踝关节姿态角 -> 电机角度 -> 踝关节姿态角（正逆运动学验证）")
    print("="*50)
    
    # 设置踝关节末端姿态角 [roll, pitch] (单位：弧度)
    endang = np.array([-24.65/180*np.pi, 5.13/180*np.pi])

    # 逆运动学：通过踝关节pitch和roll方向的角度计算两个电机输出端的旋转角度
    alpha = ParaCal.EulerToMotorAngle(endang[0], endang[1])  # 输入：roll角度psi，pitch角度phi
    print("="*50)

    # 正运动学：通过两个电机输出端的旋转角度计算踝关节末端pitch和roll方向的角度（牛顿迭代法）
    theta = ParaCal.MotorAngleToEuler(alpha)

    print("*"*50)
    print("电机角度（度）: ", alpha/np.pi*180)  # [电机5角度, 电机6角度]
    print("踝关节姿态角_参考值（弧度）: ", endang)  # [roll角度, pitch角度]
    print("踝关节姿态角_计算值（弧度）: ", theta)
    print("*"*50)
    print()
    print()
    
    # ====== 测试2：速度转换验证 ======
    print("="*50)
    print("测试2：踝关节角速度 <-> 电机角速度 转换验证")
    print("="*50)
    
    # 设置踝关节姿态角和角速度
    psi = endang[0]      # roll角度（弧度）
    phi = endang[1]      # pitch角度（弧度）
    psi_dot = 0.5        # roll方向角速度（弧度/秒）
    phi_dot = 0.3        # pitch方向角速度（弧度/秒）
    
    print("输入踝关节状态：")
    print(f"  Roll角度: {psi:.4f} rad ({psi*180/np.pi:.2f}°)")
    print(f"  Pitch角度: {phi:.4f} rad ({phi*180/np.pi:.2f}°)")
    print(f"  Roll角速度: {psi_dot:.4f} rad/s")
    print(f"  Pitch角速度: {phi_dot:.4f} rad/s")
    print()
    
    # 测试：踝关节角速度 -> 电机角速度
    motor_vel = ParaCal.AnkleVelToMotorVel(psi, phi, psi_dot, phi_dot)
    print("逆向速度运动学（踝关节角速度 -> 电机角速度）：")
    print(f"  电机5角速度: {motor_vel[0]:.6f} rad/s ({motor_vel[0]*180/np.pi:.4f} °/s)")
    print(f"  电机6角速度: {motor_vel[1]:.6f} rad/s ({motor_vel[1]*180/np.pi:.4f} °/s)")
    print()
    
    # 测试：电机角速度 -> 踝关节角速度（验证逆向转换）
    motor_angles = ParaCal.EulerToMotorAngle(psi, phi)
    ankle_vel_recovered = ParaCal.MotorVelToAnkleVel(motor_angles, motor_vel)
    print("正向速度运动学（电机角速度 -> 踝关节角速度）：")
    print(f"  Roll角速度（恢复值）: {ankle_vel_recovered[0]:.6f} rad/s")
    print(f"  Pitch角速度（恢复值）: {ankle_vel_recovered[1]:.6f} rad/s")
    print()
    
    # 验证精度
    vel_error = np.abs(ankle_vel_recovered - np.array([psi_dot, phi_dot]))
    print("速度转换精度验证：")
    print(f"  Roll角速度误差: {vel_error[0]:.2e} rad/s")
    print(f"  Pitch角速度误差: {vel_error[1]:.2e} rad/s")
    print("*"*50)
    print()
    print()
    
    # # ====== 测试2：直接从电机角度计算踝关节姿态角（正运动学） ======
    # print("="*50)
    # print("测试2：电机角度 -> 踝关节姿态角（正运动学）")
    # print("="*50)
    
    # # 输入：电机角度（单位：度）
    # # motor_angles_deg[0] 是电机5的角度
    # # motor_angles_deg[1] 是电机6的角度
    # motor_angles_deg = np.array([15.8, -15.8])  # 可以修改这里的电机角度值进行测试
    
    # # 转换为弧度
    # motor_angles_rad = motor_angles_deg / 180.0 * np.pi
    
    # print(f"输入电机角度（度）:")
    # print(f"  电机5角度: {motor_angles_deg[0]:.2f}°")
    # print(f"  电机6角度: {motor_angles_deg[1]:.2f}°")
    # print()
    
    # # 正运动学：通过两个电机输出端的旋转角度计算踝关节末端pitch和roll方向的角度（牛顿迭代法）
    # theta2 = ParaCal.MotorAngleToEuler(motor_angles_rad)
    
    # # 转换为度数
    # theta2_deg = theta2 / np.pi * 180.0
    
    # print("="*50)
    # print("输出踝关节姿态角（弧度）:")
    # print(f"  Roll角度: {theta2[0]:.6f} rad")
    # print(f"  Pitch角度: {theta2[1]:.6f} rad")
    # print()
    # print("输出踝关节姿态角（度）:")
    # print(f"  Roll角度: {theta2_deg[0]:.4f}°")
    # print(f"  Pitch角度: {theta2_deg[1]:.4f}°")
    # print("="*50)
    
