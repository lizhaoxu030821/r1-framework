这个包本来是只控制下肢12个电机的，现在我要增加上肢6个电机和腰部1个电机
当前硬件连接关系：
left_hip_yaw(id 1)、left_hip_roll(id 2)、left_hip_pitch(id 3) 连接 从站0can1 
left_knee(id 4)、left_ankle_pitch、left_ankle_roll 连接 从站0can2,ankle角度在motorcontrol代码中转换到motor5、6下发
right_hip_yaw(id 1)、right_hip_roll(id 2)、right_hip_pitch(id 3) 连接 从站1can1 
right_knee(id 4)、right_ankle_pitch、right_ankle_roll 连接 从站1can2,ankle角度在motorcontrol代码中转换到motor5、6下发

预计上肢硬件连接关系：
left_shoulder_pitch（id 1）、left_shoulder_roll(id 2)、left_elbow(id 3)连接 从站2can1
right_shoulder_pitch（id 4）、left_shoulder_roll(id 5)、left_elbow(id 6)、body_joint(id 7腰部旋转自由度)连接 从站2can2

要求：
1.依据现有motor_control包，扩展到3个从站19自由度电机控制，rviz可视化调用更改为resources/robots/robot_zky_19dof中的文件
2.将这个包设置为本地git仓库，暂不提交github
3.所有修改要有中文记录，存为md文档以备后续审查，修改代码部分也要有详细中文注释标注