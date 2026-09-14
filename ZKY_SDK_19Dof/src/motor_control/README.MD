# 电机控制模块更新说明

## 功能概览
- 适配 19 个 EtherCAT 电机：下肢 12 个、上肢 6 个、腰部旋转 1 个。
- 当前拓扑为 3 个 EtherCAT 从站：从站0左腿 6 电机，从站1右腿 6 电机，从站2上肢/腰部 7 电机。
- 通过 SOEM 底层库直接驱动，从 `app/transmit.cpp` 与 `app/motor_control.c` 获取实时状态。
- 节点订阅 `/motor_params`、`imu/data`、`camera/odom/sample`，并发布 `motor_callback` 与 `joint_states`。
- 自动记录 CSV 日志（`record_data/motor_data_record/`）。

## 构建依赖
- ROS 依赖：`roscpp`、`rospy`、`std_msgs`、`sensor_msgs`、`nav_msgs`。
- 第三方：SOEM（已在 `CMakeLists.txt` 中通过 `add_subdirectory(app/thirdparty/soem)` 构建）。
- 其他：Eigen3、Boost、pthread。

编译方式：
```bash
catkin_make --pkg motor_control
```
若已有系统级 SOEM，可在 `CMakeLists.txt` 中改为 `find_package` 并移除内置子目录。

## 关键参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `ethercat_interface` | EtherCAT 网卡名称 | `enp45s0` |

参数通过 launch 文件传入：
```xml
<param name="ethercat_interface" value="enp45s0" />
```

## 主要话题
- 订阅
  - `/motor_params` (`std_msgs/Float64MultiArray`)，新版长度为 `19 × 6`（kp、kd、pos、vel、tau、mode）。
  - 兼容旧版 `12 × 6` 下肢消息；此时上肢和腰部命令保持零输出。
  - `imu/data` (`sensor_msgs/Imu`)
  - `camera/odom/sample` (`nav_msgs/Odometry`)
- 发布
  - `motor_callback` (`std_msgs/Float64MultiArray`)：原始/滤波后的电机状态与目标参数。
  - `joint_states` (`sensor_msgs/JointState`)

## 19 DOF 关节顺序
`/motor_params`、`joint_states` 和底层全局电机索引使用同一顺序：

| 索引 | 关节名 | 硬件路由 |
|------|--------|----------|
| 0 | `left_hip_yaw_joint` | 从站0 CAN1 id1 |
| 1 | `left_hip_roll_joint` | 从站0 CAN1 id2 |
| 2 | `left_hip_pitch_joint` | 从站0 CAN1 id3 |
| 3 | `left_knee_joint` | 从站0 CAN2 id4 |
| 4 | `left_ankle_pitch_joint` | 从站0 CAN2，经并联踝转换到 motor5 |
| 5 | `left_ankle_roll_joint` | 从站0 CAN2，经并联踝转换到 motor6 |
| 6 | `right_hip_yaw_joint` | 从站1 CAN1 id1 |
| 7 | `right_hip_roll_joint` | 从站1 CAN1 id2 |
| 8 | `right_hip_pitch_joint` | 从站1 CAN1 id3 |
| 9 | `right_knee_joint` | 从站1 CAN2 id4 |
| 10 | `right_ankle_pitch_joint` | 从站1 CAN2，经并联踝转换到 motor5 |
| 11 | `right_ankle_roll_joint` | 从站1 CAN2，经并联踝转换到 motor6 |
| 12 | `left_shoulder_pitch_joint` | 从站2 CAN1 id1 |
| 13 | `left_shoulder_roll_joint` | 从站2 CAN1 id2 |
| 14 | `left_elbow_joint` | 从站2 CAN1 id3 |
| 15 | `right_shoulder_pitch_joint` | 从站2 CAN2 id4 |
| 16 | `right_shoulder_roll_joint` | 从站2 CAN2 id5 |
| 17 | `right_elbow_joint` | 从站2 CAN2 id6 |
| 18 | `body_joint` | 从站2 CAN2 id7 |

## Launch 调整
- `rviz_display.launch` 默认加载 `resources/robots/robot_zky_19dof/urdf/robot_zky_fixed_all.urdf`。
- 19 DOF URDF 根链接为 `base_link`，已移除旧版 `base_link -> pelvis` 静态 TF。
- `test1.launch`、`test2.launch`、`test3.launch` 均改用 `ethercat_interface` 参数，不再配置 USB 设备序列号/路径。
- 若需切换网卡，可在启动时追加：
  ```bash
  roslaunch motor_control test2.launch ethercat_interface:=enx000000
  ```

## 目录调整
- 新增编译 19 电机所需的 SOEM、传输与解析源文件。
- 删除废弃的 DaMiao 相关头文件（`damiao_tiny_v2.h`、`Hcanbus.h`）。

## 日志与安全
- CSV 日志位于 `record_data/motor_data_record/`，包含电机状态、目标指令、IMU 及足端数据。
- 控制循环设有限幅，越界后自动降级到安全增益 (`kp=300`, `kd=3`) 并清零速度/力矩。

## 已知事项
- 建议在具有实时补丁或低延迟内核的环境下运行，以保证 1 kHz EtherCAT 周期。
- 若改用外部 SOEM 版本，请确认编译参数和网卡驱动兼容性。
- 从站2需要支持 7 个 CAN 帧槽位的 PDO；若初始化日志提示从站2 IN/OUT 帧容量小于 7，需要同步更新从站固件或 ESI/PDO 配置。
