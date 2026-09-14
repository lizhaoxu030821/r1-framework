# 脚踝电机测试程序使用说明

## 简介

这个测试程序专门用于测试双足机器人的脚踝电机（电机5和电机6）。程序使用 `kinematics.h` 中的 `ParallelNumCal` 类进行踝关节并联机构的运动学计算，确保运动轨迹安全且符合机械结构约束。

## 功能特性

- ✅ **安全的小幅度运动**: Roll 和 Pitch 方向的运动幅度均限制在 ±4.6° 以内
- ✅ **运动学函数测试**: 使用 `EulerToMotorAngle`、`AnkleVelToMotorVel` 和 `MotorAngleToEuler` 函数
- ✅ **平滑运动控制**: 采用正弦轨迹，频率为 0.5Hz，运动平稳
- ✅ **实时反馈**: 显示目标角度和实际角度的对比
- ✅ **三阶段控制**:
  1. 缓慢移动到初始位置 (2秒)
  2. 周期性正弦运动 (20秒)
  3. 平滑返回零位 (2秒)

## 编译

在工作空间根目录下执行：

```bash
cd ~/bipedal_deploy_ZKY
catkin_make
source devel/setup.bash
```

## 使用方法

### 方法1: 使用 Launch 文件 (推荐)

#### 测试左腿脚踝:
```bash
roslaunch motor_control ankle_motor_test.launch
```

#### 测试右腿脚踝:
```bash
roslaunch motor_control ankle_motor_test.launch test_leg:=right
```

#### 自定义网络接口:
```bash
roslaunch motor_control ankle_motor_test.launch test_leg:=left ethercat_interface:=enp0s1
```

### 方法2: 直接运行节点

```bash
rosrun motor_control ankle_motor_test _test_leg:=left _ethercat_interface:=enp45s0
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `test_leg` | string | `left` | 要测试的腿，可选值: `left` 或 `right` |
| `ethercat_interface` | string | `enp45s0` | EtherCAT 网络接口名称 |

## 安全注意事项

⚠️ **运行测试前请务必确保：**

1. **机器人处于安全悬空状态**: 机器人应悬挂或支撑，脚部不接触地面
2. **周围无障碍物**: 确保脚踝活动范围内没有任何障碍物
3. **急停开关就位**: 准备好紧急停止按钮
4. **人员保持安全距离**: 测试时人员应远离机器人运动部件
5. **检查限位设置**: 确认代码中的位置限位参数 `POSITION_LIMITS` 已正确配置

## 程序运行流程

### 阶段1: 初始化 (约3秒)
- 显示测试参数
- 初始化 EtherCAT 通信
- 3秒倒计时提示

### 阶段2: 移动到初始位置 (2秒)
- 电机平滑地从当前位置移动到零位（Roll=0°, Pitch=0°）

### 阶段3: 周期性运动 (20秒)
- Roll 方向: `0.08 × sin(πt)` 弧度
- Pitch 方向: `0.08 × cos(πt)` 弧度
- 每秒打印一次当前状态（目标角度 vs 实际角度）

### 阶段4: 返回零位 (2秒)
- 平滑地将电机返回到零位
- 发送零力矩命令停止

## 输出示例

```
========================================
开始脚踝电机测试
测试腿: left
Roll 幅值: ±4.58 度
Pitch 幅值: ±4.58 度
测试频率: 0.50 Hz
测试时长: 20.0 秒
========================================
正在初始化 EtherCAT 接口 [enp45s0]...
检测到 3 个 EtherCAT 从站
控制的电机索引: 电机5 和 电机6
开始运动，请注意观察...
阶段1: 移动到初始位置...
到达初始位置
阶段2: 开始周期性运动测试...
时间: 0.0s | 目标Roll: 0.00° Pitch: 4.58° | 实际Roll: 0.02° Pitch: 4.55°
时间: 1.0s | 目标Roll: 3.71° Pitch: 1.31° | 实际Roll: 3.68° Pitch: 1.35°
时间: 2.0s | 目标Roll: 0.00° Pitch: -4.58° | 实际Roll: 0.01° Pitch: -4.56°
...
测试完成!
阶段3: 返回零位...
停止所有电机
========================================
脚踝电机测试完成!
========================================
```

## 代码结构

```
ankle_motor_test.cpp
│
├── 命名空间 ankle_test
│   ├── 安全参数配置
│   ├── 运动学对象 (kinematic_left, kinematic_right)
│   └── EtherCAT 接口配置
│
├── runAnkleMotorTest() - 主测试函数
│   ├── 初始化 EtherCAT
│   ├── 阶段1: 移动到初始位置
│   ├── 阶段2: 周期性运动
│   │   ├── 使用 EulerToMotorAngle() 计算电机角度
│   │   ├── 使用 AnkleVelToMotorVel() 计算电机速度
│   │   └── 使用 MotorAngleToEuler() 验证实际角度
│   └── 阶段3: 返回零位
│
└── main() - ROS 节点入口
    ├── 参数解析
    └── 调用测试函数
```

## 关键运动学函数说明

### 1. EulerToMotorAngle (逆运动学)
```cpp
Eigen::Vector2d motor_angles = kinematic->EulerToMotorAngle(ankle_roll, ankle_pitch);
```
- 输入: 脚踝 Roll 和 Pitch 角度（弧度）
- 输出: 电机5和电机6的目标角度（弧度）

### 2. AnkleVelToMotorVel (速度逆运动学)
```cpp
Eigen::Vector2d motor_vels = kinematic->AnkleVelToMotorVel(ankle_roll, ankle_pitch, 
                                                            ankle_roll_vel, ankle_pitch_vel);
```
- 输入: 脚踝姿态角和角速度
- 输出: 电机5和电机6的目标角速度

### 3. MotorAngleToEuler (正运动学)
```cpp
Eigen::Vector2d actual_ankle_angles = kinematic->MotorAngleToEuler(actual_motor_angles);
```
- 输入: 电机5和电机6的实际角度
- 输出: 脚踝实际的 Roll 和 Pitch 角度

## 调整测试参数

如需调整测试参数，可修改 `ankle_motor_test.cpp` 中的常量：

```cpp
namespace ankle_test {
    constexpr double ANKLE_ROLL_AMPLITUDE = 0.08;   // Roll 幅度（弧度）
    constexpr double ANKLE_PITCH_AMPLITUDE = 0.08;  // Pitch 幅度（弧度）
    constexpr double TEST_FREQUENCY = 0.5;          // 频率（Hz）
    constexpr double TEST_DURATION = 20.0;          // 测试时长（秒）
    constexpr float KP = 50.0f;                     // 位置增益
    constexpr float KD = 1.0f;                      // 速度增益
}
```

⚠️ **警告**: 修改参数后请重新编译，并确保新参数不会导致机器人碰到限位！

## 故障排查

### 问题1: EtherCAT 初始化失败
```
解决方法:
1. 检查网络接口名称是否正确: ip link show
2. 确认 EtherCAT 权限: sudo setcap cap_net_raw+ep ./devel/lib/motor_control/ankle_motor_test
3. 检查 EtherCAT 从站连接状态
```

### 问题2: 电机不响应
```
解决方法:
1. 检查电机使能状态
2. 确认电机索引配置正确（左腿: 4,5  右腿: 10,11）
3. 查看终端输出的错误信息
```

### 问题3: 运动不平滑或抖动
```
解决方法:
1. 降低 KP 和 KD 增益
2. 降低运动频率 TEST_FREQUENCY
3. 检查机械结构是否有卡滞
```

## 开发者信息

- **文件路径**: `src/motor_control/src/ankle_motor_test.cpp`
- **依赖库**: ROS, Eigen3, SOEM (EtherCAT)
- **相关头文件**: 
  - `kinematics.h` - 运动学计算
  - `transmit.h` - EtherCAT 通信

## 许可证

与主项目保持一致。

