# motor_control 19 自由度扩展修改记录

日期：2026-06-09

## 修改目标

本次修改将原本只控制下肢 12 个电机的 `motor_control` 包扩展为 19 自由度控制：

- 下肢：左腿 6 电机、右腿 6 电机，保留原有并联踝转换逻辑。
- 上肢：左肩 pitch、左肩 roll、左肘、右肩 pitch、右肩 roll、右肘，共 6 电机。
- 腰部：`body_joint` 腰部旋转 1 电机。
- RViz 默认模型切换到 `resources/robots/robot_zky_19dof`。

## 硬件映射

| 全局索引 | 关节名 | 从站 | CAN/电机说明 |
|---------|--------|------|--------------|
| 0 | `left_hip_yaw_joint` | 从站0 | CAN1 id1 |
| 1 | `left_hip_roll_joint` | 从站0 | CAN1 id2 |
| 2 | `left_hip_pitch_joint` | 从站0 | CAN1 id3 |
| 3 | `left_knee_joint` | 从站0 | CAN2 id4 |
| 4 | `left_ankle_pitch_joint` | 从站0 | CAN2，关节角转换到 motor5 |
| 5 | `left_ankle_roll_joint` | 从站0 | CAN2，关节角转换到 motor6 |
| 6 | `right_hip_yaw_joint` | 从站1 | CAN1 id1 |
| 7 | `right_hip_roll_joint` | 从站1 | CAN1 id2 |
| 8 | `right_hip_pitch_joint` | 从站1 | CAN1 id3 |
| 9 | `right_knee_joint` | 从站1 | CAN2 id4 |
| 10 | `right_ankle_pitch_joint` | 从站1 | CAN2，关节角转换到 motor5 |
| 11 | `right_ankle_roll_joint` | 从站1 | CAN2，关节角转换到 motor6 |
| 12 | `left_shoulder_pitch_joint` | 从站2 | CAN1 id1 |
| 13 | `left_shoulder_roll_joint` | 从站2 | CAN1 id2 |
| 14 | `left_elbow_joint` | 从站2 | CAN1 id3 |
| 15 | `right_shoulder_pitch_joint` | 从站2 | CAN2 id4 |
| 16 | `right_shoulder_roll_joint` | 从站2 | CAN2 id5 |
| 17 | `right_elbow_joint` | 从站2 | CAN2 id6 |
| 18 | `body_joint` | 从站2 | CAN2 id7 |

说明：需求文字中右臂第二、第三个关节写成了 `left_shoulder_roll`、`left_elbow`，代码和 URDF 按右臂实际关节名修正为 `right_shoulder_roll_joint`、`right_elbow_joint`。

## 代码修改记录

### `src/motor_control/app/config.h`

- 新增 `ZKY_SLAVE_NUMBER=3`、`ZKY_TOTAL_MOTORS=19`、`ZKY_MAX_MOTORS_PER_SLAVE=7` 等统一硬件常量。
- 将 `EtherCAT_Msg::motor` 从 6 个 CAN 帧槽位扩展到 7 个槽位，用于覆盖从站2的 `body_joint`。
- 添加中文注释说明 3 个从站、19 电机和 7 帧 PDO 的设计原因。

### `src/motor_control/app/transmit.h`

- `SLAVE_NUMBER` 改为引用统一宏 `ZKY_SLAVE_NUMBER`。
- `motorDate_recv` 数组长度改为 `ZKY_TOTAL_MOTORS`。

### `src/motor_control/app/motor_control.h`

- `rv_motor_msg` 数组长度改为 `ZKY_TOTAL_MOTORS`。

### `src/motor_control/app/motor_control.c`

- 新增反馈索引映射函数，按从站号和电机 id 映射到 19 DOF 全局索引。
- `RV_can_data_repack()` 不再使用旧的 `slave_id * 6 + motor_id - 1` 推导方式。
- `send_motor_ctrl_cmd()` 允许第 7 个 CAN 帧槽位，用于从站2的腰部电机。
- 在关键映射位置添加中文注释，标明旧 12 电机硬编码为何会导致从站2反馈丢失或越界。

### `src/motor_control/app/transmit.cpp`

- 新增 19 DOF 下发路由表 `kMotorRoutes`，集中描述全局索引到从站、CAN 帧槽位、电机 id 的关系。
- `EtherCAT_Send_Command()` 改为按路由表打包 19 个电机命令，不再使用 0~11 的 `else if` 硬编码。
- `EtherCAT_Get_State()` 改为按每个从站实际电机数量复制反馈：从站0/1 各 6 个，从站2 为 7 个。
- 新增按从站实际 PDO 字节数限长拷贝的 `readSlaveMessage()`、`writeSlaveMessage()`，避免 7 帧结构体写入旧 6 帧 PDO 时越界。
- 初始化后打印每个从站 IN/OUT PDO 可容纳的 CAN 帧数；若从站2小于 7，会输出警告。

### `src/motor_control/src/motor_control.cpp`

- `TOTAL_MOTORS` 改为 19，并新增下肢/上肢偏移常量。
- `JOINT_NAMES` 扩展为 19 个 URDF 关节名，顺序与 `/motor_params`、`joint_states`、底层路由表一致。
- `/motor_params` 新版长度为 `19 * 6`；兼容旧版 `12 * 6` 下肢消息，上肢和腰部保持零命令。
- 左右腿仍执行并联踝位置、速度和力矩转换；上肢和腰部直接透传串联关节命令。
- `joint_states` 发布 19 个关节；下肢脚踝从电机空间还原为关节空间，上肢和腰部直接使用反馈电机角。
- 预热健康检查扩展到 3 个从站，并分别统计 6/6/7 个电机反馈。
- 位置限幅、力矩限幅、目标步长限幅、跟踪误差窗口全部扩展到 19 个元素。

### `src/motor_control/src/ankle_motor_test.cpp`

- 命令数组扩展到 19 个槽位。
- 测试仍只控制脚踝 motor5/motor6，其余电机保持零命令，避免调用 19 DOF 底层发送函数时越界。

### `src/motor_control/include/legged_bridge_hw/BridgeHW.h`

- 将旧 12 元素数组改为 `ZKY_TOTAL_MOTORS`，避免后续重新启用该头文件时与底层长度不一致。

### `src/motor_control/launch/rviz_display.launch`

- 默认 URDF 切换为 `resources/robots/robot_zky_19dof/urdf/robot_zky_fixed_all.urdf`。
- 新 19 DOF URDF 根链接为 `base_link`，移除旧版 `base_link -> pelvis` 静态 TF。

### `src/motor_control/README.MD`

- 更新功能概览、话题长度、19 DOF 关节顺序、RViz 资源路径和已知事项。

### 本地 Git 仓库

- 已在 `/home/tib/code/ZKY_SDK_19Dof` 执行 `git init`，建立本地 Git 仓库。
- 当前没有配置任何 Git remote，因此不会提交或推送到 GitHub。
- 新增根目录 `.gitignore`，忽略 catkin 构建产物、运行记录和 Python 缓存。

## 安全与兼容策略

- 旧部署链路如果仍发布 `12 * 6` 的 `/motor_params`，`motor_control` 会继续控制下肢，上肢和腰部保持零输出。
- 新 19 DOF 控制链路应发布 `19 * 6` 的 `/motor_params`，字段仍为每关节 `[kp, kd, pos, vel, tau, mode]`。
- 上肢和腰部初始位置限幅采用 19 DOF URDF 中的 `[-1.57, 1.57]`，后续需要根据实机机械限位和电机零位再校准。
- 由于从站2需要 7 个 CAN 帧槽位，必须确认 EtherCAT 从站固件、ESI/PDO 映射和实际 IO 字节数支持 7 帧。若日志提示从站2 IN/OUT 容量小于 7，`body_joint` 或部分右臂电机无法完整收发。

## 后续审查清单

- 确认从站2实际 PDO 输入/输出可容纳 7 个 `Motor_Msg`。
- 实机逐个验证 19 个关节的正方向、零位和限位。
- 若上层 `zky_rl_deploy` 需要直接输出 19 DOF，需要另行扩展其 12 维策略、观测、日志和安全门控配置。
- 记录从站2 CAN2 上 id4~id7 的实际反馈顺序，确认与本表一致。
