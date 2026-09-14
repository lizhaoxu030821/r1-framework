# ZKY RL Deploy 配置说明

这些配置文件是 `zky_rl_deploy` 的安全契约，不是“先填上再说”的普通参数表。

## 使用前提

- 任何配置变更都必须先通过对应单元测试。
- 至少完成 Stage 0 dry_run/shadow 验证后，才允许讨论上机。
- `sensor_calibration.yaml` 中的 `t265_to_pelvis.verified` 在三轴姿态检查、时间对齐检查和 bag 对齐通过前必须保持 `false`。
- `robot_joints.yaml` 的 `policy_order` 只能作为 ONNX metadata 的人工审阅副本；运行时不得用 YAML 覆盖 metadata。
- `joystick_beitong.yaml` 中未实测的按钮/轴编号必须保持 `null`，控制逻辑只允许使用语义名。

## 上机门槛

- 必须先通过 `policy_order/hardware_order/npz_order/sign_convention` 的单元测试。
- `last_action_init` 必须保持 `zeros`，直到 sim2sim 对齐测试明确允许修改。
- `motion_clip_walk1subject1.yaml` 只能使用 `stale_frame_behavior: hold` 进入第一版 bringup；`catchup` 不得直接上机。
- 真实 BeyondMimic 输出前，shadow 必须至少连续 200 个控制周期无异常。
- 输出等级只能通过 `bringup_stages.yaml` 逐级提升，不能直接手改 `action_scale/kp_scale/kd_scale` 绕过 StageManager。

## 当前阶段说明

- 这些 YAML 已按 baseline v0.5 建模，但骨架节点尚未全部消费它们。
- 下一步实现代码时，必须优先把这些配置变成启动校验和单元测试，再考虑功能接管。
