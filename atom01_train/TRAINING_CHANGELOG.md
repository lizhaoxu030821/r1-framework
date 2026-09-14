# ZKY BeyondMimic 训练变更记录

## 2026-08-25（v42 扭矩、静止 reset 与统一 horizon）

- v41 `model_800.pt` 的严格 frame-0 评估暴露了显式执行器限幅不一致：腿/腰的 PhysX limit 虽是 `150 Nm`，`DelayedPDActuator` 仍继承 URDF 的 `100 Nm` motor clip；踝/手臂则是 actuator `100 Nm` 后又被 PhysX 静默裁到 `35 Nm`。旧控制器下 frame 100 约 `26.3%` 关节真实饱和。
- 对所有 ZKY 执行器显式对齐 `effort_limit` 与 `effort_limit_sim`：腿/腰 `150 Nm`，踝/手臂 `35 Nm`。W&B 和 evaluator 的饱和率现在使用真实 actuator limit，评估额外输出 computed/applied effort 及 actuator/simulator 两层限值。
- frame-0 reset 与严格评估改为根部和关节速度全零。之前 NPZ frame 0 在 `0.5x` 下会注入约 `6.04 rad/s` 的关节速度模，可能导致不可部署的初始动量依赖；物理同步的随机 phase reset 仍保留参考速度。
- 新增与起始 phase 无关的末帧 horizon：每个 episode 都在 frame 228 后保持 90 个控制步（1.8 s）就结束，避免后期 reset 收集近 10 s 重复站立奖励，而 frame-0 只有 1.8 s。frame-0 仍完整执行 229 帧，总时长约 10.92 s。
- 8 环境 PPO smoke 均通过，无 NaN/QACC/四元数异常；15 iteration smoke 确认 `motion_complete` 按预期触发。
- 保留 v41 `model_800.pt` 后停止 v41；发现 horizon 偏置后在第 39 轮停止首次 v42 短跑。已从零启动 512 环境、`cuda:1`、30000 轮正式 run `visual_balancedhorizon_v42_30k_20260825`，PID `2987976`，W&B ID `si9p5s2t`。

## 2026-08-24（v11 起身奖励重设计）

- 对比了当前框架中的 `Atom01-Getup-Mimic` 和旁边 UFO-RPO 框架的成功起身配置。
- 保留 ZKY NPZ 逐帧目标、frame-0 精确初始化和参考关节残差动作，不切换训练框架。
- 在 BeyondMimic MDP 中新增：显式根部高度跟踪、参考直立度跟踪、踝部接触支撑奖励、末帧稳定站立奖励。
- 末帧稳定奖励要求参考动作结束、根部高度至少 `0.60 m`、直立度、线速度和角速度同时达标。
- 将 v10 为避免早期截断而放宽的终止阈值恢复为位置 `0.30 m`、姿态 `0.80`；前 50 个仿真步保护仍由终止函数保留。
- 尚未启动训练；待配置导入和短评估通过后，以新实验名启动 v11，并将 W&B 作为训练曲线和物理成功率记录位置。
- 服务器 8 环境 smoke test 已成功创建场景并注册 18 个奖励项；首步发现批量四元数重力向量维度错误，已修正为按环境 batch 扩展，正式训练尚未启动。
- 第二次 smoke test 已完成真实 PPO 更新（8 环境、1 iteration），无奖励 NaN；已启动 v11 正式训练：4096 环境、cuda:1、30000 iterations、W&B 项目 `zky_getup_beyondmimic`，run name `npz_sequence_height_upright_support_v11_20260824`。
- W&B run `8fck1g4w` 在约 step 1500 平台并被 `anchor_pos` 全部截断，已停止；新增 `robot_base_height`、`reference_base_height`、`robot_upright` command metrics，直接记录真实起身高度和直立度。
- 针对 v11 早期截断，ZKY 终止改为阶段化：只在最后 9 个参考帧启用严格根部位置/姿态终止，前段保留完整起身探索窗口。
- 对齐 RoboParty 成功 get-up 结构：关闭训练阶段 anchor 位置/姿态终止，新增实际根部高度进度和实际直立度进度奖励；v11 的 `anchor_pos=1.0`、平均 147 步早期截断被确认是主要失败原因。
- v12 smoke test 首次发现 command 高度指标缺少 `quat_apply_inverse` 导入，已修正；失败发生在首步指标计算，不是仿真动力学失败。
- v12 第二次 smoke test 完成 2 次 PPO 更新且无 NaN；关闭 `randomize_joint_default_pos`，保证 frame-0 nominal 初始化与 P 控制中心一致。
- 已启动 v12 正式训练：`npz_robo_party_progress_v12_20260824`，4096 环境、cuda:1、30000 iterations、W&B `zky_getup_beyondmimic`，PID `3182169`。

## 2026-08-22

### 训练服务器

- 主机：`119.6.246.139:22222`
- 用户：`lzx`
- Conda 环境：`/data/lzx/conda_envs/robo_lzx`
- GPU：`cuda:1`
- W&B 项目：`zky_getup_beyondmimic`

### MuJoCo 播放器

- 使用框架：`/home/lizhaoxu/project/robo_party_getup/retarget/robot_retargeter`
- 修正 IsaacLab `JointPositionActionCfg` 的默认关节偏置映射：
  `target = default_joint_pos + 0.25 * action`
- 增加动作裁剪、分组 PD、力矩限制、地面修正和无窗口诊断选项。
- 正确 MJCF 路径：
  `robot_retargeter/robot_zky/mjcf/robot_zky_fixed_all.xml`
- 当前旧策略在 MuJoCo 中不再出现 NaN/QACC 爆炸，但尚未稳定完成起身。

### BeyondMimic 任务配置

- 提高全身位置、姿态、关节和关键身体跟踪奖励。
- 放宽早期根部位置/姿态终止阈值，避免倒地阶段过早 reset。
- 降低末帧保持惩罚。
- 新增 `start_at_frame_zero`，让单条起身动作按完整时间顺序训练。
- `exact_npz_frame0_v5` 中关闭 reset 时的关节、根部位姿和速度随机化：
  `joint_position_range=(0,0)`、空 `pose_range`、空 `velocity_range`。
- 目标是让每个 episode 的初始机器人状态严格匹配 NPZ 第 0 帧。

### 训练实验

- 旧实验：`nominal_v1`，曾运行至约 `model_21600.pt`，策略只能部分起身。
- `frame0_v3`：从旧策略续训，已停止。
- `fresh_reward_frame0_v4`：随机策略初始化，已停止。
- 当前实验：`exact_npz_frame0_v5`。
  - 随机初始化策略网络。
  - 机器人状态严格按 NPZ 第 0 帧初始化。
  - `4096` 个环境，最大 `30000` iterations。
  - 日志：`logs/exact_npz_frame0_v5.log`。

后续每次代码、奖励、初始化、训练命令或 MuJoCo 播放器修改，都必须在本文件新增日期和变更条目。

## 2026-08-23

### 3 万轮评估

- `exact_npz_frame0_v5` 已完成至 `model_29999.pt`。
- IsaacLab 固定 frame-0 评估显示机器人末帧高度约 `0.137 m`、直立度约 `-0.005`，实际仍未起身。
- 旧成功率定义仅统计宽松终止条件，出现 `100%` 假成功，不能作为起身成功指标。

### MuJoCo 动作映射修复

- 确认 ZKY RSL-RL runner 的 `clip_actions=None`，训练环境不会把动作裁剪到 `[-1, 1]`。
- 修复 MuJoCo 播放器：默认不裁剪策略动作。
- 新增可选参数 `--clip-actions VALUE`，仅在显式指定时裁剪。

### 起身训练根因修复

- IsaacLab 固定 frame-0 评估确认旧策略实际躺地：末帧高度约 `0.137 m`，直立度约 `-0.005`。
- 将策略动作改为参考轨迹残差：`target_joint = reference_joint + 0.25 * action`。
- 随机初始化策略的零均值输出现在先执行 NPZ 关节轨迹，策略学习接触动力学残差。
- 全局根部位置/姿态奖励权重调整为 `4.0/3.0`，标准差调整为 `0.20/0.35`。
- 全身位置/姿态奖励降为 `2.0/1.0`，避免 link 坐标误差压过根部起身目标。
- 根部位置和姿态终止阈值恢复为有判别力的 `0.30 m/0.8`，避免躺地被统计成成功。
- 评估脚本新增 `--zero-actions` 参考 PD 预检，并增加末帧高度、直立度和关节速度组成的物理成功率。
- 第一阶段参考播放速度设为 `0.5x`，参考关节、身体和根部速度同步缩放。
- episode 延长到 `11 s`，覆盖约 `9.16 s` 的慢速动作和末帧稳定时间。

### 正式残差策略训练

- PD 零动作回放仅用于诊断，不作为动力学可行性的最终判据；起身所需接触、动量和残差由 PPO 奖励学习。
- 新实验名：`npz_residual_half_speed_v6`。
- 策略网络随机初始化，机器人严格从 NPZ frame 0 初始化。
- 动作中心为当前参考关节位置，策略输出 `0.25 * action` 的动力学残差。
- 第一阶段使用 `0.5x` 参考速度、`4096` 环境、`30000` iterations、`cuda:1`。
- `v6` 第 0 轮出现 `anchor_ori termination=1.0`，随机探索过早结束；该实验停止。
- 新增根部位置/姿态终止保护窗口 `min_steps=50`（1 秒），保护期后仍使用严格阈值 `0.30 m/0.8`。
- 新实验名：`npz_residual_half_speed_v7`，随机初始化重新训练。

### 固定站立目标任务

- 用户明确目标为“从固定倒地姿态学会起身到固定站立姿态”，不要求逐帧复现原始动作。
- `goal_only=True`：reset 使用 NPZ frame 0，奖励目标、动作参考和根部目标固定使用 NPZ 最后一帧。
- 策略仍通过 PPO 奖励学习起身动力学，不把零动作 PD 结果作为成功标准。
- episode 调整为 `8 s`，新实验名：`npz_fixed_goal_v8`。
- v8 首次启动发现 `target_index` 归属错误导致初始化异常，已修正为 `MotionCommand.target_index` 并重新启动；当前 PID `154860`。

### 目标恢复为 NPZ 逐帧起身

- 用户最终目标明确为按 NPZ 数据方式完成起身，不采用仅最终站立目标的 goal-only 简化任务。
- 停止 `npz_fixed_goal_v8`，恢复 `goal_only=False`、`playback_speed=1.0`、`episode_length_s=6.0`。
- 保留 NPZ frame-0 精确初始化、参考关节残差动作、严格终止阈值和 `min_steps=50` 探索保护。
- 新实验名：`npz_sequence_residual_v9`，随机初始化策略重新训练。
- v9 首次启动因服务器未同步 `terminations.py` 的 `min_steps` 接口而在环境构建阶段退出；已补充同步后重新启动。
- v9 第二次启动发现 IsaacLab 不接受带默认值的额外终止参数；改为终止函数内部固定前 50 步保护，配置恢复为标准参数接口。

### v9 训练评估

- `npz_sequence_residual_v9` 已完成 `30000` iterations，最终模型：`model_29999.pt`。
- IsaacLab 64 环境固定 frame-0 评估：终止成功率 `0%`，物理成功率 `0%`。
- `anchor_pos_failure_rate=1.0`，末帧根部高度均值 `0.308 m`，直立度均值 `0.493`，关节误差 `5.757 rad`。
- 结论：残差动作方案仍未学会 NPZ 起身，当前模型不进入 MuJoCo 可视化验收。

### v10 探索窗口调整

- v9 全部环境因 `anchor_pos` 失败，根部误差约 `0.302 m`，探索被反复截断。
- 保留逐帧 NPZ 和参考残差动作，训练终止阈值暂调为位置 `0.80 m`、姿态 `1.80`，让 PPO 获得完整起身探索窗口。
- 物理成功标准不放宽：末帧仍要求 base 高度、upright 和关节速度同时达标。
- 新实验名：`npz_sequence_residual_v10`，随机初始化，30000 iterations。

### v10 W&B 中途评估（2026-08-24）

- 按用户要求提前停止训练，保留最新 `model_25800.pt`，未继续跑满 30000。
- W&B run：`yzxh7zgc`，状态已停止前为 running。
- 曲线显示 `error_joint_pos` 从约 `0.77` 降至 `0.35~0.41 rad`，但 `error_anchor_pos` 长期停在 `0.83~0.85 m`，`error_body_pos` 停在 `0.82~0.85 m`。
- 中后期 `time_out=1.0`、根部终止为 `0`，说明策略学会了保持当前躺倒/低根部状态以避免宽松终止，而不是学会起身。
- `model_25800.pt` IsaacLab 评估：`termination_success_rate=1.0` 但 `physical_success_rate=0.0`，末帧高度 `0.132 m`、upright `-0.051`；该 termination 成功是假成功。
- 结论：v10 未学会 NPZ 起身，继续增加训练轮数预计不会改变平台行为；下一版必须恢复可判别的根部/高度进度奖励或阶段课程。
## 2026-08-24 grounded reference revision

- v12 was stopped after W&B showed timeout while the robot remained at roughly 0.124 m height. Removing mismatch termination and adding progress rewards did not create a physical contact path.
- Diagnosis against the target ZKY MuJoCo model found the source reference penetrating the floor by up to 0.376 m. This makes direct mimic tracking physically inconsistent.
- Added `retarget/robot_retargeter/scripts/ground_zky_beyondmimic_npz.py` to ground every frame using `mj_geomDistance`, preserve joint/orientation motion, and regenerate BeyondMimic body/velocity fields.
- Generated `r1_standup_reference_full26_50hz_zky_grounded_full19.npz` and selected it in the ZKY task config. Long training is intentionally deferred until audit and an 8-env smoke test pass.
- Smoke test `grounded_smoke_v13_20260824` / W&B `qfa140a1` passed task construction and two iterations on `cuda:1`; the command, action, reward, and W&B paths are functional.
- Reverted the task's training motion back to the original MuJoCo-validated NPZ. Applied RoboParty-inspired control stabilization: ZKY-only action scale `0.15`, PPO `init_noise_std=0.30`, entropy `0.001`, learning rate `3e-4`. The grounded NPZ remains an audit artifact, not the training source.
- Formal training launched on server `119.6.246.139:22222` with `/data/lzx/conda_envs/robo_lzx`, `cuda:1`, 4096 envs, 30000 iterations, run `raw_lowexplore_robo_party_v14_20260824`. Early iteration 33 is still lying (height 0.118 m, upright -0.145, stable-stand 0), while episodes complete by timeout; continue training before judging convergence.
- v14 was stopped at approximately iteration 8898 after W&B showed a settled failure policy (height 0.149 m, upright 0.063, stable-stand 0, timeout 1.0); continuing to 30000 was not useful.
- New untrained revision replaces purely absolute get-up progress with temporal height/upright deltas and explicit reset of previous-state buffers. Smoke testing is required before launch.
- Formal training parallelism changed from 4096 to 512 environments per user request. The first smoke test caught a batched gravity-vector shape error in reset; it was fixed and `delta_reward_smoke_v15b_20260824` completed two PPO iterations without errors or NaN.
- `delta_progress_512_preflight_v15_20260824` passed 5 iterations at 512 environments with finite height/upright delta metrics and no runtime/NaN errors. Formal training is cleared to start.
- The formal v15 process was stopped immediately after launch to honor a final pre-training review. IsaacLab step ordering was checked: reward reads the previous step's valid delta, reset sets the baseline to frame-0 state, and command metrics update after reward for the following step. No unresolved shape or NaN issue remains; v15 is paused.
- Cleared formal run restarted after review with 512 environments on `cuda:1`, W&B `hnbqpr3p`; one stale duplicate process was removed, leaving only the new training instance.

## 2026-08-24 v16 grounded preflight

## 2026-08-24 RoboParty BeyondMimic architecture audit

- RoboParty comparison confirms the current ZKY actor/privileged-critic split is structurally correct. Root target observations are critic-only because they are not directly measurable on the robot.
- RoboParty recovery reward functions are present but zero-weight; copying them as active rewards would not reproduce a proven result. ZKY retains dense tracking, contact, velocity, and phase-gated standing terms instead.
- No formal run launched from this audit; deployment-observable observation parity remains a hard requirement.
- Added a small `feet_slide` penalty using the two ankle contact bodies, copied from RoboParty's contact regularization pattern.
- v22 ordered curriculum starts every episode at frame 0 and requires the reference motion to end before granting stable-stand reward.
- Fixed the frame-zero path's zero-probability adaptive sampler, which caused a CUDA device-side assert during reset.

- v15 was stopped at approximately iteration 1700: height stayed near `0.12 m`, uprightness remained negative, and `getup_stable_stand` stayed zero. This matches the prior fallen-pose plateau and is not a useful long-run candidate.
- v16 switches the training reference to `r1_standup_reference_full26_50hz_zky_grounded_full19.npz`, generated with ZKY collision geometry and a 5 mm floor clearance. Playback is `0.5x`, episode length `10 s`, and residual action scale `0.25`.
- Added a phase-gated `getup_nonfoot_contact` penalty: prone contacts are allowed initially, but torso/limb contact is penalized after the reference root rises above `0.25 m`. PPO initial noise is `0.40`.
- Added `robot_min_body_height` and `reference_min_body_height` metrics. These are preflight gates for detecting penetration or suspension before any formal run.
- Grounded preflight passed on server `cuda:1`: 8 environments and 2 PPO iterations built and updated successfully; a 20-iteration rollout covering the full `0.5x` clip completed with finite rewards and no NaN/QACC/runtime errors. The minimum simulated body height stayed positive (`0.0175 m` at the lowest sampled aggregate), and the grounded reference minimum stayed positive (`0.0197--0.1403 m` by phase).
- Formal v16 launched after the physics gate with 512 environments on `cuda:1`, W&B project `zky_getup_beyondmimic`, run `grounded_contact_phase_v16_20260824`; at iteration 18 it is healthy but not yet successful (height `~0.121 m`, upright `~-0.10`, stable-stand `0`, minimum body height `~0.033 m`).
- v16 was stopped at approximately iteration 111 after reproducing the fallen timeout plateau (`height ~0.119 m`, `upright ~-0.15`, stable-stand `0`). Phase sampling had been disabled, and its reset path would mix sampled joints with a frame-0 root. The reset now uses one sampled frame consistently for root pose/velocity and joints; v17 enables adaptive phase sampling (`adaptive_uniform_ratio=0.20`, `adaptive_alpha=0.01`).
- v17 phase-reset smoke passed with 16 environments and 5 PPO iterations. Sampled phases produced consistent reference/robot root states, finite rewards, and no NaN/QACC/runtime errors. Slightly negative body-center height is a diagnostic limitation, not collision penetration; grounded geometry preflight remains the physical gate.
- Formal v17 launched as `phase_curriculum_v17_20260824` with 512 environments on `cuda:1`. At iteration 6, phase samples yield robot height `0.266 m` and uprightness `0.516` against reference height `0.750 m`; unlike v16, this is evidence that sampled later recovery states are being represented consistently. Stable-stand remains zero and the run is not yet successful.
- MuJoCo diagnosis found a replay mapping mismatch: the player used a static default joint pose, but the ZKY training task uses the current NPZ joint frame as the residual action offset. The player now applies `reference_joint[frame] + 0.25 * action`; a 300-frame headless replay completed without NaN/QACC. The v16 `model_0` result is untrained and is not treated as a skill evaluation.
- v17 was stopped near iteration 2163 after partial height/upright improvements while `getup_stable_stand` remained zero. Stable-stand now gates on the sampled reference entering the stand-height phase (`reference anchor height >= 0.60 m`) instead of waiting for `motion_ended`, allowing adaptive late-phase episodes to provide a standing-learning signal.
- Bottom-layer audit found the actor observation lacked explicit root target position/orientation; only the privileged critic had these terms. With adaptive phase sampling, joint targets alone are phase-ambiguous. Added `motion_anchor_pos_b` and `motion_anchor_ori_b` to the policy observation; a fresh experiment is required because observation dimension changes.
- v19 actor-root-observation smoke passed (16 env, 5 PPO iterations). Formal run `actor_root_obs_v19_20260824` launched on `cuda:1` with 512 envs. Early iteration 4 reports stable-stand `0.8572`, robot height `0.4405 m`, uprightness `0.6005`, and minimum body height `0.0436 m`; encouraging but not final full-action success.
- Review correction: `motion_anchor_pos_b` and `motion_anchor_ori_b` depend on stored reference phase and target root pose, not direct onboard sensing. The actor-observation change was reverted to preserve train/deploy observability parity, and v19 was stopped before evaluation.
- v22 plateau diagnosis near iteration 1000: base height stayed below 0.2 m and timeout was 1.0. Non-foot contact gating was changed from reference height to measured robot height to remove a constant fallen-state penalty.
- v24 stage-1 curriculum limits the ordered reference to frame 40 with 2 s episodes; later stages will expand this boundary from the stage checkpoint.
- v25 stabilizes stage-1 continuation from model_800 with learning rate `1e-4` and initial action noise `0.20`.
- v26 increases bounded exploration and physical progress shaping for the frame-40 stage.
- v27 narrows the stage boundary to frame 20 and episode length to 1.5 s to isolate the first support transition.
- Restored frame-40/2 s curriculum per user request while retaining aggressive v26 reward and exploration settings.

## 2026-08-24 Reference consistency correction

- The requested `ik_hand_visual.npz` is a 12-DoF source and is not directly compatible with the 19-DoF actor.
- The ungrounded 19-DoF retarget file has frame-40 root height `0.1715 m`, matching the observed policy plateau; grounded changed it to `0.2548 m` without changing the corresponding joint pose.
- Training reference was switched to `r1_standup_reference_full26_50hz_zky_ik_hand_limited_full19.npz`.
- Source19 smoke showed only a small reference penetration (`~0.015 m`). Created `r1_standup_reference_full26_50hz_zky_source_clearance_full19.npz` with a uniform `+0.02 m` body/root lift, leaving joints unchanged; the task now uses this file.

## 2026-08-24 MuJoCo deployment alignment audit

- Stopped stale remote v22 smoke and v26 processes.
- Matched MuJoCo residual action scale to training (`0.30`).
- Matched MuJoCo effort limits to ZKY IsaacLab actuators (legs/waist `120 Nm`, ankles/arms `27 Nm`).

## 2026-08-24 Full-sequence v31/v32 correction

- Stopped the frame-40 v30 plateau after confirming the robot matched the low
  `0.19 m` reference and never received the full recovery sequence.
- Exposed all 229 frames and extended the 0.5x episode horizon to `11.0 s`.
- Increased training-only residual action scale to `0.35`, initial noise to
  `0.40`, and simulation effort limits to legs/waist `150 Nm`, ankles/arms
  `35 Nm`.
- Full-sequence smoke exposed `-0.138 m` reference body penetration in the
  uniform-clearance archive. Switched the task to the collision-grounded
  `r1_standup_reference_full26_50hz_zky_grounded_full19.npz` archive, whose
  final-frame minimum body height is positive.
- Formal W&B run `grounded_full_sequence_v32_aggressive_20260824` launched on
  512 environments, `cuda:1`.

## 2026-08-24 Root-cause audit and phase curriculum v34

- v33 reached a stable failure plateau after ~7200 iterations: robot height
  `0.19 m`, upright `~0.35`, stable-stand `0`, timeout `1.0`, while grounded
  reference height was `0.957 m`. More effort did not remove the plateau.
- Fixed the checkpoint evaluator, which assumed disabled `anchor_pos` and
  `anchor_ori` termination terms and crashed before reporting physical success.
- Enabled physically consistent phase sampling (`start_at_frame_zero=False`,
  uniform ratio `0.25`, adaptive alpha `0.01`) so PPO can learn late support
  and standing states before frame-zero sequence fine-tuning.
- Current next experiment is `phase_curriculum_v34`; frame-zero ordered
  execution remains the final deployment evaluation mode.

## 2026-08-24 30k phase curriculum retrain

- Stopped the 10k-iteration v34 trial after confirming non-zero stable-stand
  learning signals.
- Restarted a fresh 30,000-iteration run with the same grounded full-sequence
  reference and phase curriculum configuration.

## 2026-08-25 Mixed reset continuation

- v35 completed 30,000 iterations. Random-phase metrics reached intermittent
  `0.45--0.53 m` height and `0.8+` upright, but this was not a frame-zero
  success proof.
- A direct frame-zero continuation from `model_29999.pt` immediately returned
  to the old low-height plateau (`~0.135 m`, stable-stand `0`), proving that a
  hard reset-distribution switch causes catastrophic forgetting.
- Added `frame_zero_ratio` to MotionCommand and launched a continuation with
  25% frame-zero resets plus 75% physically synchronized random-phase resets:
  `mixed_reset_from_v35_20260825`.

## 2026-08-25 Task-centric get-up reward v43

- Stopped v42 after strict stationary frame-zero evaluation showed a settled
  low-pose failure (`final base height 0.201 m`, upright `0.411`, physical
  success `0`). A short smoke run is not used to judge whether get-up has
  converged; this checkpoint was rejected because the full rollout exposed a
  structural reward/reference mismatch.
- Audited the visual-grounded NPZ. Its 19 joint and 20 body mappings are
  geometrically valid, but per-frame mesh grounding lifts the floating base by
  `0.056--0.229 m` and creates non-physical root accelerations. Therefore the
  joint sequence, root orientation, and root-relative body shape remain the
  imitation guide, while global root translation and global body linear
  velocity tracking are disabled.
- Root-relative body position targets are now aligned to the measured robot
  anchor in all three axes. Added an L1 joint-position term so pose learning
  retains a gradient when the exponential tracking term saturates.
- Physical task shaping now rewards measured base-height/upright progress,
  temporal height/upright improvement, foot support, and final stable stand.
  The dense height target is `0.73 m`, matching the original CSV's final root
  height rather than the grounded archive's artificial `0.95 m` root height.
- Deployment evaluation remains stationary frame 0 and requires a complete
  229-frame rollout, final base height above `0.60 m`, upright above `0.80`,
  joint speed below `2 rad/s`, and non-zero physical success.

## 2026-08-26 v47 NPZ imitation and native-speed curriculum

- Stopped v46 at server checkpoint `model_52200.pt`. Its W&B history confirmed
  frame-zero physical success stayed zero while base height remained near
  `0.19 m`; it is retained as a failed strict-imitation baseline.
- The first v47 continuation trials from the physically successful v44
  `model_29999.pt` were stopped. A 200-iteration trial at `1e-4` degraded
  rapidly; `1e-5` retained more of v44 but still traded height for imitation.
- Added a 35k playback curriculum (`0.50x`, `0.65x`, `0.80x`, `0.90x`, `1.0x`)
  and a matched frame-zero reset curriculum (`50%` through `70%`). Milestones
  use `common_step_counter`, which advances 24 steps per PPO iteration.
- Rebalanced NPZ joint/body/root-orientation imitation with physical height,
  upright, support, and two-level stable-stand rewards. Global root tracking
  now constrains horizontal drift only; vertical tracking maps the grounded
  NPZ phase from `0.160455--0.949483 m` to a physical `0.160455--0.93 m` target.
- Per the final training decision, the formal v47 run starts from random policy
  initialization with PPO learning rate `1e-4`; it does not load v44, v45, or
  v46. Actor and action dimensions remain deployment-compatible at `101-D`
  and `19-D`.
- A fresh 2-iteration smoke passed with no checkpoint restoration, finite
  losses, no NaN/QACC, `0.50x` playback, and zero initial effort saturation.
- Formal fresh 35k training launched on server `cuda:1` with 512 environments
  as `npz_mimic_native_curriculum_v47_fresh35k_20260826`, PID `2001339`, W&B
  run `fyg6ag9p`. Strict frame-zero evaluation of every 1000th checkpoint uses
  the matching curriculum speed on `cuda:2` and appends results to
  `/tmp/zky_v47_fresh_eval.log`. The evaluator also checks the final
  `model_34999.pt` and creates `model_best_v47.pt` only after ranking completed
  evaluations by physical success first and imitation error second.

## 2026-08-27 v48 observable backward curriculum

- Stopped v47 near 32k after strict frame-zero rollouts remained at roughly
  `0.24--0.30 m` with zero physical success. The long run, not a smoke test,
  established that the 101-D policy could minimize local pose error while
  remaining low because it did not observe root height, base linear velocity,
  explicit phase, or the root-orientation target.
- Expanded the actor from 101-D to 112-D by appending base linear velocity,
  relative target root orientation, phase-height error, and normalized phase.
  The first 101 inputs and all 19 actions remain unchanged. Migrated v44
  `model_29999.pt` by copying its actor first layer and zero-initializing the
  11 new input columns; the critic remains 293-D.
- Added height-gated imitation, phase-height completion/deficit shaping, and
  stronger coupled height/upright and stand rewards. Reduced rewards that can
  be collected by staying upright in a folded low pose and strengthened the
  non-foot-contact penalty.
- Replaced failure-focused full-clip sampling with a backward curriculum that
  first learns loaded terminal support and gradually moves the reset boundary
  from frame 228 to frame 0. Late random-phase resets use a ramped `-0.05 m`
  contact correction; frame-zero deployment reset remains unmodified.
- Reduced the residual action envelope to `0.20 rad`. A two-iteration terminal
  safety check reached about `0.864 m` height and `0.926` upright without
  NaN/QACC/CUDA failure; effort saturation was about `7.1%`. This check only
  validates pipeline and initial safety and is not evidence of learned full
  229-frame tracking.
- Launched the formal added-35k continuation on 512 environments and `cuda:1`
  from `v48_initialization/model_v44_actor112.pt`, with optimizer reset and
  learning rate `2e-5`. Run name is
  `actor112_backward_npz_v48_long35k_20260827`, PID `427483`, W&B run
  `f66frzmr`. It advances from checkpoint iteration 29999 to about 64999 and
  will not be stopped for an ordinary short-term plateau; strict frame-zero
  full-sequence evaluation is required to judge learning.

## 2026-08-27 v49 ordered-skill preservation

- Stopped v48 at `model_30400.pt`. Its strict `model_30000.pt` rollout ended
  near `0.151 m`; the run was not merely under-trained. Restoring the action
  scale from `0.20` to v44's original `0.50` immediately restored the migrated
  actor's frame-zero get-up (`0.68--0.71 m`, upright `0.91--0.95`). The scale
  change had altered the physical meaning of every actor output and destroyed
  the inherited skill before PPO began.
- Full v44 rollout diagnostics showed a second issue: the old actor uses very
  large raw actions (early mean absolute action about `5.15`, maximum `37.4`)
  as bang-bang torque commands, with about `23.6%` effort saturation. Raw
  action clips from 2 through 12 all destroyed the recovery, so v49 retains
  the `0.50` scale and learns a safer control law instead of imposing a clip
  that removes required support forces.
- Changed the reset curriculum from 10% to 90% frame-zero episodes initially.
  Random late-phase resets start at only 10% and expand gradually, so training
  optimizes the deployed ordered recovery without forgetting terminal support.
  Playback remains `0.50x` for 10k iterations and reaches `1.0x` after 25k.
- Added continuous terminal-extension reward and phase/height-gated late
  residual and pre-clipping effort penalties. These terms allow strong early
  contact actions but discourage saturated residuals after the robot is up.
  Strengthened signed height/upright improvement and the non-saturating
  phase-height deficit while reducing the loose crouched-stand bonus.
- Reduced PPO update size (`1e-5` learning rate, `0.1` clip, `0.005` desired
  KL, low entropy) to preserve the inherited get-up while the 11 appended
  root/phase observations acquire useful weights.
- A 200-iteration preflight retained physical recovery (`0.676 m`, upright
  `0.948`) and improved final root-orientation error to about `0.381 rad` and
  body RMSE to `0.156 m`. It did not yet reach strict height and did not reduce
  full-rollout saturation, so the formal v49 run strengthens only the gated
  height/effort terms and continues from `model_30198.pt` for long training.
- The server root filesystem became full from stale Isaac Lab USD caches.
  Removed 11 GB of lzx-owned `/tmp/IsaacLab` generated directories; all code,
  formal checkpoints, W&B data, and logs under `/data` were preserved.
- Formal v49 long training continues from the 200-iteration preflight
  `model_30198.pt` through iteration 65197 (35,000 additional PPO updates),
  using 512 environments on `cuda:1`. Run name is
  `ordered_height_effort_v49_long35k_20260827`, PID `761168`, W&B run
  `5tjk2c1l`. A low-frequency watcher (PID `769091`) evaluates every 1000th
  checkpoint on `cuda:2`; its generated temporary USD data is redirected to
  `/data/lzx/tmp/zky_v49_eval` to avoid filling the root filesystem again.

## 2026-08-27 v50 overlapping speed curriculum

- Stopped v49 near iteration 43566 after a reproducible discontinuity. At
  iteration ~40198 the scalar playback speed changed from `0.50x` to `0.65x`
  at the same milestone that random resets expanded from frames 190--228 to
  160--228. `model_40000` succeeded at `0.50x` (height `0.914 m`, physical
  success `1.0`) but the same checkpoint at `0.65x` fell to `0.229 m`. After
  1000 updates, `model_41000` also failed when evaluated back at `0.50x`,
  proving the distribution jump first caused timing failure and then PPO
  catastrophic forgetting.
- Rolled back to the best pre-transition checkpoint `model_38000.pt`, whose
  strict `0.50x` rollout reached `0.936 m`, upright `0.999`, joint MAE
  `0.211 rad`, body RMSE `0.044 m`, and physical success `1.0`.
- Replaced the global scalar speed switch with a per-environment, per-episode
  speed. Speed is sampled only at reset and remains fixed throughout that
  episode, so no robot changes timing in the middle of a contact transition.
  Each stage keeps 25% of episodes exactly at `0.50x`, assigns 25% to the
  current maximum, and samples the rest across the overlap range.
- The maximum speed now expands in small steps:
  `0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00`.
  Frame-zero resets remain fixed at 90%; the auxiliary random reset range is
  fixed at frames 190--228, so reset difficulty and speed no longer change at
  the same milestone.
- Added commanded playback speed as a deployable actor observation. Expanded
  the actor from 112-D to 113-D with a zero-initialized new input column while
  keeping the critic 293-D and action 19-D. The migrated actor reproduced the
  `model_38000` `0.50x` behavior exactly.
- A 200-iteration mixed `0.50--0.525x` preflight passed strict rollouts at both
  endpoints: `0.50x` height `0.921 m`, success `1.0`; `0.525x` height
  `0.931 m`, success `1.0`. This validates preservation plus the first speed
  adaptation step, rather than treating training aggregates as proof.
- Formal added-35k v50 training starts from preflight `model_38199.pt`, PID
  `2206072`, W&B run `rl25vqeg`, using 512 environments on `cuda:1`. The
  dual-speed watcher PID `2245716` evaluates fixed `0.50x` and the active
  maximum speed every 1000 iterations on `cuda:2`.
