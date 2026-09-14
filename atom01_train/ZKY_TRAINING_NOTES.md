# ZKY Training Package Notes

本文档按当前机器路径 `/home/a/yi/robo` 整理。当前包是从 Roboparty Atom01 训练包改出的 robot_zky 训练包，任务入口仍大量沿用 `atom01` 文件名，但当前主任务 ID 是 `zky`。

## 1. 当前训练包结构

```text
/home/a/yi/robo
├── IsaacLab/                         # 本地 IsaacLab
├── robot_zky/                        # 中科院机器人原始/备份资产
│   ├── urdf/                         # robot_zky.urdf 等
│   ├── mjcf/                         # robot_zky.xml 等
│   └── meshes/                       # 12 DoF 下肢 mesh
└── modules/atom01_train/
    ├── robolab/                      # IsaacLab 扩展和训练任务
    │   ├── scripts/rsl_rl/           # train.py, play.py, 课程训练脚本
    │   ├── scripts/mujoco/           # MuJoCo sim2sim 脚本
    │   ├── robolab/tasks/direct/base # 当前 zky 任务主要代码
    │   ├── robolab/assets/robots     # IsaacLab 机器人 ArticulationCfg
    │   └── data/robots/roboparty/atom01
    │       ├── urdf/atom01.urdf      # 当前实际是 robot_zky 内容
    │       ├── mjcf/atom01.xml       # MuJoCo 平地模型
    │       ├── mjcf/atom01_terrain.xml
    │       └── meshes/
    ├── rsl_rl/                       # 本包内 rsl_rl
    ├── logs/rsl_rl/zky/              # 训练日志和 checkpoint
    └── outputs/                      # 脚本输出
```

当前已知训练结果：

```text
/home/a/yi/robo/modules/atom01_train/logs/rsl_rl/zky/2026-07-18_14-04-59_push_0p80
├── model_26000.pt
├── model_27000.pt
├── params/agent.yaml
└── params/env.yaml
```

建议默认使用最新的：

```text
logs/rsl_rl/zky/2026-07-18_14-04-59_push_0p80/model_27000.pt
```

## 2. zky 任务调用链

主链路如下：

```text
train.py / play.py
  -> import robolab.tasks
  -> robolab/tasks/__init__.py
  -> robolab/tasks/direct/base/__init__.py 注册 gym task: zky
  -> BaseEnv
  -> ATOM01FlatEnvCfg
  -> ATOM01FlatAgentCfg
  -> ATOM01_CFG
  -> data/robots/roboparty/atom01/urdf/atom01.urdf
```

重点事实：

- `zky` 只注册了 direct/base 这个直接强化学习任务。
- `manager_based`、`direct.attn_enc`、`direct.interrupt` 被 blacklist，不是当前 zky 主训练入口。
- 类名还是 `ATOM01...`，但 experiment_name/wandb_project 已改成 `zky`。
- 当前动作维度是 12，单帧 actor 观测是 45，历史长度是 10，所以 actor 输入是 450。
- MuJoCo 回放脚本 `sim2sim_zky_clean.py` 也按 12 动作、45 单帧观测、10 帧历史来加载 policy。

## 3. 主要应该改哪些文件

### 机器人模型和动力学

训练侧 IsaacLab 使用：

```text
modules/atom01_train/robolab/data/robots/roboparty/atom01/urdf/atom01.urdf
modules/atom01_train/robolab/data/robots/roboparty/atom01/meshes/
modules/atom01_train/robolab/robolab/assets/robots/roboparty.py
```

MuJoCo 查看和 sim2sim 使用：

```text
modules/atom01_train/robolab/data/robots/roboparty/atom01/mjcf/atom01.xml
modules/atom01_train/robolab/data/robots/roboparty/atom01/mjcf/atom01_terrain.xml
modules/atom01_train/robolab/scripts/mujoco/sim2sim_zky_clean.py
```

如果你改机器人模型，训练 URDF 和 MuJoCo MJCF 必须保持关节名、关节方向、默认角、限位、足底碰撞、质量惯量尽量一致。

### 训练环境

```text
modules/atom01_train/robolab/robolab/tasks/direct/base/atom01_env_cfg.py
```

主要改：

- reward 权重和 reward term
- `action_space`、`observation_space`、`state_space`
- terrain 类型和 terrain generator
- 终止碰撞 body：`terminate_contacts_body_names`
- 脚部 body：`feet_body_names`
- action scale 和每个关节的 action scale multiplier
- 噪声大小

```text
modules/atom01_train/robolab/robolab/tasks/direct/base/base_config.py
```

主要改：

- 仿真 dt、decimation、episode length
- 命令范围：`lin_vel_x`、`lin_vel_y`、`ang_vel_z`
- domain randomization：摩擦、质量、COM、PD gain、joint 参数
- 推机器人扰动：`push_robot`

注意：`robolab/scripts/rsl_rl/train_zky_push_curriculum.sh` 会直接修改 `base_config.py` 里的 `push_robot` 范围。

```text
modules/atom01_train/robolab/robolab/tasks/direct/base/base_env.py
```

主要改：

- actor/critic observation 拼接顺序
- action 如何变成关节 position target
- reset、done 条件
- height scan 是否进入观测

```text
modules/atom01_train/robolab/robolab/tasks/direct/base/mdp/rewards.py
```

主要改 reward 函数实现，例如速度跟踪、姿态、脚滑、脚高、能耗、关节偏离等。

### PPO 和对称增强

```text
modules/atom01_train/robolab/robolab/tasks/direct/base/agents/atom01_agent_cfg.py
```

主要改：

- `max_iterations`、`save_interval`、`num_steps_per_env`
- PPO 超参
- actor/critic hidden dims
- observation/action 镜像映射
- `experiment_name = "zky"`

如果你改了关节顺序、动作数量、观测维度，必须同步改这里的：

```text
NUM_ACTIONS
POLICY_OBS_DIM
CRITIC_OBS_DIM_FLAT
OBS_HISTORY_LENGTH
generate_joint_mirror()
```

### 任务注册

```text
modules/atom01_train/robolab/robolab/tasks/direct/base/__init__.py
modules/atom01_train/robolab/robolab/tasks/__init__.py
```

一般不需要动。除非你要新增任务名，比如 `zky_rough` 或 `zky_v2`。

### sim2sim / MuJoCo 部署适配

```text
modules/atom01_train/robolab/scripts/mujoco/sim2sim_zky_clean.py
```

主要改：

- `POLICY_JOINT_NAMES`
- `POLICY_DEFAULT_POS`
- PD gain：`KP_BY_JOINT`、`KD_BY_JOINT`
- torque limit：`TAU_LIMIT_BY_JOINT`
- action scale 和脚踝/髋 roll 修正参数
- MuJoCo XML 选择逻辑
- IMU/观测拼接是否和 `BaseEnv.compute_current_observations()` 一致

## 4. 运行前环境

从训练目录启动：

```bash
cd /home/a/yi/robo/modules/atom01_train
```

推荐用本仓库 IsaacLab 的 python 包装器：

```bash
../../IsaacLab/isaaclab.sh -p robolab/scripts/tools/list_envs.py
```

如果你已经激活了正确 conda 环境，也可以把下面命令里的 `../../IsaacLab/isaaclab.sh -p` 换成 `python`。

## 5. 训练命令

普通从头训练：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/rsl_rl/train.py \
  --task zky \
  --headless \
  --logger tensorboard \
  --num_envs 4096 \
  --run_name zky_new
```

小规模冒烟测试：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/rsl_rl/train.py \
  --task zky \
  --headless \
  --logger tensorboard \
  --num_envs 64 \
  --max_iterations 2 \
  --run_name smoke
```

从已知 checkpoint 续训：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/rsl_rl/train.py \
  --task zky \
  --headless \
  --resume \
  --load_run '^2026-07-18_14-04-59_push_0p80$' \
  --checkpoint '^model_27000\.pt$' \
  --logger tensorboard \
  --num_envs 4096 \
  --max_iterations 2000 \
  --run_name continue_from_27000
```

课程式推扰续训：

```bash
cd /home/a/yi/robo/modules/atom01_train
conda activate <your_isaaclab_env>
PYTHON_BIN=python \
INITIAL_RUN=2026-07-18_14-04-59_push_0p80 \
INITIAL_CHECKPOINT=model_27000.pt \
PHASE_ITERS=2000 \
bash robolab/scripts/rsl_rl/train_zky_push_curriculum.sh --num_envs 4096
```

注意：上面这个课程脚本当前写法默认 `PYTHON_BIN` 是单个可执行文件，不适合直接写 `../../IsaacLab/isaaclab.sh -p` 这种两个 token 的命令。更稳的是激活 IsaacLab conda 环境后用 `PYTHON_BIN=python`。

## 6. IsaacLab 里查看策略

使用已有 checkpoint，在 IsaacLab 里 play：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/rsl_rl/play.py \
  --task zky \
  --num_envs 1 \
  --load_run '^2026-07-18_14-04-59_push_0p80$' \
  --checkpoint '^model_27000\.pt$' \
  --vx 0.4 \
  --vy 0.0 \
  --dyaw 0.0 \
  --plane \
  --debug_info
```

如果只想无界面跑几步检查加载：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/rsl_rl/play.py \
  --task zky \
  --headless \
  --num_envs 1 \
  --load_run '^2026-07-18_14-04-59_push_0p80$' \
  --checkpoint '^model_27000\.pt$' \
  --vx 0.4 \
  --max_steps 100 \
  --plane
```

## 7. MuJoCo 查看 sim2sim

平地、有窗口：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/mujoco/sim2sim_zky_clean.py \
  --load_model logs/rsl_rl/zky/2026-07-18_14-04-59_push_0p80/model_27000.pt \
  --vx 0.4 \
  --vy 0.0 \
  --dyaw 0.0
```

复杂地形：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/mujoco/sim2sim_zky_clean.py \
  --load_model logs/rsl_rl/zky/2026-07-18_14-04-59_push_0p80/model_27000.pt \
  --terrain \
  --vx 0.4
```

无窗口调试并输出 CSV：

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p robolab/scripts/mujoco/sim2sim_zky_clean.py \
  --load_model logs/rsl_rl/zky/2026-07-18_14-04-59_push_0p80/model_27000.pt \
  --no_render \
  --max_steps 4000 \
  --debug_interval 100 \
  --debug_com \
  --joint_log_csv outputs/zky_mujoco_debug.csv \
  --vx 0.4
```

常用 MuJoCo 调参参数：

```text
--foot_friction 0.9
--foot_torsional_friction 0.2
--foot_rolling_friction 0.2
--foot_solref_timeconst 0.02
--foot_solref_dampratio 1.5
--action_scale 0.25
--actuator_delay_steps 0
--warmup_time 5.0
--ramp_time 2.0
--root_z 0.9
```

如果 MuJoCo 窗口启动失败，先用 `--no_render` 验证模型和 policy 是否能加载；如果报 `mujoco_viewer is not installed`，说明当前 Python 环境缺窗口 viewer 包。

## 8. TensorBoard

```bash
cd /home/a/yi/robo/modules/atom01_train
../../IsaacLab/isaaclab.sh -p -m tensorboard.main \
  --logdir logs/rsl_rl/zky \
  --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

## 9. 修改时的同步检查清单

- URDF 关节名是否和 `POLICY_JOINT_NAMES`、reward 正则、asset config 一致。
- 训练默认关节角 `roboparty.py` 是否和 MuJoCo `POLICY_DEFAULT_POS` 一致。
- IsaacLab PD gain/torque limit 是否和 MuJoCo PD gain/torque limit 接近。
- 足底碰撞 body/geom 是否能被 `.*ankle_roll.*` 和 `left_foot_collision*` / `right_foot_collision*` 找到。
- 观测维度改动后，`atom01_env_cfg.py`、`base_env.py`、`atom01_agent_cfg.py`、`sim2sim_zky_clean.py` 要同步。
- 动作顺序改动后，mirror mapping、MuJoCo joint mapping、部署配置都要同步。

## 10. 2026-08-02 速度课程训练交接

当前部署基线是：

```text
logs/rsl_rl/zky/2026-07-23_20-17-17/model_20000.pt
```

MuJoCo 平地评估显示该策略可稳定前进，但能力不对称：`vx=0.5` 时约前进
`6.09 m / 20 s`、横向约 `-0.05 m`；`vx=2.0` 时能站住但横偏和航向漂移很大；
后退、侧移和原地转向尚未学会。因此不能直接将其作为全命令范围策略部署。

已停止并弃用以下全向续训结果，不要部署其 checkpoint：

```text
logs/rsl_rl/zky/2026-08-02_20-34-23_omni_speed_2p5/model_25000.pt
```

该轮训练将全部速度命令一次性打开，并且正向恢复 reward 可被持续倾斜触发，导致
策略退化为站立/倾斜恢复，而不是行走。相应的 recovery reward 现已全部置零。

为保留原始前进步态并让新命令维度有合理归一化，已从基线生成课程训练初始化模型：

```text
logs/rsl_rl/zky/2026-07-23_20-17-17/model_20000_curriculum_init.pt
```

该 checkpoint 已做 MuJoCo `vx=0.5` 冒烟验证，20 秒位置约为
`[6.08, -0.04, 0.90]`，与原始基线一致。它将新出现的 `vy`、`yaw rate` 输入从零开始
学习，同时保持 `vx` 输入的原始响应。生成脚本是：

```text
robolab/scripts/rsl_rl/recondition_zky_checkpoint.py
```

课程定义在：

```text
robolab/robolab/tasks/direct/base/atom01_env_cfg.py
```

阶段按环境公共仿真步数切换；每个命令会保持 6--10 秒，前 1--2 秒站立后再以 1.5--2 秒
平滑升至目标速度：

| 阶段 | 起始步数 | vx (m/s) | vy (m/s) | yaw rate (rad/s) | 命令比例（前进/后退/侧移/转向） |
| --- | ---: | --- | --- | --- | --- |
| 0 | 0 | 0.1 .. 0.6 | 0 | 0 | 90 / 0 / 0 / 0 |
| 1 | 36,000 | -0.3 .. 1.0 | -0.15 .. 0.15 | -0.3 .. 0.3 | 45 / 10 / 10 / 10 |
| 2 | 84,000 | -0.6 .. 1.5 | -0.4 .. 0.4 | -0.7 .. 0.7 | 40 / 15 / 15 / 15 |
| 3 | 144,000 | -0.8 .. 2.0 | -0.6 .. 0.6 | -1.0 .. 1.0 | 35 / 15 / 15 / 15 |
| 4 | 216,000 | -1.0 .. 2.5 | -0.8 .. 0.8 | -1.5 .. 1.5 | 30 / 15 / 15 / 15 |

课程采样逻辑位于 `base_env.py` 的 `DirectionalVelocityCommand`。训练 TensorBoard 中应检查：

```text
Curriculum/velocity_phase
Curriculum/max_vx
Curriculum/max_abs_vy
Curriculum/max_abs_yaw
Episode_Reward/track_lin_vel_xy_exp
Episode_Reward/lateral_vel_y_l2
Episode_Reward/track_ang_vel_z_exp
Episode_Reward/yaw_rate_l2
```

开始正式训练的命令如下。`--reset_optimizer` 是必要的：此初始化模型只继承网络参数和
观测归一化，不继承旧策略的 PPO 优化器状态。

本轮已于 `2026-08-02 23:35` 启动，当前运行目录、终端会话和文本日志分别是：

```text
logs/rsl_rl/zky/2026-08-02_23-35-01_velocity_curriculum_2p5
tmux session: zky_velocity_curriculum
outputs/zky_velocity_curriculum_train.log
```

查看训练终端：

```bash
tmux attach -t zky_velocity_curriculum
```

```bash
cd /home/k205-2/Isaaclab/ZKY/ZKY_lab_1/modules/atom01_train
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/k205-2/miniconda3/envs/isaaclab/bin/python robolab/scripts/rsl_rl/train.py \
  --task zky --headless --resume --reset_optimizer \
  --load_run '^2026-07-23_20-17-17$' \
  --checkpoint '^model_20000_curriculum_init\\.pt$' \
  --logger tensorboard --num_envs 4096 --max_iterations 20001 \
  --run_name velocity_curriculum_2p5
```

每 500 次迭代保存一次。首个可比较 checkpoint 是阶段 0 完成后的 checkpoint；之后应对
`vx=0.5/1.0/2.0`、`vx=-0.5`、`vy=+/-0.5`、`dyaw=+/-1.0` 分别使用
`sim2sim_zky_clean.py --no_render --max_steps 4000` 评估，确认前进横偏、侧移方向和转向
方向均正确后再考虑样机部署。

## 11. 2026-08-03 本轮 MuJoCo 复测结论

本节所有有效测试均使用上一轮最后一个稳定 checkpoint：

```text
logs/rsl_rl/zky/2026-08-02_23-35-01_velocity_curriculum_2p5/model_37500.pt
```

`model_40000.pt` 已经明显发散：训练末尾 value loss 约 `3.54e6`、symmetry loss
约 `1.12e5`、平均回报 `-7755`，MuJoCo 中会立即失稳，因此禁止部署或作为下一轮初始化。

### 连续前进与低速长时稳定性

20 秒平地扫描（`vy=0, dyaw=0`）的实测前向速度随输入大致上升，但并不连续：

| 输入 vx (m/s) | 实测 vx (m/s) | 结论 |
| ---: | ---: | --- |
| 0.10 | 0.000 | 等效站立 |
| 0.20 | 0.171 | 可行，但有航向漂移 |
| 0.30 | - | 约 7.9 s 跌倒 |
| 0.40 | 0.358 | 可行，但有横偏 |
| 0.50 | 0.460 | 最稳定的低速点 |
| 1.00 | 0.851 | 可行 |
| 1.50 | 1.304 | 可行，但开始明显横偏 |
| 2.00 | 1.662 | 可行，横偏和偏航大 |
| 2.50 | 1.853 | 可行，但未达到 2 m/s |

进一步细扫确认 `vx=0.27--0.32 m/s` 是确定性跌倒带；`0.26` 和 `0.33` 又可站住。
根因之一是 `BaseEnv.command` 将 `vx` 吸附到
`(0.2, 0.5, 1.0, 1.5, 2.0, 2.5)`，训练和部署都没有真正学习中间命令。

120 秒低速测试表明：`vx=0/0.05/0.10` 均近似站立；`vx=0.20` 虽不跌倒，但末端约
`(x, y, yaw)=(18.73 m, 7.14 m, 33.2 deg)`；`vx=0.40` 末端约
`(38.99 m, -10.25 m, -26.2 deg)`；`vx=0.50` 相对最好，末端约
`(52.29 m, -2.01 m, -6.0 deg)`。因此当前策略不具备可部署的低速连续长期直行能力。

### 横偏补偿与 2 m/s 目标

在 MuJoCo 中，前进时需要负 `vy` 才能减小横偏。实用的初始查表为：

| 输入 vx (m/s) | 初始 vy 补偿 (m/s) | 实测 vx (m/s) |
| ---: | ---: | ---: |
| 0.5 | -0.05 | 0.466 |
| 1.0 | -0.05 | 0.872 |
| 1.5 | -0.20 | 1.346 |
| 2.0 | -0.30 | 1.730 |
| 2.5 | -0.30 | 1.996 |

线性近似 `vy = 0.045 - 0.150 * vx` 的拟合度约 `R^2=0.893`，但低中速不完全线性，
部署侧应优先用查表加线性插值。`vx=2.5, vy=-0.3` 可接近 2 m/s，但目前不宜直接上样机：
关节力矩已打到 `120 Nm` 上限，且仍有横滚问题。

### 高速横滚

| 输入命令 | 横滚 RMS | 横滚峰值 | 横滚峰峰值 |
| --- | ---: | ---: | ---: | --- |
| vx=1.5, vy=0.0 | 3.81 deg | 7.48 deg | 12.95 deg |
| vx=2.0, vy=0.0 | 4.57 deg | 8.75 deg | 14.48 deg |
| vx=2.5, vy=0.0 | 5.25 deg | 10.35 deg | 16.94 deg |
| vx=2.5, vy=-0.3 | 5.64 deg | 11.47 deg | 20.84 deg |

横滚随速度单调变差，当前横向补偿会轻微加剧该问题。原始图和 CSV 位于：

```text
outputs/zky_forward_speed_continuity.png
outputs/zky_vy_compensation_sweep.png
outputs/zky_high_speed_roll_analysis.png
outputs/zky_forward_speed_summary.csv
outputs/zky_vy_compensation_summary.csv
outputs/zky_high_speed_roll_summary.csv
```

## 12. 2026-08-03 连续速度与横滚修复训练

下一轮从 `model_37500.pt` 继承策略网络和观测归一化、重置 PPO optimizer，不继承已经
发散的优化器状态。配置改动如下：

- 取消 `vx` 离散吸附（`walking_lin_vel_x_targets=()`），直接训练连续命令。
- 命令死区从 `0.10` 降为 `0.05 m/s`，并在第一阶段覆盖 `0.05--0.75 m/s`，专门修复
  `0.27--0.32 m/s` 失稳带和低速空档。
- 课程按 `0/45k/120k/225k/330k` 环境步逐步引入后退、侧移、转向和最高 `2.5 m/s`；高速度
  只在后半程出现，避免过早破坏低速步态。
- 加强 `vy`、yaw tracking 和平面姿态约束；新增独立 roll angle / roll rate 惩罚，并增强
  力矩、动作一阶及二阶平滑惩罚。
- 关闭会发散的 mirror loss；保留镜像数据增强。学习率降为 `1e-4`、`desired_kl=0.008`、
  `max_grad_norm=0.7`，本轮训练 30001 迭代、每 500 迭代保存。

启动命令：

```bash
cd /home/k205-2/Isaaclab/ZKY/ZKY_lab_1/modules/atom01_train
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
/home/k205-2/miniconda3/envs/isaaclab/bin/python robolab/scripts/rsl_rl/train.py \\
  --task zky --headless --resume --reset_optimizer \\
  --load_run '^2026-08-02_23-35-01_velocity_curriculum_2p5$' \\
  --checkpoint '^model_37500\\.pt$' \\
  --logger tensorboard --num_envs 4096 --max_iterations 30001 \\
  --run_name velocity_continuous_roll_repair
```

本轮已于 `2026-08-03 11:35` 启动，运行目录、可见终端和文本日志为：

```text
logs/rsl_rl/zky/2026-08-03_11-35-08_velocity_continuous_roll_repair
tmux session: zky_velocity_continuous_roll_repair
outputs/zky_velocity_continuous_roll_repair_train.log
```

查看实时训练终端：

```bash
tmux attach -t zky_velocity_continuous_roll_repair
```

监控重点：`Episode_Reward/roll_orientation_l2`、`Episode_Reward/roll_ang_vel_l2`、
`Episode_Reward/lateral_vel_y_l2`、`Episode_Reward/yaw_rate_l2`、value loss、平均 episode
长度以及 symmetry loss（关闭后应不再出现）。每完成一个课程阶段后，用 MuJoCo 对
`vx=0.1/0.2/0.28/0.3/0.4/0.5/1.0/1.5/2.0/2.5` 做连续性扫描，对 `vx=1.5/2.0/2.5` 做
横滚统计，再决定是否进入样机。

## 13. 2026-08-04 高速续训：从 52000 到实机 2.0--2.5 m/s

实机已经验证 `model_52000.pt` 的低中速策略稳定，但最大实测速度约为 `1.5 m/s`。
这说明 sim-to-real 的基础链路有效；下一步不是替换控制尺度或取消随机化，而是让策略在
训练中持续、密集地见到高速前进命令。上一轮最大命令仅为 `2.5 m/s`，且纯前进只占一部分
采样，`>=2.0 m/s` 的样本比例不足，无法稳定地学习大步幅高速步态。

### 本轮备份与初始化

启动前已在本目录初始化本地 Git 仓库，并提交上一轮的不可变备份：

```text
training_backups/2026-08-03_velocity_continuous_roll_repair/
```

其中包含实际保存的 `env.yaml`、`agent.yaml`、源配置和命令采样器；部署基线
`model_52000.pt` 的 SHA-256 为
`47c0844e51b5bf108897f20fb4967ceef9f551a85264702fcb66f3f07930d1f6`。

本轮从该 checkpoint 加载网络与观测归一化，并使用 `--reset_optimizer`。不能从 67500
继续，因为 52000 已经在实机验证且其低中速策略更适合作为高速扩展的锚点。

### 高速课程设计

本轮训练 `40001` 次迭代，每次 `24` 个环境步，共约 `960k` 环境步。PPO runner 从
checkpoint 延续总迭代号，因此日志显示 `52000--92000`；速度课程使用本轮新环境的相对
步数并从零开始。前进命令上限逐级为 `1.6 -> 2.1 -> 2.65 -> 3.1 -> 3.5 m/s`：

| 阶段 | 本轮相对迭代 | 日志总迭代 | 前进命令上限 | 高速前进过采样 |
| --- | ---: | ---: | ---: | --- |
| 0 | 0--3749 | 52000--55749 | 1.60 | 无 |
| 1 | 3750--9999 | 55750--61999 | 2.10 | 35% 纯前进样本从 1.4 m/s 起抽样 |
| 2 | 10000--19999 | 62000--71999 | 2.65 | 55% 从 1.85 m/s 起抽样 |
| 3 | 20000--29999 | 72000--81999 | 3.10 | 70% 从 2.20 m/s 起抽样 |
| 4 | 30000--40000 | 82000--92000 | 3.50 | 75% 从 2.60 m/s 起抽样 |

最终阶段中，纯前进模式占 60%，其中 75% 强制落在 `2.6--3.5 m/s`；因此约 45% 的所有
重采样都是高速纯前进，而低速、后退、侧移、转向和组合命令仍保留以避免遗忘。
`3.5 m/s` 是训练命令上限，不是直接下发给实机的初始目标；它为模型、传动和接触损失后的
`2.0--2.5 m/s` 实现速度提供余量。

速度跟踪核从 `std=0.55` 扩至 `0.70`、权重从 `3.2` 提至 `4.0`，避免高速初期误差过大时
奖励梯度消失；横向误差与平面姿态惩罚略加强。动作尺度仍为已完成 sim2sim 标定的 `0.25`，
执行器延迟、质量、质心和增益随机化全部保留，不能为了仿真速度牺牲实机鲁棒性。

### 启动与验收

```bash
cd /home/k205-2/Isaaclab/ZKY/ZKY_lab_1/modules/atom01_train
tmux attach -t zky_high_speed_3p5
```

训练会从 `model_52000.pt` 启动。每个阶段完成后，对 `vx=1.5/2.0/2.5/3.0/3.5` 做 MuJoCo
长时测试，并记录实际机体坐标系速度、跌倒、最大横滚、关节力矩饱和和脚滑。候选策略必须在
低摩擦 + 两步执行器延迟测试中无跌倒、最大横滚不高于 `12 deg`，再以受保护模式按
`1.5 -> 1.8 -> 2.0 -> 2.2 -> 2.5 m/s` 逐档上样机；任一档出现持续力矩饱和、脚滑或横滚
放大即回退，不直接尝试 `3.5 m/s` 训练命令。
