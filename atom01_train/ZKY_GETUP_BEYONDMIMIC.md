# ZKY Get-up BeyondMimic

## Scope

The supplied `r1_standup_reference_full26_50hz_zky_ik_hand_visual.npz` was
generated from an unlimited, visualization-only IK result and is not used for
training. The training task uses the constrained replacement generated with
`retarget/robot_retargeter`:

`r1_standup_reference_full26_50hz_zky_ik_hand_limited_full19.npz`

It is a 229-frame, 50 Hz BeyondMimic archive with all 19 ZKY joints and 20 rigid
bodies. The task therefore trains a 19-action whole-body policy. Joint and body
names are stored in the NPZ and resolved explicitly by the motion loader.

The loader maps NPZ columns to Isaac Lab articulation order by name instead of
assuming a converter-specific column order.

The constrained clip respects the ZKY MJCF limits used by `robot_retargeter`.
The SDK's deployment safety limits are narrower and the preflight reports those
differences as warnings. Passing the preflight means the clip is structurally
suitable for simulation; it is not approval to send the reference or a learned
policy to hardware.

## Training plan

The nominal first stage disables pushes, rigid-body material randomization,
mass/CoM randomization, actuator-gain randomization, and debug visualization.
W&B records PPO optimization metrics, episode returns and lengths, individual
reward terms, termination counts, motion tracking errors, and adaptive sampling
statistics under the `zky_getup_beyondmimic` project.

1. Train nominal reference tracking without pushes or broad dynamics
   randomization.
2. Replay checkpoints in Isaac Sim and require completion from sampled phases,
   stable standing after frame 228, and no joint-limit saturation.
3. Add pushes and dynamics randomization gradually, then repeat evaluation.
4. Export ONNX, validate observations/actions and joint order, and run MuJoCo
   sim-to-sim before any shadow-only hardware integration.

## Preflight

Run from `atom01_train`:

```bash
python robolab/scripts/tools/beyondmimic/validate_zky_getup_motion.py
```

## Runtime prerequisite

Run training on the configured GPU server with the Conda environment at
`/data/lzx/conda_envs/robo_lzx`. A one-iteration Isaac Sim smoke test has
already completed successfully there.

## Training command

```bash
python robolab/scripts/rsl_rl/train.py \
  --task ZKY-Getup-BeyondMimic-v0 \
  --headless \
  --device cuda:1 \
  --logger wandb \
  --log_project_name zky_getup_beyondmimic \
  --num_envs 2048 \
  --run_name nominal_v1
```
