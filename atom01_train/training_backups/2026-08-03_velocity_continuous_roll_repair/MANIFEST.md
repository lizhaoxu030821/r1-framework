# 2026-08-03 Velocity Continuous Roll Repair

This immutable source snapshot is the configuration used by the completed
`velocity_continuous_roll_repair` run. It is stored before switching the live
configuration to the high-speed curriculum.

Run directory: `logs/rsl_rl/zky/2026-08-03_11-35-08_velocity_continuous_roll_repair`

Reference checkpoints:

- `model_52000.pt` (deployed baseline): `47c0844e51b5bf108897f20fb4967ceef9f551a85264702fcb66f3f07930d1f6`
- `model_67500.pt` (completed final): `6229ccd191a00ad24cc277ae2633fc9f3c0801962cf416c1f1463b4f0e934cf6`

The snapshot includes the source environment, runner configuration, command
sampler, and the exact saved `env.yaml`/`agent.yaml`. Checkpoints remain in the
run directory and are intentionally not duplicated into Git.
