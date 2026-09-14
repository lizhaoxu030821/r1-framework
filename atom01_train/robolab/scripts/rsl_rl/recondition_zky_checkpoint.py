"""Prepare a forward-only ZKY checkpoint for omnidirectional curriculum training."""

import argparse
from pathlib import Path

import torch


COMMAND_OFFSET = 6
NORMALIZATION_EPS = 1.0e-2


def recondition_group(state, prefix: str, linear_key: str, single_obs: int, command_stds: tuple[float, ...]):
    mean = state[f"{prefix}._mean"]
    var = state[f"{prefix}._var"]
    std = state[f"{prefix}._std"]
    weight = state[linear_key]
    history_length = mean.shape[1] // single_obs

    for frame in range(history_length):
        for command_axis, target_std in enumerate(command_stds):
            index = frame * single_obs + COMMAND_OFFSET + command_axis
            if command_axis == 0:
                weight[:, index] *= (target_std + NORMALIZATION_EPS) / (std[0, index] + NORMALIZATION_EPS)
            else:
                weight[:, index] = 0.0
                mean[0, index] = 0.0
            var[0, index] = target_std**2
            std[0, index] = target_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    recondition_group(state, "actor_obs_normalizer", "actor.0.weight", 45, (1.0, 0.8, 1.5))
    recondition_group(state, "critic_obs_normalizer", "critic.0.weight", 84, (1.0, 0.8, 1.5))
    checkpoint["infos"] = {
        "source_checkpoint": str(args.input.resolve()),
        "command_normalization_stds": (1.0, 0.8, 1.5),
        "lateral_and_yaw_input_weights_reset": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Wrote curriculum initialization checkpoint: {args.output}")


if __name__ == "__main__":
    main()
