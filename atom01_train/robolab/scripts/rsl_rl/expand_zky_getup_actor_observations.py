"""Expand a ZKY get-up checkpoint while preserving its existing actor exactly."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--new-actor-observations", type=int, default=113)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    key = "actor.0.weight"
    old_weight = state[key]
    old_observations = old_weight.shape[1]
    expanded_weight = old_weight.new_zeros((old_weight.shape[0], args.new_actor_observations))
    preserved_observations = min(old_observations, args.new_actor_observations)
    expanded_weight[:, :preserved_observations] = old_weight[:, :preserved_observations]
    state[key] = expanded_weight
    checkpoint["infos"] = {
        **(checkpoint.get("infos") or {}),
        "source_checkpoint": str(args.input.resolve()),
        "old_actor_observations": old_observations,
        "new_actor_observations": args.new_actor_observations,
        "preserved_actor_observations": preserved_observations,
        "appended_actor_weights_zero_initialized": args.new_actor_observations > old_observations,
        "trailing_actor_observations_removed": max(0, old_observations - args.new_actor_observations),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"Resized {key} from {tuple(old_weight.shape)} to {tuple(expanded_weight.shape)}; "
        f"wrote {args.output}"
    )


if __name__ == "__main__":
    main()
