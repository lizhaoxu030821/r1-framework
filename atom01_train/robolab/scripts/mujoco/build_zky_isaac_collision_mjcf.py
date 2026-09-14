"""Build a ZKY MJCF with explicit URDF/IsaacLab-style link collision geoms."""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tree = ET.parse(args.source)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF has no worldbody")

    collision_count = 0
    for body in worldbody.iter("body"):
        body_name = body.get("name")
        if not body_name:
            continue

        # Remove legacy foot-only collision duplicates before rebuilding.
        for geom in list(body.findall("geom")):
            if (geom.get("name") or "").endswith("_collision"):
                body.remove(geom)

        visual = next(
            (
                geom
                for geom in body.findall("geom")
                if geom.get("type") == "mesh" and geom.get("mesh") == body_name
            ),
            None,
        )
        if visual is None:
            raise RuntimeError(f"Body {body_name!r} has no matching mesh geom")

        visual.set("name", f"{body_name}_visual")
        visual.set("group", "1")
        visual.set("contype", "0")
        visual.set("conaffinity", "0")

        collision = copy.deepcopy(visual)
        collision.set("name", f"{body_name}_collision")
        collision.set("group", "3")
        collision.set("rgba", "0.2 0.8 0.2 0.0")
        collision.set("contype", "0")
        collision.set("conaffinity", "1")
        collision.set("condim", "3")
        collision.set("friction", "1.0 0.3 0.3")
        collision.set("margin", "0.0")
        collision.set("solref", "0.02 1")
        collision.set("solimp", "0.90 0.95 0.001 0.5 2")
        body.insert(list(body).index(visual) + 1, collision)
        collision_count += 1

    if collision_count != 20:
        raise RuntimeError(f"Expected 20 link collision geoms, generated {collision_count}")

    ground = worldbody.find("geom[@name='ground']")
    if ground is None:
        raise RuntimeError("MJCF has no ground geom")
    ground.set("contype", "1")
    ground.set("conaffinity", "0")
    ground.set("condim", "3")
    ground.set("friction", "1.0 0.3 0.3")
    ground.set("solref", "0.02 1")
    ground.set("solimp", "0.90 0.95 0.001 0.5 2")

    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="unicode", xml_declaration=False)
    print(f"Wrote {args.output} with {collision_count} explicit collision geoms")


if __name__ == "__main__":
    main()
