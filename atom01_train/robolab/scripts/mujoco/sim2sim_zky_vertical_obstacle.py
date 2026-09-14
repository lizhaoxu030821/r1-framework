"""MuJoCo sim2sim replay with the vertical-obstacle wall used by zky_vertical_obstacle.

The controller and observation construction stay in ``sim2sim_zky_clean``.  This
entry point only injects a temporary wall into the existing Atom01 MJCF before
delegating to that replay loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "robolab", ROOT / "rsl_rl", ROOT.parent.parent / "IsaacLab" / "source"):
    if path.exists():
        sys.path.insert(0, str(path))

import mujoco


def parse_wall_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--obstacle_height", type=float, default=0.25)
    parser.add_argument("--obstacle_x", type=float, default=1.50)
    parser.add_argument("--obstacle_width", type=float, default=8.0)
    parser.add_argument("--obstacle_thickness", type=float, default=0.60)
    return parser.parse_known_args()


def make_wall_xml(xml_path: Path, height: float, obstacle_x: float, width: float, thickness: float) -> str:
    if height <= 0.0 or width <= 0.0 or thickness <= 0.0:
        raise ValueError("Obstacle height, width, and thickness must be positive.")

    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"MJCF has no worldbody: {xml_path}")

    wall = ET.Element(
        "geom",
        {
            "name": "vertical_obstacle_wall",
            "type": "box",
            "size": f"{thickness / 2:.6f} {width / 2:.6f} {height / 2:.6f}",
            "pos": f"{obstacle_x:.6f} 0.0 {height / 2:.6f}",
            "rgba": "0.85 0.20 0.08 1.0",
            "friction": "0.9 0.2 0.2",
            "condim": "3",
            "contype": "1",
            "conaffinity": "15",
        },
    )
    worldbody.insert(0, wall)

    temp = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".xml", prefix="atom01_vertical_obstacle_", dir=xml_path.parent, delete=False
    )
    temp_path = temp.name
    temp.close()
    tree.write(temp_path, encoding="utf-8", xml_declaration=True)
    return temp_path


def main() -> None:
    wall_args, remaining = parse_wall_args()
    sys.argv = [sys.argv[0], *remaining]

    clean_path = Path(__file__).with_name("sim2sim_zky_clean.py")
    clean_spec = importlib.util.spec_from_file_location("sim2sim_zky_clean", clean_path)
    if clean_spec is None or clean_spec.loader is None:
        raise RuntimeError(f"Cannot load clean MuJoCo replay module: {clean_path}")
    clean = importlib.util.module_from_spec(clean_spec)
    clean_spec.loader.exec_module(clean)

    terrain = "--terrain" in remaining
    xml_name = "atom01_terrain.xml" if terrain else "atom01.xml"
    xml_path = Path(clean.ISAAC_DATA_DIR) / "robots" / "roboparty" / "atom01" / "mjcf" / xml_name
    temp_xml = make_wall_xml(
        xml_path,
        wall_args.obstacle_height,
        wall_args.obstacle_x,
        wall_args.obstacle_width,
        wall_args.obstacle_thickness,
    )

    original_loader = mujoco.MjModel.from_xml_path

    def load_with_wall(path):
        if Path(path).resolve() == xml_path.resolve():
            return original_loader(temp_xml)
        return original_loader(path)

    mujoco.MjModel.from_xml_path = load_with_wall
    print(
        "[INFO] vertical-obstacle MuJoCo wall: "
        f"x={wall_args.obstacle_x:.3f} m, height={wall_args.obstacle_height:.3f} m, "
        f"width={wall_args.obstacle_width:.3f} m, thickness={wall_args.obstacle_thickness:.3f} m"
    )
    try:
        clean.main()
    finally:
        mujoco.MjModel.from_xml_path = original_loader
        try:
            os.unlink(temp_xml)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
