#!/usr/bin/env python3
"""Render onboard / overview cameras for armor recognition tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.model_io import load_infantry


def render_camera(model, data, camera: str, width: int, height: int) -> np.ndarray:
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", choices=("red", "blue"), default="red")
    parser.add_argument("--camera", default="onboard")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--yaw", type=float, default=0.0, help="yaw command in degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="pitch command in degrees")
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "captures" / "onboard.png")
    args = parser.parse_args()

    model, data = load_infantry(args.team)
    data.ctrl[0] = np.deg2rad(args.yaw)
    data.ctrl[1] = np.deg2rad(args.pitch)
    mujoco.mj_step(model, data, nstep=50)

    image = render_camera(model, data, args.camera, args.width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
    except Exception:
        from PIL import Image

        Image.fromarray(image).save(args.output)
    else:
        iio.imwrite(args.output, image)
    print(f"wrote {args.output} camera={args.camera} team={args.team}")


if __name__ == "__main__":
    main()
