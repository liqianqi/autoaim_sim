#!/usr/bin/env python3
"""Dump several stills so we can check armor lights and pose."""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.model_io import load_infantry

OUT = ROOT / "captures"


def save(model, data, camera: str, path: Path, width=1280, height=720) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio

    iio.imwrite(path, image)
    print(f"wrote {path}")


def main() -> None:
    for team in ("red", "blue"):
        model, data = load_infantry(team)
        prefix = OUT / team
        for cam in ("overview", "armor_front", "armor_left", "onboard", "enemy_view"):
            save(model, data, cam, prefix / f"{cam}.png")


if __name__ == "__main__":
    main()
