#!/usr/bin/env python3
"""Load the infantry MJCF, optionally swapping the armor team color."""

from __future__ import annotations

from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def infantry_xml_path(team: str = "red") -> Path:
    src = MODEL_DIR / "infantry.xml"
    if team == "red":
        return src
    if team != "blue":
        raise ValueError(f"unknown team {team!r}")
    text = src.read_text().replace("materials_red.xml", "materials_blue.xml")
    dst = MODEL_DIR / "_infantry_blue_generated.xml"
    dst.write_text(text)
    return dst


def load_infantry(team: str = "red"):
    model = mujoco.MjModel.from_xml_path(str(infantry_xml_path(team)))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data
