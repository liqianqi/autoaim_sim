#!/usr/bin/env python3
"""Grab the onboard camera and publish BGR frames for the auto-aim module."""

from __future__ import annotations

import struct
import sys
from multiprocessing import shared_memory
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_model import DEFAULT_CAMERA, CameraInfo, camera_info_from_model

SHM_NAME = "autoaim_onboard"
HEADER_SIZE = 128
MAGIC = b"AIM1"


def _shm_size(width: int, height: int) -> int:
    return HEADER_SIZE + width * height * 3


class OnboardFeed:
    """Offscreen renderer + optional shared-memory publisher (BGR8)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int | None = None,
        height: int | None = None,
        camera: str = DEFAULT_CAMERA,
        shm: bool = True,
    ):
        self.model = model
        self.camera = camera
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.cam_id < 0:
            raise ValueError(f"camera {camera!r} not in model")
        self.info: CameraInfo = camera_info_from_model(model, camera, width, height)
        self.renderer = mujoco.Renderer(model, height=self.info.height, width=self.info.width)
        self.seq = 0
        self.shm = None
        if shm:
            size = _shm_size(self.info.width, self.info.height)
            try:
                self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=size)
            except FileExistsError:
                stale = shared_memory.SharedMemory(name=SHM_NAME)
                stale.close()
                stale.unlink()
                self.shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=size)

    def grab(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera)
        rgb = self.renderer.render()
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def publish(self, data: mujoco.MjData) -> np.ndarray:
        bgr = self.grab(data)
        self.seq += 1
        if self.shm is None:
            return bgr
        buf = self.shm.buf
        buf[24:28] = struct.pack("<I", 1)
        header = struct.pack(
            "<4sIIIdI",
            MAGIC,
            self.info.width,
            self.info.height,
            self.seq,
            float(data.time),
            1,
        )
        xmat = np.asarray(data.cam_xmat[self.cam_id], dtype=np.float32).reshape(9)
        xpos = np.asarray(data.cam_xpos[self.cam_id], dtype=np.float32).reshape(3)
        buf[:28] = header
        buf[32:68] = xmat.tobytes()
        buf[68:80] = xpos.tobytes()
        buf[HEADER_SIZE : HEADER_SIZE + bgr.size] = bgr.tobytes()
        buf[24:28] = struct.pack("<I", 0)
        return bgr

    def close(self) -> None:
        self.renderer.close()
        if self.shm is None:
            return
        self.shm.close()
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass
        self.shm = None


def attach_shm(name: str = SHM_NAME) -> shared_memory.SharedMemory:
    return shared_memory.SharedMemory(name=name)


def read_frame(shm: shared_memory.SharedMemory) -> tuple[int, float, np.ndarray] | None:
    magic, width, height, seq, stamp, writing = struct.unpack_from("<4sIIIdI", shm.buf, 0)
    if magic != MAGIC or writing:
        return None
    image = np.frombuffer(shm.buf, dtype=np.uint8, offset=HEADER_SIZE, count=width * height * 3)
    return seq, stamp, image.reshape(height, width, 3).copy()
