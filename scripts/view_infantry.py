#!/usr/bin/env python3
"""3D viewer + separate onboard camera window. Space toggles target strafe+spin."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 这台机器 X 的 GLX 默认走 Intel iGPU (Mesa)，MuJoCo 离屏渲染 1280x720 要 130 ms/帧。
# 用 PRIME render offload 把 GLX 上下文放到 NVIDIA 上，降到 2 ms。必须在 import mujoco/glfw 之前设置。
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_model import camera_info_from_model, save_camera_yaml
from scripts.model_io import load_infantry
from scripts.onboard_feed import SHM_NAME, OnboardFeed

SPACE = 32


def yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def quat_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class TargetPatrol:
    """Sentry: left-right ping-pong at fixed range, plus continuous yaw spin."""

    def __init__(self, half_range: float = 0.40, speed: float = 4.65, spin: float = 20.0):
        self.half_range = half_range
        self.speed = speed
        self.spin = spin
        self.active = False
        self.origin = np.zeros(3)
        self.yaw0 = np.pi
        self.t0 = 0.0
        self.last_t = 0.0
        self.x = 0.0
        self.direction = 1.0

    def toggle(self, data: mujoco.MjData) -> None:
        if data.mocap_pos.shape[0] < 1:
            print("model has no mocap target")
            return
        if self.active:
            self.active = False
            print("target stopped")
            return
        self.origin = np.array(data.mocap_pos[0], dtype=np.float64)
        self.yaw0 = quat_yaw(data.mocap_quat[0])
        self.t0 = float(data.time)
        self.last_t = self.t0
        self.x = 0.0
        self.direction = 1.0
        self.active = True
        print("target moving: left-right + yaw")

    def apply(self, data: mujoco.MjData) -> None:
        if not self.active or data.mocap_pos.shape[0] < 1:
            return
        now = float(data.time)
        step = now - self.last_t
        self.last_t = now
        self.x += self.direction * self.speed * step
        if self.x >= self.half_range:
            self.x = self.half_range
            self.direction = -1.0
        elif self.x <= -self.half_range:
            self.x = -self.half_range
            self.direction = 1.0
        data.mocap_pos[0] = (self.origin[0] + self.x, self.origin[1], self.origin[2])
        data.mocap_quat[0] = yaw_quat(self.yaw0 + self.spin * (now - self.t0))


def print_intrinsics(info) -> None:
    print("onboard camera (ideal pinhole, no distortion)")
    print(f"  {info.width}x{info.height}  fovy={info.fovy_deg} deg  {info.color_space}")
    print(f"  fx={info.fx:.6f}  fy={info.fy:.6f}  cx={info.cx:.3f}  cy={info.cy:.3f}")
    print(f"  D={list(info.D)}")
    print(f"  K=\n{info.K}")


def poll_camera_window(window: str, bgr: np.ndarray) -> str:
    """Return 'ok', 'space', 'quit', or 'unavailable'."""
    try:
        import cv2
    except ImportError:
        return "unavailable"
    cv2.imshow(window, bgr)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        return "quit"
    if key == SPACE:
        return "space"
    return "ok"


def run(model, data, args) -> None:
    patrol = TargetPatrol()
    info = camera_info_from_model(model, width=args.width, height=args.height)
    yaml_path = save_camera_yaml(info)
    print_intrinsics(info)
    print(f"wrote {yaml_path}")
    print("Space: target start/stop left-right + yaw")
    print("Esc in camera window: quit (camera-only mode)")

    cam_period = 1.0 / max(args.fps, 1.0)
    last_cam = 0.0
    camera_ok = not args.no_cam_window
    feed: OnboardFeed | None = None

    def key_callback(keycode: int) -> None:
        if keycode == SPACE:
            patrol.toggle(data)

    def ensure_feed() -> OnboardFeed:
        nonlocal feed
        if feed is None:
            feed = OnboardFeed(model, width=args.width, height=args.height, shm=not args.no_shm)
            if feed.shm is not None:
                print(f"shared memory: {SHM_NAME}  (python aim/camera_example.py)")
        return feed

    def pump_camera() -> bool:
        nonlocal last_cam, camera_ok
        now = time.time()
        if now - last_cam < cam_period:
            return True
        last_cam = now
        bgr = ensure_feed().publish(data)
        if args.no_cam_window or not camera_ok:
            return True
        try:
            action = poll_camera_window("onboard", bgr)
        except Exception as exc:
            print(f"camera window unavailable: {exc}")
            camera_ok = False
            return True
        if action == "unavailable":
            camera_ok = False
            return True
        if action == "space":
            patrol.toggle(data)
        return action != "quit"

    try:
        if args.camera_only:
            while True:
                if patrol.active:
                    patrol.apply(data)
                mujoco.mj_step(model, data)
                if not pump_camera():
                    break
                time.sleep(model.opt.timestep)
            return

        try:
            # 不能写 `import mujoco.viewer`：那会把 mujoco 变成 run() 的局部名，
            # 让上面 --camera-only 分支里的 mujoco.mj_step 报 UnboundLocalError。
            import importlib

            importlib.import_module("mujoco.viewer")
        except Exception as exc:
            raise SystemExit(f"MuJoCo viewer unavailable: {exc}") from exc

        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            ensure_feed()
            while viewer.is_running():
                step_start = time.time()
                if patrol.active:
                    patrol.apply(data)
                mujoco.mj_step(model, data)
                viewer.sync()
                if not pump_camera():
                    break
                leftover = model.opt.timestep - (time.time() - step_start)
                if leftover > 0:
                    time.sleep(leftover)
    finally:
        if feed is not None:
            feed.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", choices=("red", "blue"), default="red")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument("--camera-only", action="store_true", help="only the onboard window")
    parser.add_argument("--no-cam-window", action="store_true", help="publish shm, no OpenCV window")
    parser.add_argument("--no-shm", action="store_true")
    args = parser.parse_args()

    model, data = load_infantry(args.team)
    print(f"loaded infantry team={args.team} ngeom={model.ngeom} nmocap={model.nmocap}")
    run(model, data, args)


if __name__ == "__main__":
    main()
