#!/usr/bin/env python3
"""Example: read the live onboard camera the same way an auto-aim module would.

1. In another terminal:  python scripts/view_infantry.py --team red
2. Then:                 python aim/camera_example.py
   C++:                  cmake -S aim/cpp -B aim/cpp/build && cmake --build aim/cpp/build
                         ./aim/cpp/build/camera_example
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.onboard_feed import attach_shm, read_frame


def load_intrinsics(path: Path = ROOT / "models" / "onboard_camera.yaml"):
    cam = yaml.safe_load(path.read_text())
    K = np.asarray(cam["K"], dtype=np.float64)
    D = np.asarray(cam["D"], dtype=np.float64)
    return cam, K, D


def on_frame(bgr: np.ndarray, K: np.ndarray, D: np.ndarray, seq: int, stamp: float) -> None:
    """Replace this with your detector + solvePnP."""
    h, w = bgr.shape[:2]
    print(f"seq={seq:5d}  t={stamp:7.3f}s  {w}x{h}  mean={bgr.mean():.1f}")
    # corners = detect_armor(bgr)
    # ok, rvec, tvec = cv2.solvePnP(object_pts, corners, K, D)


def main() -> None:
    cam, K, D = load_intrinsics()
    print("camera", cam["name"], f"{cam['width']}x{cam['height']}")
    print("K =\n", K)
    print("D =", D)

    try:
        shm = attach_shm()
    except FileNotFoundError:
        raise SystemExit("先开仿真: python scripts/view_infantry.py --team red")

    last = -1
    try:
        while True:
            got = read_frame(shm)
            if got is None:
                time.sleep(0.003)
                continue
            seq, stamp, bgr = got
            if seq == last:
                time.sleep(0.003)
                continue
            last = seq

            on_frame(bgr, K, D, seq, stamp)

            cv2.imshow("aim input (onboard)", bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        shm.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
