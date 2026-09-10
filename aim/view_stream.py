#!/usr/bin/env python3
"""Show or dump the live onboard shared-memory stream."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.onboard_feed import attach_shm, read_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, help="write one frame and exit")
    args = parser.parse_args()

    try:
        shm = attach_shm()
    except FileNotFoundError:
        raise SystemExit("shared memory autoaim_onboard not found; start scripts/view_infantry.py first")

    last = -1
    try:
        while True:
            got = read_frame(shm)
            if got is None:
                time.sleep(0.005)
                continue
            seq, stamp, bgr = got
            if seq == last:
                time.sleep(0.005)
                continue
            last = seq
            if args.output:
                import cv2

                args.output.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.output), bgr)
                print(f"wrote {args.output} seq={seq} t={stamp:.3f}")
                return
            import cv2

            cv2.imshow("onboard (shm)", bgr)
            if cv2.waitKey(1) == 27:
                return
    finally:
        shm.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
