#!/usr/bin/env python3
"""Onboard pinhole camera: OpenCV intrinsics, zero distortion, frame conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_CAMERA = "onboard"

# Number-plate quad in the armor body frame (meters). +X faces outward.
ARMOR_NUMBER_CORNERS = np.array(
    [
        [0.001, -0.060, -0.060],
        [0.001, 0.060, -0.060],
        [0.001, 0.060, 0.060],
        [0.001, -0.060, 0.060],
    ],
    dtype=np.float64,
)

# MuJoCo camera: X right, Y up, Z back. OpenCV: X right, Y down, Z forward.
R_CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraInfo:
    name: str
    width: int
    height: int
    fovy_deg: float
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    D: tuple[float, float, float, float, float]
    color_space: str

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["K"] = self.K.tolist()
        payload["D"] = list(self.D)
        payload["notes"] = {
            "distortion": "MuJoCo is ideal pinhole. D is exactly zero; do not undistort.",
            "principal_point": "Image center: cx=width/2, cy=height/2 (OpenGL convention).",
            "mujoco_frame": "X right, Y up, Z back; looks along -Z.",
            "opencv_frame": "X right, Y down, Z forward. R_cv = diag(1,-1,-1) @ R_mj.",
            "solvePnP": "Use this K and D=0. Object points: armor number quad, meters.",
            "armor_number_corners_m": ARMOR_NUMBER_CORNERS.tolist(),
            "armor_frame": "Armor body: +X outward, +Z up. 120mm square number plate.",
        }
        return payload


def pinhole_from_fovy(width: int, height: int, fovy_deg: float, name: str = DEFAULT_CAMERA) -> CameraInfo:
    fy = (0.5 * height) / np.tan(np.deg2rad(fovy_deg) * 0.5)
    fx = fy
    return CameraInfo(
        name=name,
        width=width,
        height=height,
        fovy_deg=float(fovy_deg),
        fx=float(fx),
        fy=float(fy),
        cx=width / 2.0,
        cy=height / 2.0,
        distortion_model="plumb_bob",
        D=(0.0, 0.0, 0.0, 0.0, 0.0),
        color_space="BGR8",
    )


def camera_info_from_model(model, name: str = DEFAULT_CAMERA, width: int | None = None, height: int | None = None) -> CameraInfo:
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cam_id < 0:
        raise ValueError(f"camera {name!r} not in model")
    res = model.cam_resolution[cam_id]
    width = int(width or (res[0] if res[0] > 0 else DEFAULT_WIDTH))
    height = int(height or (res[1] if res[1] > 0 else DEFAULT_HEIGHT))
    return pinhole_from_fovy(width, height, float(model.cam_fovy[cam_id]), name=name)


def mujoco_cam_to_opencv(xpos: np.ndarray, xmat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-to-OpenCV-camera (R, t) for solvePnP / projectPoints.

    p_cv = R @ p_world + t
    """
    R_wm = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    t_w = np.asarray(xpos, dtype=np.float64).reshape(3)
    R_w_cv = R_wm @ R_CV_FROM_MJ
    R_cw = R_w_cv.T
    t_cw = -R_cw @ t_w
    return R_cw, t_cw


def project_opencv(K: np.ndarray, R_cw: np.ndarray, t_cw: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    x = (R_cw @ pts.T).T + t_cw
    uv = (K @ x.T).T
    return uv[:, :2] / uv[:, 2:3]


def body_points_world(model, data, body_name: str, points_local: np.ndarray) -> np.ndarray:
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"body {body_name!r} not in model")
    R = data.xmat[bid].reshape(3, 3)
    return (R @ np.asarray(points_local, dtype=np.float64).T).T + data.xpos[bid]


def save_camera_yaml(info: CameraInfo, path: Path | None = None) -> Path:
    import yaml

    path = path or (ROOT / "models" / "onboard_camera.yaml")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(info.to_dict(), sort_keys=False, allow_unicode=True))
    return path


def main() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.model_io import load_infantry

    model, _data = load_infantry("red")
    info = camera_info_from_model(model)
    path = save_camera_yaml(info)
    print(path)
    print(f"K=\n{info.K}")
    print(f"D={info.D}")
    print(f"{info.width}x{info.height} fovy={info.fovy_deg}")


if __name__ == "__main__":
    main()
