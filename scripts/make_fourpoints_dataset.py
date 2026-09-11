#!/usr/bin/env python3
"""从 MuJoCo 机载相机录序列，写成 yolov5_fourpoints 的 9 列四点数据集。

标签一行: cls x1 y1 x2 y2 x3 y3 x4 y4  (归一化，点序左上-左下-右下-右上)
类别: cls = color * 6 + digit，color: B=0 R=1，digit: G,1,2,3,4,5
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.camera_model import (  # noqa: E402
    body_points_world,
    camera_info_from_model,
    mujoco_cam_to_opencv,
    project_opencv,
)
from scripts.model_io import MODEL_DIR  # noqa: E402

CLASS_NAMES = ["B_G", "B_1", "B_2", "B_3", "B_4", "B_5", "R_G", "R_1", "R_2", "R_3", "R_4", "R_5"]
DIGIT_IDS = {"G": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
# 只标对面靶车。文件名/类别按画面灯条颜色：红灯=R_*，蓝灯=B_*。
TARGET_ARMORS = ("target_armor_f", "target_armor_b", "target_armor_l", "target_armor_r")

# 灯条外角：与 5 小时前模型一致，18mm x 60mm，中心 y=±0.0642。
# 装甲体坐标：+X 朝外，+Y 朝右（正视时），+Z 朝上。
# 点序投影后按图像重排成 LT-LB-RB-RT，和 data36 一致。
GLOW_Y = 0.0642 + 0.009
GLOW_Z = 0.030
ARMOR_KPTS_LOCAL = np.array(
    [
        [0.0, -GLOW_Y, GLOW_Z],
        [0.0, -GLOW_Y, -GLOW_Z],
        [0.0, GLOW_Y, -GLOW_Z],
        [0.0, GLOW_Y, GLOW_Z],
    ],
    dtype=np.float64,
)

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
TARGET_Z = 0.085


def yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def write_sentry_texture(path: Path, size: int = 700) -> None:
    """data36 哨兵贴纸：白色炮塔线稿 + 底座梯形 + 四角定位孔，不是字母 G。"""
    img = np.full((size, size, 3), 10, dtype=np.uint8)
    s = size / 700.0
    thick = max(3, int(round(15 * s)))
    body = (np.array(
        [
            [205, 255],
            [225, 168],
            [355, 148],
            [418, 168],
            [430, 198],
            [555, 205],
            [560, 268],
            [430, 278],
            [405, 345],
            [330, 405],
            [255, 405],
            [205, 330],
        ],
        dtype=np.int32,
    ) * s).astype(np.int32)
    base = (np.array(
        [
            [245, 448],
            [445, 448],
            [500, 545],
            [190, 545],
        ],
        dtype=np.int32,
    ) * s).astype(np.int32)
    cv2.polylines(img, [body], True, (255, 255, 255), thick, cv2.LINE_AA)
    cv2.polylines(img, [base], True, (255, 255, 255), thick, cv2.LINE_AA)
    for x, y in ((95, 95), (605, 95), (95, 605), (605, 605)):
        cv2.circle(img, (int(x * s), int(y * s)), max(3, int(round(9 * s))), (255, 255, 255), -1, cv2.LINE_AA)
    Image.fromarray(img).save(path)


def ensure_digit_textures() -> None:
    tex_dir = ROOT / "assets" / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    sentry = tex_dir / "G.png"
    write_sentry_texture(sentry)
    print(f"wrote {sentry}")
    font = ImageFont.truetype(str(FONT_PATH), 640)
    for ch in ("1", "2", "3", "4", "5"):
        path = tex_dir / f"{ch}.png"
        if path.exists():
            continue
        canvas = Image.new("RGB", (700, 700), (10, 10, 10))
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        scale = 599.0 / max(h, 1)
        sized = ImageFont.truetype(str(FONT_PATH), max(8, int(640 * scale)))
        bbox = draw.textbbox((0, 0), ch, font=sized)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (700 - w) // 2 - bbox[0]
        y = 49 - bbox[1]
        if y < 8 or y + h > 692:
            y = (700 - h) // 2 - bbox[1]
        draw.text((x, y), ch, font=sized, fill=(255, 255, 255))
        canvas.save(path)
        print(f"wrote {path}")


def load_scene(armor_color: str, digit: str):
    """armor_color 是画面里靶车灯条颜色。要拍红灯就开蓝方（对面是红），反之亦然。"""
    if armor_color not in ("red", "blue"):
        raise ValueError(f"unknown armor color {armor_color!r}")
    ego_team = "blue" if armor_color == "red" else "red"
    src = (MODEL_DIR / "infantry.xml").read_text()
    mat_src = "materials_red.xml" if ego_team == "red" else "materials_blue.xml"
    mat = (MODEL_DIR / mat_src).read_text().replace('file="3.png"', f'file="{digit}.png"')
    mat_name = f"_materials_{armor_color}_{digit}_gen.xml"
    xml_name = f"_infantry_{armor_color}_{digit}_gen.xml"
    (MODEL_DIR / mat_name).write_text(mat)
    (MODEL_DIR / xml_name).write_text(src.replace("materials_red.xml", mat_name))
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / xml_name))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def set_scene(model, data, target_xy: np.ndarray, target_yaw: float, gimbal_yaw: float, gimbal_pitch: float) -> None:
    data.qpos[11] = gimbal_yaw
    data.qpos[12] = gimbal_pitch
    if model.nmocap >= 1:
        data.mocap_pos[0] = (float(target_xy[0]), float(target_xy[1]), TARGET_Z)
        data.mocap_quat[0] = yaw_quat(target_yaw)
    mujoco.mj_forward(model, data)


def cls_id(armor_color: str, digit: str) -> int:
    color = 1 if armor_color == "red" else 0
    return color * 6 + DIGIT_IDS[digit]


def order_lt_lb_rb_rt(uv: np.ndarray) -> np.ndarray:
    pts = np.asarray(uv, dtype=np.float64).reshape(4, 2)
    top = pts[np.argsort(pts[:, 1])[:2]]
    bot = pts[np.argsort(pts[:, 1])[2:]]
    lt = top[np.argmin(top[:, 0])]
    rt = top[np.argmax(top[:, 0])]
    lb = bot[np.argmin(bot[:, 0])]
    rb = bot[np.argmax(bot[:, 0])]
    return np.stack([lt, lb, rb, rt], axis=0)


def quad_area(uv: np.ndarray) -> float:
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def collect_labels(model, data, info, armor_color: str, digit: str) -> list[tuple[int, np.ndarray]]:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "onboard")
    R_cw, t_cw = mujoco_cam_to_opencv(data.cam_xpos[cam_id], data.cam_xmat[cam_id])
    labels = []
    w, h = info.width, info.height
    for body in TARGET_ARMORS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            continue
        n_world = data.xmat[bid].reshape(3, 3)[:, 0]
        n_cam = R_cw @ n_world
        if n_cam[2] >= -0.42:
            continue
        world = body_points_world(model, data, body, ARMOR_KPTS_LOCAL)
        cam = (R_cw @ world.T).T + t_cw
        if np.any(cam[:, 2] < 0.20):
            continue
        uv = project_opencv(info.K, R_cw, t_cw, world)
        if np.any(~np.isfinite(uv)):
            continue
        if np.any(uv[:, 0] < 2) or np.any(uv[:, 0] > w - 3):
            continue
        if np.any(uv[:, 1] < 2) or np.any(uv[:, 1] > h - 3):
            continue
        uv = order_lt_lb_rb_rt(uv)
        xs, ys = uv[:, 0], uv[:, 1]
        if xs.max() - xs.min() < 14 or ys.max() - ys.min() < 8:
            continue
        if quad_area(uv) < 80:
            continue
        labels.append((cls_id(armor_color, digit), uv))
    return labels


def write_label(path: Path, labels: list[tuple[int, np.ndarray]], width: int, height: int) -> None:
    lines = []
    for cls, uv in labels:
        n = uv.copy()
        n[:, 0] /= width
        n[:, 1] /= height
        n = np.clip(n, 0.0, 1.0)
        vals = " ".join(f"{v:.6f}" for v in n.reshape(-1))
        lines.append(f"{cls} {vals}\n")
    path.write_text("".join(lines))


def draw_labels(bgr: np.ndarray, labels: list[tuple[int, np.ndarray]]) -> np.ndarray:
    out = bgr.copy()
    for cls, uv in labels:
        color = (255, 160, 0) if cls < 6 else (0, 64, 255)
        pts = uv.astype(np.int32)
        for i in range(4):
            cv2.line(out, pts[i], pts[(i + 1) % 4], color, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            cv2.circle(out, p, 4, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(out, str(i), p + np.array([5, -5]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, CLASS_NAMES[cls], pts[0] + np.array([0, -12]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_data_yaml(out_dir: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (out_dir / "data12.yaml").write_text(
        f"""# MuJoCo infantry four-point dataset (yolov5_fourpoints)
path: {out_dir}
train: train/images
val: test/images

ncolor: 2
ndigit: 6
nc: 12
names:
{names}
"""
    )


@dataclass
class Clip:
    team: str
    digit: str
    distance: float
    split: str
    frames: int
    seed: int


def build_clips(teams: list[str], digits: list[str], distances: list[float], test_distance: float, frames: int) -> list[Clip]:
    clips = []
    seed = 0
    for team in teams:
        for digit in digits:
            for dist in distances:
                split = "test" if abs(dist - test_distance) < 1e-6 else "train"
                clips.append(Clip(team, digit, dist, split, frames, seed))
                seed += 1
    return clips


def run_clip(model, data, info, renderer, clip: Clip, out_dir: Path, preview_dir: Path, writers: dict) -> tuple[int, int]:
    rng = np.random.default_rng(clip.seed)
    yaw0 = np.pi + rng.uniform(-0.25, 0.25)
    gimbal_yaw = rng.uniform(-0.08, 0.08)
    gimbal_pitch = rng.uniform(-0.05, 0.04)
    spin = rng.uniform(1.4, 2.4)
    amp = rng.uniform(0.40, 0.60)
    speed = rng.uniform(0.95, 1.35)
    saved = 0
    empty = 0
    for i in range(clip.frames):
        t = i / max(clip.frames - 1, 1) * 2.4
        x = amp * np.sin(speed * t)
        yaw = yaw0 + spin * t
        set_scene(model, data, np.array([x, clip.distance]), yaw, gimbal_yaw, gimbal_pitch)
        renderer.update_scene(data, camera="onboard")
        rgb = renderer.render()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        labels = collect_labels(model, data, info, clip.team, clip.digit)
        vis = draw_labels(bgr, labels)
        key = f"{clip.team}_{clip.digit}"
        if key in writers:
            writers[key].write(vis)
        if not labels:
            empty += 1
            continue
        stem = f"{clip.team}_{clip.digit}_d{int(round(clip.distance * 100)):03d}_f{i:03d}"
        split_dir = out_dir / clip.split
        cv2.imwrite(str(split_dir / "images" / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        write_label(split_dir / "labels" / f"{stem}.txt", labels, info.width, info.height)
        dist_cm = int(round(clip.distance * 100))
        if i in (0, clip.frames // 2) and dist_cm in (135, 145):
            cv2.imwrite(str(preview_dir / f"{stem}.jpg"), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if i == 0 and dist_cm == 135:
                uv = np.vstack([p for _, p in labels])
                x0, y0 = np.maximum(uv.min(0).astype(int) - 50, 0)
                x1, y1 = uv.max(0).astype(int) + 50
                crop = vis[y0:y1, x0:x1]
                if crop.size:
                    crop = cv2.resize(crop, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(preview_dir / f"{stem}_zoom.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        saved += 1
    return saved, empty


def make_mosaic(preview_dir: Path) -> None:
    files = sorted(preview_dir.glob("*_f000.jpg"))
    if not files:
        files = sorted(preview_dir.glob("*.jpg"))[:12]
    if not files:
        return
    imgs = [cv2.imread(str(p)) for p in files[:12]]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    h, w = 240, 426
    tiles = [cv2.resize(im, (w, h)) for im in imgs]
    while len(tiles) < 12:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, 12, 4)]
    cv2.imwrite(str(preview_dir / "mosaic.jpg"), np.vstack(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/home/ubuntu/armor_sim_fourpoints"))
    parser.add_argument("--teams", default="red,blue", help="画面里靶车灯条颜色，红=R_* 蓝=B_*")
    parser.add_argument("--digits", default="G,1,2,3,4,5")
    parser.add_argument("--distances", default="1.35,1.40,1.45,1.50")
    parser.add_argument("--test-distance", type=float, default=1.50)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    teams = [x.strip() for x in args.teams.split(",") if x.strip()]
    digits = [x.strip() for x in args.digits.split(",") if x.strip()]
    distances = [float(x) for x in args.distances.split(",") if x.strip()]
    for d in digits:
        if d not in DIGIT_IDS:
            raise SystemExit(f"unknown digit {d}")

    ensure_digit_textures()
    out = args.out
    preview = out / "preview"
    for split in ("train", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out)

    clips = build_clips(teams, digits, distances, args.test_distance, args.frames)
    saved = empty = 0
    writers: dict[str, cv2.VideoWriter] = {}

    grouped: dict[tuple[str, str], list[Clip]] = {}
    for clip in clips:
        grouped.setdefault((clip.team, clip.digit), []).append(clip)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for (team, digit), team_clips in grouped.items():
        print(f"scene team={team} digit={digit} clips={len(team_clips)}")
        model, data = load_scene(team, digit)
        info = camera_info_from_model(model, width=args.width, height=args.height)
        renderer = mujoco.Renderer(model, height=info.height, width=info.width)
        key = f"{team}_{digit}"
        if digit == "3" and any(abs(c.distance - 1.40) < 1e-6 for c in team_clips):
            writers[key] = cv2.VideoWriter(str(preview / f"{key}_d140.mp4"), fourcc, 15.0, (info.width, info.height))
        try:
            for clip in team_clips:
                s, e = run_clip(model, data, info, renderer, clip, out, preview, writers)
                saved += s
                empty += e
                print(f"  {clip.split} d={clip.distance:.1f} saved={s} empty={e}")
        finally:
            renderer.close()
            if key in writers:
                writers[key].release()

    make_mosaic(preview)
    n_train = len(list((out / "train" / "images").glob("*.jpg")))
    n_test = len(list((out / "test" / "images").glob("*.jpg")))
    counts = np.zeros(12, dtype=int)
    for split in ("train", "test"):
        for p in (out / split / "labels").glob("*.txt"):
            for line in p.read_text().splitlines():
                if line.strip():
                    counts[int(float(line.split()[0]))] += 1
    print(f"done out={out} train={n_train} test={n_test} labeled_frames={saved} skipped_empty={empty}")
    print("class counts:")
    for i, n in enumerate(counts):
        print(f"  {i:2d} {CLASS_NAMES[i]:4s} {n}")


if __name__ == "__main__":
    main()
