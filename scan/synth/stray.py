"""Synthetic Stray Scanner recordings of box rooms with known size (tests of the LiDAR tier; PLAN step 3.1).

A camera stands in the room and turns a full circle; depth is ray-cast against the box walls, floor and ceiling and
written in Stray Scanner's format (odometry.csv in ARKit axes, depth/ and confidence/ PNGs, rgb.mp4).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0])


def _R_cv(yaw: float, pitch: float) -> np.ndarray:
    """Camera-to-world rotation (OpenCV camera axes) in an ARKit world (y up), looking along yaw/pitch."""
    fwd = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(fwd, right), fwd], 1)


def write_box_room(out: Path, size=(4.0, 2.6, 3.0), cam=(2.0, 1.4, 1.5), n=24, hw=(48, 64), f=50.0,
                   rgb_scale=5, pitch_deg=(-25.0, 25.0), noise_m=0.0, seed=0) -> Path:
    """Box room x in [0, Lx], y (up) in [0, H], z in [0, Lz]; one recording of n frames turning 360 degrees while
    pitching between pitch_deg. Depth hw = (H, W) px with focal f px; colour frames rgb_scale times larger."""
    Lx, H, Lz = size
    h, w = hw
    rng = np.random.default_rng(seed)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    p = np.array(cam, float)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays = np.stack([(u - w / 2) / f, (v - h / 2) / f, np.ones_like(u)], -1)  # z = 1: t is the z-depth
    lo, hi = np.zeros(3), np.array([Lx, H, Lz])
    W, Hc = w * rgb_scale, h * rgb_scale
    vid = cv2.VideoWriter(str(out / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, Hc))
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    for i in range(n):
        yaw = 2 * np.pi * i / n
        pitch = np.deg2rad(np.interp(np.sin(3 * yaw), [-1, 1], pitch_deg))
        R = _R_cv(yaw, pitch)
        d = rays @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d > 0, (hi - p) / d, np.where(d < 0, (lo - p) / d, np.inf))
        depth = t.min(-1) + noise_m * rng.standard_normal((h, w))
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(out / "depth" / f"{i:06d}.png")
        Image.fromarray(np.full((h, w), 2, np.uint8)).save(out / "confidence" / f"{i:06d}.png")
        shade = (255 * (1 - depth / depth.max())).astype(np.uint8)
        vid.write(cv2.resize(cv2.merge([shade, shade, shade]), (W, Hc), interpolation=cv2.INTER_NEAREST))
        q = Rotation.from_matrix(R @ ARKIT_TO_CV).as_quat()  # ARKit camera axes
        fs = f * rgb_scale
        rows.append(f"{i / 30:.6f}, {i:06d}, {p[0]}, {p[1]}, {p[2]}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, "
                    f"{fs}, {fs}, {W / 2}, {Hc / 2}, , ")
    vid.release()
    (out / "odometry.csv").write_text("\n".join(rows) + "\n")
    return out
