"""LiDAR tier: Stray Scanner exports -> frames carrying ARKit depth, confidence and camera poses.

Stray Scanner (free App Store app, Pro iPhones) writes one folder per recording:
  odometry.csv           timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy: camera-to-world pose and intrinsics
                         of the 1920x1440 colour frame, one row per frame. World: ARKit's (gravity-aligned, +y up).
                         Camera axes: OpenCV (x right, y down, z ahead), as written; the ARKit camera flip (y, z)
                         makes two frames of the same wall disagree by ~30 cm instead of ~1 cm (E25a)
  depth/NNNNNN.png       uint16 millimetres, 256x192 (ARKit scene depth: z-depth)
  confidence/NNNNNN.png  uint8 0/1/2 (low / medium / high)
  rgb.mp4                the colour frames, one per odometry row
Layouts (ingest): a bare export at the capture root or in <capture>/lidar/ is one walk through the home (UNSPLIT: the
pipeline splits it into rooms at doorways, scan/layout/segment.py); <capture>/lidar/<room>/ holds one recording per
room (each its own ARKit session, so rooms are placed by door matching, as per-room photo runs).

No learned model is involved: depth and poses come from the phone. The pipeline gets the same per-view arrays the
MapAnything backend returns (lidar_views), so alignment, layout, openings, stitching and output are shared.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from scan.errors import InputError
from scan.geometry.cloud import unproject
from scan.types import Capture, Frame, PhotoMeta

ROOM_NAME = re.compile(r"^[a-z0-9_]+$")
UNSPLIT = "home"  # room id of a bare export until segmentation names its rooms
WORLD_UP = np.array([0.0, 1.0, 0.0])  # ARKit world: gravity-aligned, +y up


def recordings(root: Path) -> dict[str, Path]:
    """room -> recording folder for the accepted layouts."""
    if (root / "odometry.csv").is_file():
        return {UNSPLIT: root}
    lidar = root / "lidar"
    if (lidar / "odometry.csv").is_file():
        return {UNSPLIT: lidar}
    recs = {d.name: d for d in sorted(lidar.iterdir()) if d.is_dir() and (d / "odometry.csv").is_file()}
    if not recs:
        raise InputError(f"{lidar}: no Stray Scanner recording (a folder with odometry.csv) found.")
    return recs


def read_odometry(path: Path) -> dict[str, np.ndarray]:
    """Columns of odometry.csv; poses as 4x4 camera-to-world (OpenCV camera axes, as written: E25a)."""
    rows = [r.split(",") for r in path.read_text().strip().splitlines()[1:]]
    a = np.array([[float(x) for x in r[:13]] for r in rows])
    T = np.repeat(np.eye(4)[None], len(a), 0)
    T[:, :3, :3] = Rotation.from_quat(a[:, 5:9]).as_matrix()  # (qx, qy, qz, qw)
    T[:, :3, 3] = a[:, 2:5]
    return {"t": a[:, 0], "frame": a[:, 1].astype(int), "T_wc": T, "f": a[:, 9:11], "c": a[:, 11:13]}


def pick_frames(t: np.ndarray, n_max: int, min_dt: float) -> np.ndarray:
    """Indices spread evenly over the recording's time, at most n_max, at least min_dt seconds apart."""
    n = int(min(n_max, max(1, (t[-1] - t[0]) // min_dt + 1))) if min_dt > 0 else n_max
    want = np.linspace(t[0], t[-1], n)
    return np.unique(np.searchsorted(t, want).clip(0, len(t) - 1))


def _rgb_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    cap, out, i = cv2.VideoCapture(str(video)), {}, 0
    while len(out) < len(wanted) and cap.grab():
        if i in wanted:
            ok, bgr = cap.retrieve()
            if ok:
                out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return out


def load_recording(rec: Path, room: str, cfg: dict[str, Any]) -> tuple[list[Frame], list[str]]:
    lc, warnings = cfg["lidar"], []
    odo = read_odometry(rec / "odometry.csv")
    idx = pick_frames(odo["t"], lc["max_frames_per_room"], lc["min_dt_s"])
    rgb = _rgb_frames(rec / "rgb.mp4", {int(odo["frame"][i]) for i in idx})
    key = hashlib.sha256((rec / "odometry.csv").read_bytes()).hexdigest()
    frames = []
    for i in idx:
        fid = int(odo["frame"][i])
        dpath, cpath = rec / "depth" / f"{fid:06d}.png", rec / "confidence" / f"{fid:06d}.png"
        if fid not in rgb or not dpath.is_file():
            warnings.append(f"{rec.name}: frame {fid:06d} has no colour or depth image; skipped")
            continue
        depth = np.asarray(Image.open(dpath), np.float32) / 1000.0
        conf = np.asarray(Image.open(cpath), np.uint8) if cpath.is_file() else np.full(depth.shape, 2, np.uint8)
        H0, W0 = rgb[fid].shape[:2]
        s = cfg["ingest"]["working_max_side"] / max(H0, W0)
        img = cv2.resize(rgb[fid], (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_AREA)
        K = np.array([[odo["f"][i, 0] * s, 0, odo["c"][i, 0] * s], [0, odo["f"][i, 1] * s, odo["c"][i, 1] * s],
                      [0, 0, 1.0]])
        meta = PhotoMeta(sha256=f"{key}:{fid}", full_size=(W0, H0), orientation="portrait" if H0 > W0 else "landscape",
                         device="Stray Scanner (ARKit LiDAR)", lens=None, f35_mm=float("nan"),
                         intrinsics_source="arkit", captured_at=None,
                         sharpness=float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
        # one name per frame (views are matched to frames by name): rgb.mp4#<frame>
        frames.append(Frame(image_path=rec / f"rgb.mp4#{fid:06d}", K=K, rgb=img, room_hint=room, T_wc=odo["T_wc"][i],
                            depth=depth, depth_conf=conf, timestamp=float(odo["t"][i]), meta=meta))
    if len(frames) < 2:
        raise InputError(f"{rec}: fewer than 2 usable frames (colour + depth) in the recording.")
    return frames, warnings


def ingest_lidar(cap: Capture, cfg: dict[str, Any]) -> None:
    for room, rec in recordings(cap.root).items():
        if not ROOM_NAME.match(room):
            cap.warnings.append(f"{rec}: room name '{room}' should be lowercase letters, digits and _")
        frames, warns = load_recording(rec, room, cfg)
        cap.rooms[room] = frames
        cap.warnings += warns
    cap.devices = ["Stray Scanner (ARKit LiDAR)"]


def lidar_views(frames: list[Frame], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """The per-view arrays the geometry core reads (as from the MapAnything backend), at depth resolution.
    Points are the phone's depth through the phone's pose. mask: high confidence, within the sensor's range."""
    lc = cfg["lidar"]
    h, w = frames[0].depth.shape
    Ks, pts, rgbs = [], [], []
    for f in frames:
        sx, sy = w / f.rgb.shape[1], h / f.rgb.shape[0]
        K = f.K.copy()
        K[0] *= sx
        K[1] *= sy
        Ks.append(K)
        pts.append(unproject(f.depth, K, f.T_wc))
        rgbs.append(cv2.resize(f.rgb, (w, h), interpolation=cv2.INTER_AREA))
    depth = np.stack([f.depth for f in frames]).astype(np.float32)
    conf = np.stack([f.depth_conf for f in frames])
    K = np.stack(Ks)
    return {
        "pts": np.stack(pts).astype(np.float32),
        "depth": depth,
        "conf": (conf / 2.0).astype(np.float32),
        "mask": (conf >= lc["min_confidence"]) & (depth > 0.1) & (depth <= lc["max_depth_m"]),
        "T_wc": np.stack([f.T_wc for f in frames]),
        "K_model": K,
        "K_exif": K.copy(),
        "rgb": np.stack(rgbs),
        "names": np.array([f.image_path.name for f in frames]),
        "rooms": np.array([f.room_hint for f in frames]),
    }
