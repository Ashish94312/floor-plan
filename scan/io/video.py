"""Video tier input (ARCHITECTURE §10.1): clips -> sharp, well-spread keyframes as Frames.

Per-room clips (video/<room>/*.mov) map onto the photo pipeline: each room's clip becomes a handful
of keyframes (one sharpest frame per time slice, so they cover the clip and avoid motion blur).
Video files carry no focal length, and stabilisation crops the view, so the focal is estimated from
vanishing points over the keyframes (median), falling back to a nominal value with a warning.
A single continuous walkthrough needs room segmentation (C4) and is not supported yet.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from scan.errors import InputError
from scan.io.photos import intrinsics
from scan.types import Frame, PhotoMeta

VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation:format=duration:format_tags=com.apple.quicktime.model",
                          "-of", "json", str(path)], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise InputError(f"{path}: cannot read video ({out.stderr.strip()[:200]})")
    d = json.loads(out.stdout)
    st = d["streams"][0]
    rot = next((sd.get("rotation", 0) for sd in st.get("side_data_list", []) if "rotation" in sd), 0)
    W, H = (st["height"], st["width"]) if abs(int(rot)) % 180 == 90 else (st["width"], st["height"])
    return {"W": W, "H": H, "duration": float(d["format"]["duration"]),
            "model": d["format"].get("tags", {}).get("com.apple.quicktime.model")}


def decode(path: Path, fps: float, max_side: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """All frames at `fps`, upright (ffmpeg applies the rotation), long side = max_side."""
    info = probe(path)
    k = max_side / max(info["W"], info["H"])
    w, h = int(round(info["W"] * k / 2) * 2), int(round(info["H"] * k / 2) * 2)
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"fps={fps},scale={w}:{h}",
                        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=False)
    if p.returncode != 0 or not p.stdout:
        raise InputError(f"{path}: video decode failed ({p.stderr.decode()[:200]})")
    frames = np.frombuffer(p.stdout, np.uint8).reshape(-1, h, w, 3)
    return frames, np.arange(len(frames)) / fps, info


def keyframes(frames: np.ndarray, k: int, min_sharpness: float, min_gap: int = 0) -> list[int]:
    """One sharpest frame per equal time slice (coverage + no motion blur), at least `min_gap` frames
    after the previous pick (neighbouring slices could otherwise pick near-duplicates at their border)."""
    sharp = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])
    picks = []
    for sl in np.array_split(np.arange(len(frames)), k):
        if picks:
            sl = sl[sl >= picks[-1] + min_gap]
        if len(sl) and sharp[sl].max() >= min_sharpness:
            picks.append(int(sl[np.argmax(sharp[sl])]))
    return picks, sharp


def ingest_video(cap, cfg: dict) -> None:
    vc = cfg["video"]
    vdir = cap.root / "video"
    loose = [f for f in vdir.iterdir() if f.suffix.lower() in VIDEO_EXT]
    if loose:
        raise InputError(f"{vdir}: a single walkthrough clip ({loose[0].name}) needs room segmentation, not supported yet. "
                         "Record one clip per room into video/<room>/ instead.")
    room_dirs = sorted(d for d in vdir.iterdir() if d.is_dir() and not d.name.startswith("."))
    clips = {d.name: sorted(f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXT) for d in room_dirs}
    clips = {r: c for r, c in clips.items() if c}
    if not clips:
        raise InputError(f"{vdir}: no clips. Expected video/<room>/*.mov")
    per_room = vc["keyframes_per_room"]
    from scan.geometry.vanishing import focal_px

    raw, fs = {}, []
    for room, files in clips.items():
        fr_list, names, sharps = [], [], []
        for f in files:
            frames, ts, info = decode(f, vc["decode_fps"], cfg["ingest"]["working_max_side"])
            picks, sharp = keyframes(frames, max(1, per_room // len(files)), vc["min_sharpness"],
                                     min_gap=round(vc["min_keyframe_gap_s"] * vc["decode_fps"]))
            for i in np.argsort(sharp)[::-1][: vc["vp_frames_per_clip"]]:  # focal: many sharp frames, not just keyframes
                f_px = focal_px(frames[i], seed=int(i))
                if f_px is not None:
                    fs.append(f_px / math.hypot(*frames[i].shape[:2]))  # focal per unit diagonal
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
            for i in picks:
                fr_list.append(frames[i])
                names.append((f, ts[i], digest, info))
                sharps.append(sharp[i])
        raw[room] = (fr_list, names, sharps)
    # focal from vanishing points, pooled over all clips (same phone + mode)
    diag_mm = cfg["ingest"]["intrinsics"]["film_diagonal_mm"]
    nominal = vc["default_f35_mm"] / diag_mm
    plausible = [x for x in fs if 0.6 * nominal <= x <= 1.6 * nominal]
    if vc.get("force_f35_mm"):
        f_rel, source = vc["force_f35_mm"] / diag_mm, "default"
    elif len(plausible) >= vc["min_vp_frames"]:
        f_rel, source = float(np.median(plausible)), "vanishing_points"
    else:
        f_rel, source = nominal, "default"
        cap.warnings.append(f"video focal: only {len(plausible)} frames gave vanishing points; assuming "
                            f"{vc['default_f35_mm']} mm (35 mm eq.), wider intervals")
    f35 = f_rel * diag_mm
    for room, (fr_list, names, sharps) in raw.items():
        frames_out = []
        for fr, (f, t, digest, info), sh in zip(fr_list, names, sharps):
            H, W = fr.shape[:2]
            K = intrinsics(f35, W, H, diag_mm)
            meta = PhotoMeta(sha256=hashlib.sha256(f"{digest}@{t:.3f}".encode()).hexdigest(), full_size=(W, H),
                             orientation="portrait" if H > W else "landscape", device=info["model"], lens=None,
                             f35_mm=round(f35, 2), intrinsics_source=source, captured_at=None, sharpness=float(sh))
            frames_out.append(Frame(image_path=f.parent / f"{f.stem}_t{t:06.2f}", K=K, rgb=np.ascontiguousarray(fr),
                                    room_hint=room, timestamp=float(t), meta=meta))
        if len(frames_out) < cfg["ingest"]["min_images_per_room"]:
            raise InputError(f"{vdir / room}: only {len(frames_out)} sharp keyframes; record slower / steadier.")
        cap.rooms[room] = frames_out
    cap.devices = sorted({i[3]["model"] for _, n, _ in raw.values() for i in n if i[3]["model"]})
    cap.warnings.append(f"video: {sum(len(v) for v in cap.rooms.values())} keyframes from {sum(len(c) for c in clips.values())} "
                        f"clips; focal {f35:.1f} mm (35 mm eq.) from {source} ({len(plausible)} frames)")
