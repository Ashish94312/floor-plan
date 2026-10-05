"""E24 (NEW APPROACH, separate track): classical structure-from-motion as a geometry reference with no learned depth.

Everything in the pipeline's 3D comes from MapAnything's learned depth. COLMAP triangulates from camera motion only:
right shape up to one overall size. This script, for one room's clip:
  1. frames: sharp decoded frames (every `stride`-th at video.decode_fps) plus the model run's keyframes
  2. COLMAP (pycolmap): SIFT, sequential matching, incremental mapping, focal FIXED to the vanishing-point focal
  3. coverage: registered frames, points, reprojection error
  4. comparison with the model: each SfM point seen in a model keyframe is paired with the model's 3D point at that
     pixel (plan frame: floor z=0, walls on x / y). Similarity fit (RANSAC), then the size ratio SfM/model along x,
     y and z separately: equal = the model's shape matches pure geometry; unequal = the model stretches or squeezes
     the room along that axis (E22v/E23b: the hall comes out ~8% short east-west but right north-south in every run).
No tape is used. The model's side comes from e24_export_model.py (run in its own process: torch and pycolmap each
ship an OpenMP runtime and abort in one process).

Usage: uv run python scripts/experiments/e24_sfm_room.py <capture> <room> [stride] [--reuse] [section.key=value ...]
  --reuse: keep the frames and COLMAP model of an earlier run (compare only)
Output: <capture>/experiments/E24_sfm/<room>/ (images/, database.db, sparse/, report.txt)
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import yaml

from scan.config import load_config
from scan.io.video import VIDEO_EXT, decode

cap_dir, room = Path(sys.argv[1]), sys.argv[2]
nums = [a for a in sys.argv[3:] if "=" not in a and not a.startswith("--")]
reuse = "--reuse" in sys.argv
stride = int(nums[0]) if nums else 2
extra: dict = {}
for a in (a for a in sys.argv[3:] if "=" in a):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
out = cap_dir / "experiments" / "E24_sfm" / room
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def umeyama(A, B):
    """Similarity A ~ s R B + t (least squares)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, D, Vt = np.linalg.svd(A0.T @ B0 / len(A))
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (B0**2).sum(1).mean())
    return s, R, ma - s * R @ mb


# --- the model's reconstruction of this room (plan frame) and its keyframes, exported in a torch process
npz = out / "model_views.npz"
if not npz.exists():
    subprocess.run([sys.executable, str(Path(__file__).with_name("e24_export_model.py")), str(cap_dir), room,
                    *[a for a in sys.argv[3:] if "=" in a]], check=True)
m_ = np.load(npz)
v = {k: m_[k] for k in ("names", "pts", "mask", "K_exif", "depth", "T_wc")}
K = m_["K_work"].astype(float)  # working resolution, vanishing-point focal
vc = cfg["video"]
clip = next(p for p in sorted((cap_dir / "video" / room).iterdir()) if p.suffix.lower() in VIDEO_EXT)
key_t = {round(float(m.group(1)), 2): i for i, n in enumerate(v["names"])
         if (m := re.search(r"_t(\d+\.\d+)$", str(n))) and str(n).startswith(clip.stem)}

if reuse and (out / "sparse" / "0").exists():
    recs = {0: pycolmap.Reconstruction(out / "sparse" / "0")}
    keep = sorted(out.joinpath("images").iterdir())
    say(f"{cap_dir.name} {room}: reusing {len(keep)} frames and the COLMAP model of an earlier run")
else:
    # --- 1. frames
    frames, ts, _ = decode(clip, vc["decode_fps"], cfg["ingest"]["working_max_side"])
    sharp = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])
    keep = {i for i in range(0, len(frames), stride) if sharp[i] >= vc["min_sharpness"]}
    keep |= {i for i, t in enumerate(ts) if round(float(t), 2) in key_t}
    img_dir = out / "images"
    shutil.rmtree(out / "images", ignore_errors=True)
    shutil.rmtree(out / "sparse", ignore_errors=True)
    (out / "database.db").unlink(missing_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    for i in sorted(keep):
        cv2.imwrite(str(img_dir / f"{i:04d}_t{ts[i]:06.2f}.jpg"), frames[i][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    H, W = frames.shape[1:3]
    say(f"{cap_dir.name} {room}: {clip.name}, {len(keep)} frames of {len(frames)} ({W}x{H}), focal {K[0, 0]:.1f} px fixed, "
        f"{len(key_t)} model keyframes")

    # --- 2. COLMAP
    db = out / "database.db"
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "SIMPLE_PINHOLE"
    ro.camera_params = f"{K[0, 0]},{K[0, 2]},{K[1, 2]}"
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                              device=pycolmap.Device.cpu)
    po = pycolmap.SequentialPairingOptions()
    po.overlap = 12
    pycolmap.match_sequential(db, pairing_options=po)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    (out / "sparse").mkdir(exist_ok=True)
    recs = pycolmap.incremental_mapping(db, img_dir, out / "sparse", options=opts)

# --- 3. coverage
if not recs:
    say("SfM: no reconstruction")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    sys.exit(0)
sizes = sorted(((r.num_reg_images(), k) for k, r in recs.items()), reverse=True)
rec = recs[sizes[0][1]]
say(f"SfM: {len(recs)} model(s), registered {[s for s, _ in sizes]} of {len(keep)} frames; largest: "
    f"{rec.num_points3D()} points, mean reprojection error {rec.compute_mean_reprojection_error():.2f} px")

# --- 4. SfM points <-> the model's 3D points at the same keyframe pixels, for both ray models
fm = v["K_exif"]  # the model's views: true focal at model resolution (content box)
for RAYS in ("model", "exif"):
    say(f"model points along the {RAYS} rays:")
    A, B = [], []  # SfM point, model point
    for img in rec.images.values():
        m = re.search(r"_t(\d+\.\d+)\.jpg$", img.name)
        if not m or round(float(m.group(1)), 2) not in key_t:
            continue
        i = key_t[round(float(m.group(1)), 2)]
        pts2d = [(p.xy, p.point3D_id) for p in img.points2D if p.has_point3D()]
        if not pts2d:
            continue
        xy = np.array([q[0] for q in pts2d], float)
        px = (xy - K[:2, 2]) * (fm[i][0, 0] / K[0, 0]) + fm[i][:2, 2]  # working-resolution pixel -> model pixel
        Hm, Wm = v["mask"][i].shape
        c_, r_ = np.clip(np.round(px[:, 0]).astype(int), 0, Wm - 1), np.clip(np.round(px[:, 1]).astype(int), 0, Hm - 1)
        ok = v["mask"][i][r_, c_] & (px[:, 0] >= 0) & (px[:, 0] < Wm) & (px[:, 1] >= 0) & (px[:, 1] < Hm)
        if RAYS == "model":  # the pipeline's video points: model depth along the model's own rays
            P = v["pts"][i][r_, c_].astype(float)
        else:  # model depth along the TRUE rays (vanishing-point focal)
            d = v["depth"][i][r_, c_].astype(float)
            cam = np.stack([(c_ - fm[i][0, 2]) / fm[i][0, 0], (r_ - fm[i][1, 2]) / fm[i][1, 1], np.ones(len(c_))], 1) * d[:, None]
            P = cam @ v["T_wc"][i][:3, :3].T + v["T_wc"][i][:3, 3]
        for (_, pid), p, o in zip(pts2d, P, ok):
            if o:
                A.append(rec.points3D[pid].xyz)
                B.append(p)
    A, B = np.array(A, float), np.array(B, float)
    say(f"pairs (SfM point seen in a model keyframe, model point at that pixel): {len(A)}")
    if len(A) < 30:
        say("too few pairs to compare shapes")
        continue
    # similarity SfM -> model (B ~ s R A + t), refit on the points within 3x the median residual
    keepm = np.ones(len(A), bool)
    for _ in range(3):
        s, R, t = umeyama(B[keepm], A[keepm])
        res = np.linalg.norm(B - (s * A @ R.T + t), axis=1)
        keepm = res < 3 * np.median(res[keepm])
    s, R, t = umeyama(B[keepm], A[keepm])
    Am = s * A[keepm] @ R.T + t  # SfM points in the model's plan frame, at the model's overall size
    Bm = B[keepm]
    say(f"similarity fit: {keepm.sum()} inliers, median residual {np.median(np.linalg.norm(Bm - Am, axis=1)):.3f} m")
    # per-axis size: slope of model vs SfM coordinates (centred) along the plan axes; 1 = same shape
    for k, ax in enumerate("xyz"):
        a, b = Am[:, k] - Am[:, k].mean(), Bm[:, k] - Bm[:, k].mean()
        slope = float((a @ b) / (a @ a))
        say(f"   {ax}: model / SfM size {slope:.3f}  (spread of the points along {ax}: {np.ptp(Am[:, k]):.2f} m)")
say("model layout walls: " + ", ".join(str(w) for w in m_["walls"]))
(out / "report.txt").write_text("\n".join(lines) + "\n")
say(str(out / "report.txt"))
