"""E22u (step 1). Does the video hold what the keyframes miss?

The model sees ~12 keyframes per room (sharpest per time slice); the clips hold 4 fps x 75-160 s = 300-640 frames.
Every decoded frame is localised in the stitched plan frame by PnP: SIFT matches to the room's keyframes (all views
of its model run), whose pixels have 3D points, and the true focal (K at working resolution). Checked on the
keyframes themselves (their PnP pose vs the model run's pose). Then, per frame, which plan walls and which floor
cells are in view: sample points (walls: every 10 cm at 0.3 / 0.9 / 1.5 m; floor box: 10 cm grid) projected into the
frame, occluded when the stitched cloud has a surface clearly in front (coarse depth buffer). A frame 'sees' a wall
when >= 30% of its samples are visible. Compares all frames with the keyframes: walls the clip filmed but the
keyframes did not are what better keyframe choice (step 2) can recover; walls no frame saw are capture gaps.

Usage: uv run python scripts/experiments/coverage_audit.py <capture> <out_dir> [box x0 x1 y0 y1] [section.key=value ...]
"""

import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scan.config import load_config
from scan.io.video import VIDEO_EXT, decode, frame_name
from scan.pipeline import run
from scan.stitch.links import lookup

cap_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
nums = [a for a in sys.argv[3:] if "=" not in a]
box = tuple(map(float, nums[:4])) if len(nums) >= 4 else None
extra: dict = {}
for a in (a for a in sys.argv[3:] if "=" in a):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
out_dir.mkdir(parents=True, exist_ok=True)
cap, clouds, summary = run(cap_dir, cfg, out=out_dir / "_run", log=lambda *a: None)
rays_mode, vc = cfg["geometry"]["rays"], cfg["video"]
rooms = [r for r, c in clouds.items() if c.frame_id == "stitched"]
sift = cv2.SIFT_create(nfeatures=3000)
flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})

# --- plan walls and sample points (stitched frame) ---
walls = []  # (room, wall_id, start, end, samples Nx3)
for r in rooms:
    for w in clouds[r].layout.walls:
        a, b = np.array(w.start, float), np.array(w.end, float)
        n = max(2, int(np.linalg.norm(b - a) / 0.1))
        xy = a + (b - a) * np.linspace(0.02, 0.98, n)[:, None]
        S = np.vstack([np.c_[xy, np.full(n, z)] for z in (0.3, 0.9, 1.5)])
        walls.append((r, w.wall_id, a, b, S))
targets = [(f"{r}:{wid}", S) for r, wid, _a, _b, S in walls]
if box:
    gx, gy = np.meshgrid(np.arange(box[0], box[1], 0.1), np.arange(box[2], box[3], 0.1))
    targets.append(("box (floor)", np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 0.05)]))

scene = np.vstack([c.points for c in clouds.values() if c.frame_id == "stitched"])
scene = scene[np.random.default_rng(0).choice(len(scene), min(len(scene), 400_000), replace=False)]
ends = np.array([p for _r, _w, a, b, _S in walls for p in (a, b)])
xy_lo, xy_hi = ends.min(0) - 0.5, ends.max(0) + 0.5  # a hand-held camera stands inside the plan (+0.5 m), 0.3-2.5 m up
budget = cfg["geometry"]["max_joint_views"]  # views per model run (memory), E22u step 2 fills it by coverage


def visible_frac(R, t, K, hw, S, zbuf, cellpx):
    """Fraction of sample points S in front of the camera, inside the image and not behind a surface of the scene."""
    Xc = S @ R.T + t
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    ok = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hw[0])
    ci, cj = (uv[ok, 1] // cellpx).astype(int), (uv[ok, 0] // cellpx).astype(int)
    front = zbuf[ci, cj]
    vis = np.zeros(len(S), bool)
    vis[np.flatnonzero(ok)] = ~(front < z[ok] - 0.25)  # nothing clearly closer in that image cell
    return float(vis.mean())


def depth_buffer(R, t, K, hw, cellpx):
    Xc = scene @ R.T + t
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hw[0])
    zb = np.full((hw[0] // cellpx + 1, hw[1] // cellpx + 1), np.inf)
    np.minimum.at(zb, ((uv[ok, 1] // cellpx).astype(int), (uv[ok, 0] // cellpx).astype(int)), z[ok])
    return zb


def seen_cells(R, t, K, hw, zb, cellpx):
    """10 cm floor-plan cells of the scene surfaces this frame sees in front (nearest surface per image cell)."""
    Xc = scene @ R.T + t
    z = Xc[:, 2]
    uv = Xc[:, :2] / np.maximum(z, 1e-6)[:, None] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    ok = (z > 0.1) & (uv[:, 0] >= 0) & (uv[:, 0] < hw[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < hw[0])
    i = np.flatnonzero(ok)
    front = z[i] <= zb[(uv[i, 1] // cellpx).astype(int), (uv[i, 0] // cellpx).astype(int)] + 0.10
    ij = np.floor(scene[i[front], :2] / 0.1).astype(np.int64)
    return np.unique(ij[:, 0] * 100_000 + ij[:, 1])


results, extra_names = {}, []
for r in rooms:
    c = clouds[r]
    v = c.run_views
    K = cap.rooms[r][0].K.astype(np.float64)  # true focal (vanishing points), working resolution
    clip = next(p for p in sorted((cap_dir / "video" / r).iterdir()) if p.suffix.lower() in VIDEO_EXT)
    frames, ts, _ = decode(clip, vc["decode_fps"], cfg["ingest"]["working_max_side"])
    hw = frames.shape[1:3]
    sharp = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])
    implausible = 0
    key_t = {float(m.group(1)) for n in v["names"] if (m := re.search(r"_t(\d+\.\d+)$", str(n))) and str(n).startswith(clip.stem)}
    db = []  # per keyframe: (descriptors, 3D points)
    for i in range(len(v["depth"])):
        kp, des = sift.detectAndCompute(cv2.cvtColor(v["rgb"][i], cv2.COLOR_RGB2GRAY), None)
        if des is None:
            continue
        P, okp = lookup(v, i, np.array([k.pt for k in kp]), rays_mode)
        if okp.sum() >= 20:
            db.append((des[okp], P[okp]))
    rows = []
    for fi, (img, t_s) in enumerate(zip(frames, ts)):
        kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
        if des is None or len(kp) < 20:
            rows.append(None)
            continue
        px = np.array([k.pt for k in kp])
        obj, im = [], []
        for d_k, P_k in db:
            for m in (x for x in flann.knnMatch(des, d_k, k=2) if len(x) == 2):
                if m[0].distance < 0.75 * m[1].distance:
                    obj.append(P_k[m[0].trainIdx])
                    im.append(px[m[0].queryIdx])
        if len(obj) < 25:
            rows.append(None)
            continue
        obj, im = np.array(obj, np.float64), np.array(im, np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, im, K, None, iterationsCount=1000, reprojectionError=4.0,
                                                 confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 25:
            rows.append(None)
            continue
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], im[inl], K, None, rvec, tvec)
        R = cv2.Rodrigues(rvec)[0]
        t = tvec.ravel()
        C = -R.T @ t
        fwd = R.T @ [0, 0, 1.0]
        if np.any(C[:2] < xy_lo) or np.any(C[:2] > xy_hi) or not 0.3 <= C[2] <= 2.5:
            implausible += 1
            rows.append(None)
            continue
        zb = depth_buffer(R, t, K, hw, 16)
        vis = {name: visible_frac(R, t, K, hw, S, zb, 16) for name, S in targets}
        rows.append({"t": float(t_s), "C": C, "yaw": float(np.degrees(np.arctan2(fwd[1], fwd[0]))),
                     "pitch": float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))), "inliers": len(inl), "vis": vis,
                     "key": any(abs(t_s - k) < 0.13 for k in key_t), "cells": seen_cells(R, t, K, hw, zb, 16),
                     "sharp": float(sharp[fi])})
    results[r] = rows
    tracked = [x for x in rows if x]
    print(f"\n== {r}: {clip.name}, {len(frames)} frames at {vc['decode_fps']} fps, tracked {len(tracked)} "
          f"({len(tracked) / len(frames):.0%}; {implausible} implausible poses dropped), keyframes among them "
          f"{sum(x['key'] for x in tracked)} of {len(key_t)}")
    # check: PnP pose of a keyframe vs its pose in the model run
    errs = []
    for i, n in enumerate(v["names"]):
        m = re.search(r"_t(\d+\.\d+)$", str(n))
        if not m or not str(n).startswith(clip.stem):
            continue
        x = next((x for x in tracked if abs(x["t"] - float(m.group(1))) < 0.13), None)
        if x:
            Tw = v["T_wc"][i]
            f_run = Tw[:3, :3] @ [0, 0, 1.0]
            dy = (x["yaw"] - np.degrees(np.arctan2(f_run[1], f_run[0])) + 180) % 360 - 180
            errs.append((np.linalg.norm(x["C"] - Tw[:3, 3]), abs(dy)))
    if errs:
        e = np.array(errs)
        print(f"   keyframe check (PnP vs model run): position median {np.median(e[:, 0]):.3f} m (max {e[:, 0].max():.2f}), "
              f"heading median {np.median(e[:, 1]):.1f} deg (max {e[:, 1].max():.1f})")
    yaw_all = np.array([x["yaw"] for x in tracked])
    yaw_key = np.array([x["yaw"] for x in tracked if x["key"]])
    hist = lambda y: np.histogram((y + 22.5) % 360, bins=np.arange(0, 361, 45))[0]
    print(f"   heading (8 x 45 deg from +x): all {hist(yaw_all).tolist()}  keyframes {hist(yaw_key).tolist()}")
    print(f"   {'target':24s} frames seeing it (>=30%): all / keyframes   best view: all / keyframes")
    for name, _S in targets:
        fa = [x["vis"][name] for x in tracked]
        fk = [x["vis"][name] for x in tracked if x["key"]]
        n_all, n_key = sum(f >= 0.3 for f in fa), sum(f >= 0.3 for f in fk)
        if n_all or name.startswith((r, "box")):
            print(f"   {name:24s} {n_all:4d} / {n_key:2d}        {max(fa, default=0):.0%} / {max(fk, default=0):.0%}")
    # step 2: add frames by coverage. Start from what the run's keyframes see; add the tracked frame that sees the
    # most scene cells not yet seen (sharpness breaks ties), at least min_keyframe_gap_s from every chosen frame,
    # until the run is full. A tracked frame shares >= 25 PnP inliers with the keyframes, so it overlaps them.
    covered = set().union(*[set(x["cells"]) for x in tracked if x["key"]]) if any(x["key"] for x in tracked) else set()
    chosen_t = [x["t"] for x in tracked if x["key"]]
    free_slots = max(0, budget - len(v["depth"]))
    cand = [x for x in tracked if not x["key"] and x["sharp"] >= vc["min_sharpness"]]
    print(f"   coverage: keyframes see {len(covered)} cells; all tracked frames {len(set().union(*[set(x['cells']) for x in tracked]))}; "
          f"run has {len(v['depth'])} views, {free_slots} free")
    for _ in range(free_slots):
        ok_c = [x for x in cand if min((abs(x["t"] - t0) for t0 in chosen_t), default=9e9) >= vc["min_keyframe_gap_s"]]
        if not ok_c:
            break
        best = max(ok_c, key=lambda x: (len(set(x["cells"]) - covered), x["sharp"]))
        gain = len(set(best["cells"]) - covered)
        if gain == 0:
            break
        covered |= set(best["cells"])
        chosen_t.append(best["t"])
        extra_names.append(frame_name(clip, best["t"]))
        sees = [n for n, f in best["vis"].items() if f >= 0.3]
        print(f"   + {frame_name(clip, best['t'])}: +{gain} cells, camera ({best['C'][0]:+.2f}, {best['C'][1]:+.2f}) "
              f"heading {best['yaw']:+.0f} deg, sees {', '.join(sees) or '-'}")

# --- figure: every tracked camera (grey, heading tick) and keyframes (red) over the plan walls ---
fig, axes = plt.subplots(1, len(rooms), figsize=(6.5 * len(rooms), 6.5))
for ax, r in zip(np.atleast_1d(axes), rooms):
    for rr, wid, a, b, _S in walls:
        ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="k" if rr == r else "0.75", lw=2 if rr == r else 1)
        if rr == r:
            ax.text(*(a + b) / 2, wid.split("-")[-1], fontsize=7)
    if box:
        ax.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec="tab:purple", ls="--"))
    tr = [x for x in results[r] if x]
    for x in tr:
        d = 0.18 * np.array([np.cos(np.radians(x["yaw"])), np.sin(np.radians(x["yaw"]))])
        col = "red" if x["key"] else plt.cm.viridis(x["t"] / max(1e-6, tr[-1]["t"]))
        ax.plot([x["C"][0], x["C"][0] + d[0]], [x["C"][1], x["C"][1] + d[1]], "-", color=col, lw=1.6 if x["key"] else 0.6)
        ax.plot(x["C"][0], x["C"][1], "o", color=col, ms=3 if x["key"] else 1)
    ax.set_title(f"{r}: {len(tr)} tracked frames (colour = time), red = keyframes")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out_dir / "coverage.png", dpi=110)
np.save(out_dir / "coverage.npy", {r: [x and {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items()}
                                       for x in rows] for r, rows in results.items()}, allow_pickle=True)
print(f"\n{out_dir / 'coverage.png'}")
print("video.extra_keyframes=[" + ",".join(extra_names) + "]")
(out_dir / "extra_keyframes.txt").write_text("[" + ",".join(extra_names) + "]\n")
