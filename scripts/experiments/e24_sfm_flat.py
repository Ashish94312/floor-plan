"""E24 (NEW APPROACH, separate track), step 1: one structure-from-motion model over all rooms' clips.

COLMAP over every room's clip at once: sharp frames (every `stride`-th at video.decode_fps) plus the model's keyframes,
one camera (same phone, vanishing-point focal fixed). Pairs: each frame with the next `overlap` frames of its own clip
and 2^j frames ahead beyond that (COLMAP's quadratic sequential overlap: keeps a clip connected
across short featureless or blurred stretches; without it the hall broke into 4 pieces, E24 step 1), and across clips every frame within `window_s` of a link time (stitch link frames: two clips see the same
doorway) against every frame within `window_s` of the other side's link time. If the clips land in one model, all
rooms share ONE geometric frame and ONE size: no learned depth involved.

Then, per room: the room's SfM points seen in its model keyframes, paired with the model's 3D point at that pixel;
similarity SfM -> model gives that room's model size per SfM unit (s_r). Across rooms in one SfM model, s_r / mean(s)
is each room's size in the model RELATIVE to the others: 1 = the model sizes the rooms consistently; 0.92 = the model
makes this room 8% small next to the others (E23b: the hall). Plus per-axis model / SfM size. No tape.

Usage: uv run python scripts/experiments/e24_sfm_flat.py <capture> [stride] [--reuse] [--keep-features] [--cross=N]
                                                         [section.key=value ...]
  --reuse: analysis only (frames + COLMAP model of an earlier run)
  --keep-features: keep the frames and the feature database of an earlier run (match new pairs, map again)
  --cross=N: also pair every N-th frame of each clip with every N-th frame of every other clip (E24 step 1 try 3: the
             link windows alone joined the bedroom to the others only through a few far doorway views)
Output: <capture>/experiments/E24_sfm/flat/ (images/<room>/, database.db, pairs.txt, sparse/, report.txt)
"""

import json
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

OVERLAP, WINDOW_S = 12, 5.0
cap_dir = Path(sys.argv[1])
nums = [a for a in sys.argv[2:] if "=" not in a and not a.startswith("--")]
stride = int(nums[0]) if nums else 2
reuse = "--reuse" in sys.argv
keep_features = "--keep-features" in sys.argv
cross = next((int(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--cross=")), 0)
overrides = [a for a in sys.argv[2:] if "=" in a and not a.startswith("--")]
extra: dict = {}
for a in overrides:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
vc = cfg["video"]
base = cap_dir / "experiments" / "E24_sfm"
out = base / "flat"
out.mkdir(parents=True, exist_ok=True)
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


def t_of(name: str) -> float:
    return float(re.search(r"_t(\d+\.\d+)", name).group(1))


rooms = sorted(d.name for d in (cap_dir / "video").iterdir() if d.is_dir() and not d.name.startswith("."))
if not all((base / r / "model_views.npz").exists() for r in rooms) or not (base / "links.json").exists():
    subprocess.run([sys.executable, str(Path(__file__).with_name("e24_export_model.py")), str(cap_dir), "all", *overrides],
                   check=True)  # torch in its own process (OpenMP clash with pycolmap)
V = {r: dict(np.load(base / r / "model_views.npz")) for r in rooms}
links = json.loads((base / "links.json").read_text())
K = V[rooms[0]]["K_work"].astype(float)
clips = {r: next(p for p in sorted((cap_dir / "video" / r).iterdir()) if p.suffix.lower() in VIDEO_EXT) for r in rooms}
key_t = {r: {round(t_of(str(n)), 2): i for i, n in enumerate(V[r]["names"]) if str(n).startswith(clips[r].stem)} for r in rooms}

img_dir = out / "images"
db = out / "database.db"


def existing_names():
    return {r: sorted(f"{r}/{p.name}" for p in (img_dir / r).iterdir()) for r in rooms}


if reuse and (out / "sparse" / "0").exists():
    names = existing_names()
    say(f"{cap_dir.name}: reusing frames and the COLMAP model of an earlier run")
else:
    if keep_features and db.exists():
        names = existing_names()
        shutil.rmtree(out / "sparse", ignore_errors=True)
        say(f"{cap_dir.name}: keeping {sum(map(len, names.values()))} frames and their features")
    else:
        names = {}
        for d in ("images", "sparse"):
            shutil.rmtree(out / d, ignore_errors=True)
        db.unlink(missing_ok=True)
        for r in rooms:
            frames, ts, _ = decode(clips[r], vc["decode_fps"], cfg["ingest"]["working_max_side"])
            sharp = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])
            keep = {i for i in range(0, len(frames), stride) if sharp[i] >= vc["min_sharpness"]}
            keep |= {i for i, t in enumerate(ts) if round(float(t), 2) in key_t[r]}
            (img_dir / r).mkdir(parents=True, exist_ok=True)
            names[r] = []
            for i in sorted(keep):
                n = f"{r}/{i:04d}_t{ts[i]:06.2f}.jpg"
                cv2.imwrite(str(img_dir / n), frames[i][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                names[r].append(n)
            say(f"{r}: {clips[r].name}, {len(names[r])} frames of {len(frames)}, {len(key_t[r])} model keyframes")
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "SIMPLE_PINHOLE"
        ro.camera_params = f"{K[0, 0]},{K[0, 2]},{K[1, 2]}"
        pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                                  device=pycolmap.Device.cpu)
    # pairs: sequential within a clip (next OVERLAP frames + 2^j ahead), around every link across clips, and
    # optionally a sparse grid between every two clips. Pairs already matched in the database are not redone.
    pairs = set()
    for r in rooms:
        for k, a in enumerate(names[r]):
            pairs |= {(a, b) for b in names[r][k + 1:k + 1 + OVERLAP]}
            j = 1
            while (1 << j) < len(names[r]):
                if (1 << j) > OVERLAP and k + (1 << j) < len(names[r]):
                    pairs.add((a, names[r][k + (1 << j)]))
                j += 1
    for ra, na, rb, nb in links:
        ta, tb = t_of(na), t_of(nb)
        A = [n for n in names[ra] if abs(t_of(n) - ta) <= WINDOW_S]
        B = [n for n in names[rb] if abs(t_of(n) - tb) <= WINDOW_S]
        pairs |= {(a, b) for a in A for b in B}
    if cross:
        for ra in rooms:
            for rb in rooms:
                if ra < rb:
                    pairs |= {(a, b) for a in names[ra][::cross] for b in names[rb][::cross]}
    (out / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
    say(f"pairs: {len(pairs)} ({len(links)} links across clips, +-{WINDOW_S:g} s each side"
        + (f"; every {cross}th frame across clips" if cross else "") + ")")
    po = pycolmap.ImportedPairingOptions()
    po.match_list_path = str(out / "pairs.txt")
    pycolmap.match_image_pairs(db, pairing_options=po, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    (out / "sparse").mkdir(exist_ok=True)
    pycolmap.incremental_mapping(db, img_dir, out / "sparse", options=opts)

recs = {int(p.name): pycolmap.Reconstruction(p) for p in sorted((out / "sparse").iterdir()) if p.is_dir()}
if not recs:
    say("SfM: no reconstruction")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    sys.exit(0)
for k, rec in recs.items():
    per = {r: sum(1 for im in rec.images.values() if im.name.startswith(r + "/") and getattr(im, "has_pose", True))
           for r in rooms}
    say(f"SfM model {k}: {rec.num_reg_images()} frames registered ({', '.join(f'{r} {n}/{len(names[r])}' for r, n in per.items())}), "
        f"{rec.num_points3D()} points, reprojection {rec.compute_mean_reprojection_error():.2f} px")
# per SfM model, per room: SfM points seen in the room's model keyframes <-> the model's points at those pixels. Also
# how far those points are from the keyframe cameras: far = seen through a doorway, where the runs disagree (E22w)
for mk, rec in recs.items():
    scales = {}
    say(f"--- SfM model {mk}")
    for r in rooms:
        v = V[r]
        A, B, D = [], [], []
        for im in rec.images.values():
            if not im.name.startswith(r + "/") or round(t_of(im.name), 2) not in key_t[r]:
                continue
            i = key_t[r][round(t_of(im.name), 2)]
            p2 = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
            if not p2:
                continue
            fm = v["K_exif"][i]
            px = (np.array([q[0] for q in p2], float) - K[:2, 2]) * (fm[0, 0] / K[0, 0]) + fm[:2, 2]
            Hm, Wm = v["mask"][i].shape
            c_, r_ = np.clip(np.round(px[:, 0]).astype(int), 0, Wm - 1), np.clip(np.round(px[:, 1]).astype(int), 0, Hm - 1)
            ok = v["mask"][i][r_, c_] & (px[:, 0] >= 0) & (px[:, 0] < Wm) & (px[:, 1] >= 0) & (px[:, 1] < Hm)
            P = v["pts"][i][r_, c_].astype(float)  # the pipeline's video points (model rays)
            dist = np.linalg.norm(P - v["T_wc"][i][:3, 3], axis=1)
            for (_, pid), p, o, d in zip(p2, P, ok, dist):
                if o:
                    A.append(rec.points3D[pid].xyz)
                    B.append(p)
                    D.append(d)
        if len(A) < 30:
            if A:
                say(f"{r}: {len(A)} pairs, too few")
            continue
        A, B, D = np.array(A, float), np.array(B, float), np.array(D, float)
        keepm = np.ones(len(A), bool)
        for _ in range(3):
            s, R, t = umeyama(B[keepm], A[keepm])
            res = np.linalg.norm(B - (s * A @ R.T + t), axis=1)
            keepm = res < 3 * np.median(res[keepm])
        s, R, t = umeyama(B[keepm], A[keepm])
        Am, Bm = s * A[keepm] @ R.T + t, B[keepm]
        slopes = []
        for k in range(3):
            a, b = Am[:, k] - Am[:, k].mean(), Bm[:, k] - Bm[:, k].mean()
            slopes.append(float((a @ b) / (a @ a)))
        scales[r] = s
        say(f"{r}: {keepm.sum()} of {len(A)} pairs, model size per SfM unit {s:.4f}, median residual "
            f"{np.median(np.linalg.norm(Bm - Am, axis=1)):.3f} m, per-axis model/SfM x {slopes[0]:.3f} y {slopes[1]:.3f} "
            f"z {slopes[2]:.3f}; points {np.median(D[keepm]):.1f} m from the cameras (p90 {np.percentile(D[keepm], 90):.1f})")
    if len(scales) > 1:
        g = float(np.exp(np.mean(np.log(list(scales.values())))))
        say("rooms' model size relative to the others (this SfM model; 1 = consistent): "
            + ", ".join(f"{r} {s_ / g:.3f}" for r, s_ in scales.items()))
say("model walls: " + "; ".join(f"{r}: {', '.join(str(w) for w in V[r]['walls'])}" for r in rooms))
(out / "report.txt").write_text("\n".join(lines) + "\n")
say(str(out / "report.txt"))
