"""E22n. Video focal by self-calibration (independent of the model, vanishing points and metadata).

For two views of a rigid scene with the true intrinsics K, E = K^T F K is an essential matrix: its two
non-zero singular values are equal (Mendonca-Cipolla). Sweep the 35 mm-equivalent focal, score each
value by (s1 - s2) / (s1 + s2) of E over many frame pairs with parallax, take the best. Principal point
at the image centre. Pairs a homography explains (pure rotation / one plane) are skipped: F is
undetermined there.

Usage: uv run python scripts/experiments/video_selfcal.py <clip or photo dir> [...]
  clip: frames 0.5-1.5 s apart.  photo dir (validation, known EXIF focal): every pair of photos.
"""

import itertools
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from scan.config import load_config
from scan.io.video import VIDEO_EXT, decode

cfg = load_config()
DIAG = cfg["ingest"]["intrinsics"]["film_diagonal_mm"]
F35 = np.arange(18.0, 42.01, 0.25)
sift = cv2.SIFT_create(nfeatures=5000)


def feats(img):
    k, d = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    return np.float64([p.pt for p in k]), d


def pair_F(fa, fb, min_inl=150):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None:
        return None
    good = [m for m, n in (x for x in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if len(x) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < min_inl:
        return None
    A, B = pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]]
    F, mF = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, 1.0, 0.9999)
    if F is None or F.shape != (3, 3) or mF.sum() < min_inl:
        return None
    _H, mH = cv2.findHomography(A, B, cv2.RANSAC, 2.0)
    if mH is not None and mH.sum() > 0.8 * mF.sum():  # rotation-only / planar: F says nothing about f
        return None
    return F, int(mF.sum())


def costs(F, W, H):
    out = []
    for f35 in F35:
        f = f35 / DIAG * math.hypot(W, H)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        s = np.linalg.svd(K.T @ F @ K, compute_uv=False)
        out.append((s[0] - s[1]) / (s[0] + s[1]))
    return np.array(out)


def run(frames, pairs, label):
    H, W = frames[0].shape[:2]
    fe = {}
    C, n_ok = [], 0
    for i, j in pairs:
        for k in (i, j):
            if k not in fe:
                fe[k] = feats(frames[k])
        r = pair_F(fe[i], fe[j])
        if r is None:
            continue
        C.append(costs(r[0], W, H))
        n_ok += 1
    if not C:
        print(f"{label}: no usable pairs")
        return
    C = np.array(C)
    med = np.median(C, 0)
    per_pair = F35[np.argmin(C, 1)]
    print(f"{label}: {n_ok}/{len(pairs)} pairs | median-cost minimum {F35[np.argmin(med)]:.2f} mm | per-pair best "
          f"p25/50/75 {np.percentile(per_pair, 25):.1f} / {np.median(per_pair):.1f} / {np.percentile(per_pair, 75):.1f} mm"
          f" | cost at 26/27/28/30/32 mm: " + " ".join(f"{med[np.argmin(abs(F35 - x))]:.4f}" for x in (26, 27, 28, 30, 32)))


for arg in map(Path, sys.argv[1:]):
    if arg.is_file() and arg.suffix.lower() in VIDEO_EXT:
        fps = 4
        frames, ts, info = decode(arg, fps, 1024)
        sharp = np.array([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames])
        ok = np.flatnonzero(sharp >= np.percentile(sharp, 40))[::2]
        pairs = [(i, j) for i in ok for j in ok if 2 <= j - i <= 6][:240]
        run(frames, pairs, f"{arg.parent.name}/{arg.name} ({len(frames)} frames)")
    else:
        from scan.io.photos import load_photo

        photos = [load_photo(p, arg.name, cfg)[0] for p in sorted(arg.iterdir())
                  if p.suffix.lower() in cfg["ingest"]["image_extensions"]]
        groups = {}
        for p in photos:  # one orientation + size per run (EXIF focal per photo printed)
            groups.setdefault(p.rgb.shape, []).append(p)
        for shape, ps in groups.items():
            print(f"  {arg.name} {shape}: EXIF f35 {sorted({p.meta.f35_mm for p in ps})}")
            run([p.rgb for p in ps], list(itertools.combinations(range(len(ps)), 2)), f"{arg.name} {shape}")
