"""E22w (option A). Same physical points, same size in every room's model run?

Links (stitch.links) measure the size ratio of two rooms' runs from one designated frame pair per room pair. Here:
every own keyframe of room A against every own keyframe of room B (link frames of the other clip excluded), SIFT +
RANSAC fundamental matrix, each matched pixel's 3D point in A's run and in B's run (stitched frame). Pooled per room
pair: RANSAC similarity A ~ s B (tolerance 5% of the distance to the cameras), i.e. how much bigger A's version of
the same points is. Also per frame pair (>= 15 inliers), with where the two cameras stand, to see whether the ratio
depends on which part of a run the points come from. Then the loop: s_ab * s_bc * s_ca = 1 if the runs are
consistent.

Usage: uv run python scripts/experiments/cross_ratios.py <capture> [section.key=value ...]
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import yaml

from scan.config import load_config
from scan.geometry.repose import similarity_ransac, umeyama
from scan.pipeline import run
from scan.stitch.links import lookup, match_pixels

cap_dir = Path(sys.argv[1])
extra: dict = {}
for a in sys.argv[2:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
cap, clouds, _s = run(cap_dir, cfg, out=cap_dir / "experiments" / "E22w_cross" / "_run", log=lambda *a: None)
rays = cfg["geometry"]["rays"]
rooms = [r for r, c in clouds.items() if c.frame_id == "stitched"]
stem = {r: {f.image_path.name.rpartition("_t")[0] for f in cap.rooms[r] if not f.link} for r in rooms}


def own(r):
    v = clouds[r].run_views
    return [i for i, n in enumerate(v["names"]) if str(n).rpartition("_t")[0] in stem[r]]


ratios = {}
for a, b in itertools.combinations(rooms, 2):
    va, vb = clouds[a].run_views, clouds[b].run_views
    A3, B3, D, per = [], [], [], []
    for i in own(a):
        for j in own(b):
            pa, pb = match_pixels(va["rgb"][i], vb["rgb"][j])
            if len(pa) < 12:
                continue
            Pa, oka = lookup(va, i, pa, rays)
            Pb, okb = lookup(vb, j, pb, rays)
            ok = oka & okb
            if ok.sum() < 12:
                continue
            ca, cb = va["T_wc"][i][:3, 3], vb["T_wc"][j][:3, 3]
            d = np.maximum(np.linalg.norm(Pa[ok] - ca, axis=1), np.linalg.norm(Pb[ok] - cb, axis=1))
            A3.append(Pa[ok])
            B3.append(Pb[ok])
            D.append(d)
            sim = similarity_ransac(Pa[ok], Pb[ok], 0.05 * np.maximum(d, 0.3))
            if sim is not None and sim[3].sum() >= 15:
                s_ij = umeyama(Pa[ok][sim[3]], Pb[ok][sim[3]])[0]
                per.append((str(va["names"][i]), str(vb["names"][j]), ca, cb, int(sim[3].sum()), s_ij, float(np.median(d))))
    if not A3:
        print(f"{a} ~ s {b}: no shared points")
        continue
    A3, B3, D = map(np.concatenate, (A3, B3, D))
    sim = similarity_ransac(A3, B3, 0.05 * np.maximum(D, 0.3), iters=1000)
    s = float(sim[0]) if sim is not None else float("nan")
    ratios[(a, b)] = s
    print(f"\n{a} ~ s x {b}: pooled s = {s:.3f} ({int(sim[3].sum()) if sim is not None else 0} of {len(A3)} matches agree), "
          f"{len(per)} frame pairs with >= 15 inliers")
    for na, nb, ca, cb, n, s_ij, d in sorted(per, key=lambda x: x[5]):
        print(f"    s {s_ij:.3f}  {n:3d} pts at {d:.1f} m   {na} cam ({ca[0]:+.1f},{ca[1]:+.1f})  <->  {nb} cam ({cb[0]:+.1f},{cb[1]:+.1f})")
if len(rooms) == 3 and len(ratios) == 3:
    (a, b), (_, c) = list(ratios)[0], list(ratios)[1]
    loop = ratios[(a, b)] * ratios[(b, c)] / ratios[(a, c)]
    print(f"\nloop s({a},{b}) s({b},{c}) / s({a},{c}) = {loop:.3f} (1 = consistent)")
