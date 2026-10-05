"""E22v (step 3 test). Does the model's depth error grow with distance? Checked on the floor, without tape.

A floor pixel's depth follows from geometry alone: its true ray (K from vanishing points) from a camera at height h
meets the plane z = 0 at z-depth h / (-d_z) along the ray (d = ray in the levelled frame, camera-z component 1). The
model gives its own depth for the same pixel. Their ratio, binned by the geometric distance, shows whether the model
is short (or long) at range. Ratio ~1 everywhere: the floor line adds nothing over the model's depth. Ratio falling
with distance: far walls come out short, and walls measured from where the floor meets them (camera height + angle)
would fix it. The camera height and tilt come from the model run (floor plane fit, mostly near points).

Usage: uv run python scripts/experiments/floor_range.py <capture> [section.key=value ...]
"""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import yaml

from scan.config import load_config
from scan.pipeline import run

cap_dir = Path(sys.argv[1])
extra: dict = {}
for a in sys.argv[2:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, _s = run(cap_dir, cfg, out=cap_dir / "experiments" / "E22v_floor" / "_run", log=lambda *a: None)
bins = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
print(f"model depth / floor-geometry depth on floor pixels, median per distance bin (m): {bins.tolist()}")
for r, c in clouds.items():
    v = c.views
    ratios, dists = [], []
    for i in range(len(v["depth"])):
        floor = v["mask"][i] & (np.abs(v["pts"][i][..., 2]) < 0.04)  # model points on the floor (aligned frame, floor z=0)
        if floor.sum() < 200:
            continue
        rr, cc = np.nonzero(floor)
        Kinv = np.linalg.inv(v["K_exif"][i])  # true focal
        d_cam = (Kinv @ np.stack([cc, rr, np.ones_like(cc)]).astype(float)).T  # z component 1 -> t is the z-depth
        R, C = v["T_wc"][i][:3, :3], v["T_wc"][i][:3, 3]
        d_w = d_cam @ R.T
        down = d_w[:, 2] < -0.05
        z_geo = C[2] / -d_w[down, 2]
        horiz = z_geo * np.linalg.norm(d_w[down, :2], axis=1)
        ratios.append(v["depth"][i][rr[down], cc[down]] / z_geo)
        dists.append(horiz)
    if not ratios:
        continue
    q, d = np.concatenate(ratios), np.concatenate(dists)
    h = np.median([v["T_wc"][i][2, 3] for i in range(len(v["depth"]))])
    row = []
    for a, b in pairwise(bins):
        m = (d >= a) & (d < b)
        row.append(f"{np.median(q[m]):.3f}({m.sum() // 1000}k)" if m.sum() > 500 else "  -  ")
    print(f"{r:9s} camera height {h:.2f} m  " + "  ".join(row))
