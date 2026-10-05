"""E22t. Who owns the passage? (video_e: the kitchen's outline took the hall's passage)

Ownership after placement (layout.room.layout_rooms): a free-space cell goes to the room with the most of its OWN
views seeing through it. Counts, over a box of cells (the passage), how many views of each room see through them, and
splits each room's views by where its camera stands: inside a box (e.g. south of the hall/kitchen partition = in
the hall) or not. If the kitchen's claim on the passage comes from kitchen frames filmed standing in the hall, the
claim is footage of the hall, not of the kitchen.

Usage: uv run python scripts/experiments/passage_ownership.py <capture> x0 x1 y0 y1 [section.key=value ...]
  (box in the stitched frame, metres; camera 'outside' = y below y0 of the box minus 0.1 m)
"""

import sys
from pathlib import Path

import numpy as np
import yaml

from scan.config import load_config
from scan.layout.room import _grid, carve_free_space, view_rays
from scan.pipeline import run

cap_dir = Path(sys.argv[1])
x0, x1, y0, y1 = map(float, sys.argv[2:6])
extra: dict = {}
for a in sys.argv[6:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
out = cap_dir / "experiments" / "E22t_hall" / "_ownership_run"
_cap, clouds, _s = run(cap_dir, cfg, out=out, log=lambda *a: None)
lc = cfg["layout"]
cell = lc["cell_m"]
rays = {r: view_rays(c, cfg) for r, c in clouds.items() if c.frame_id == "stitched"}
allxy = np.vstack([np.array([cx for cx, _ in v]) for v in rays.values()] + [c.points[:, :2] for c in clouds.values()])
lo, shape = _grid(allxy, cell, margin=0.5)
ii, jj = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing="ij")  # free-space images: rows = y, cols = x
X, Y = lo[0] + (jj + 0.5) * cell, lo[1] + (ii + 0.5) * cell
box = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
print(f"passage box x {x0}..{x1}, y {y0}..{y1}: {box.sum()} cells")
for r, v in rays.items():
    names = clouds[r].views["names"]
    per = []
    for k, (cxy, P) in enumerate(v):
        f = carve_free_space([(cxy, P)], lo, shape, cell, lc["ray_stop_short_m"])
        per.append((str(names[k]) if k < len(names) else str(k), cxy, float((f[box] > 0).mean())))
    tot = carve_free_space(v, lo, shape, cell, lc["ray_stop_short_m"])
    south = [p for p in per if p[1][1] < y0 - 0.1]
    print(f"{r:9s} views {len(v):2d}  mean views/cell in box {tot[box].mean():.2f}  "
          f"views seeing >=20% of box: {sum(p[2] >= 0.2 for p in per)} "
          f"(of which camera south of the box: {sum(p[2] >= 0.2 for p in south)})")
    for name, cxy, frac in per:
        if frac >= 0.05:
            print(f"    {name:22s} camera ({cxy[0]:+.2f}, {cxy[1]:+.2f})  sees {frac:.0%} of the box")
