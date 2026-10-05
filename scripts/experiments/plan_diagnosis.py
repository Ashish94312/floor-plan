"""E22n. Why does a stitched video plan have walls where there are none, and outer walls off one line?

Re-runs the pipeline (model cache hits) and draws, per room, in the stitched frame:
  left:  vertical-surface points, each 4 cm cell coloured by the HIGHEST point in it. Real walls reach
         the ceiling; furniture (fridge, wardrobe, sofa, door leaves) stops lower. Room outline + cameras.
  right: floor-level free space each room's cameras saw (lines of sight carved, as the layout does),
         with door/opening segments: shows where an outline leaks through a doorway.
Also prints each room's outer wall lines (north/south/east/west) so collinear walls can be compared.

Usage: uv run python scripts/experiments/plan_diagnosis.py <capture> <out.png> [section.key=value ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scan.config import load_config
from scan.geometry.align import normals
from scan.pipeline import run

cap_dir, out_png = Path(sys.argv[1]), Path(sys.argv[2])
extra: dict = {}
for a in sys.argv[3:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, summary = run(cap_dir, cfg, out=out_png.parent / f"_{out_png.stem}_run", log=print)
res = summary["result"]

colours = {"bedroom1": "tab:orange", "hall": "tab:blue", "kitchen": "tab:green"}
fig, axes = plt.subplots(1, len(clouds), figsize=(7 * len(clouds), 7.5))
cell = 0.04
for ax, (r, c) in zip(np.atleast_1d(axes), clouds.items()):
    P = c.points
    N = normals(P, cfg["alignment"]["normal_radius_m"])
    top = c.planes.ceiling_z or np.percentile(P[:, 2], 98)
    vert = (np.abs(N[:, 2]) < 0.3) & (P[:, 2] > 0.3) & (P[:, 2] < top - 0.05)
    V = P[vert]
    ij = np.floor(V[:, :2] / cell).astype(int)
    key = ij[:, 0] * 100000 + ij[:, 1]
    order = np.lexsort((V[:, 2], key))
    last = np.r_[key[order][1:] != key[order][:-1], True]  # highest point per cell
    top_pts = V[order][last]
    sc = ax.scatter(top_pts[:, 0], top_pts[:, 1], s=2, c=top_pts[:, 2], cmap="viridis", vmin=0.3, vmax=top)
    for r2, c2 in clouds.items():  # every room's outline for context, this room's bold
        if c2.layout is None or c2.frame_id != c.frame_id:
            continue
        poly = np.array(c2.layout.polygon + c2.layout.polygon[:1])
        ax.plot(poly[:, 0], poly[:, 1], "-", c=colours.get(r2, "gray"), lw=2.5 if r2 == r else 1, alpha=1 if r2 == r else 0.5)
        if r2 == r:
            for w in c2.layout.walls:
                m = (np.array(w.start) + np.array(w.end)) / 2
                ax.text(m[0], m[1], f"{w.wall_id.split('-')[1]} {w.length_m:.2f} ({w.support})", fontsize=7, color="k",
                        ha="center", bbox={"fc": "white", "alpha": 0.6, "lw": 0})
    cams = c.views["T_wc"][:, :3, 3]
    look = c.views["T_wc"][:, :3, 2]
    ax.quiver(cams[:, 0], cams[:, 1], look[:, 0], look[:, 1], color="red", scale=12, width=0.004)
    ax.scatter(cams[:, 0], cams[:, 1], c="red", s=25, zorder=5)
    if c.run_views is not None:
        lk = [i for i, n in enumerate(c.run_views["names"]) if n not in set(c.views["names"])]
        lc = c.run_views["T_wc"][lk, :3, 3]
        ax.scatter(lc[:, 0], lc[:, 1], c="magenta", marker="s", s=40, zorder=5, label="link frames")
    for o in c.openings:
        w = next(w for w in c.layout.walls if w.wall_id == o.wall_id)
        a, b = np.array(w.start), np.array(w.end)
        u = (b - a) / np.linalg.norm(b - a)
        p0, p1 = a + u * o.offset_m, a + u * (o.offset_m + o.width_m)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", c="red", lw=5, alpha=0.6)
        ax.text(*(p0 + p1) / 2, f"{o.type} {o.width_m:.2f}", fontsize=7, color="red")
    plt.colorbar(sc, ax=ax, fraction=0.04, label="highest wall point in the cell [m]")
    ax.set_title(f"{r}: ceiling {top:.2f} m, walls {len(c.layout.walls)}, placed by {c.placed_by}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
fig.suptitle(f"{cap_dir.name}: {res.stitched_plan.placement_method}. Colour = highest vertical-surface point per 4 cm "
             "cell (walls reach the ceiling, furniture does not). Red = cameras + view direction, magenta = link frames")
fig.savefig(out_png, dpi=100, bbox_inches="tight")

print("\nouter wall lines (stitched frame): axis, coordinate, span, support")
for r, c in clouds.items():
    for w in c.layout.walls:
        (x0, y0), (x1, y1) = w.start, w.end
        if abs(y1 - y0) < abs(x1 - x0):
            print(f"  {w.wall_id:12s} y = {(y0 + y1) / 2:+.3f}  x {min(x0, x1):+.2f}..{max(x0, x1):+.2f}  support {w.support}")
        else:
            print(f"  {w.wall_id:12s} x = {(x0 + x1) / 2:+.3f}  y {min(y0, y1):+.2f}..{max(y0, y1):+.2f}  support {w.support}")
print(out_png)
