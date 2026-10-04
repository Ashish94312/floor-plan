"""Top-down overview of a whole capture in one frame: wall points and cameras coloured by room, room
outlines, openings. Used to explain why a stitched plan differs from the photo plan (E22l).

Usage: uv run python scripts/experiments/joint_overview.py <capture> <out.png> [section.key=value ...]
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
scratch = out_png.parent / f"_{out_png.stem}_run"
_cap, clouds, summary = run(cap_dir, cfg, out=scratch, log=lambda *a: None)
res = summary["result"]

colours = {"bedroom1": "tab:orange", "hall": "tab:blue", "kitchen": "tab:green"}
fig, ax = plt.subplots(figsize=(11, 9))
for r, c in clouds.items():
    col = colours.get(r, "tab:gray")
    P = c.points[:: max(1, len(c.points) // 200000)]
    N = normals(P, cfg["alignment"]["normal_radius_m"])
    top = c.planes.ceiling_z or np.percentile(P[:, 2], 98)
    wall = (np.abs(N[:, 2]) < 0.3) & (P[:, 2] > 0.5) & (P[:, 2] < top - 0.2)
    W = P[wall][:: max(1, wall.sum() // 30000)]
    ax.scatter(W[:, 0], W[:, 1], s=0.3, c=col, alpha=0.35)
    cams = c.views["T_wc"][:, :3, 3]
    ax.scatter(cams[:, 0], cams[:, 1], marker="^", s=60, c=col, edgecolors="k", label=f"{r} cameras ({len(cams)})")
    if c.layout is not None:
        poly = np.array(c.layout.polygon + c.layout.polygon[:1])
        ax.plot(poly[:, 0], poly[:, 1], "-", c=col, lw=2)
        for w in c.layout.walls:
            m = (np.array(w.start) + np.array(w.end)) / 2
            ax.text(m[0], m[1], f"{w.wall_id.split('-')[1]} {w.length_m:.2f}{' open' if w.kind == 'open' else ''}",
                    fontsize=7, color=col, ha="center")
for room in res.rooms:
    for o in room.openings:
        w = next(w for w in clouds[room.room_id].layout.walls if w.wall_id == o.wall_id)
        a, b = np.array(w.start), np.array(w.end)
        u = (b - a) / np.linalg.norm(b - a)
        off = o.offset_along_wall.value
        p0, p1 = a + u * off, a + u * (off + o.width.value)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "-", c="red" if o.type != "window" else "cyan", lw=5, alpha=0.8)
        ax.text(*(p0 + p1) / 2, f"{o.opening_id.split('-')[1]} {o.type} {o.width.value:.2f}", fontsize=7, color="red")
sp = res.stitched_plan
ax.set_title(f"{cap_dir.name}: {sp.placement_method}, adjacency "
             f"{[(a.room_a, a.room_b) for a in sp.adjacency]}, overlaps {len(sp.overlaps)}\n"
             "points = wall surfaces seen by that room's frames; lines = room outlines; red = doors/openings, cyan = windows")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.savefig(out_png, dpi=110, bbox_inches="tight")
print(out_png)
