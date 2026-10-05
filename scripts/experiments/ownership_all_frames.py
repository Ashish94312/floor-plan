"""E22x (option C). Who owns the passage when every tracked frame votes, not only the keyframes?

Uses the coverage audit's per-frame data (E22u: PnP camera position + the 10 cm plan cells of the surfaces each frame
sees in front). Each frame carves its lines of sight to those cells (layout.room.carve_free_space, as the layout does
for the model's views); a cell goes to the room with the most of its own frames seeing through it (ownership). Shown
for the keyframes alone and for all tracked frames, as the share of each box's seen cells owned by each room.

Usage: uv run python scripts/experiments/ownership_all_frames.py <coverage.npy>
"""

import sys

import numpy as np

from scan.config import load_config
from scan.layout.room import _grid, carve_free_space

data = np.load(sys.argv[1], allow_pickle=True).item()
cfg = load_config()
lc = cfg["layout"]
cell = lc["cell_m"]
boxes = {"passage": (-1.0, -0.1, 0.65, 1.80), "hall proper": (-3.3, -0.2, -1.6, 0.45), "kitchen proper": (-3.85, -1.1, 0.65, 1.80)}


def centres(ids):
    ids = np.asarray(ids, np.int64)
    ix = np.floor((ids + 50_000) / 100_000).astype(np.int64)
    iy = ids - ix * 100_000
    return np.c_[(ix + 0.5) * 0.1, (iy + 0.5) * 0.1, np.zeros(len(ids))]


frames = {r: [x for x in rows if x] for r, rows in data.items()}
allxy = np.vstack([np.vstack([centres(x["cells"])[:, :2] for x in fr] + [np.array([x["C"][:2] for x in fr])]) for fr in frames.values()])
lo, shape = _grid(allxy, cell, margin=0.5)
ii, jj = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), indexing="ij")
X, Y = lo[0] + (jj + 0.5) * cell, lo[1] + (ii + 0.5) * cell
rooms = list(frames)
for label, pick in (("keyframes only", lambda x: x["key"]), ("all tracked frames", lambda x: True)):
    free = {r: carve_free_space([(np.array(x["C"][:2]), centres(x["cells"])) for x in fr if pick(x)], lo, shape, cell,
                                lc["ray_stop_short_m"]) for r, fr in frames.items()}
    stack = np.stack([free[r] for r in rooms]).astype(float)
    owner = np.argmax(stack, 0)
    seen = stack.max(0) >= lc["min_views_free"]
    n = {r: sum(pick(x) for x in fr) for r, fr in frames.items()}
    print(f"\n{label} ({', '.join(f'{r} {n[r]}' for r in rooms)} frames)")
    for name, (x0, x1, y0, y1) in boxes.items():
        m = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & seen
        share = {r: (owner[m] == k).mean() for k, r in enumerate(rooms)} if m.any() else {}
        views = {r: stack[k][m].mean() for k, r in enumerate(rooms)} if m.any() else {}
        print(f"  {name:15s} {m.sum():5d} seen cells   owned: " + "  ".join(f"{r} {share[r]:.0%}" for r in rooms)
              + "   mean frames/cell: " + "  ".join(f"{r} {views[r]:.1f}" for r in rooms))
