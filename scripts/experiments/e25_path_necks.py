"""E25: how wide are the necks the walk passes through? (why segmentation did or did not split a recording)

Clearance (distance to the nearest cell no line of sight crossed) sampled along the camera path every 5 cm; stretches
under 0.6 m are the necks walked through. A doorway shows as clearance <= door_max_m / 2 (neck <= 1.0 m wide).

  uv run python scripts/experiments/e25_path_necks.py <recording>
"""

import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import ndimage

from scan.config import load_config
from scan.io.ingest import ingest
from scan.layout.room import view_rays
from scan.layout.segment import free_space_rooms
from scan.pipeline import align_rooms, geometry_lidar

rec = Path(sys.argv[1])
cfg = load_config()
cap = ingest(rec, "lidar", cfg)
clouds, _ = geometry_lidar(cap, cfg, {}, lambda *a: None)
align_rooms(clouds, cfg, lambda *a: None, check=False)
c = next(iter(clouds.values()))
rays = view_rays(c, cfg)
cams = np.array([x for x, _ in rays])
labels, lo, cell, _, _ = free_space_rooms(rays, cams, cfg)
clear = ndimage.distance_transform_edt(labels > 0) * cell
pts = np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.05))) for a, b in pairwise(cams)])
ij = np.floor((pts - lo) / cell).astype(int)
prof = clear[ij[:, 1], ij[:, 0]]
# necks: contiguous stretches of the walk with clearance < 0.6 m; width = 2 x the stretch's minimum
lab, n = ndimage.label(prof < 0.6)
w = np.array([2 * prof[lab == k].min() for k in range(1, n + 1) if (lab == k).sum() >= 3])
print(f"{rec.name}: path clearance p10/p50 {np.percentile(prof, 10):.2f}/{np.percentile(prof, 50):.2f} m; "
      f"{len(w)} necks walked through (stretches < 0.6 m clearance)")
print("  neck widths m:", " ".join(f"{x:.2f}" for x in sorted(w)))
print(f"  <= 1.0 m (doorway by segment.door_max_m): {(w <= 1.0).sum()}, > 1.0 m: {(w > 1.0).sum()}")
