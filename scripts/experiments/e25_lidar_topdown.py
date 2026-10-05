"""E25: top-down view of a LiDAR run's aligned cloud with the camera path (diagnostic, no model).

Reads <out>/debug/room1/cloud.ply (aligned frame: floor z = 0) and the alignment T from <out>/debug/geometry.json,
puts the recording's camera path (odometry.csv) into the same frame. Panels: wall points by height band
(0.3-1.5 m, 1.5-2.0, > 2.0 m: lintels over doors), floor points, layout polygon.

  uv run python scripts/experiments/e25_lidar_topdown.py <recording> <out> <png>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

rec, out, png = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
g = json.loads((out / "debug" / "geometry.json").read_text())
room = next(iter(g["rooms"]))
T = np.array(next(iter(g["alignment"].values()))["T"])
P = np.asarray(o3d.io.read_point_cloud(str(out / "debug" / room / "cloud.ply")).points)
odo = np.genfromtxt(rec / "odometry.csv", delimiter=",", skip_header=1)
cams = odo[:, 2:5] @ T[:3, :3].T + T[:3, 3]
poly = np.array(g["rooms"][room]["layout"]["polygon"])

fig, ax = plt.subplots(1, 3, figsize=(21, 8))
z = P[:, 2]
for a, (lo, hi, title) in zip(ax, [(0.3, 1.5, "0.3-1.5 m"), (1.5, 2.0, "1.5-2.0 m"), (2.0, 9, "> 2.0 m (lintels, ceiling)")]):
    s = P[(z >= lo) & (z < hi)]
    a.scatter(P[z < 0.1, 0], P[z < 0.1, 1], s=0.2, c="0.85")
    a.scatter(s[:, 0], s[:, 1], s=0.3, c=s[:, 2], cmap="viridis")
    a.plot(cams[:, 0], cams[:, 1], "r-", lw=0.8)
    a.plot(cams[0, 0], cams[0, 1], "go")
    a.plot(*np.vstack([poly, poly[:1]]).T, "k--", lw=1)
    a.set_title(f"{rec.name}: walls {title}, floor grey, path red (start green)")
    a.set_aspect("equal")
    a.grid(alpha=0.3)
png.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(png, dpi=80)
print(f"z range {z.min():.2f}..{z.max():.2f}, pts {len(P)}, path {np.linalg.norm(np.diff(cams[:, :2], axis=0), axis=1).sum():.1f} m -> {png}")
