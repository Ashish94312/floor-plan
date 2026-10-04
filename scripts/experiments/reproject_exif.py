"""E10. Re-project MapAnything depth with the EXIF focal instead of the model's predicted rays.

MapAnything's output focal is ~13% longer than the EXIF focal we feed it (E6-E9). A too-long focal
shrinks lateral extents in each camera, which hits ceiling height (vertical offsets) harder than
wall lengths (mostly along depth). This keeps predicted depth + poses, swaps only the rays.

Usage: uv run python scripts/experiments/reproject_exif.py <ma_raw.npz> <out_ma_raw.npz>
"""

import sys
from pathlib import Path

import numpy as np

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
r = dict(np.load(src))
pts, E = r["pts"], r["extrinsic"]  # (S,H,W,3) world; (S,3,4) cam-from-world
S, H, W, _ = pts.shape
f = 26 / 43.27 * np.hypot(W, H)  # EXIF 26 mm-equivalent at the working resolution
u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
out = np.empty_like(pts)
for i in range(S):
    R, t = E[i, :, :3], E[i, :, 3]
    z = (pts[i] @ R.T + t)[..., 2]  # predicted depth_z
    cam = np.stack([(u - W / 2) / f * z, (v - H / 2) / f * z, z], -1)
    out[i] = (cam - t) @ R  # back to world with the predicted pose
print(f"EXIF f = {f:.1f}px; model f = {r['intrinsic'][:, 0, 0].mean():.1f}px (ratio {r['intrinsic'][:, 0, 0].mean() / f:.3f})")
r["pts"] = out
dst.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(dst, **r)
