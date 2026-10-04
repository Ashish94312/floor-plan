"""Draw a detected opening's outline onto the frames that see it (what did the pipeline call a door?).

Usage: uv run python scripts/experiments/show_opening.py <capture> <room> <opening_id> <out.jpg> [section.key=value ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scan.config import load_config
from scan.geometry.cloud import ray_K
from scan.pipeline import run

cap_dir, room, oid, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
extra: dict = {}
for a in sys.argv[5:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, summary = run(cap_dir, cfg, out=out.parent / f"_{out.stem}_run", log=lambda *a: None)
r = next(x for x in summary["result"].rooms if x.room_id == room)
o = next(x for x in r.openings if x.opening_id == oid)
c = clouds[room]
w = next(x for x in c.layout.walls if x.wall_id == o.wall_id)
a, b = np.array(w.start), np.array(w.end)
u = (b - a) / np.linalg.norm(b - a)
off, wd, ht = o.offset_along_wall.value, o.width.value, o.height.value
z0 = o.sill_height.value if o.sill_height else 0.0
corners = np.array([[*(a + u * off), z0], [*(a + u * (off + wd)), z0], [*(a + u * (off + wd)), z0 + ht], [*(a + u * off), z0 + ht]])
v = c.views
tiles = []
for i in range(len(v["depth"])):
    T, K = v["T_wc"][i], ray_K(v, i, cfg["geometry"]["rays"])
    cam = (corners - T[:3, 3]) @ T[:3, :3]
    if (cam[:, 2] < 0.2).any():
        continue
    px = np.c_[K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2], K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]]
    H, W = v["depth"][i].shape
    if not ((px[:, 0].mean() > 0) & (px[:, 0].mean() < W) & (px[:, 1].mean() > 0) & (px[:, 1].mean() < H)):
        continue
    im = np.ascontiguousarray(v["rgb"][i][..., ::-1])
    cv2.polylines(im, [px.round().astype(np.int32)], True, (0, 0, 255), 2)
    cv2.putText(im, str(v["names"][i]), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    tiles.append(cv2.resize(im, (W, H)))
print(f"{oid}: {o.type} on {o.wall_id}, offset {off:.2f}, width {wd:.2f}, height {ht:.2f}, seen in {len(tiles)} views")
if tiles:
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(out)
