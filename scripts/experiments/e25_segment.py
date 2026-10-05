"""E25: room segmentation of one LiDAR walk-through, without the detector (fast loop on scan/layout/segment.py).

ingest -> LiDAR geometry -> alignment -> free space, cores, watershed rooms (lintels + detector doorways, magenta);
plots free space, cores and rooms with the camera path coloured by its room.

  uv run python scripts/experiments/e25_segment.py <recording> <png> [--plan=<out>/result.json] [key=value overrides]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from scan.config import load_config
from scan.damage.detect import detect
from scan.device import select_device
from scan.geometry.align import normals
from scan.io.ingest import ingest
from scan.layout.room import view_rays, wall_points
from scan.layout.segment import door_segments, free_space_rooms
from scan.pipeline import align_rooms, geometry_lidar

rec, png = Path(sys.argv[1]), Path(sys.argv[2])
plan = next((Path(a.split("=", 1)[1]) for a in sys.argv[3:] if a.startswith("--plan=")), None)  # result.json to overlay
over = {}
for kv in [a for a in sys.argv[3:] if not a.startswith("--plan=")]:
    k, v = kv.split("=")
    d = over
    *path, last = k.split(".")
    for q in path:
        d = d.setdefault(q, {})
    d[last] = float(v)
cfg = load_config(overrides=over)
cap = ingest(rec, "lidar", cfg)
clouds, _ = geometry_lidar(cap, cfg, {}, print)
align_rooms(clouds, cfg, print, check=False)
c = next(iter(clouds.values()))
rays = view_rays(c, cfg)
cams = np.array([x for x, _ in rays])
walls = wall_points(c.points, normals(c.points, cfg["alignment"]["normal_radius_m"]), c.planes.ceiling_z, cfg)
doors = door_segments(c.views, c.frames, detect(c.frames, cfg, select_device(None), True), cfg)  # cached by the pipeline
labels, lo, cell, cam_lab, passages = free_space_rooms(rays, cams, cfg, walls, doors)
free = labels > 0
clear = ndimage.distance_transform_edt(free) * cell
cores, n = ndimage.label(clear > cfg["segment"]["door_max_m"] / 2)
ext = [lo[0], lo[0] + labels.shape[1] * cell, lo[1], lo[1] + labels.shape[0] * cell]
fig, ax = plt.subplots(1, 3, figsize=(21, 8))
for a, img, title in zip(ax, [clear, cores, labels], ["clearance to nearest obstacle (m)", f"cores ({n})", f"rooms ({labels.max()})"]):
    a.imshow(np.ma.masked_equal(img, 0), origin="lower", extent=ext, cmap="viridis" if a is ax[0] else "tab10")
    a.scatter(cams[:, 0], cams[:, 1], c=cam_lab, cmap="tab10", s=12, edgecolors="k", linewidths=0.3)
    a.plot(cams[:, 0], cams[:, 1], "r-", lw=0.5)
    for d0, d1, _ in doors:  # detector doorways
        a.plot([d0[0], d1[0]], [d0[1], d1[1]], "m-", lw=3)
    if plan is not None:  # final room outlines (same aligned frame)
        import json

        for rm in json.loads(plan.read_text())["rooms"]:
            P = np.array(rm["polygon"] + rm["polygon"][:1])
            a.plot(P[:, 0], P[:, 1], "k-", lw=1.5)
            a.text(*np.mean(P[:-1], 0), rm["room_id"], fontsize=8, ha="center")
    a.set_title(f"{rec.name}: {title}")
    a.set_aspect("equal")
png.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(png, dpi=80)
print({int(k): (int((cam_lab == k).sum()), round(float((labels == k).sum() * cell**2), 2)) for k in np.unique(labels) if k},
      "-> views, m2 per room", png)

# contact sheet: 4 frames per room, turned upright by gravity (image axis closest to world up goes to the top)
up = np.array([0.0, 0, 1])  # aligned frame
rows = []
for k in [int(x) for x in dict.fromkeys(cam_lab) if x]:
    idx = np.flatnonzero(cam_lab == k)
    tiles = []
    for i in idx[np.linspace(0, len(idx) - 1, 4).astype(int)]:
        R = c.views["T_wc"][i][:3, :3]
        img = c.frames[i].rgb
        kx, ky = R[:, 0] @ up, R[:, 1] @ up  # image right / image down, in world up
        turns = 1 if kx > abs(ky) else 3 if -kx > abs(ky) else 2 if ky > 0 else 0
        img = np.ascontiguousarray(np.rot90(img, turns))
        h = 240
        tiles.append(np.asarray(matplotlib.image.pil_to_array(
            __import__("PIL.Image").Image.fromarray(img).resize((round(img.shape[1] * h / img.shape[0]), h)))))
    w = max(t.shape[1] for t in tiles)
    rows.append(np.hstack([np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]))
w = max(r.shape[1] for r in rows)
sheet = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows])
plt.imsave(png.with_name(png.stem + "_frames.jpg"), sheet)
print("rooms top to bottom:", [int(x) for x in dict.fromkeys(cam_lab) if x])
