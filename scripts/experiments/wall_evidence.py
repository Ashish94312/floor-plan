"""E22n. What is a plan wall in the frames? For each named wall, the views with the most pixels on it,
those pixels tinted red (vertical surface within 6 cm of the wall line, inside its span).

Usage: uv run python scripts/experiments/wall_evidence.py <capture> <out.jpg> <wall_id> [...] [section.key=value ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scan.config import load_config
from scan.geometry.cloud import view_points
from scan.pipeline import run

cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
walls = [a for a in sys.argv[3:] if "=" not in a]
extra: dict = {}
for a in (a for a in sys.argv[3:] if "=" in a):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, _s = run(cap_dir, cfg, out=out.parent / f"_{out.stem}_run", log=lambda *a: None)
rays = cfg["geometry"]["rays"]

rows = []
for wid in walls:
    room = wid.rsplit("-W", 1)[0]
    c = clouds[room]
    w = next(w for w in c.layout.walls if w.wall_id == wid)
    a, b = np.array(w.start), np.array(w.end)
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    n = np.array([-u[1], u[0]])
    hits = []
    for i in range(len(c.views["depth"])):
        P = view_points(c.views, i, rays)
        s, d = (P[..., :2] - a) @ u, (P[..., :2] - a) @ n
        m = c.views["mask"][i] & (np.abs(d) < 0.06) & (s > 0) & (s < L) & (P[..., 2] > 0.3)
        hits.append((int(m.sum()), i, m, P[..., 2]))
    tiles = []
    for cnt, i, m, z in sorted(hits, key=lambda h: -h[0])[:3]:
        img = c.views["rgb"][i].copy()
        img[m] = (0.45 * img[m] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        zt = np.nanpercentile(z[m], 95) if cnt else float("nan")
        cv2.putText(img, f"{wid} {w.length_m:.2f}m", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(img, f"{c.views['names'][i][-10:]} {cnt}px top {zt:.2f}m", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)
        tiles.append(img)
    rows.append(np.concatenate(tiles, 1))
W = max(r.shape[1] for r in rows)
sheet = np.concatenate([np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0))) for r in rows], 0)
cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out)
