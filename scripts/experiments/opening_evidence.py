"""E23a. What made an opening? The frames whose pixels voted 'see-through' inside it.

For each named opening (result.json): every view's pixels whose ray meets the wall plane inside the opening's
rectangle; 'through' when the model's depth is beyond the wall by more than openings.depth_margin_m (blue), 'on' when
within it (red). The 3 views with the most through pixels are drawn with the rectangle outlined, and how far behind
the wall the through pixels were put (median, m) is printed: a real opening shows the room or outdoors beyond it, a
reflection (glossy tiles, glass) shows depth about one camera-to-wall distance behind the wall.

Usage: uv run python scripts/experiments/opening_evidence.py <capture> <out.jpg> <opening_id> [...] [section.key=value ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scan.config import load_config
from scan.layout.openings import ray_K
from scan.pipeline import run

cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
ids = [a for a in sys.argv[3:] if "=" not in a]
extra: dict = {}
for a in (a for a in sys.argv[3:] if "=" in a):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, summary = run(cap_dir, cfg, out=out.parent / f"_{out.stem}_run", log=lambda *a: None)
res = summary["result"].model_dump()
margin = cfg["openings"]["depth_margin_m"]
rows = []
for oid in ids:
    room = next(r for r in res["rooms"] if any(o["opening_id"] == oid for o in r.get("openings", [])))
    o = next(o for o in room["openings"] if o["opening_id"] == oid)
    w = next(w for w in room["walls"] if w["wall_id"] == o["wall_id"])
    a, b = np.array(w["start"], float), np.array(w["end"], float)
    L = np.linalg.norm(b - a)
    ud = (b - a) / L
    n = np.array([-ud[1], ud[0], 0.0])
    u0, u1 = o["offset_along_wall"]["value"], o["offset_along_wall"]["value"] + o["width"]["value"]
    z0 = o["sill_height"]["value"] if o.get("sill_height") else 0.0
    z1 = z0 + o["height"]["value"]
    v = clouds[room["room_id"]].views
    scored = []
    for i in range(len(v["depth"])):
        H, W = v["depth"][i].shape
        K = ray_K(v, i, cfg["geometry"]["rays"])
        R, C = v["T_wc"][i][:3, :3], v["T_wc"][i][:3, 3]
        yy, xx = np.mgrid[0:H, 0:W]
        d = (np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones_like(xx, float)], -1)) @ R.T
        denom = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((np.r_[a, 0.0] - C) @ n) / denom  # z-depth where the pixel's ray meets the wall plane
        P = C + t[..., None] * d
        uu, zz = (P[..., :2] - a) @ ud, P[..., 2]
        inside = (t > 0.3) & (uu >= u0) & (uu <= u1) & (zz >= z0) & (zz <= z1) & v["mask"][i]
        through = inside & (v["depth"][i] > t + margin)
        on = inside & (np.abs(v["depth"][i] - t) <= margin)
        if through.sum():
            scored.append((int(through.sum()), i, through, on, float(np.median((v["depth"][i] - t)[through])),
                           float(np.median(t[inside]))))
    scored.sort(key=lambda x: -x[0])
    print(f"{oid} ({o['type']} on {o['wall_id']}, {o['width']['value']:.2f} x {o['height']['value']:.2f} m, sill {z0:.2f}): "
          f"{len(scored)} views see through")
    for n_px, i, through, on, behind, dist in scored[:5]:
        print(f"   {v['names'][i]}: {n_px} px through, {int(on.sum())} on the wall; wall {dist:.2f} m away, "
              f"through pixels put {behind:.2f} m behind it")
    for n_px, i, through, on, behind, _ in scored[:3]:
        img = v["rgb"][i].astype(float)
        img[through] = 0.45 * img[through] + 0.55 * np.array([40, 90, 255])
        img[on] = 0.45 * img[on] + 0.55 * np.array([255, 40, 40])
        img = np.ascontiguousarray(img.astype(np.uint8)[..., ::-1])
        cv2.putText(img, f"{oid} {v['names'][i]} behind {behind:.2f} m", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        rows.append(img)
if rows:
    h = min(r.shape[0] for r in rows)
    rows = [cv2.resize(r, (round(r.shape[1] * h / r.shape[0]), h)) for r in rows]
    while len(rows) % 3:
        rows.append(np.zeros_like(rows[0]))
    grid = np.vstack([np.hstack(rows[k:k + 3]) for k in range(0, len(rows), 3)])
    cv2.imwrite(str(out), grid)
    print(out)
