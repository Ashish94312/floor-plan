"""E23a. Which photos and detector boxes made a detector opening (e.g. a window on a wall with no window)?

Runs the pipeline (caches), lifts the OWLv2 boxes onto the surfaces (damage.detect.surface_boxes) and, for the named
surface and class, draws in each contributing photo every box of that class: yellow = lifted onto the named surface,
grey = elsewhere. Prints each box's score and where on the surface it landed.

Usage: uv run python scripts/experiments/detector_boxes.py <capture> <out.jpg> <surface_id> <class> [section.key=value ...]
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scan.config import load_config
from scan.damage.detect import detect, surface_boxes
from scan.device import select_device
from scan.pipeline import run

cap_dir, out, sid, cls = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
extra: dict = {}
for a in sys.argv[5:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
cap, clouds, _s = run(cap_dir, cfg, out=out.parent / f"_{out.stem}_run", log=lambda *a: None)
dev = select_device(None)
frames = {f.image_path.name: f for fs in cap.rooms.values() for f in fs}
dets = detect(list(frames.values()), cfg, dev, True)
boxes = surface_boxes(cap, clouds, cfg, dev, True)
hits = [b for bs in boxes.values() for b in bs if b.surface_id == sid and b.cls == cls]
print(f"{cls} boxes lifted onto {sid}: {len(hits)} from {len({b.photo for b in hits})} photos")
for b in hits:
    print(f"   {b.photo}: score {b.score:.2f}, along {b.u0:.2f}..{b.u1:.2f} m, height {b.v0:.2f}..{b.v1:.2f} m")
tiles = []
for photo in sorted({b.photo for b in hits}):
    img = np.ascontiguousarray(frames[photo].rgb[..., ::-1].copy())
    for d in dets.get(photo, []):
        if d["cls"] != cls:
            continue
        x0, y0, x1, y1 = map(int, d["box"])
        on = any(abs(b.score - d["score"]) < 1e-6 for b in hits if b.photo == photo)
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 230, 255) if on else (160, 160, 160), 3)
        cv2.putText(img, f"{d['score']:.2f}", (x0 + 4, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
    cv2.putText(img, photo, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    tiles.append(cv2.resize(img, (640, round(640 * img.shape[0] / img.shape[1]))))
if tiles:
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]))
    print(out)
