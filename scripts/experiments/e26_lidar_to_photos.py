"""E26: a LiDAR walk-through as a PHOTO capture, so the photo tier (MapAnything) can be measured against LiDAR truth.

Same frames, nothing from the LiDAR geometry goes in: per room, n frames evenly spread in time, turned upright, as
JPEGs with the ARKit focal as EXIF FocalLengthIn35mmFilm (an integer in EXIF: ~0.6% rounding, as on iPhone photos).
Rooms (photo folders) come from the LiDAR run's room split: the grouping a photo user gives with room folders.

  uv run python scripts/experiments/e26_lidar_to_photos.py <recording> <out capture dir> [n_per_room=8]
  uv run scan <out capture dir> --per-room
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import IFD

from scan.config import load_config
from scan.damage.detect import detect
from scan.device import select_device
from scan.io.ingest import ingest
from scan.io.lidar import UNSPLIT, read_odometry
from scan.layout.segment import split_recording
from scan.pipeline import align_rooms, geometry_lidar

rec, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
cfg = load_config()
cap = ingest(rec, "lidar", cfg)
clouds, _ = geometry_lidar(cap, cfg, {}, lambda *a: None)
align_rooms(clouds, cfg, lambda *a: None, check=False)
whole = clouds[UNSPLIT]
rooms = split_recording(whole, cfg, print, detect(whole.frames, cfg, select_device(None), True))  # cached

odo = read_odometry(rec / "odometry.csv")
W0, H0 = 1920, 1440  # Stray Scanner colour frame; ARKit intrinsics are given at this size
f35 = float(np.median(odo["f"][:, 0])) * cfg["ingest"]["intrinsics"]["film_diagonal_mm"] / math.hypot(W0, H0)
manifest = {"recording": rec.name, "f35_exact_mm": round(f35, 3), "f35_exif_mm": round(f35), "rooms": {}}
for rid, c in rooms.items():
    frames = sorted(c.frames, key=lambda f: f.timestamp)
    pick = [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int))]
    d = out / "photos" / rid
    d.mkdir(parents=True, exist_ok=True)
    for f in pick:
        img = Image.fromarray(np.ascontiguousarray(np.rot90(f.rgb, f.upright_turns)))
        ex = Image.Exif()
        ex[0x010F], ex[0x0110] = "Apple", "iPhone (Stray Scanner rgb.mp4)"
        sub = ex.get_ifd(IFD.Exif)
        sub[0xA405] = round(f35)  # FocalLengthIn35mmFilm
        img.save(d / f"{f.image_path.name.split('#')[1]}.jpg", quality=95, exif=ex)
    manifest["rooms"][rid] = [f.image_path.name for f in pick]
    print(f"{rid}: {len(pick)} of {len(frames)} frames -> {d}")
(out / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"f35 {f35:.2f} mm (EXIF {round(f35)})")
