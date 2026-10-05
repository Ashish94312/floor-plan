"""E25g: does the detector need upright frames? LiDAR frames come in sensor orientation (sideways when the phone is
held portrait). Same frames, OWLv2 as configured, as-is vs turned upright by gravity (ARKit world +y); no cache.

  uv run python scripts/experiments/e25_detector_upright.py <recording> [n_frames=40]
"""

import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from scan.config import load_config
from scan.damage.detect import detect
from scan.device import select_device
from scan.io.ingest import ingest

rec = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
cfg = load_config()
cap = ingest(rec, "lidar", cfg)
frames = [f for fs in cap.rooms.values() for f in fs]
frames = [frames[i] for i in np.linspace(0, len(frames) - 1, n).astype(int)]
up = np.array([0.0, 1.0, 0.0])
upright = []
for f in frames:
    R = f.T_wc[:3, :3]
    kx, ky = R[:, 0] @ up, R[:, 1] @ up  # image right / image down, in world up
    turns = 1 if kx > abs(ky) else 3 if -kx > abs(ky) else 2 if ky > 0 else 0
    g = copy.copy(f)
    g.rgb = np.ascontiguousarray(np.rot90(f.rgb, turns))
    upright.append(g)
dev = select_device(None)
for name, fs in (("as-is (sensor)", frames), ("upright", upright)):
    d = detect(fs, cfg, dev, use_cache=False)
    c = Counter(x["cls"] for v in d.values() for x in v)
    print(f"{rec.name} {name:15s} {n} frames: {dict(sorted(c.items()))}, frames with a door {sum(any(x['cls'] == 'door' for x in v) for v in d.values())}")
