"""Contact sheet of the keyframes the pipeline picks per room (debug for video captures).

Usage: uv run python scripts/experiments/keyframe_sheet.py <capture> [section.key=value ...]
Writes <capture>/keyframes_<room>.jpg (timestamp on each frame).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from scan.config import load_config
from scan.io.ingest import ingest

cap_dir = Path(sys.argv[1])
extra: dict = {}
for a in sys.argv[2:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cap = ingest(cap_dir, None, load_config(overrides=extra))
for room, frames in cap.rooms.items():
    ims = []
    for f in frames:
        im = cv2.resize(f.rgb, (180, 320) if f.rgb.shape[0] > f.rgb.shape[1] else (320, 180))
        cv2.putText(im, f"{f.timestamp:.1f}s", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        ims.append(im)
    while len(ims) % 6:
        ims.append(np.zeros_like(ims[0]))
    sheet = np.vstack([np.hstack(ims[i:i + 6]) for i in range(0, len(ims), 6)])
    out = cap_dir / f"keyframes_{room}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(out, len(frames), [round(f.timestamp, 1) for f in frames])
