"""E22c. How many sharp keyframes can each clip supply? Decodes every clip once and counts the
keyframes `keyframes()` would pick at several budgets (CPU only, no model).

Usage: uv run python scripts/experiments/video_sharpness.py captures/home01_video_a captures/home01_video_b
"""

import sys
from pathlib import Path

import numpy as np

from scan.config import load_config
from scan.io.video import VIDEO_EXT, decode, keyframes

cfg = load_config()
vc = cfg["video"]
budgets = [8, 12, 16, 20, 24, 28, 32, 40]
print(f"| capture | room | s | frames | sharp >= {vc['min_sharpness']} | median sharp | " + " | ".join(f"k{k}" for k in budgets) + " |")
print("|---|---|---|---|---|---|" + "---|" * len(budgets))
for cap in map(Path, sys.argv[1:]):
    for clip in sorted(p for p in (cap / "video").rglob("*") if p.suffix.lower() in VIDEO_EXT):
        frames, ts, info = decode(clip, vc["decode_fps"], cfg["ingest"]["working_max_side"])
        _, sharp = keyframes(frames, 1, vc["min_sharpness"])
        counts = [len(keyframes(frames, k, vc["min_sharpness"])[0]) for k in budgets]
        print(f"| {cap.name} | {clip.parent.name} | {info['duration']:.1f} | {len(frames)} | {(sharp >= vc['min_sharpness']).sum()} | "
              f"{np.median(sharp):.0f} | " + " | ".join(map(str, counts)) + " |")
