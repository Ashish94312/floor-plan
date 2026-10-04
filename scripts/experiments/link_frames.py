"""E22l. Find the frames where two rooms' clips see the same thing (doorway / the room beyond it) and
show the best pairs side by side, so a person can check them.

Usage: uv run python scripts/experiments/link_frames.py <capture> [n_per_pair] [min_inliers]
Writes <capture>/links.jpg and prints the pairs (room A time <-> room B time, verified matches).
"""

import itertools
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from scan.io.video import VIDEO_EXT, decode, keyframes, link_features, link_pairs

cap_dir = Path(sys.argv[1])
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 3
min_inl = int(sys.argv[3]) if len(sys.argv) > 3 else 30
clips = {d.name: next(f for f in sorted(d.iterdir()) if f.suffix.lower() in VIDEO_EXT)
         for d in sorted((cap_dir / "video").iterdir()) if d.is_dir()}
data = {}
t0 = time.perf_counter()
for room, f in clips.items():
    fr, ts, _ = decode(f, 1.0, 1024)
    _, sharp = keyframes(fr, 1, 0)
    data[room] = (fr, ts, link_features(fr, sharp, range(len(fr)), 480, 30))
    print(f"{room}: {len(fr)} candidate frames, {len(data[room][2])} with features ({time.perf_counter() - t0:.1f} s)")

rows = []
for a, b in itertools.combinations(clips, 2):
    t1 = time.perf_counter()
    pairs = link_pairs(data[a][2], data[b][2], min_inliers=min_inl)
    print(f"{a} <-> {b}: {len(pairs)} verified pairs ({time.perf_counter() - t1:.1f} s)")
    shown = []
    for inl, fa, fb in pairs:
        if any(abs(fa - x) < 3 and abs(fb - y) < 3 for x, y in shown):
            continue  # neighbours of a pair already shown
        shown.append((fa, fb))
        ta, tb = data[a][1][fa], data[b][1][fb]
        print(f"   {a} {ta:6.1f} s  <->  {b} {tb:6.1f} s   {inl} inliers")
        ims = []
        for room, i, t in ((a, fa, ta), (b, fb, tb)):
            im = cv2.resize(data[room][0][i], (216, 384))
            cv2.putText(im, f"{room} {t:.0f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            ims.append(im)
        sep = np.full((384, 6, 3), 255, np.uint8)
        row = np.hstack([ims[0], sep, ims[1]])
        cv2.putText(row, f"{inl} matches", (150, 376), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows.append(row)
        if len(shown) == n_show:
            break
if rows:
    gap = np.zeros((384, 20, 3), np.uint8)
    per_line = 3
    while len(rows) % per_line:
        rows.append(np.zeros_like(rows[0]))
    lines = [np.hstack([x for r in rows[i:i + per_line] for x in (r, gap)][:-1]) for i in range(0, len(rows), per_line)]
    cv2.imwrite(str(cap_dir / "links.jpg"), cv2.cvtColor(np.vstack(lines), cv2.COLOR_RGB2BGR))
    print(cap_dir / "links.jpg")
