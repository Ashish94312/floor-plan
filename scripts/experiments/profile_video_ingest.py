"""Where does video ingest spend its time? cProfile of scan.io.ingest on one video capture, grouped by stage.

    uv run python scripts/experiments/profile_video_ingest.py captures/home01_video_e
"""

import cProfile
import pstats
import sys
import time
from pathlib import Path

from scan.config import load_config
from scan.io.ingest import ingest

cap_dir = Path(sys.argv[1])
cfg = load_config()
prof = cProfile.Profile()
t = time.perf_counter()
prof.enable()
cap = ingest(cap_dir, None, cfg)
prof.disable()
print(f"ingest total {time.perf_counter() - t:.1f} s, {sum(len(v) for v in cap.rooms.values())} frames")
st = pstats.Stats(prof)
for fn in ("decode", "keyframes", "focal_px", "link_features", "link_pairs", "_add_links", "run", "Laplacian",
           "cvtColor", "detectAndCompute", "knnMatch", "findFundamentalMat", "_ransac", "_segments"):
    rows = [(k, v) for k, v in st.stats.items() if k[2] == fn]
    if rows:
        cum = max(v[3] for _, v in rows)
        calls = sum(v[1] for _, v in rows)
        print(f"  {fn:20s} {cum:7.1f} s  ({calls} calls)")
