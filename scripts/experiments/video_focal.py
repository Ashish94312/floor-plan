"""E22b. Is the vanishing-point video focal right? Re-run one room at fixed focals and compare
scale-free shape ratios (ceiling / longest wall) with the tape.

Usage: uv run python scripts/experiments/video_focal.py captures/home01_video_b hall 26 28.3 32.9
Builds <capture>_<room>/ (symlinks to the room's clips), writes out_f<F>/ per focal.
A wrong focal stretches heights and lateral extents (∝ 1/f) but not depth, so the ratio moves with f;
a pure scale error leaves it unchanged.
"""

import sys
from pathlib import Path

from scan.config import load_config
from scan.eval.gt import load_gt
from scan.pipeline import run

src, room = Path(sys.argv[1]), sys.argv[2]
focals = [float(f) for f in sys.argv[3:]]
sub = src.parent / f"{src.name}_{room}"
(sub / "video").mkdir(parents=True, exist_ok=True)
link = sub / "video" / room
if not link.exists():
    link.symlink_to((src / "video" / room).resolve())

gt = load_gt(Path("data/ground_truth") / f"{src.name.split('_')[0]}.yaml")[room]
tape = [w.length_m for w in gt.walls if w.length_m]
tape_ratio = gt.ceiling_m / max(tape)
print(f"tape: ceiling {gt.ceiling_m:.3f}, longest wall {max(tape):.3f}, ratio {tape_ratio:.3f}")

for f in focals:
    cfg = load_config(overrides={"video": {"force_f35_mm": f}})
    _cap, _clouds, summary = run(sub, cfg, out=sub / f"out_f{f:g}", log=lambda *a: None)
    r = summary["result"].rooms[0]
    walls = sorted((w.length.value for w in r.walls), reverse=True)
    ceil = r.ceiling_height.value if r.ceiling_height else float("nan")
    print(f"f35 {f:5.1f} mm: ceiling {ceil:.3f} ({100 * (ceil / gt.ceiling_m - 1):+.0f}%), "
          f"walls {[round(w, 2) for w in walls[:3]]}, ratio {ceil / walls[0]:.3f} "
          f"(tape {tape_ratio:.3f}), area {r.floor_area.value:.2f}", flush=True)
