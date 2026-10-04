"""E15. Ray source per room, for per-room AND joint runs: inter-view residual + size vs tape.

Usage: uv run python scripts/experiments/rays_eval.py <out_dir> [<out_dir2> ...] [--plots DIR]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spike_phase0 as sp  # noqa: E402
from icp_eval import TAPE, residual, view_clouds  # noqa: E402

from scan import cache  # noqa: E402
from scan.config import load_config  # noqa: E402

args = sys.argv[1:]
plots = None
if "--plots" in args:
    i = args.index("--plots")
    plots = Path(args[i + 1])
    del args[i : i + 2]
cfg = load_config()
for out_dir in map(Path, args):
    g = json.loads((out_dir / "geometry.json").read_text())
    for run, info in g["geometry"]["runs"].items():
        pred = cache.load(cfg, "mapanything", info["key"])
        rooms = pred["rooms"]
        for room in dict.fromkeys(rooms):
            idx = np.flatnonzero(rooms == room)
            sub = {k: v[idx] for k, v in pred.items()}
            thr = np.percentile(sub["conf"][sub["mask"]], cfg["geometry"]["conf_percentile"])
            for mode in ("model", "exif"):
                clouds = view_clouds(sub, sub["T_wc"], mode, thr)
                med, w2 = residual(clouds)
                m = sp.measure_room(np.concatenate(clouds), np.linalg.inv(sub["T_wc"])[:, :3, :], np.random.default_rng(0), down_axis=1)
                tv = TAPE[room]
                errs = [100 * (m[k] - x) / x for k, x in zip(["short_side_m", "long_side_m", "ceiling_height_m"], tv)]
                print(f"{run:9s} {room:9s} {mode:6s} residual {100 * med:4.1f} cm ({100 * w2:3.0f}% < 2 cm) | "
                      f"short {m['short_side_m']:.3f} ({errs[0]:+5.1f}%)  long {m['long_side_m']:.3f} ({errs[1]:+5.1f}%)  "
                      f"ceiling {m['ceiling_height_m']:.3f} ({errs[2]:+5.1f}%)")
                if plots is not None:
                    d = plots / f"{run}_{room}_{mode}"
                    d.mkdir(parents=True, exist_ok=True)
                    sp.save_plots(d, m)
