"""Measure room width / length / ceiling from a saved cloud (MapAnything ma_raw.npz or VGGT raw.npz)
with the spike's heuristic at one peak threshold, and compare to the tape.

Usage: uv run python scripts/experiments/measure_cloud.py <raw.npz> <rel_threshold> [short_cm long_cm ceiling_cm]
       (tape defaults to bedroom1: 239 291.5 279.25)
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spike_phase0 as sp
src, rel = sys.argv[1], float(sys.argv[2])
orig = sp.peak_extremes
sp.peak_extremes = lambda x, bin_m=0.01, rel_=None, **k: orig(x, bin_m=bin_m, rel=rel)   # one threshold for walls + floor/ceiling
r = np.load(src)
if "pts" in r:      # MapAnything
    keep = r["mask"] & (r["conf"] > np.percentile(r["conf"][r["mask"]], 30))
    P, E, down = r["pts"][keep], r["extrinsic"], 1
else:               # VGGT raw (rotated portrait -> camera +x is down)
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    s = float(r["scale"]); E = r["extrinsic"].copy(); E[:, :, 3] *= s
    pts = unproject_depth_map_to_point_map(r["depth"][..., None] * s, E, r["intrinsic"])
    P, down = pts[r["depth_conf"] >= np.percentile(r["depth_conf"], 50)], 0
room = sp.measure_room(P, E, np.random.default_rng(0), down_axis=down)
out = Path(src).parent / f"rel{rel}"; out.mkdir(exist_ok=True); sp.save_plots(out, room)
t = [float(v) / 100 for v in sys.argv[3:6]] if len(sys.argv) >= 6 else [2.39, 2.915, 2.7925]
tape = {"short_side_m": t[0], "long_side_m": t[1], "ceiling_height_m": t[2]}
print(f"{Path(src).parent.name:6s} rel={rel}: " + "  ".join(f"{k.split('_')[0]} {room[k]:.3f} ({100*(room[k]-t)/t:+.1f}%)" for k, t in tape.items()))
