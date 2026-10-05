"""E22y. Where along a plan wall are there wall points, and where did the openings module put its openings?

Per named wall: vertical-surface points within the layout's coverage window of the wall line, counted in 10 cm bins
along the wall (start -> end), plus each opening on that wall (from where to where along it). A drawn wall span with
no points that is not inside an opening is a wall the plan invents.

Usage: uv run python scripts/experiments/wall_profile.py <capture> <wall_id> [...] [section.key=value ...]
"""

import sys
from pathlib import Path

import numpy as np
import yaml

from scan.config import load_config
from scan.pipeline import run

cap_dir = Path(sys.argv[1])
names = [a for a in sys.argv[2:] if "=" not in a]
extra: dict = {}
for a in (a for a in sys.argv[2:] if "=" in a):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, summary = run(cap_dir, cfg, out=cap_dir / "experiments" / "E22y_walls" / "_run", log=lambda *a: None)
res = summary["result"].model_dump()
win = cfg["layout"]["coverage_window_m"]
for wid in names:
    r = wid.rsplit("-", 1)[0]
    lay = clouds[r].layout
    w = next(w for w in lay.walls if w.wall_id == wid)
    pts = lay._debug["walls_pts"]
    a, b = np.array(w.start), np.array(w.end)
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    n = np.array([-u[1], u[0]])
    rel = pts[:, :2] - a
    along, across = rel @ u, rel @ n
    m = (np.abs(across) < win) & (along > -0.05) & (along < L + 0.05)
    bins = np.histogram(along[m], bins=np.arange(0, L + 0.1, 0.1))[0]
    print(f"\n{wid} ({w.kind}) {a.round(2).tolist()} -> {b.round(2).tolist()}, {L:.2f} m, coverage {w.coverage}, support {w.support}")
    print("  points per 10 cm along the wall: " + " ".join(f"{x:d}" for x in bins))
    room = next(x for x in res["rooms"] if x["room_id"] == r)
    for o in room.get("openings", []):
        if o.get("wall_id") == wid:
            u0 = o["offset_along_wall"]["value"]
            wd = o["width"]["value"] if isinstance(o["width"], dict) else o["width"]
            print(f"  opening {o.get('opening_id')} {o.get('type')}: {u0} .. {u0 + wd if u0 is not None else '?'} m along, width {wd}")
