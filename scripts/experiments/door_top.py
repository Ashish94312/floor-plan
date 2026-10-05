"""E22z. Is a door's height measured (wall seen above it: the lintel) or cut off by the field of view?

For each door / opening of the named rooms: open and wall votes (layout.openings.wall_votes) summed over the opening's
width, per 10 cm row from the floor up to the top of the vote grid. Wall votes in the rows above the reported height =
the lintel was seen, the height is a measurement. No votes at all above it = nothing was seen there: the height is
where the view ended.

Usage: uv run python scripts/experiments/door_top.py <capture> <room> [...] [section.key=value ...]
"""

import sys
from pathlib import Path

import numpy as np
import yaml

from scan.config import load_config
from scan.layout.openings import wall_votes
from scan.pipeline import run

cap_dir = Path(sys.argv[1])
rooms = [a for a in sys.argv[2:] if "=" not in a]
extra: dict = {}
for a in (a for a in sys.argv[2:] if "=" in a):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
_cap, clouds, summary = run(cap_dir, cfg, out=cap_dir / "experiments" / "E22z_doors" / "_run", log=lambda *a: None)
res = summary["result"].model_dump()
oc = cfg["openings"]
for r in rooms:
    c = clouds[r]
    ceil = c.planes.ceiling_z or float(np.percentile(c.points[:, 2], 98))
    room = next(x for x in res["rooms"] if x["room_id"] == r)
    for o in room.get("openings", []):
        if o["type"] == "window":
            continue
        w = next(w for w in c.layout.walls if w.wall_id == o["wall_id"])
        open_v, wall_v, L = wall_votes(w, c.views, ceil, cfg)
        u0, wd = o["offset_along_wall"]["value"], o["width"]["value"]
        cols = slice(int(u0 / oc["cell_u_m"]), max(int(u0 / oc["cell_u_m"]) + 1, int((u0 + wd) / oc["cell_u_m"])))
        ov, wv = open_v[:, cols].sum(1), wall_v[:, cols].sum(1)
        print(f"\n{r} {o['opening_id']} {o['type']} on {o['wall_id']}: width {wd:.2f}, height {o['height']['value']:.2f} m; "
              f"vote grid up to {ceil:.2f} m (ceiling {'plane' if c.planes.ceiling_z else 'not found: 98th pct'})")
        for k in range(len(ov) - 1, -1, -1):
            z = (k + 0.5) * oc["cell_z_m"]
            if k % 2 == 0 or z > o["height"]["value"] - 0.3:
                print(f"   z {z:4.2f}  open {int(ov[k]):4d}  wall {int(wv[k]):4d}")
