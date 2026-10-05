"""E27: is a stitched plan clean? Overlaps between rooms, how many separate pieces, and each room's gap to its nearest
neighbour (rooms meet across a wall: a gap up to stitch.adjacency_max_gap_m, 0.35 m, is a wall; more is missing space).

  uv run python scripts/experiments/e27_plan_check.py <result.json> [<result.json> ...]
"""

import json
import sys
from itertools import combinations
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

for f in map(Path, sys.argv[1:]):
    r = json.loads(f.read_text())
    P = {rm["room_id"]: Polygon(rm["polygon"]) for rm in r["rooms"]}
    ov = [(a, b, round(P[a].intersection(P[b]).area, 3)) for a, b in combinations(P, 2) if P[a].intersection(P[b]).area > 0.02]
    walls = unary_union([p.buffer(0.35 / 2) for p in P.values()])  # rooms grown by half a wall
    pieces = 1 if walls.geom_type == "Polygon" else len(walls.geoms)
    gaps = {a: round(min(P[a].distance(P[b]) for b in P if b != a), 2) for a in P} if len(P) > 1 else {}
    print(f"{f.parent.parent.name}/{f.parent.name}: {len(P)} spaces ({', '.join(f'{k} {v.area:.1f}' for k, v in P.items())} m2), "
          f"overlaps {ov or 0}, pieces across walls {pieces}, gap to nearest {gaps}")
