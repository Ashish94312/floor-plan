"""E26: MapAnything (photo tier on a LiDAR walk's frames) vs the LiDAR plan of the same walk (reference).

Per room (same room ids: e26_lidar_to_photos.py keeps the LiDAR split's grouping): floor area, the room's short and
long side (extent of its outline on its own axes; sorted, so the plans' axis choice does not matter), ceiling.
Error = MapAnything / LiDAR - 1; 'in' = the LiDAR value lies inside MapAnything's 90% interval.

  uv run python scripts/experiments/e26_compare.py <recording> [<recording> ...]
"""

import json
import sys
from pathlib import Path

import numpy as np


def rooms(path: Path) -> dict:
    r = json.loads(path.read_text())
    out = {}
    for rm in r["rooms"]:
        P = np.array(rm["polygon"])
        short, long_ = sorted(P.max(0) - P.min(0))
        c = rm["ceiling_height"]
        out[rm["room_id"]] = {"area": rm["floor_area"], "short": short, "long": long_, "ceiling": c,
                              "status": rm["status"], "method": rm["layout_method"]}
    return out, r["stitched_plan"]["footprint_area"]


errs = {"area": [], "short": [], "long": [], "ceiling": []}
for rec in sys.argv[1:]:
    base = Path("captures/lidar_samples") / rec
    ref, ref_fp = rooms(base / "out" / "result.json")
    ma, ma_fp = rooms(base / "photo_from_lidar" / "out" / "result.json")
    print(f"\n### {rec}: footprint LiDAR {ref_fp['value']:.2f} m², MapAnything {ma_fp['value']:.2f} m² "
          f"({100 * (ma_fp['value'] / ref_fp['value'] - 1):+.1f}%)\n")
    print("| Room | Area LiDAR → MA | Short side | Long side | Ceiling | MA layout |")
    print("|---|---|---|---|---|---|")
    for rid, a in ref.items():
        b = ma.get(rid)
        if b is None:
            print(f"| {rid} | missing in MapAnything run | | | | |")
            continue
        cells = []
        for k in ("area", "short", "long", "ceiling"):
            if k == "area":
                va, vb = a[k]["value"], b[k]["value"]
                inside = b[k]["lo"] <= va <= b[k]["hi"]
            elif k == "ceiling":
                if not a[k] or not b[k]:
                    cells.append(f"{a[k]['value']:.2f} → –" if a[k] else f"– → {b[k]['value']:.2f}" if b[k] else "– → –")
                    continue
                va, vb = a[k]["value"], b[k]["value"]
                inside = b[k]["lo"] <= va <= b[k]["hi"]
            else:
                va, vb, inside = a[k], b[k], None
            e = vb / va - 1
            errs[k].append(e)
            cells.append(f"{va:.2f} → {vb:.2f} ({100 * e:+.1f}%{', in' if inside else ', out' if inside is False else ''})")
        print(f"| {rid} | {' | '.join(cells)} | {b['method']}, {b['status']} |")
print("\nMean |error| over rooms: " + ", ".join(f"{k} {100 * np.mean(np.abs(v)):.1f}% (n={len(v)})" for k, v in errs.items() if v))
