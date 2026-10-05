"""Fix-loop prediction (FIX_DECLARATION.md): simulate the planned photo-tier scale fix on cached model outputs.

Fix: one global scale per joint run, k = exp(median over rooms of the room's median log(f_exif / f_model,i)),
then a calibrated level c (calibrate.fit_size_bias over the corrected captures, leave-one-physical-room-out).
Applied here to the CURRENT eval.json values (all lengths scale linearly), so no pipeline code is touched.

    uv run python scripts/experiments/orientation_scale_predict.py captures/home01_photo_a captures/home01_photo_b
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np

from scan.uncertainty.calibrate import fit_size_bias, size_rows


def votes(cap: Path) -> tuple[float, dict]:
    key = json.loads((cap / "out/debug/geometry.json").read_text())["geometry"]["runs"]["joint"]["key"]
    z = np.load(f".cache/mapanything/{key}.npz", allow_pickle=True)
    v = np.log(z["K_exif"][:, 0, 0] / z["K_model"][:, 0, 0])  # frame i votes log(f_true / f_model,i)
    rooms = {r: float(np.median(v[z["rooms"] == r])) for r in dict.fromkeys(z["rooms"])}
    return math.exp(float(np.median(list(rooms.values())))), {r: round(math.exp(x), 3) for r, x in rooms.items()}


def scaled(ev: dict, f: float) -> dict:
    ev = copy.deepcopy(ev)
    for r in ev["rooms"].values():
        for m in [w for w in (r.get("walls") or []) if isinstance(w, dict)] + [r.get("ceiling")]:
            if isinstance(m, dict) and "pred" in m and m.get("gt"):
                for k in ("pred", "lo", "hi"):
                    m[k] *= f
                m["err_pct"] = 100 * (m["pred"] / m["gt"] - 1)
                m["covered"] = m["lo"] <= m["gt"] <= m["hi"]
    return ev


def table(ev: dict, label: str) -> None:
    walls, cov = [], []
    for room, r in ev["rooms"].items():
        for m in [w for w in (r.get("walls") or []) if isinstance(w, dict) and "err_pct" in w]:
            walls.append(abs(m["err_pct"]) <= 8)
            cov.append(m["covered"])
            print(f"  {label:8s} {room:9s} {m['gt_id']:3s} pred {m['pred']:.3f} tape {m['gt']:.3f} {m['err_pct']:+6.1f}%")
        c = r.get("ceiling")
        if isinstance(c, dict) and "err_pct" in c:
            cov.append(c["covered"])
            print(f"  {label:8s} {room:9s} ceil pred {c['pred']:.3f} tape {c['gt']:.3f} {c['err_pct']:+6.1f}%")
    print(f"  {label}: walls within ±8% {sum(walls)}/{len(walls)}, intervals covering {sum(cov)}/{len(cov)}"
          " (same relative widths as now)")


caps = [Path(a) for a in sys.argv[1:]]
evs = {c.name: json.loads((c / "out/eval.json").read_text()) for c in caps}
ks = {}
for c in caps:
    ks[c.name], per_room = votes(c)
    print(f"{c.name}: k = {ks[c.name]:.3f}  (room medians of f_exif/f_model: {per_room})")
corr = {n: scaled(ev, ks[n]) for n, ev in evs.items()}
fit = fit_size_bias([row for ev in corr.values() for row in size_rows(ev)])
for mode, f in fit.items():
    print(f"level c ({mode}): x{f['factor']}  sigma_log {f['sigma_log']}  rooms {f['rooms']} / physical {f['physical_rooms']}")
    for row in f["loo"]:
        print(f"  LOO {row['room']:28s} after k: {row['raw_pct']:+6.2f}%  after k and c from the other rooms: {row['corrected_pct']:+6.2f}%")
c_all = next(iter(fit.values()))["factor"]
for n, ev in evs.items():
    print(f"\n{n} BEFORE"); table(ev, "before")
    print(f"{n} AFTER (k x c, c fitted on all)"); table(scaled(corr[n], c_all), "after")
