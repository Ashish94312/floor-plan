"""E22. Video tier: how many keyframes per room? Runs the full pipeline at several budgets and scores
each run against the tape (data/ground_truth/<property>.yaml).

Usage: uv run python scripts/experiments/video_keyframes.py captures/home01_video_a 6 8 12 16 [section.key=value ...]
Writes <capture>/out_k<N>[_<value>...]/ (result.json, plan.png, eval.md, debug/) per budget and prints one table.
Extra section.key=value arguments override config/default.yaml (e.g. geometry.rays=model).
"""

import json
import sys
from pathlib import Path

import yaml

from scan.config import load_config
from scan.eval.gates import evaluate
from scan.eval.gt import load_gt
from scan.eval.report import markdown
from scan.pipeline import run

cap_dir = Path(sys.argv[1])
budgets = [int(k) for k in sys.argv[2:] if "=" not in k] or [6, 8, 12, 16]
extra: dict = {}
tag = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--tag=")), None)  # output name instead of the values
for a in (a for a in sys.argv[2:] if "=" in a and not a.startswith("--tag=")):
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
suffix = f"_{tag}" if tag else "".join(f"_{v}" for sec in extra.values() for v in sec.values())
gt_path = Path("data/ground_truth") / f"{cap_dir.name.split('_')[0]}.yaml"
gt = load_gt(gt_path)

rows = []
for k in budgets:
    cfg = load_config(overrides=extra | {"video": extra.get("video", {}) | {"keyframes_per_room": k}})
    out = cap_dir / f"out_k{k}{suffix}"
    _cap, _clouds, summary = run(cap_dir, cfg, out=out, log=lambda *a: None)
    res = summary["result"]
    ev = evaluate(res, gt, cfg)
    (out / "eval.json").write_text(json.dumps(ev, indent=2, default=str))
    (out / "eval.md").write_text(markdown(ev))
    rooms = {r.room_id: r for r in res.rooms}
    s = ev["summary"]
    walls_ok = next(v for g, v in ev["gates"].items() if g.startswith("walls within"))
    rows.append({
        "k": k, "joint": summary["geometry"]["joint"],
        "area": {r: round(rooms[r].floor_area.value, 2) for r in rooms},
        "nwalls": {r: len(rooms[r].walls) for r in rooms},
        "ceiling": {r: (round(rooms[r].ceiling_height.value, 3) if rooms[r].ceiling_height else None) for r in rooms},
        "walls_ok": walls_ok, "mean_err": s["mean_abs_wall_err_pct"], "max_err": s["max_abs_wall_err_pct"],
        "adj_equal": ev["adjacency"]["equal"], "overlaps": ev["overlaps"],
        "footprint": round(res.stitched_plan.footprint_area.value, 2),
        "placed": res.stitched_plan.placement_method, "timing": summary["timing_s"],
    })
    print(json.dumps(rows[-1]), flush=True)

print("\n| k | joint | areas m² | walls | ceilings m | walls ok | mean / max err % | adj | overl. | footprint |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['k']} | {r['joint']} | {r['area']} | {r['nwalls']} | {r['ceiling']} | {r['walls_ok']} | "
          f"{r['mean_err']} / {r['max_err']} | {'=' if r['adj_equal'] else '≠'} | {r['overlaps']} | {r['footprint']} |")
