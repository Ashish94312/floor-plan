"""E24 (NEW APPROACH, separate track), step 0: export one room's model reconstruction for the SfM comparison.

Runs the pipeline (torch / MapAnything, cache) and saves what e24_sfm_room.py needs, so that script never imports
torch: torch and pycolmap each ship an OpenMP runtime and abort when both load in one process (E24, OMP #15).

Usage: uv run python scripts/experiments/e24_export_model.py <capture> <room | all> [section.key=value ...]
Output: <capture>/experiments/E24_sfm/<room>/model_views.npz (all: one per room, and links.json: the link frames)
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from scan.config import load_config
from scan.pipeline import run

cap_dir, room = Path(sys.argv[1]), sys.argv[2]
extra: dict = {}
for a in sys.argv[3:]:
    key, val = a.split("=", 1)
    sec, name = key.split(".", 1)
    extra.setdefault(sec, {})[name] = yaml.safe_load(val)
cfg = load_config(overrides=extra)
base = cap_dir / "experiments" / "E24_sfm"
base.mkdir(parents=True, exist_ok=True)
cap, clouds, _s = run(cap_dir, cfg, out=base / "_run", log=lambda *a: None)
for r in (list(clouds) if room == "all" else [room]):
    out = base / r
    out.mkdir(parents=True, exist_ok=True)
    v = clouds[r].views
    lay = clouds[r].layout
    np.savez_compressed(
        out / "model_views.npz",
        names=np.array([str(n) for n in v["names"]]), pts=v["pts"], mask=v["mask"], K_exif=v["K_exif"],
        depth=v["depth"], T_wc=v["T_wc"],
        K_work=cap.rooms[r][0].K, walls=np.array([f"{w.wall_id} {w.length_m:.3f}" for w in lay.walls]),
    )
    print(out / "model_views.npz")
(base / "links.json").write_text(json.dumps([list(x) for x in cap.links]))
