"""Video ingest speed-up check: the working-tree scan/io/video.py against the committed one (git HEAD), each in a
fresh process (link matching carries random state across runs in one process, so in-process runs differ).

    uv run python scripts/experiments/video_ingest_speed.py captures/home01_video_e

Prints both times and whether keyframes (pixels), intrinsics and link pairs are identical.
"""

import json
import subprocess
import sys

CHILD = """
import hashlib, json, subprocess, sys, time, types
from pathlib import Path
import scan.io.video as mod
from scan.config import load_config
from scan.types import Capture
root, which = Path(sys.argv[1]), sys.argv[2]
if which == "head":
    src = subprocess.run(["git", "show", "HEAD:scan/io/video.py"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("video_head")
    mod.__dict__.update({"__file__": __import__("scan.io.video").io.video.__file__})
    code = compile(src, "video_head", "exec")
    eval(code, mod.__dict__)
cap = Capture(capture_id=root.name, root=root, tier="video")
t = time.perf_counter()
mod.ingest_video(cap, load_config())
dt = time.perf_counter() - t
out = {"t": dt, "links": cap.links, "rooms": {r: [[f.image_path.name, hashlib.sha256(f.rgb.tobytes()).hexdigest(),
       [round(x, 6) for x in f.K.ravel()]] for f in fs] for r, fs in cap.rooms.items()}}
print(json.dumps(out))
"""


def run(which):
    p = subprocess.run([sys.executable, "-c", CHILD, sys.argv[1], which], capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


a, b = run("head"), run("new")
same = a["links"] == b["links"] and a["rooms"] == b["rooms"]
print(f"committed {a['t']:.1f} s, new {b['t']:.1f} s ({a['t'] / b['t']:.1f}x); identical keyframes, K and links: {same}")
if not same:
    for r in a["rooms"]:
        print(r, [x[0] for x in a["rooms"][r]] == [x[0] for x in b["rooms"][r]], a["rooms"][r] == b["rooms"][r])
    print(a["links"], b["links"])
