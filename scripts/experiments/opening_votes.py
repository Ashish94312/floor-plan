"""E17b. Plot open-vote ratio per wall (visibility voting) to check doors/windows show up.

Usage: uv run python scripts/experiments/opening_votes.py <capture_dir> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan.config import load_config  # noqa: E402
from scan.layout.openings import wall_votes  # noqa: E402
from scan.pipeline import run  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

cap, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cfg = load_config()
_, clouds, _ = run(cap, cfg, out=out / "_run", log=lambda *_: None)
for r, c in clouds.items():
    ceil = c.planes.ceiling_z or 2.8
    walls = c.layout.walls
    fig, axes = plt.subplots(1, len(walls), figsize=(3.2 * len(walls), 3.4), squeeze=False)
    for k, w in enumerate(walls):
        o, wv, L = wall_votes(w, c.views, ceil, cfg)
        ratio = np.where(o + wv > 0, o / np.maximum(o + wv, 1), np.nan)
        axes[0, k].imshow(ratio, origin="lower", extent=[0, L, 0, ceil], aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[0, k].set_title(f"{w.wall_id} ({w.kind})\nred = seen through, green = wall", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{r}_open_votes.png", dpi=90)
    plt.close(fig)
    print("wrote", out / f"{r}_open_votes.png")
