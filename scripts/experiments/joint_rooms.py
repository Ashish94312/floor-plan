"""E9. Measure each room of a joint multi-room MapAnything run, and plot all rooms in one frame.

Usage: uv run python scripts/experiments/joint_rooms.py <ma_raw.npz> room=short,long,ceiling[cm] ...
  e.g. bedroom1=239,291.5,279.25 hall=230,370,280.25
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spike_phase0 as sp  # noqa: E402

src = Path(sys.argv[1])
tapes = {a.split("=")[0]: [float(v) / 100 for v in a.split("=")[1].split(",")] for a in sys.argv[2:]}
r = np.load(src)
rooms, E = r["rooms"], r["extrinsic"]
keep = r["mask"] & (r["conf"] > np.percentile(r["conf"][r["mask"]], 30))
rng = np.random.default_rng(0)

# Per room: only points seen from that room's photos, gravity from that room's cameras.
for room in sorted(set(rooms)):
    idx = rooms == room
    P = r["pts"][idx][keep[idx]]
    m = sp.measure_room(P, E[idx], rng, down_axis=1)
    out = src.parent / room
    out.mkdir(exist_ok=True)
    sp.save_plots(out, m)
    t = tapes.get(room)
    errs = "" if t is None else "  ".join(
        f"{k.split('_')[0]} {m[k]:.3f} ({100 * (m[k] - tv) / tv:+.1f}%)"
        for k, tv in zip(["short_side_m", "long_side_m", "ceiling_height_m"], t)
    )
    print(f"{room:9s} imgs={idx.sum()}  {errs}")

# All rooms in one frame: shared gravity + Manhattan axes, wall band coloured by room.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P_all, lab = r["pts"][keep], np.repeat(rooms, keep.reshape(len(rooms), -1).sum(1))
m_all = sp.measure_room(P_all, E, rng, down_axis=1)
up = -E[:, 1, :3].mean(0)
up /= np.linalg.norm(up)
h = P_all @ up
fh, ch = m_all["_plot"]["fh"], m_all["_plot"]["ch"]
e1 = np.cross(up, [1.0, 0, 0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(up, e1)
a = np.deg2rad(m_all["manhattan_deg"])
rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
band = (h > fh + 0.5 * (ch - fh)) & (h < ch - 0.10)
q = np.stack([P_all[band] @ e1, P_all[band] @ e2], 1) @ rot
cam = np.linalg.inv(np.concatenate([E, np.tile([[[0, 0, 0, 1.0]]], (len(E), 1, 1))], 1))[:, :3, 3]
cq = np.stack([cam @ e1, cam @ e2], 1) @ rot
fig, ax = plt.subplots(figsize=(9, 9))
for room, c in zip(sorted(set(rooms)), ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    sel = lab[band] == room
    ax.scatter(q[sel, 0][::3], q[sel, 1][::3], s=0.3, c=c, label=f"{room} points")
    ax.scatter(cq[rooms == room, 0], cq[rooms == room, 1], marker="^", s=60, c=c, edgecolors="k", label=f"{room} cameras")
ax.set_aspect("equal")
ax.legend(markerscale=6)
ax.set_title(f"Joint reconstruction, top-down (upper wall band). Ceiling (all) {m_all['ceiling_height_m']:.3f} m")
fig.tight_layout()
fig.savefig(src.parent / "joint_topdown.png", dpi=120)
print(f"joint ceiling (all points) {m_all['ceiling_height_m']:.3f} m -> {src.parent / 'joint_topdown.png'}")
