"""Video: every room on one scale, from one model run over all rooms (E23b).

Per-room model runs give the best shapes (12+ views each), but each run sets its own size: per room ~6-10% apart
(D32, E22v: the home01_video_e hall run ~10% small next to a right-sized bedroom). Repairs after the fact failed:
shared points between rooms are all 4-6 m away and disagree by range (E22w); a room's uniform size error is
invisible from inside it (E22v); doors anchor only rooms whose door top was seen (E22z).

One model run over frames of all rooms has ONE scale for all of them (E22k: joint, bedroom and hall came out
equally small). It cannot hold enough views for good shapes (memory: geometry.max_joint_views), so it is used only
for size: the rooms' link frames (they see across doorways and tie the rooms together) plus evenly spaced keyframes
of every room, up to the budget. A frame in both the joint run and a room's run has two depth maps of the same
pixels; their ratio is that room's size relative to the joint run. Per frame the median pixel ratio; per room the
median over its frames (spread = the room's uncertainty). Each room is then scaled to the joint run's size, and
what remains is one size error for the whole flat.
"""

from __future__ import annotations

import numpy as np


def joint_frames(cap, budget: int) -> list:
    """Frames for the joint run: every room's link frames, then evenly spaced own keyframes, up to `budget`."""
    rooms = list(cap.rooms)
    links = [f for r in rooms for f in cap.rooms[r] if f.link]
    picks = {r: [f for f in cap.rooms[r] if f.link] for r in rooms}
    free = max(0, budget - len(links))
    per = {r: free // len(rooms) + (1 if k < free % len(rooms) else 0) for k, r in enumerate(rooms)}
    for r in rooms:
        own = sorted((f for f in cap.rooms[r] if not f.link), key=lambda f: f.timestamp or 0.0)
        if per[r] and own:
            idx = np.unique(np.round(np.linspace(0, len(own) - 1, min(per[r], len(own)))).astype(int))
            picks[r] += [own[i] for i in idx]
    return [f for r in rooms for f in sorted(picks[r], key=lambda f: f.timestamp or 0.0)]


def room_factors(joint: dict, runs: dict[str, dict], min_pixels: int = 500) -> dict[str, dict]:
    """joint: the joint run's prediction; runs: room -> its run views (depth, mask, names), current size.
    -> {room: {c (multiply the room by it), sigma_log, frames, per_frame}}. Rooms sharing no frame are left out."""
    jn = {str(n): i for i, n in enumerate(joint["names"])}
    out = {}
    for r, v in runs.items():
        per = []
        for i, n in enumerate(v["names"]):
            j = jn.get(str(n))
            if j is None or joint["depth"][j].shape != v["depth"][i].shape:
                continue
            ok = joint["mask"][j] & v["mask"][i] & (v["depth"][i] > 0.1) & (joint["depth"][j] > 0.1)
            if ok.sum() >= min_pixels:
                per.append(float(np.median(np.log(joint["depth"][j][ok] / v["depth"][i][ok]))))
        if per:
            med = float(np.median(per))
            spread = float(1.4826 * np.median(np.abs(np.array(per) - med))) if len(per) > 1 else float("nan")
            out[r] = {"c": float(np.exp(med)), "sigma_log": spread / np.sqrt(len(per)), "frames": len(per),
                      "per_frame": [round(float(np.exp(x)), 4) for x in per]}
    return out
