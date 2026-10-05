"""Video: one scale for the whole flat (E22p).

After the camera re-solve (geometry.repose) each room's shape is fixed by geometry, its overall size by the
votes of its own frames (the model's size sense, f_true / f_model). Those per-room sizes carry per-room
errors, so rooms drawn side by side disagree (bedroom ~12% too big next to a right-sized hall, E22o).

Link frames see the same physical points from two rooms, so the ratio of the two rooms' sizes is MEASURED
(3D similarity of the shared points, stitch.links): log c_b - log c_a = log s_ab, with c the correction
of each room's size. That leaves one unknown per connected group of rooms, its level, and every frame of
every room in the group votes for it: frame i of room r says log c_r = u_i, so the level is the median of
u_i - delta_r. Its uncertainty combines the votes' spread within each room and the disagreement between
the rooms' medians (one room's frames share that room's bias, so they are not independent votes). What this cannot remove is a
bias shared by all frames (the model's size sense on the whole flat): that needs a reference from outside
the video.
"""

from __future__ import annotations

import numpy as np


def flat_scale(rooms: list[str], votes: dict[str, np.ndarray], edges: list[dict], min_inliers: int = 8) -> dict[str, dict]:
    """votes[r]: frames' log votes for room r's size relative to its current size. edges: link edges
    (a, b, scale_b_in_a = s with A ~ s B, scale_inliers). -> {room: {c, sigma_log, group, votes}}."""
    usable = [e for e in edges if e["a"] in rooms and e["b"] in rooms and np.isfinite(e["scale_b_in_a"])
              and e["scale_inliers"] >= min_inliers]
    parent = {r: r for r in rooms}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for e in usable:
        parent[find(e["a"])] = find(e["b"])
    out = {}
    for g, root in enumerate(dict.fromkeys(find(r) for r in rooms)):
        members = [r for r in rooms if find(r) == root]
        idx = {r: k for k, r in enumerate(members)}
        es = [e for e in usable if e["a"] in idx]
        A = np.zeros((len(es) + 1, len(members)))
        b = np.zeros(len(es) + 1)
        for k, e in enumerate(es):
            w = np.sqrt(e["scale_inliers"])
            A[k, idx[e["b"]]], A[k, idx[e["a"]]], b[k] = w, -w, w * np.log(e["scale_b_in_a"])
        A[-1, 0] = 1.0  # gauge: first member's delta = 0 (the level below absorbs it)
        delta = np.linalg.lstsq(A, b, rcond=None)[0]
        per = {r: np.asarray(votes.get(r, []), float) - delta[idx[r]] for r in members}
        # frames of one room share its bias, so they are not independent: the room is the unit of evidence.
        # Level = median over rooms of each room's median vote (one badly sized room cannot move it; its size
        # still follows from its measured link ratio). Uncertainty = spread within rooms combined with the
        # disagreement between the rooms' medians.
        meds = {r: float(np.median(x)) for r, x in per.items() if len(x)}
        level = float(np.median(list(meds.values()))) if meds else 0.0
        within = float(1.4826 * np.median(np.abs(np.concatenate([per[r] - m for r, m in meds.items()])))) if meds else np.nan
        between = float(np.sqrt(np.mean([(m - level) ** 2 for m in meds.values()]))) if len(meds) > 1 else 0.0
        spread = float(np.hypot(within, between))
        for r in members:
            out[r] = {"c": float(np.exp(level + delta[idx[r]])), "sigma_log": spread, "group": g,
                      "votes": len(votes.get(r, []))}
    return out


def scale_room(c, factor: float) -> None:
    """Scale a room's reconstruction about the origin of its aligned frame (floor at z=0 stays there)."""
    c.points = c.points * factor
    for v in (x for x in (c.views, c.run_views) if x is not None):
        v["pts"] = (v["pts"] * factor).astype(np.float32)
        v["depth"] = (v["depth"] * factor).astype(np.float32)
        v["T_wc"] = v["T_wc"].copy()
        v["T_wc"][:, :3, 3] *= factor
