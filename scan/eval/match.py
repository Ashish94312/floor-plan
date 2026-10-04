"""Match predicted walls to ground-truth walls (ARCHITECTURE §13).

Both lists run clockwise seen from above; only the starting wall can differ (the plan's W1 is
provisional until doors are detected). Try every rotation and keep the one with the lowest cost:
relative length error on measured walls + a penalty when open/wall kinds disagree. Choosing the best
rotation is slightly optimistic, so the chosen shift is reported.
"""

from __future__ import annotations


def match_walls(pred_walls, gt_walls) -> dict | None:
    n = len(pred_walls)
    if n != len(gt_walls) or n == 0:
        return None
    best = None
    for shift in range(n):
        cost = 0.0
        for k, g in enumerate(gt_walls):
            p = pred_walls[(k + shift) % n]
            if g.length_m is not None:
                cost += abs(p.length.value - g.length_m) / g.length_m
            if (getattr(p, "kind", "wall") == "open") != (g.kind == "open"):
                cost += 1.0
        if best is None or cost < best[0] - 1e-9:
            best = (cost, shift)
    cost, shift = best
    pairs = [(pred_walls[(k + shift) % n], g) for k, g in enumerate(gt_walls)]
    return {"shift": shift, "cost": round(cost, 4), "pairs": pairs}
