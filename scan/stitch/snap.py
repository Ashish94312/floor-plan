"""Structural wall snapping in a shared frame (photo-tier drift correction, D6 / D25).

After a joint reconstruction every room is placed, but each room's walls are fitted on its own, so
the same physical wall can appear twice: two facing walls a few cm too far apart (a gap where the
partition is), or two rooms' outer walls a few cm off one line. Every wall line is one FACE of a
physical wall whose centre lies t/2 behind it. Faces whose centre estimates agree within a tolerance
are linked (facing each other with overlapping spans = shared partition; facing the same way side by
side = one outer wall), each linked group gets one centre line, set by its BEST-SEEN face (most
supporting points); the other faces snap to it (D25: a support-weighted mean or median let short or
poorly-seen faces drag a well-seen wall). Faces are moved to centre - outward*t/2.
Polygons, lengths and areas are rebuilt from the snapped lines.
"""

from __future__ import annotations

import numpy as np


def _lines(layout):
    """Per wall: orientation, fixed coordinate, outward sign, span along the wall, weight."""
    out = []
    for w in layout.walls:
        (x0, y0), (x1, y1) = w.start, w.end
        if abs(y1 - y0) < abs(x1 - x0):  # horizontal wall at y = const; clockwise -> outward = sign(dx) in y
            out.append({"o": "h", "c": (y0 + y1) / 2, "s": float(np.sign(x1 - x0)), "span": tuple(sorted((x0, x1)))})
        else:  # vertical wall at x = const; outward = -sign(dy) in x
            out.append({"o": "v", "c": (x0 + x1) / 2, "s": -float(np.sign(y1 - y0)), "span": tuple(sorted((y0, y1)))})
        out[-1]["w"] = 1.0 + float(w.support)
    return out


def _find(p, i):
    while p[i] != i:
        p[i] = p[p[i]]
        i = p[i]
    return i


def snap_walls(layouts: dict, cfg: dict) -> list[dict]:
    """Snap in place. Returns one record per moved wall: room, wall_id, shift_m."""
    sc = cfg["stitch"]
    t, tol = sc["wall_thickness_m"], sc["snap_tolerance_m"]
    items = []  # (room, wall index, line)
    for r, lay in layouts.items():
        if lay.method != "free_space_carving":
            continue
        for k, ln in enumerate(_lines(lay)):
            items.append((r, k, ln))
    parent = list(range(len(items)))
    for i, (ra, _, a) in enumerate(items):
        for j in range(i + 1, len(items)):
            rb, _, b = items[j]
            if ra == rb or a["o"] != b["o"]:
                continue
            centre_a, centre_b = a["c"] + a["s"] * t / 2, b["c"] + b["s"] * t / 2
            if abs(centre_a - centre_b) > tol:
                continue
            overlap = min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0])
            facing = a["s"] != b["s"] and a["s"] * (b["c"] - a["c"]) > 0
            partition = facing and overlap >= sc["adjacency_min_overlap_m"]
            collinear = a["s"] == b["s"] and -sc["snap_max_along_gap_m"] <= overlap <= 0.05
            if partition or collinear:
                parent[_find(parent, i)] = _find(parent, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(_find(parent, i), []).append(i)
    moves = []
    new_c: dict[tuple[str, int], float] = {}
    for idx in groups.values():
        if len({items[i][0] for i in idx}) < 2:
            continue
        w = np.array([items[i][2]["w"] for i in idx])
        centres = np.array([items[i][2]["c"] + items[i][2]["s"] * t / 2 for i in idx])
        c = float(centres[int(np.argmax(w))])  # the best-seen face sets the line; the others snap to it
        for i in idx:
            r, k, ln = items[i]
            new_c[(r, k)] = c - ln["s"] * t / 2
    for r, lay in layouts.items():
        lines = _lines(lay) if lay.method == "free_space_carving" else None
        if not lines or not any((r, k) in new_c for k in range(len(lines))):
            continue
        for k, ln in enumerate(lines):
            if (r, k) in new_c:
                moves.append({"room": r, "wall_id": lay.walls[k].wall_id, "shift_m": round(new_c[(r, k)] - ln["c"], 4)})
                ln["c"] = new_c[(r, k)]
        _rebuild(lay, lines)
    return moves


def _rebuild(lay, lines) -> None:
    """Corners from consecutive (alternating) snapped lines -> walls, polygon, area."""
    n = len(lines)
    corners = []
    for k in range(n):  # corner at the START of wall k = intersection of wall k-1 and wall k
        a, b = lines[k - 1], lines[k]
        corners.append((a["c"], b["c"]) if a["o"] == "v" else (b["c"], a["c"]))
    for k, w in enumerate(lay.walls):
        s, e = corners[k], corners[(k + 1) % n]
        w.start = (round(float(s[0]), 4), round(float(s[1]), 4))
        w.end = (round(float(e[0]), 4), round(float(e[1]), 4))
        w.length_m = round(float(np.hypot(e[0] - s[0], e[1] - s[1])), 4)
    lay.polygon = [w.start for w in lay.walls]
    P = np.array(lay.polygon)
    lay.floor_area_m2 = round(abs(0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(np.roll(P[:, 0], -1), P[:, 1]))), 4)
