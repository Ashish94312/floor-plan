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
    # a face with no wall points is not a measurement: its place came from the free space, or the layout pushed it
    # camera_wall_margin_m behind the cameras (a camera in the doorway pushes it into the next room), E23a
    unseen = cfg["layout"]["min_wall_support"]
    pushed = cfg["layout"]["camera_wall_margin_m"]
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
            overlap = min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0])
            centre_a, centre_b = a["c"] + a["s"] * t / 2, b["c"] + b["s"] * t / 2
            if abs(centre_a - centre_b) > tol + (pushed if min(a["w"], b["w"]) - 1 < unseen else 0.0):
                continue
            # opposite faces of two rooms: facing across a gap, or crossed (the rooms overlap, which no partition
            # allows: the two faces are one wall misplaced), E23a
            partition = a["s"] != b["s"] and overlap >= sc["adjacency_min_overlap_m"]
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


def join_declared(layouts: dict, cfg: dict, same_line: list) -> list[dict]:
    """Room pairs the user declared (captures/<capture>/hints.yaml `same_line`): their outer walls that continue each
    other (same side, end to end) are one wall line whatever their offset, up to stitch.same_line_max_offset_m. The
    line goes to the MEAN of the rooms' estimates, not the best-seen face: each estimate is off by its room's size
    error (similar per room, D32), which no number of wall points reduces (E23c: kitchen + passage and hall span the
    same width, 3.82 vs 3.41 m, tape 3.70). Runs last (openings and damage were measured on walls where the room's
    points are; carry them over with surface_anchors / reanchor). Returns moves like snap_walls."""
    sc = cfg["stitch"]
    pairs = {frozenset(p) for p in same_line if len(set(p)) == 2}
    lines = {r: _lines(lay) for r, lay in layouts.items() if lay.method == "free_space_carving"}
    moves = []
    for r1, r2 in (tuple(p) for p in pairs):
        if r1 not in lines or r2 not in lines:
            continue
        for k1, a in enumerate(lines[r1]):
            for k2, b in enumerate(lines[r2]):
                overlap = min(a["span"][1], b["span"][1]) - max(a["span"][0], b["span"][0])
                if (a["o"] != b["o"] or a["s"] != b["s"] or not -sc["snap_max_along_gap_m"] <= overlap <= 0.05
                        or abs(a["c"] - b["c"]) > sc["same_line_max_offset_m"]):
                    continue
                mid = (a["c"] + b["c"]) / 2
                for r, k, ln in ((r1, k1, a), (r2, k2, b)):
                    moves.append({"room": r, "wall_id": layouts[r].walls[k].wall_id, "shift_m": round(mid - ln["c"], 4),
                                  "declared": True})
                    ln["c"] = mid
    for r in {m["room"] for m in moves}:
        _rebuild(layouts[r], lines[r])
    return moves


def surface_anchors(layouts: dict, clouds: dict) -> dict:
    """Plan position of every opening's and wall damage's near edge (on its wall line), to carry them over when
    walls move (join_declared)."""
    out = {}
    for r, lay in layouts.items():
        walls = {w.wall_id: w for w in lay.walls}
        for o in clouds[r].openings or []:
            if o.wall_id in walls:
                out[("o", r, o.opening_id)] = _point_at(walls[o.wall_id], o.offset_m)
        for d in clouds[r].damage or []:
            if d.surface_id in walls:
                out[("d", r, d.damage_id)] = _point_at(walls[d.surface_id], d.from_left_m)
    return out


def reanchor(layouts: dict, clouds: dict, anchors: dict) -> None:
    """Offsets along the (moved) walls from the anchored plan positions, kept on the wall."""
    for r, lay in layouts.items():
        walls = {w.wall_id: w for w in lay.walls}
        for o in clouds[r].openings or []:
            p = anchors.get(("o", r, o.opening_id))
            if p is not None:
                w = walls[o.wall_id]
                o.offset_m = round(min(max(_along(w, p), 0.0), max(w.length_m - o.width_m, 0.0)), 3)
        for d in clouds[r].damage or []:
            p = anchors.get(("d", r, d.damage_id))
            if p is not None:
                w = walls[d.surface_id]
                d.from_left_m = round(min(max(_along(w, p), 0.0), max(w.length_m - d.width_m, 0.0)), 3)


def _point_at(w, offset: float) -> np.ndarray:
    a, b = np.array(w.start, float), np.array(w.end, float)
    return a + (b - a) / np.linalg.norm(b - a) * offset


def _along(w, p: np.ndarray) -> float:
    a, b = np.array(w.start, float), np.array(w.end, float)
    return float((p - a) @ ((b - a) / np.linalg.norm(b - a)))


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


def merge_open_boundaries(layouts: dict, pairs: list[tuple[str, str]]) -> list[dict]:
    """An open boundary (no wall) has no thickness: move both rooms' faces to their common midline and
    rebuild the polygons, so the rooms meet at one line. pairs: (wall_id, partner wall_id)."""
    lines = {r: _lines(lay) for r, lay in layouts.items() if lay.method == "free_space_carving"}
    where = {w.wall_id: (r, k) for r, lay in layouts.items() for k, w in enumerate(lay.walls)}
    moves, touched = [], set()
    for a_id, b_id in pairs:
        (ra, ka), (rb, kb) = where[a_id], where[b_id]
        if ra not in lines or rb not in lines:
            continue
        mid = (lines[ra][ka]["c"] + lines[rb][kb]["c"]) / 2
        for r, k, wid in ((ra, ka, a_id), (rb, kb, b_id)):
            moves.append({"room": r, "wall_id": wid, "shift_m": round(mid - lines[r][k]["c"], 4)})
            lines[r][k]["c"] = mid
            touched.add(r)
    for r in touched:
        _rebuild(layouts[r], lines[r])
    return moves
