"""Structural wall snapping (D25): shared partitions and collinear outer walls in a shared frame."""

from __future__ import annotations

import numpy as np
import pytest

from scan.config import load_config
from scan.layout.room import Layout, _walls_clockwise
from scan.stitch.snap import snap_walls

CFG = load_config()
T = CFG["stitch"]["wall_thickness_m"]


def layout(room, poly, support):
    V = np.array(poly, float)
    _, walls = _walls_clockwise(V, [(support, 0.02)] * len(V), room)
    P = np.array([w.start for w in walls])
    area = abs(0.5 * float(np.dot(P[:, 0], np.roll(P[:, 1], -1)) - np.dot(np.roll(P[:, 0], -1), P[:, 1])))
    return Layout(room, [w.start for w in walls], walls, area, 2.7, "ok", "free_space_carving", None)


def wall_at(lay, axis, value, tol=0.02):
    """The wall whose fixed coordinate (x for vertical, y for horizontal) is near value."""
    for w in lay.walls:
        horiz = abs(w.end[1] - w.start[1]) < abs(w.end[0] - w.start[0])
        k = 1 if horiz else 0
        if ("y" if horiz else "x") == axis and abs(w.start[k] - value) < tol:
            return w
    return None


def test_partition_gap_closed_to_wall_thickness():
    # room A x 0..3, room B x 3.30..6: a 30 cm gap where one partition should be. A is better seen.
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=5000)
    b = layout("b", [(3.3, 0), (6, 0), (6, 3), (3.3, 3)], support=500)
    snap_walls({"a": a, "b": b}, CFG)
    assert wall_at(a, "x", 3.0) is not None  # better-seen face stays
    assert wall_at(b, "x", 3.0 + T) is not None  # other face moved to one wall thickness away
    assert b.floor_area_m2 == pytest.approx((6 - 3 - T) * 3, abs=1e-6)


def test_collinear_outer_walls_snap_to_better_seen():
    # north walls side by side: A at y=3.00 (well seen), B at y=3.15 (poorly seen)
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=8000)
    b = layout("b", [(3 + T, 0), (6, 0), (6, 3.15), (3 + T, 3.15)], support=300)
    snap_walls({"a": a, "b": b}, CFG)
    assert wall_at(b, "y", 3.0) is not None and wall_at(a, "y", 3.0) is not None


def test_crossed_partition_with_an_unseen_face_takes_the_seen_face():
    # bedroom-like room B's west face (no wall points: pushed behind a camera in the doorway) lies 0.185 m INSIDE
    # room A, whose east face is well seen (E23a): the rooms overlap; B's face goes to A's face + one wall thickness
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=5000)
    b = layout("b", [(2.815, 0), (6, 0), (6, 3), (2.815, 3)], support=5000)
    wall_at(b, "x", 2.815).support = 4
    snap_walls({"a": a, "b": b}, CFG)
    assert wall_at(a, "x", 3.0) is not None and wall_at(b, "x", 3.0 + T) is not None


def test_crossed_seen_faces_beyond_tolerance_untouched():
    # two well-seen faces crossed by 0.30 m: their wall centres disagree by 0.42 m, more than snap_tolerance_m
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=5000)
    b = layout("b", [(2.7, 0), (6, 0), (6, 3), (2.7, 3)], support=5000)
    snap_walls({"a": a, "b": b}, CFG)
    assert wall_at(b, "x", 2.7) is not None


def test_far_apart_walls_untouched():
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=1000)
    b = layout("b", [(3.8, 0), (6, 0), (6, 3.6), (3.8, 3.6)], support=1000)  # 80 cm away, 60 cm higher
    moves = snap_walls({"a": a, "b": b}, CFG)
    assert all(abs(m["shift_m"]) < 1e-9 for m in moves if m["room"] == "a")
    assert wall_at(b, "y", 3.6) is not None and wall_at(b, "x", 3.8) is not None


# ---- open boundaries (no wall between two rooms: open kitchen, archway) --------------------------

from scan.layout.room import classify_open_walls
from scan.stitch.snap import merge_open_boundaries


def wall_points_on(lay, skip=(), step=0.02):
    """Dense vertical-surface points on every wall of `lay` except the ids in `skip`."""
    pts = []
    for w in lay.walls:
        if w.wall_id in skip:
            continue
        a, b = np.array(w.start), np.array(w.end)
        L = np.linalg.norm(b - a)
        for s_ in np.arange(0, L, step):
            xy = a + (b - a) * s_ / L
            for z in (0.5, 1.0, 1.5, 2.0, 2.5):
                pts.append((xy[0], xy[1], z))
    return np.array(pts)


def two_rooms(open_on_a=True, open_on_b=True):
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=5000)
    b = layout("b", [(3 + T, 0), (6, 0), (6, 3), (3 + T, 3)], support=5000)
    a_shared = wall_at(a, "x", 3.0).wall_id
    b_shared = wall_at(b, "x", 3.0 + T).wall_id
    a._debug = {"walls_pts": wall_points_on(a, skip=(a_shared,) if open_on_a else ())}
    b._debug = {"walls_pts": wall_points_on(b, skip=(b_shared,) if open_on_b else ())}
    return a, b, a_shared, b_shared


def test_open_boundary_detected_and_merged_to_one_line():
    a, b, sa, sb = two_rooms()
    opened, pairs = classify_open_walls({"a": a, "b": b}, CFG)
    assert set(opened) == {sa, sb}
    merge_open_boundaries({"a": a, "b": b}, pairs)
    xa, xb = wall_at(a, "x", 3.0 + T / 2, tol=0.01), wall_at(b, "x", 3.0 + T / 2, tol=0.01)
    assert xa is not None and xb is not None  # both faces on the common midline: no wall thickness


def test_boundary_seen_as_wall_by_one_room_stays_a_wall():
    a, b, _, _ = two_rooms(open_on_a=True, open_on_b=False)  # room a never looked at it; b sees a wall
    opened, _ = classify_open_walls({"a": a, "b": b}, CFG)
    assert opened == []


def test_poorly_seen_outer_wall_is_not_open():
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=5000)
    north = wall_at(a, "y", 3.0).wall_id
    a._debug = {"walls_pts": wall_points_on(a, skip=(north,))}
    opened, _ = classify_open_walls({"a": a}, CFG)
    assert opened == []  # nothing behind it: poorly seen, not an open boundary
