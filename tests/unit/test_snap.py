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


def test_far_apart_walls_untouched():
    a = layout("a", [(0, 0), (3, 0), (3, 3), (0, 3)], support=1000)
    b = layout("b", [(3.8, 0), (6, 0), (6, 3.6), (3.8, 3.6)], support=1000)  # 80 cm away, 60 cm higher
    moves = snap_walls({"a": a, "b": b}, CFG)
    assert all(abs(m["shift_m"]) < 1e-9 for m in moves if m["room"] == "a")
    assert wall_at(b, "y", 3.6) is not None and wall_at(b, "x", 3.8) is not None
