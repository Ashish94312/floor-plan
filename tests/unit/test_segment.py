"""Room segmentation of one walk-through (scan/layout/segment.py, E25): rooms split at door-wide necks only."""

import numpy as np
import pytest

from scan.config import load_config
from scan.layout.segment import free_space_rooms

CFG = load_config()


def _perimeter(x0, x1, y0, y1, step=0.02):
    t = np.arange(0, 1, step / max(x1 - x0, y1 - y0))
    return np.vstack([np.c_[x0 + (x1 - x0) * t, np.full_like(t, y0)], np.c_[x0 + (x1 - x0) * t, np.full_like(t, y1)],
                      np.c_[np.full_like(t, x0), y0 + (y1 - y0) * t], np.c_[np.full_like(t, x1), y0 + (y1 - y0) * t]])


def _two_rooms(gap: tuple[float, float]):
    """Rooms A x 0-4 and B x 4.1-7 (y 0-3), wall between with an opening at y in gap. Each camera sees its own
    room's walls and, through the opening, the other room's far wall. Rays: (camera xy, surface points xyz)."""
    a, b = _perimeter(0, 4, 0, 3), _perimeter(4.1, 7, 0, 3)
    rays = []
    ys = np.linspace(0, 3, 300)
    for cx, own, far_x in ((2.0, a, 7.0), (5.5, b, 0.0)):
        far = np.c_[np.full_like(ys, far_x), ys]
        for cy in (1.2, 1.5, 1.8):  # a few camera positions per room
            y_at_wall = cy + (far[:, 1] - cy) * (4.05 - cx) / (far_x - cx)
            P = np.vstack([own, far[(y_at_wall > gap[0]) & (y_at_wall < gap[1])]])
            rays.append((np.array([cx, cy]), np.c_[P, np.ones(len(P))]))
    return rays, np.array([r[0] for r in rays])


def test_door_splits_two_rooms():
    rays, cams = _two_rooms((1.1, 1.9))  # 0.8 m door
    labels, _, cell, cam_lab, _ = free_space_rooms(rays, cams, CFG)
    assert len(set(cam_lab[:3])) == 1 and len(set(cam_lab[3:])) == 1 and cam_lab[0] != cam_lab[3]
    area = {k: (labels == k).sum() * cell**2 for k in (cam_lab[0], cam_lab[3])}
    assert area[cam_lab[0]] == pytest.approx(12.0, rel=0.1) and area[cam_lab[3]] == pytest.approx(8.7, rel=0.1)


def test_wide_opening_stays_one_space():
    rays, cams = _two_rooms((0.4, 2.6))  # 2.2 m opening: open plan, not a doorway
    _, _, _, cam_lab, _ = free_space_rooms(rays, cams, CFG)
    assert len(set(cam_lab)) == 1


def test_lintel_over_wide_opening_splits():
    # 1.5 m opening: wider than a door, so one space by width alone; wall seen above door height over it (a lintel)
    # means a doorway with a wall above it, not an open side
    rays, cams = _two_rooms((0.75, 2.25))
    assert len(set(free_space_rooms(rays, cams, CFG)[3])) == 1
    y = np.arange(0.75, 2.25, 0.01)
    lintel = np.vstack([np.c_[np.full_like(y, 4.05), y, np.full_like(y, z)] for z in (2.3, 2.4)])  # >= 2 per cell
    _, _, _, cam_lab, _ = free_space_rooms(rays, cams, CFG, lintel)
    assert cam_lab[0] != cam_lab[3] and len(set(cam_lab[:3])) == 1 and len(set(cam_lab[3:])) == 1


def test_region_resampled_to_layout_grid():
    from scan.layout.room import region_on_grid

    labels = np.zeros((20, 40), np.int32)  # 5 cm cells from (0, 0): label 3 on x 0.5-1.0, y 0.25-0.75
    labels[5:15, 10:20] = 3
    own = region_on_grid((labels, np.array([0.0, 0.0]), 0.05, 3), np.array([-0.1, 0.0]), (110, 50), 0.02)
    ys, xs = np.nonzero(own)
    assert own.sum() * 0.02**2 == pytest.approx(0.5 * 0.5, rel=0.1)
    assert -0.1 + xs.min() * 0.02 == pytest.approx(0.5, abs=0.03) and ys.max() * 0.02 == pytest.approx(0.73, abs=0.03)


def test_doorway_from_detector_box():
    from types import SimpleNamespace

    from scan.geometry.cloud import unproject
    from scan.layout.segment import door_segments

    # aligned frame (z up): camera at (0, 0, 1.4) looking along +x; wall x = 2 with a 0.8 m opening (|y| < 0.4)
    # through which the next room's wall at x = 5 is seen
    H, W, f = 48, 64, 40.0
    u, _ = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    y_at_wall = -(u - W / 2) / f * 2.0  # camera x (image right) = world -y
    depth = np.where(np.abs(y_at_wall) < 0.4, 5.0, 2.0)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = [0, -1, 0], [0, 0, -1], [1, 0, 0], [0, 0, 1.4]
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    pts = unproject(depth, K, T)
    views = {"names": np.array(["a", "b"]), "depth": np.stack([depth] * 2), "mask": np.ones((2, H, W), bool),
             "pts": np.stack([pts] * 2)}
    frames = [SimpleNamespace(image_path=SimpleNamespace(name=n), rgb=np.zeros((H, W, 3)), upright_turns=0) for n in "ab"]
    box = {"cls": "door", "score": 0.5, "box": [24.0, 12.0, 40.0, 47.0]}  # hugs the opening
    doors = door_segments(views, frames, {"a": [box], "b": [box]}, CFG)
    assert len(doors) == 1
    a, b, n = doors[0]
    assert n == 2 and a[0] == pytest.approx(2.0, abs=0.05) and b[0] == pytest.approx(2.0, abs=0.05)  # on the wall line
    assert 0.8 <= abs(b[1] - a[1]) <= 1.0  # jamb to jamb, slightly wide (the strips reach past the jambs)
    assert door_segments(views, frames[:1], {"a": [box]}, CFG) == []  # one frame only: no consensus


def test_corridor_between_rooms_is_a_passage():
    # rooms A (x 0-4) and B (x 7.1-11), linked by a 0.9 m corridor (y 1.05-1.95) that reaches into both
    spaces = {"A": (0, 4, 0, 3, (2.0, 1.5)), "corr": (3.5, 7.6, 1.05, 1.95, (5.5, 1.5)), "B": (7.1, 11, 0, 3, (9.0, 1.5))}
    rays = []
    for x0, x1, y0, y1, (cx, cy) in spaces.values():
        P = _perimeter(x0, x1, y0, y1)
        for dx in (-0.3, 0.0, 0.3):
            rays.append((np.array([cx + dx, cy]), np.c_[P, np.ones(len(P))]))
    cams = np.array([r[0] for r in rays])
    labels, _, cell, cam_lab, passages = free_space_rooms(rays, cams, CFG)
    a, corr, b = cam_lab[1], cam_lab[4], cam_lab[7]
    assert len({a, corr, b}) == 3 and passages == {corr}
    assert (labels == corr).sum() * cell**2 > 1.5  # the corridor between the rooms' bodies


def test_overlap_goes_to_the_room_whose_segment_holds_it():
    from scan.layout.room import Layout, _walls_clockwise, resolve_overlaps

    def lay(rid, x0, x1):
        V = np.array([[x0, 0], [x0, 3], [x1, 3], [x1, 0]], float)
        poly, walls = _walls_clockwise(V, [(50, 0.01)] * 4, rid)
        return Layout(rid, poly, walls, abs(x1 - x0) * 3, 2.5, "ok", "free_space_carving", None)

    lays = {"a": lay("a", 0, 3.0), "b": lay("b", 2.8, 6.0)}  # walls moved onto wall points overlap by 0.2 m
    lo, cell = np.array([-0.5, -0.5]), 0.02
    xs = lo[0] + (np.arange(400) + 0.5) * cell
    mask_a = np.tile(xs < 3.0, (200, 1))  # the segments meet at x = 3.0: the strip is a's
    resolve_overlaps(lays, {"a": (mask_a, lo, cell), "b": (~mask_a, lo, cell)})
    xb = [p[0] for p in lays["b"].polygon]
    assert min(xb) == pytest.approx(3.0) and lays["b"].floor_area_m2 == pytest.approx(9.0)
    assert lays["a"].floor_area_m2 == pytest.approx(9.0)  # a keeps it
    cut = [w for w in lays["b"].walls if w.start[0] == pytest.approx(3.0) and w.end[0] == pytest.approx(3.0)]
    assert len(cut) == 1 and cut[0].support == 0  # the new edge: the neighbour's boundary, no wall points of its own
    assert all(w.support == 50 for w in lays["b"].walls if w is not cut[0])


def test_self_crossing_outline_is_repaired():
    from shapely.geometry import Polygon

    from scan.layout.room import Layout, WallSeg, repair_outlines

    # a branching outline that crosses itself (bow tie): keep the larger valid piece
    poly = [(0.0, 0.0), (4.0, 3.0), (4.0, 0.0), (0.0, 2.0)]
    walls = [WallSeg(f"p-W{k}", poly[k - 1], poly[k % 4], 1.0, 30, 0.01) for k in range(1, 5)]
    lays = {"p": Layout("p", poly, walls, 0.0, 2.5, "ok", "free_space_carving", None),
            "ok": Layout("ok", [(5, 0), (5, 2), (7, 2), (7, 0)], [], 4.0, 2.5, "ok", "free_space_carving", None)}
    assert repair_outlines(lays) == ["p"]
    assert Polygon(lays["p"].polygon).is_valid and lays["p"].floor_area_m2 > 0
    assert lays["ok"].polygon == [(5, 0), (5, 2), (7, 2), (7, 0)]  # valid outlines untouched
