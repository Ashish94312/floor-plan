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
    labels, _, cell, cam_lab = free_space_rooms(rays, cams, CFG)
    assert len(set(cam_lab[:3])) == 1 and len(set(cam_lab[3:])) == 1 and cam_lab[0] != cam_lab[3]
    area = {k: (labels == k).sum() * cell**2 for k in (cam_lab[0], cam_lab[3])}
    assert area[cam_lab[0]] == pytest.approx(12.0, rel=0.1) and area[cam_lab[3]] == pytest.approx(8.7, rel=0.1)


def test_wide_opening_stays_one_space():
    rays, cams = _two_rooms((0.4, 2.6))  # 2.2 m opening: open plan, not a doorway
    _, _, _, cam_lab = free_space_rooms(rays, cams, CFG)
    assert len(set(cam_lab)) == 1


def test_lintel_over_wide_opening_splits():
    # 1.5 m opening: wider than a door, so one space by width alone; wall seen above door height over it (a lintel)
    # means a doorway with a wall above it, not an open side
    rays, cams = _two_rooms((0.75, 2.25))
    assert len(set(free_space_rooms(rays, cams, CFG)[3])) == 1
    y = np.arange(0.75, 2.25, 0.01)
    lintel = np.vstack([np.c_[np.full_like(y, 4.05), y, np.full_like(y, z)] for z in (2.3, 2.4)])  # >= 2 per cell
    _, _, _, cam_lab = free_space_rooms(rays, cams, CFG, lintel)
    assert cam_lab[0] != cam_lab[3] and len(set(cam_lab[:3])) == 1 and len(set(cam_lab[3:])) == 1


def test_region_resampled_to_layout_grid():
    from scan.layout.room import region_on_grid

    labels = np.zeros((20, 40), np.int32)  # 5 cm cells from (0, 0): label 3 on x 0.5-1.0, y 0.25-0.75
    labels[5:15, 10:20] = 3
    own = region_on_grid((labels, np.array([0.0, 0.0]), 0.05, 3), np.array([-0.1, 0.0]), (110, 50), 0.02)
    ys, xs = np.nonzero(own)
    assert own.sum() * 0.02**2 == pytest.approx(0.5 * 0.5, rel=0.1)
    assert -0.1 + xs.min() * 0.02 == pytest.approx(0.5, abs=0.03) and ys.max() * 0.02 == pytest.approx(0.73, abs=0.03)
