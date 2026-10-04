"""Unseen wall behind the cameras (video_d bedroom, E22f): the room must contain every camera."""

from __future__ import annotations

import pytest

from scan.config import load_config
from scan.layout.room import layout_rooms
from tests.unit.test_layout import synthetic_room

CFG = load_config()
POLY = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.5), (0.0, 2.5)]
# every shot taken from near the west wall (x = 0), looking east: the west wall is never filmed
CAMS = [(0.3, 0.6, 1.4, 0), (0.3, 1.25, 1.4, 20), (0.3, 1.9, 1.4, -20), (0.3, 1.25, 1.4, -30)]


def _x_extent(lay):
    xs = [p[0] for p in lay.polygon]
    return min(xs), max(xs)


def test_unseen_wall_is_placed_behind_the_cameras():
    c = synthetic_room("r", POLY, CAMS)
    lay = layout_rooms({"r": c}, CFG)["r"]
    x0, x1 = _x_extent(lay)
    assert x1 == pytest.approx(3.0, abs=0.05)  # filmed east wall: from data
    assert 0.3 - CFG["layout"]["camera_wall_margin_m"] - 0.02 <= x0 <= 0.12  # behind the cameras, never past them
    assert any("behind the cameras" in w for w in lay.warnings)

    off = load_config(overrides={"layout": {"min_wall_support": 0}})  # every wall counts as seen: no push
    x0_off, _ = _x_extent(layout_rooms({"r": synthetic_room("r", POLY, CAMS)}, off)["r"])
    assert x0_off > x0 + 0.1  # without the rule the outline stops in front of the cameras
