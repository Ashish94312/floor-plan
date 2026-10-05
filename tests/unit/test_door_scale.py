"""E22z: a room's size from doors whose top was seen."""

from __future__ import annotations

import numpy as np

from scan.config import load_config
from scan.stitch.door_scale import door_scale, lintel_seen


def _grid(top_m, above):
    """Vote grid (rows = 5 cm up, 50 columns): seen through up to top_m, then `above` (wall / none) up to 2.5 m."""
    nz = 50
    open_v, wall_v = np.zeros((nz, 50)), np.zeros((nz, 50))
    k = round(top_m / 0.05)
    open_v[:k] = 20
    if above == "wall":
        wall_v[k:] = 20
    return open_v, wall_v


def test_lintel_seen_only_when_the_wall_above_the_top_was_seen():
    cfg = load_config()
    assert lintel_seen(*_grid(1.90, "wall"), slice(0, 50), 1.90, cfg)
    assert not lintel_seen(*_grid(1.90, "none"), slice(0, 50), 1.90, cfg)  # view ended at the top: not a measurement
    o, w = _grid(1.90, "none")
    w[38] = 20  # a single row of wall above (e.g. a beam edge) is not enough
    assert not lintel_seen(o, w, slice(0, 50), 1.90, cfg)
    o, w = _grid(1.90, "wall")
    o[38], w[38] = 11, 5  # the row just above the reported top straddles the edge (hall door, E22z)
    assert lintel_seen(o, w, slice(0, 50), 1.90, cfg)


def test_door_scale_weighs_the_door_against_the_rooms_own_uncertainty():
    cfg = load_config()
    cfg["stitch"]["door_height_prior_m"] = 2.05
    assert door_scale([], 0.06, cfg) is None
    f, s = door_scale([1.90], 0.06, cfg)
    full = 2.05 / 1.90
    assert 1.0 < f < full  # moves toward the door, not all the way
    assert s < 0.06  # combined evidence is tighter than the room alone
    f_sure, _ = door_scale([1.90], 0.01, cfg)
    f_unsure, _ = door_scale([1.90], 0.5, cfg)
    assert abs(f_sure - 1) < 0.01 and abs(f_unsure - full) < 0.01
    assert door_scale([1.60], 0.05, cfg) is None  # 22% off a 2.0-2.1 m door: not a door (bedroom, E22z)
    assert door_scale([1.60, 1.90], 0.05, cfg)[0] == door_scale([1.90], 0.05, cfg)[0]
