"""Tier 1 step 1.4: room layout by free-space carving, on ray-cast synthetic rooms with exact answers."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from scan.config import load_config
from scan.geometry.align import Alignment, room_planes
from scan.layout.room import layout_rooms
from scan.types import RoomCloud, Scale

CFG = load_config()
REPO = Path(__file__).resolve().parents[2]
H_ROOM = 2.7
IMG_W, IMG_H, F = 320, 240, 190.0  # ~80 x 65 deg, like the 1x lens at the model's working size


def render(poly, cam, yaw_deg):
    """Ray-cast a z-depth image of a rectilinear room (polygon walls, floor z=0, ceiling H_ROOM)."""
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    R = np.stack([right, down, fwd], 1)  # columns = camera x, y, z in world
    u, v = np.meshgrid(np.arange(IMG_W) + 0.5, np.arange(IMG_H) + 0.5)
    dc = np.stack([(u - IMG_W / 2) / F, (v - IMG_H / 2) / F, np.ones_like(u)], -1)  # camera-frame ray, z = 1
    dw = dc @ R.T
    t = np.full(u.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for zp in (0.0, H_ROOM):
            tz = (zp - cam[2]) / dw[..., 2]
            t = np.where((tz > 0) & (tz < t), tz, t)
        P = np.array(poly + poly[:1], float)
        for a, b in itertools.pairwise(P):
            e = b - a
            den = dw[..., 0] * e[1] - dw[..., 1] * e[0]
            tw = ((a[0] - cam[0]) * e[1] - (a[1] - cam[1]) * e[0]) / den
            s = ((a[0] - cam[0]) * dw[..., 1] - (a[1] - cam[1]) * dw[..., 0]) / den
            hit_z = cam[2] + tw * dw[..., 2]
            ok = (tw > 0) & (s >= 0) & (s <= 1) & (hit_z >= 0) & (hit_z <= H_ROOM) & (tw < t)
            t = np.where(ok, tw, t)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cam
    return t, T  # t is the z-depth because camera-frame rays have z = 1


def synthetic_room(room_id, poly, cams):
    K = np.array([[F, 0, IMG_W / 2], [0, F, IMG_H / 2], [0, 0, 1.0]])
    depth, T_wc = zip(*(render(poly, np.array(c[:3], float), c[3]) for c in cams))
    depth, T_wc = np.stack(depth).astype(np.float32), np.stack(T_wc)
    S = len(cams)
    from scan.geometry.cloud import unproject

    pts = np.stack([unproject(depth[i], K, T_wc[i]) for i in range(S)]).astype(np.float32)
    views = {
        "pts": pts, "depth": depth, "conf": np.ones_like(depth), "mask": np.isfinite(depth),
        "T_wc": T_wc, "K_model": np.stack([K] * S), "K_exif": np.stack([K] * S),
        "rgb": np.zeros((*depth.shape, 3), np.uint8), "names": np.array([f"v{i}" for i in range(S)]),
        "rooms": np.array([room_id] * S),
    }
    P = pts[views["mask"]]
    P = P[:: max(1, len(P) // 150000)]
    c = RoomCloud(room_id, P, np.zeros_like(P), [], Scale(1.0, 0.05), np.eye(4), frame_id=room_id, views=views)
    c.alignment = Alignment(np.eye(4), 0, 0, 0, 1, 0, 0)
    c.planes = room_planes(P, CFG)
    return c


def corner_cams(poly, z=1.4, inset=0.35):
    """Protocol-style shots: from near each corner, looking at the room's centre."""
    P = np.array(poly, float)
    centre = P.mean(0)
    cams = []
    for p in P:
        q = p + inset * np.sign(centre - p)
        yaw = np.degrees(np.arctan2(*(centre - q)[::-1]))
        cams.append((q[0], q[1], z, yaw))
    return cams


def lengths(lay):
    return sorted(w.length_m for w in lay.walls)


def test_rectangle_room():
    poly = [(0, 0), (3, 0), (3, 4), (0, 4)]
    lay = layout_rooms({"r": synthetic_room("r", poly, corner_cams(poly))}, CFG)["r"]
    assert lay.status == "ok" and len(lay.walls) == 4
    assert lengths(lay) == pytest.approx([3, 3, 4, 4], abs=0.03)
    assert lay.floor_area_m2 == pytest.approx(12.0, rel=0.02)
    assert lay.ceiling_height_m == pytest.approx(H_ROOM, abs=0.015)


def test_l_shaped_room():
    poly = [(0, 0), (4, 0), (4, 2), (2.5, 2), (2.5, 3.5), (0, 3.5)]
    cams = [(0.4, 0.4, 1.4, 45), (3.6, 0.4, 1.4, 150), (0.4, 3.1, 1.4, -45), (2.1, 3.1, 1.4, -120),
            (3.6, 1.6, 1.4, -150), (1.2, 1.5, 1.4, 0)]
    lay = layout_rooms({"r": synthetic_room("r", poly, cams)}, CFG)["r"]
    assert lay.status == "ok" and len(lay.walls) == 6
    assert lengths(lay) == pytest.approx(sorted([4, 2, 1.5, 1.5, 2.5, 3.5]), abs=0.03)
    assert lay.floor_area_m2 == pytest.approx(4 * 2 + 2.5 * 1.5, rel=0.02)


def test_walls_listed_clockwise_from_south_wall():
    poly = [(0, 0), (3, 0), (3, 4), (0, 4)]
    lay = layout_rooms({"r": synthetic_room("r", poly, corner_cams(poly))}, CFG)["r"]
    w1 = lay.walls[0]
    assert max(w1.start[1], w1.end[1]) < 0.1  # W1 = south wall (y ~ 0), starting at its west end
    xs = np.array(lay.polygon)
    signed = 0.5 * np.sum(xs[:, 0] * np.roll(xs[:, 1], -1) - np.roll(xs[:, 0], -1) * xs[:, 1])
    assert signed < 0  # clockwise seen from above


def _home01_cached():
    from tests.unit.test_geometry import _cached_capture_available

    return _cached_capture_available()


@pytest.mark.skipif(not (REPO / "captures/home01_photo_a/photos").is_dir(), reason="data bundle not present")
def test_home01_layout_against_tape(tmp_path):
    if not _home01_cached():
        pytest.skip("needs cached model outputs for home01_photo_a")
    from scan.pipeline import run

    _, clouds, _ = run(REPO / "captures/home01_photo_a", CFG, out=tmp_path, log=lambda *_: None)
    bed, hall = clouds["bedroom1"].layout, clouds["hall"].layout  # (kitchen: too few photos to score yet)
    assert len(bed.walls) == 4 and len(hall.walls) == 6
    assert lengths(bed) == pytest.approx([2.39, 2.39, 2.915, 2.915], rel=0.05)
    hall_long = sorted(lengths(hall))[-2:]  # the two long walls (W2/W4 370 and W1 361 incl. notch)
    assert hall_long == pytest.approx([3.61, 3.70], rel=0.05)


def test_rectilinear_snaps_staircase_and_drops_jogs():
    from scan.layout.room import _rectilinear, _vertices

    # a 3 x 2 rectangle traced with a 5 cm jog on the top edge and slightly noisy corners
    c = np.array([(0, 0), (3.0, 0.01), (3.0, 2.0), (1.6, 2.0), (1.6, 2.05), (0.0, 2.05)], float)
    segs = _rectilinear(c, min_edge=0.15)
    V = _vertices(segs)
    assert len(V) == 4  # the 5 cm jog is merged away
    xs, ys = sorted({round(x, 2) for x, _ in V}), sorted({round(y, 2) for _, y in V})
    assert xs == [0.0, 3.0] and len(ys) == 2 and ys[0] == pytest.approx(0.0, abs=0.01) and 2.0 <= ys[1] <= 2.05


def test_small_corner_notch_filled_big_l_kept():
    from scan.layout.room import _fill_small_notches, _rectilinear, _signed_area, _vertices

    notch = np.array([(0, 0), (3, 0), (3, 2.5), (0.8, 2.5), (0.8, 2.0), (0, 2.0)], float)  # 0.8 x 0.5 corner block
    segs, filled = _fill_small_notches(_rectilinear(notch, 0.15), 0.6, 1.0)
    assert len(segs) == 4 and filled == pytest.approx(0.4, abs=0.01)
    ell = np.array([(0, 0), (4, 0), (4, 2), (2.5, 2), (2.5, 3.5), (0, 3.5)], float)  # real L (2.25 m2 concavity)
    segs, filled = _fill_small_notches(_rectilinear(ell, 0.15), 0.6, 1.0)
    assert len(segs) == 6 and filled == 0 and abs(_signed_area(_vertices(segs))) == pytest.approx(4 * 2 + 2.5 * 1.5)


def test_refined_lines_meeting_leave_no_zero_length_wall():
    from scan.layout.room import _drop_short_edges, _signed_area, _vertices

    # 4 x 3 room whose top edge was a jog; refinement moved both top lines to ~y=3, so the jog is 4 cm long
    segs = [["h", 0.0, 4, 80, 0.01], ["v", 4.0, 3, 60, 0.01], ["h", 3.0, 2, 50, 0.01],
            ["v", 2.0, 0.3, 5, 0.02], ["h", 3.04, 2, 10, 0.02], ["v", 0.0, 3, 60, 0.01]]
    out = _drop_short_edges(segs, 0.15)
    V = _vertices([s[:3] for s in out])
    assert len(out) == 4 and abs(_signed_area(V)) == pytest.approx(12.0)  # top line at the better-supported y=3.0
    assert min(np.linalg.norm(V - np.roll(V, 1, 0), axis=1)) > 2.9
