"""Tier 1 step 1.7: openings by visibility voting, on a ray-cast room with a known door and window."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scan.config import load_config
from scan.geometry.align import Alignment, room_planes
from scan.geometry.cloud import unproject
from scan.layout.openings import detect_openings, reindex_by_main_door
from scan.layout.room import layout_rooms
from scan.types import RoomCloud, Scale

CFG = load_config()
H, IW, IH, F = 2.7, 320, 240, 190.0
ROOM = [(0, 0), (3, 0), (3, 4), (0, 4)]
OUTER = [(-2, -2), (5, -2), (5, 6), (-2, 6)]  # backstop: what you see through the holes
# holes: (wall index in ROOM order, along-wall start, end, z0, z1)
DOOR = (0, 1.0, 1.9, 0.0, 2.05)  # south wall: 0.90 m wide door
WINDOW = (1, 1.5, 2.4, 0.9, 2.1)  # east wall: 0.90 m wide window, sill 0.90


def ray_cast(cam, yaw_deg, holes):
    yaw = np.radians(yaw_deg)
    fwd, right, down = np.array([np.cos(yaw), np.sin(yaw), 0]), np.array([np.sin(yaw), -np.cos(yaw), 0]), np.array([0, 0, -1.0])
    R = np.stack([right, down, fwd], 1)
    u, v = np.meshgrid(np.arange(IW) + 0.5, np.arange(IH) + 0.5)
    dw = np.stack([(u - IW / 2) / F, (v - IH / 2) / F, np.ones_like(u)], -1) @ R.T
    t = np.full(u.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for zp in (0.0, H):
            tz = (zp - cam[2]) / dw[..., 2]
            t = np.where((tz > 0) & (tz < t), tz, t)
        for poly, cut in ((ROOM, holes), (OUTER, [])):
            P = np.array(poly + poly[:1], float)
            for k, (a, b) in enumerate(itertools.pairwise(P)):
                e = b - a
                den = dw[..., 0] * e[1] - dw[..., 1] * e[0]
                tw = ((a[0] - cam[0]) * e[1] - (a[1] - cam[1]) * e[0]) / den
                s = ((a[0] - cam[0]) * dw[..., 1] - (a[1] - cam[1]) * dw[..., 0]) / den
                hz = cam[2] + tw * dw[..., 2]
                ok = (tw > 0) & (s >= 0) & (s <= 1) & (hz >= 0) & (hz <= H) & (tw < t)
                L = np.linalg.norm(e)
                for wk, s0, s1, z0, z1 in cut:
                    if wk == k:
                        ok &= ~((s * L >= s0) & (s * L <= s1) & (hz >= z0) & (hz <= z1))
                t = np.where(ok, tw, t)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cam
    return t, T


def room_with_holes(holes):
    cams = [(0.4, 3.6, 1.4, -60), (2.6, 3.6, 1.4, -120), (1.5, 3.0, 1.4, -90), (0.5, 1.5, 1.4, -10), (2.0, 2.0, 1.4, -45),
            (1.5, 0.6, 1.4, 90)]
    K = np.array([[F, 0, IW / 2], [0, F, IH / 2], [0, 0, 1.0]])
    depth, T = zip(*(ray_cast(np.array(c[:3]), c[3], holes) for c in cams))
    depth, T = np.stack(depth).astype(np.float32), np.stack(T)
    S = len(cams)
    pts = np.stack([unproject(depth[i], K, T[i]) for i in range(S)]).astype(np.float32)
    # confidence falls with distance, as MapAnything's does: the cloud/carving filter then drops most
    # far through-hole pixels, like on real data (high-confidence through-door views can still leak, WORKLOG)
    conf = (1.0 / (1.0 + np.nan_to_num(depth, posinf=99.0))).astype(np.float32)
    views = {"pts": pts, "depth": depth, "conf": conf, "mask": np.isfinite(depth), "T_wc": T,
             "K_model": np.stack([K] * S), "K_exif": np.stack([K] * S), "rgb": np.zeros((*depth.shape, 3), np.uint8),
             "names": np.array([f"v{i}" for i in range(S)]), "rooms": np.array(["r"] * S)}
    inside = pts[views["mask"]]
    inside = inside[(inside[:, 0] > -0.1) & (inside[:, 0] < 3.1) & (inside[:, 1] > -0.1) & (inside[:, 1] < 4.1)]
    inside = inside[:: max(1, len(inside) // 150000)]
    c = RoomCloud("r", inside, np.zeros_like(inside), [], Scale(1.0, 0.05), np.eye(4), frame_id="r", views=views)
    c.alignment = Alignment(np.eye(4), 0, 0, 0, 1, 0, 0)
    c.planes = room_planes(inside, CFG)
    c.layout = layout_rooms({"r": c}, CFG)["r"]
    return c


def test_door_and_window_found_with_sizes():
    c = room_with_holes([DOOR, WINDOW])
    ops = detect_openings({"r": c.layout}, {"r": c}, CFG)["r"]
    doors = [o for o in ops if o.type == "door"]
    wins = [o for o in ops if o.type == "window"]
    assert len(doors) == 1 and len(wins) == 1, [(o.type, o.width_m) for o in ops]
    assert doors[0].width_m == pytest.approx(0.90, abs=0.05) and doors[0].height_m == pytest.approx(2.05, abs=0.1)
    assert wins[0].width_m == pytest.approx(0.90, abs=0.05) and wins[0].sill_m == pytest.approx(0.90, abs=0.1)
    reindex_by_main_door(c.layout, ops)
    assert doors[0].wall_id == "r-W1"  # D11: W1 = the main-door wall


def test_no_holes_no_openings():
    c = room_with_holes([])
    assert detect_openings({"r": c.layout}, {"r": c}, CFG)["r"] == []
