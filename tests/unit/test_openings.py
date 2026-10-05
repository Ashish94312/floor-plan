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


def test_no_openings_on_an_unseen_wall():
    """A wall with almost no points (never filmed, placed behind the cameras) gives no openings: it would
    otherwise read as one wide doorway (video_d bedroom, E22j)."""
    c = room_with_holes([DOOR, WINDOW])
    door_wall = next(o.wall_id for o in detect_openings({"r": c.layout}, {"r": c}, CFG)["r"] if o.type == "door")
    next(w for w in c.layout.walls if w.wall_id == door_wall).support = CFG["layout"]["min_wall_support"] - 1
    ops = detect_openings({"r": c.layout}, {"r": c}, CFG)["r"]
    assert [o.type for o in ops] == ["window"]


def _votes(top_m, above):
    """Wall vote grid (2 cm x 5 cm cells), 2 m wide, 2.5 m high: see-through over u 0.5..1.3 m from the floor up to
    top_m, then wall (`above` = "wall") or nothing seen ("none") above it; wall votes elsewhere."""
    oc = CFG["openings"]
    nu, nz = round(2.0 / oc["cell_u_m"]), round(2.5 / oc["cell_z_m"])
    open_v, wall_v = np.zeros((nz, nu)), np.full((nz, nu), 5.0)
    u0, u1, k = round(0.5 / oc["cell_u_m"]), round(1.3 / oc["cell_u_m"]), round(top_m / oc["cell_z_m"])
    open_v[:k, u0:u1], wall_v[:k, u0:u1] = 5, 0
    if above == "none":
        wall_v[k:, u0:u1] = 0
    return open_v, wall_v


def test_low_floor_gap_with_its_top_seen_is_not_a_door():
    from scan.layout.openings import extract

    o, w = _votes(0.75, "wall")  # wall seen above a 0.75 m gap: under furniture, not a doorway (E23a)
    assert extract(None, o, w, 2.0, 2.7, CFG) == []
    o, w = _votes(0.75, "none")  # nothing seen above: the view ended there, the height is unknown -> kept
    assert [d["type"] for d in extract(None, o, w, 2.0, 2.7, CFG)] == ["door"]
    o, w = _votes(2.05, "wall")
    assert [d["type"] for d in extract(None, o, w, 2.0, 2.7, CFG)] == ["door"]


def test_door_is_shared_onto_a_partition_face_too_poorly_seen_to_vote():
    from scan.layout.openings import OpeningSeg, _share_partition_openings
    from scan.layout.room import Layout, WallSeg

    def room(rid, x0, x1, support_east, support_west):
        walls = [WallSeg(f"{rid}-W1", (x0, 0.0), (x1, 0.0), x1 - x0, 500, 0.02),
                 WallSeg(f"{rid}-W2", (x1, 0.0), (x1, 3.0), 3.0, support_east, 0.02),
                 WallSeg(f"{rid}-W3", (x1, 3.0), (x0, 3.0), x1 - x0, 500, 0.02),
                 WallSeg(f"{rid}-W4", (x0, 3.0), (x0, 0.0), 3.0, support_west, 0.02)]
        return Layout(rid, [w.start for w in walls], walls, 9.0, 2.7, "ok", "free_space_carving", None)

    t = CFG["stitch"]["wall_thickness_m"]
    for b_support, copied in ((4, True), (500, False)):
        layouts = {"a": room("a", 0.0, 3.0, 500, 500), "b": room("b", 3.0 + t, 6.0, 500, b_support)}
        door = OpeningSeg("a-O1", "door", "a-W2", 1.0, 0.9, 2.05, None, 4, connects_to="b")
        result = {"a": [door], "b": []}
        _share_partition_openings(layouts, result, t, CFG["layout"]["min_wall_support"])
        assert bool(result["b"]) == copied
        if copied:  # same place: b's west wall runs from y=3 down to y=0, so the door sits 3 - 1.9 = 1.1 m from its start
            d = result["b"][0]
            assert d.wall_id == "b-W4" and d.connects_to == "a" and d.offset_m == pytest.approx(1.1) and d.width_m == pytest.approx(0.9)


def test_window_reflected_in_the_wall_next_to_it_is_dropped():
    from scan.layout.openings import OpeningSeg, reflections
    from scan.layout.room import WallSeg

    # kitchen-like corner (E23a): sink window on the west wall W2 (runs north to the corner at (0, 1.22)), its image
    # in the tiles of the north wall W3 (starts at that corner)
    walls = {"k-W2": WallSeg("k-W2", (0.0, 0.0), (0.0, 1.22), 1.22, 500, 0.02),
             "k-W3": WallSeg("k-W3", (0.0, 1.22), (3.8, 1.22), 3.8, 500, 0.02)}
    real = OpeningSeg("k-O2", "window", "k-W2", 0.33, 0.91, 0.94, 1.14, 4)
    image = OpeningSeg("k-O3", "window", "k-W3", 0.04, 1.38, 0.86, 1.14, 4)
    assert reflections([real, image], walls, {"k-O2": 600.0, "k-O3": 0.0}) == [(image, real)]
    assert reflections([real, image], walls, {"k-O2": 600.0, "k-O3": 50.0}) == []  # both see-through: corner window
    away = OpeningSeg("k-O4", "window", "k-W3", 2.2, 1.0, 0.9, 1.14, 4)  # same height, far from the corner
    assert reflections([real, away], walls, {"k-O2": 600.0, "k-O4": 0.0}) == []
    low = OpeningSeg("k-O5", "window", "k-W3", 0.04, 1.0, 0.5, 0.3, 4)  # at the corner but another height
    assert reflections([real, low], walls, {"k-O2": 600.0, "k-O5": 0.0}) == []
    # neither seen through (grilles and glass read as wall): the dimmer detection is the image
    real.score, image.score = 0.45, 0.365
    assert reflections([real, image], walls, {"k-O2": 0, "k-O3": 0}) == [(image, real)]
    image.score = None  # no score to compare: keep both
    assert reflections([real, image], walls, {"k-O2": 0, "k-O3": 0}) == []


def test_detector_door_with_a_well_seen_solid_wall_behind_it_is_dropped():
    from scan.layout.openings import OpeningSeg, drop_one_sided_doors
    from scan.layout.room import Layout, WallSeg

    def room(rid, x0, x1, support_east, support_west):
        walls = [WallSeg(f"{rid}-W1", (x0, 0.0), (x1, 0.0), x1 - x0, 500, 0.02),
                 WallSeg(f"{rid}-W2", (x1, 0.0), (x1, 3.0), 3.0, support_east, 0.02),
                 WallSeg(f"{rid}-W3", (x1, 3.0), (x0, 3.0), x1 - x0, 500, 0.02),
                 WallSeg(f"{rid}-W4", (x0, 3.0), (x0, 0.0), 3.0, support_west, 0.02)]
        return Layout(rid, [w.start for w in walls], walls, 9.0, 2.7, "ok", "free_space_carving", None)

    t = CFG["stitch"]["wall_thickness_m"]
    det = ["detector (closed door / covered window): box extent, wider interval"]
    for a_support, a_has_door, kept in ((5000, False, False), (5000, True, True), (4, False, True)):
        layouts = {"a": room("a", 0.0, 3.0, a_support, 500), "b": room("b", 3.0 + t, 6.0, 500, 4)}
        door = OpeningSeg("b-O1", "door", "b-W4", 1.0, 0.9, 1.75, None, 2, notes=list(det), score=0.3)
        ops = {"a": [OpeningSeg("a-O1", "door", "a-W2", 1.1, 0.9, 1.8, None, 3, notes=list(det))] if a_has_door else [],
               "b": [door]}
        drop_one_sided_doors(layouts, ops, CFG)
        assert (door in ops["b"]) == kept  # dropped only when a saw its face well and found nothing there
