"""Step 1.8: placing separately reconstructed rooms by visibility-scored door matching."""

from __future__ import annotations

import numpy as np

from scan.config import load_config
from scan.geometry.align import Alignment, room_planes
from scan.geometry.cloud import unproject
from scan.layout.openings import detect_openings
from scan.layout.room import Layout, _walls_clockwise
from scan.stitch.doors import _rot, place_rooms
from scan.types import RoomCloud, Scale

CFG = load_config()
H, IW, IH, F = 2.7, 320, 240, 190.0
# room A x 0..3, room B x 3.12..6 (12 cm partition). Real door in the partition at y 1.0..1.9.
# Decoy: A's west wall has a same-width door at y 2.0..2.9 opening onto a far backdrop (not room B).
WALLS = [  # (segment a, b, holes [(s0, s1, z0, z1)] in metres along the segment)
    ((0, 0), (3, 0), []), ((3, 0), (3, 4), [(1.0, 1.9, 0, 2.05)]), ((3, 4), (0, 4), []), ((0, 4), (0, 0), [(1.1, 2.0, 0, 2.05)]),
    ((3.12, 0), (6, 0), []), ((6, 0), (6, 4), []), ((6, 4), (3.12, 4), []), ((3.12, 4), (3.12, 0), [(2.1, 3.0, 0, 2.05)]),
    ((-6, -3), (9, -3), []), ((9, -3), (9, 7), []), ((9, 7), (-6, 7), []), ((-6, 7), (-6, -3), []),  # far backdrop
]


def cast(cam, yaw):
    y = np.radians(yaw)
    R = np.stack([[np.sin(y), -np.cos(y), 0], [0, 0, -1.0], [np.cos(y), np.sin(y), 0]], 1)
    u, v = np.meshgrid(np.arange(IW) + 0.5, np.arange(IH) + 0.5)
    d = np.stack([(u - IW / 2) / F, (v - IH / 2) / F, np.ones_like(u)], -1) @ R.T
    t = np.full(u.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for zp in (0.0, H):
            tz = (zp - cam[2]) / d[..., 2]
            t = np.where((tz > 0) & (tz < t), tz, t)
        for a, b, holes in WALLS:
            a, b = np.array(a, float), np.array(b, float)
            e = b - a
            den = d[..., 0] * e[1] - d[..., 1] * e[0]
            tw = ((a[0] - cam[0]) * e[1] - (a[1] - cam[1]) * e[0]) / den
            s = ((a[0] - cam[0]) * d[..., 1] - (a[1] - cam[1]) * d[..., 0]) / den
            hz = cam[2] + tw * d[..., 2]
            ok = (tw > 0) & (s >= 0) & (s <= 1) & (hz >= 0) & (hz <= H) & (tw < t)
            L = np.linalg.norm(e)
            for s0, s1, z0, z1 in holes:
                ok &= ~((s * L >= s0) & (s * L <= s1) & (hz >= z0) & (hz <= z1))
            t = np.where(ok, tw, t)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cam
    return t, T


def room(rid, cams, box, M=None):
    """Render a room's views, then express everything in a frame moved by M (its own reconstruction frame)."""
    M = np.eye(4) if M is None else M
    K = np.array([[F, 0, IW / 2], [0, F, IH / 2], [0, 0, 1.0]])
    depth, T = zip(*(cast(np.array(c[:3], float), c[3]) for c in cams))
    depth = np.stack(depth).astype(np.float32)
    T = np.einsum("ij,sjk->sik", M, np.stack(T))
    S = len(cams)
    pts = np.stack([unproject(depth[i], K, T[i]) for i in range(S)]).astype(np.float32)
    conf = (1.0 / (1.0 + np.nan_to_num(depth, posinf=99.0))).astype(np.float32)
    views = {"pts": pts, "depth": depth, "conf": conf, "mask": np.isfinite(depth), "T_wc": T,
             "K_model": np.stack([K] * S), "K_exif": np.stack([K] * S), "rgb": np.zeros((*depth.shape, 3), np.uint8),
             "names": np.array([f"{rid}{i}" for i in range(S)]), "rooms": np.array([rid] * S)}
    P = pts[views["mask"]]
    Pl = P @ np.linalg.inv(M)[:3, :3].T + np.linalg.inv(M)[:3, 3]  # crop in house coordinates
    keep = (Pl[:, 0] > box[0] - 0.1) & (Pl[:, 0] < box[1] + 0.1) & (Pl[:, 1] > -0.1) & (Pl[:, 1] < 4.1)
    P = P[keep][:: max(1, keep.sum() // 150000)]
    c = RoomCloud(rid, P, np.zeros_like(P), [], Scale(1.0, 0.05), np.eye(4), frame_id=rid, views=views)
    c.alignment = Alignment(np.eye(4), 0, 0, 0, 1, 0, 0)
    c.planes = room_planes(P, CFG)
    return c


def layout(rid, poly, support):
    V = np.array(poly, float)
    _, walls = _walls_clockwise(V, [(support, 0.02)] * len(V), rid)
    return Layout(rid, [w.start for w in walls], walls, 0.0, 2.7, "ok", "free_space_carving", None)


def test_door_matching_recovers_placement_and_rejects_decoy():
    M = np.eye(4)
    M[:2, :2] = _rot(1)  # room B reconstructed in its own frame: rotated 90 deg ...
    M[:2, 3] = (10.0, -5.0)  # ... and shifted
    a = room("a", [(0.4, 0.4, 1.4, 45), (2.6, 0.4, 1.4, 120), (2.6, 3.6, 1.4, -150), (0.4, 3.6, 1.4, -30),
                   (1.5, 1.45, 1.4, 0), (1.5, 2.45, 1.4, 180)], (0, 3))
    b = room("b", [(3.5, 0.4, 1.4, 45), (5.6, 0.4, 1.4, 120), (5.6, 3.6, 1.4, -150), (3.5, 3.6, 1.4, -30),
                   (4.6, 1.45, 1.4, 180)], (3.12, 6), M)
    clouds = {"a": a, "b": b}
    # true outlines (this test is about door matching; carving leaks through doors are a separate known issue)
    poly_b = (np.array([(3.12, 0), (3.12, 4), (6, 4), (6, 0)]) @ M[:2, :2].T + M[:2, 3]).tolist()
    layouts = {"a": layout("a", [(0, 0), (0, 4), (3, 4), (3, 0)], 5000), "b": layout("b", poly_b, 5000)}
    ops = detect_openings(layouts, clouds, CFG)
    assert len([o for o in ops["a"] if o.type == "door"]) == 2  # real door + decoy
    place = place_rooms(clouds, layouts, ops, CFG)
    edges = place.pop("_edges")
    assert len(edges) == 1 and place["b"]["placed"]
    # B's frame -> house: inverse of M (rooted at A, whose frame is the house frame here)
    Minv = np.linalg.inv(M)
    assert np.allclose(place["b"]["R"], Minv[:2, :2], atol=1e-6)
    assert np.allclose(place["b"]["t"], Minv[:2, 3], atol=0.08)
