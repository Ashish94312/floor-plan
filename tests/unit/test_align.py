"""Tier 1 step 1.3: alignment on a synthetic room with a known answer."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scan.config import load_config
from scan.geometry.align import apply, estimate_alignment, room_planes

CFG = load_config()
W, L, H = 3.0, 4.0, 2.7


def box_room(rng, ceiling=True, doubled_floor=0.06, step=0.03, noise=0.004):
    """Points on floor, ceiling, 4 walls, plus a bed (top 0.55 m) and part of the floor doubled."""
    g = lambda a, b: np.arange(a, b, step)
    X, Y = np.meshgrid(g(0, W), g(0, L))
    floor = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    parts = [floor]
    copy = floor[floor[:, 0] < W / 2].copy()  # one view's floor sits a few cm high (E15 doubling)
    copy[:, 2] += doubled_floor
    parts.append(copy[::2])
    if ceiling:
        parts.append(np.c_[X.ravel(), Y.ravel(), np.full(X.size, H)])
    for x0, x1, y0, y1 in [(0, W, 0, 0), (0, W, L, L), (0, 0, 0, L), (W, W, 0, L)]:
        length = np.hypot(x1 - x0, y1 - y0)
        s, Z = np.meshgrid(g(0, length) / length, g(0, H))  # same metric spacing as floor/ceiling
        x = x0 + (x1 - x0) * s.ravel()
        y = y0 + (y1 - y0) * s.ravel()
        parts.append(np.c_[x, y, Z.ravel()])
    BX, BY = np.meshgrid(g(0.2, 1.6), g(0.2, 2.2))
    parts.append(np.c_[BX.ravel(), BY.ravel(), np.full(BX.size, 0.55)])  # bed top
    P = np.concatenate(parts)
    return P + rng.normal(0, noise, P.shape)


def cameras(n=6):
    """Phones held upright at 1.4 m: camera x = world x, camera y (image down) = -z, camera z = +y."""
    T = np.repeat(np.eye(4)[None], n, 0)
    T[:, :3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    T[:, :3, 3] = np.c_[np.linspace(0.5, 2.5, n), np.full(n, 0.5), np.full(n, 1.4)]
    return T


def disturb(P, T, yaw=30, tilt=3, shift=(1.0, -2.0, 0.7)):
    R = Rotation.from_euler("zx", [yaw, tilt], degrees=True).as_matrix()
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, shift
    return apply(M, P), np.einsum("ij,sjk->sik", M, T)


def test_recovers_floor_ceiling_and_wall_axes():
    rng = np.random.default_rng(0)
    P, T = disturb(box_room(rng), cameras())
    a = estimate_alignment(P, T, CFG)
    A = apply(a.T, P)
    planes = room_planes(A, CFG)
    assert planes.status == "ok"
    assert planes.floor_z == pytest.approx(0.0, abs=0.02)
    assert planes.ceiling_height_m == pytest.approx(H, abs=0.015)  # G2 tolerance, despite bed + doubled floor
    assert a.tilt_correction_deg < 0.5  # cameras were level with the true floor
    assert a.manhattan_support > 0.9
    # walls axis-aligned: the x = 0 wall's points share one x to within noise
    band = A[(A[:, 2] > 1.0) & (A[:, 2] < 2.4)]
    xs = np.sort(band[:, 0])
    left = xs[: len(xs) // 10]
    assert np.std(left) < 0.02


def test_cameras_define_up_even_when_room_is_flipped():
    rng = np.random.default_rng(1)
    P, T = disturb(box_room(rng), cameras(), yaw=-75, tilt=-5)
    a = estimate_alignment(P, T, CFG)
    A = apply(a.T, P)
    cams = np.einsum("ij,sjk->sik", a.T, T)[:, :3, 3]
    assert np.all(cams[:, 2] > 1.0)  # cameras end up ~1.4 m above the floor, not below it
    assert room_planes(A, CFG).ceiling_height_m == pytest.approx(H, abs=0.015)


def test_missing_ceiling_is_partial_with_warning():
    rng = np.random.default_rng(2)
    P, T = disturb(box_room(rng, ceiling=False), cameras())
    a = estimate_alignment(P, T, CFG)
    planes = room_planes(apply(a.T, P), CFG)
    assert planes.status == "partial" and planes.ceiling_height_m is None
    assert "ceiling not visible" in planes.warnings[0]


def test_shared_floor_replaces_an_unseen_room_floor():
    from scan.geometry.align import RoomPlanes, use_shared_floor

    p = RoomPlanes(floor_z=0.22, ceiling_z=2.74, ceiling_height_m=2.52, floor_rms_m=0.02, ceiling_rms_m=0.02, status="ok")
    use_shared_floor(p, tol_m=0.08)
    assert p.floor_z == 0.0 and p.ceiling_height_m == pytest.approx(2.74)
    assert "shared floor" in p.warnings[0]
    q = RoomPlanes(floor_z=0.03, ceiling_z=2.80, ceiling_height_m=2.77, floor_rms_m=0.02, ceiling_rms_m=0.02, status="ok")
    use_shared_floor(q, tol_m=0.08)
    assert q.ceiling_height_m == 2.77 and not q.warnings  # floor seen: untouched


def test_ceiling_mismatch_is_warned_not_overwritten():
    from types import SimpleNamespace

    from scan.geometry.align import RoomPlanes, ceiling_consistency

    def room(h):
        return SimpleNamespace(planes=RoomPlanes(0.0, h, h, 0.01, 0.01, "ok"))

    rooms = {"bed": room(2.78), "hall": room(2.80), "kitchen": room(2.52)}
    ceiling_consistency(rooms, tol_m=0.10)
    assert rooms["kitchen"].planes.ceiling_height_m == 2.52 and "differs" in rooms["kitchen"].planes.warnings[0]
    assert not rooms["hall"].planes.warnings
