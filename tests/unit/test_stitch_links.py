"""E22n: placing separately reconstructed rooms from link frames (both rooms' clips see the same thing)."""

from __future__ import annotations

import numpy as np

from scan.config import load_config
from scan.geometry.cloud import unproject
from scan.stitch.links import fit_rigid_2d, link_edges
from scan.types import RoomCloud, Scale
from tests.unit.test_stitch_doors import IH, IW, F, cast

CFG = load_config()
K = np.array([[F, 0, IW / 2], [0, F, IH / 2], [0, 0, 1.0]])
TEX = np.random.default_rng(1).integers(0, 256, (400, 400, 3), dtype=np.uint8)


def texture(P: np.ndarray) -> np.ndarray:
    """Colour fixed to the world position (8 cm blocks on every surface): both rooms' cameras see the same pattern."""
    ix = np.floor((P[..., 0] + P[..., 2] * 0.37) / 0.08).astype(int) % 400
    iy = np.floor((P[..., 1] - P[..., 2] * 0.61) / 0.08).astype(int) % 400
    return TEX[ix, iy]


def room(rid: str, cams: list, M: np.ndarray) -> RoomCloud:
    """Room whose model run holds `cams`, expressed in its own reconstruction frame (house frame moved by M)."""
    depth, T = zip(*(cast(np.array(c[:3], float), c[3]) for c in cams))
    depth = np.stack(depth)
    true_pts = np.stack([unproject(np.nan_to_num(d, posinf=50.0), K, t) for d, t in zip(depth, T)])
    T = np.einsum("ij,sjk->sik", M, np.stack(T))
    S = len(cams)
    v = {"pts": np.stack([unproject(depth[i], K, T[i]) for i in range(S)]).astype(np.float32), "depth": depth.astype(np.float32),
         "mask": np.isfinite(depth), "T_wc": T, "K_model": np.stack([K] * S), "K_exif": np.stack([K] * S),
         "rgb": texture(true_pts), "names": np.array([f"{rid}{i}" for i in range(S)]), "rooms": np.array([rid] * S)}
    return RoomCloud(rid, np.zeros((0, 3)), np.zeros((0, 3)), [], Scale(1.0, 0.05), np.eye(4), frame_id=rid, views=v, run_views=v)


def test_fit_rigid_2d_recovers_rotation_and_shift_despite_outliers():
    rng = np.random.default_rng(0)
    B = rng.uniform(-2, 2, (60, 2))
    th = np.radians(93.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    A = B @ R.T + [1.5, -0.7] + rng.normal(0, 0.02, B.shape)
    A[:20] = rng.uniform(-3, 3, (20, 2))  # a third wrong matches
    R_, t_, inl = fit_rigid_2d(A, B, thr=0.1)
    assert inl[20:].all() and inl[:20].sum() <= 2
    assert np.allclose(R_, R, atol=0.02) and np.allclose(t_, [1.5, -0.7], atol=0.03)


def test_link_frames_place_room_b_in_room_a():
    """Room A x 0..3, room B x 3.12..6, door in the partition at y 1.0..1.9. A's link frame looks through the
    door at B's east wall; B's link frame looks at the same wall from inside B. B's reconstruction frame is the
    house frame turned 90 deg and shifted: the link placement must undo exactly that."""
    a = room("A", [(1.0, 1.5, 1.4, 20), (2.0, 1.45, 1.5, 0)], np.eye(4))
    th = np.radians(90)
    M = np.eye(4)
    M[:2, :2] = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    M[:3, 3] = [-1.0, 4.0, 0.0]
    b = room("B", [(5.0, 3.0, 1.5, 200), (3.6, 1.6, 1.5, 5)], M)
    edges = link_edges({"A": a, "B": b}, [("A", "A1", "B", "B1")], CFG)
    assert len(edges) == 1
    e = edges[0]
    assert e["inliers"] >= 30 and e["theta_deg"] == 270.0  # B's frame turned +90 -> turn back by -90
    Minv = np.linalg.inv(M)
    assert np.allclose(e["R"], Minv[:2, :2], atol=1e-9) and np.allclose(e["t"], Minv[:2, 3], atol=0.02)
    assert abs(e["dz_m"]) < 0.02 and abs(e["scale_b_in_a"] - 1) < 0.02
    assert not link_edges({"A": a, "B": b}, [("A", "A0", "B", "B0")], CFG)  # frames that share nothing: no edge


def test_lintel_barrier_stops_carving_through_a_door():
    """Per-room layouts carve through an open 0.9 m door (wider than the 0.70 m neck cut) into the next room.
    Cutting the free space along high wall points (the lintel over the door) keeps each room to itself (E22n)."""
    from shapely.geometry import Polygon

    from scan.config import load_config
    from scan.layout.room import layout_rooms
    from tests.unit.test_stitch_doors import room as door_room

    a = door_room("A", [(0.4, 0.4, 1.5, 45), (0.4, 3.6, 1.5, -45), (2.0, 1.45, 1.5, 0), (2.6, 3.6, 1.5, 225),
                        (2.6, 0.4, 1.5, 135)], (0, 3))
    b = door_room("B", [(3.5, 0.4, 1.5, 45), (5.6, 3.6, 1.5, 225), (5.6, 0.4, 1.5, 135), (3.5, 3.6, 1.5, -45)], (3.12, 6))
    a.frame_id = b.frame_id = "stitched"
    before = layout_rooms({"A": a, "B": b}, CFG)
    after = layout_rooms({"A": a, "B": b}, load_config(overrides={"layout": {"lintel_min_z_m": 2.15}}))

    def overlap(lay):
        return Polygon(lay["A"].polygon).intersection(Polygon(lay["B"].polygon)).area

    assert overlap(before) > 0.3  # A's outline reaches into B through the door
    assert overlap(after) < 0.05
    assert abs(after["A"].floor_area_m2 - 12.0) < 0.5 and abs(after["B"].floor_area_m2 - 11.52) < 0.5


def test_link_measures_the_size_ratio_of_two_rooms():
    """Room B reconstructed 10% too small (its frame scaled 0.9): the link's 3D similarity must say A ~ (1/0.9) B."""
    a = room("A", [(1.0, 1.5, 1.4, 20), (2.0, 1.45, 1.5, 0)], np.eye(4))
    M = np.eye(4)
    M[:3, :3] *= 0.9
    M[:3, 3] = [0.3, -0.2, 0.0]
    b = room("B", [(5.0, 3.0, 1.5, 200), (3.6, 1.6, 1.5, 5)], M)
    e = link_edges({"A": a, "B": b}, [("A", "A1", "B", "B1")], CFG)[0]
    assert e["scale_inliers"] >= 30 and abs(e["scale_b_in_a"] - 1 / 0.9) < 0.02


def test_flat_scale_puts_linked_rooms_on_one_scale():
    from scan.stitch.scale import flat_scale

    edges = [{"a": "A", "b": "B", "scale_b_in_a": 1.2, "scale_inliers": 50},
             {"a": "B", "b": "C", "scale_b_in_a": 0.8, "scale_inliers": 50}]
    # every frame agrees with the links; A's frames say A is 5% too small; D has no link
    votes = {"A": [0.05] * 5, "B": [0.05 + np.log(1.2)] * 5, "C": [0.05 + np.log(0.96)] * 4, "D": [0.10, 0.12, 0.08]}
    out = flat_scale(["A", "B", "C", "D"], votes, edges)
    assert np.isclose(out["B"]["c"] / out["A"]["c"], 1.2) and np.isclose(out["C"]["c"] / out["B"]["c"], 0.8)
    assert np.isclose(out["A"]["c"], np.exp(0.05)) and out["A"]["sigma_log"] < 1e-9
    assert out["D"]["group"] != out["A"]["group"] and np.isclose(out["D"]["c"], np.exp(0.10))
    # one room's frames biased +20%: outvoted by the median of the group, it shows as spread
    votes["B"] = [0.05 + np.log(1.2) + 0.2] * 5
    out = flat_scale(["A", "B", "C"], votes, edges)
    assert np.isclose(out["A"]["c"], np.exp(0.05))
    assert np.isclose(out["A"]["sigma_log"], np.sqrt(0.2**2 / 3))  # between-room disagreement, not hidden by the median
