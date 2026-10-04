"""Tier 1 step 1.2: back-projection, cloud building and the model-output cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scan import cache
from scan.config import load_config
from scan.geometry.cloud import build_cloud, unproject, view_points

REPO = Path(__file__).resolve().parents[2]


def K(f=100.0, w=40, h=30):
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def test_unproject_flat_wall_identity_pose():
    d = np.full((30, 40), 2.0)
    P = unproject(d, K(), np.eye(4))
    assert np.allclose(P[..., 2], 2.0)
    # pixel (row 15, col 20) has centre (20.5, 15.5), half a pixel right/below (cx, cy) = (20, 15)
    assert np.allclose(P[15, 20, :2], [2.0 * 0.5 / 100, 2.0 * 0.5 / 100])
    # lateral extent = depth * width / f
    assert P[0, -1, 0] - P[0, 0, 0] == pytest.approx(2.0 * 39 / 100)


def test_unproject_applies_camera_to_world_pose():
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # camera z-axis -> world x-axis
    T[:3, 3] = [1.0, 2.0, 3.0]
    P = unproject(np.full((30, 40), 2.0), K(), T)
    assert np.allclose(P[..., 0], 1.0 + 2.0)  # 2 m along camera z = world +x, plus translation


def test_longer_focal_shrinks_lateral_extent():
    """E12 mechanism: a focal 12% too long compresses everything off the optical axis by 12%."""
    d = np.full((30, 40), 3.0)
    wide = unproject(d, K(100.0), np.eye(4))
    narrow = unproject(d, K(112.0), np.eye(4))
    ext = lambda P: np.ptp(P[..., 1])  # vertical extent ~ ceiling height
    assert ext(narrow) / ext(wide) == pytest.approx(100 / 112)


def fake_pred(S=2, H=30, W=40):
    rng = np.random.default_rng(0)
    depth = np.full((S, H, W), 2.0, np.float32)
    T = np.stack([np.eye(4)] * S)
    pts = np.stack([unproject(depth[i], K(), T[i]) for i in range(S)]).astype(np.float32)
    return {
        "pts": pts,
        "depth": depth,
        "conf": rng.random((S, H, W)).astype(np.float32),
        "mask": np.ones((S, H, W), bool),
        "T_wc": T,
        "K_model": np.stack([K()] * S),
        "K_exif": np.stack([K(90.0)] * S),
        "rgb": np.full((S, H, W, 3), 128, np.uint8),
        "names": np.array([f"v{i}.jpg" for i in range(S)]),
        "rooms": np.array(["r"] * S),
    }


def test_view_points_selects_ray_source():
    p = fake_pred()
    assert np.array_equal(view_points(p, 0, "model"), p["pts"][0])
    exif = view_points(p, 0, "exif")  # f=90 < 100 -> wider
    assert np.ptp(exif[..., 0]) > np.ptp(p["pts"][0][..., 0])
    with pytest.raises(ValueError):
        view_points(p, 0, "bogus")


def test_build_cloud_confidence_filter_and_voxel():
    p = fake_pred()
    pts, cols = build_cloud(p, [0, 1], "model", conf_percentile=30, voxel_m=0)
    n_px = 2 * 30 * 40
    assert len(pts) == pytest.approx(0.7 * n_px, rel=0.02) and cols.max() <= 1.0
    thin, _ = build_cloud(p, [0, 1], "model", conf_percentile=30, voxel_m=0.05)
    assert len(thin) < len(pts) / 2  # two identical views + 5 cm grid merge heavily


def test_cache_roundtrip_and_key(tmp_path):
    cfg = {"cache": {"dir": str(tmp_path)}}
    a = cache.cache_key(images=["x", "y"], f_scale=1.0)
    assert a == cache.cache_key(f_scale=1.0, images=["x", "y"])  # order of kwargs irrelevant
    assert a != cache.cache_key(images=["y", "x"], f_scale=1.0)  # image order matters
    assert a != cache.cache_key(images=["x", "y"], f_scale=0.9)
    p = fake_pred()
    cache.save(cfg, "t", a, p)
    back = cache.load(cfg, "t", a)
    assert all(np.array_equal(p[k], back[k]) for k in p)
    assert cache.load(cfg, "t", "missing") is None
    assert not list(tmp_path.rglob("*.tmp.npz"))


def test_cache_hit_takes_labels_from_frames(tmp_path):
    from scan.geometry.mapanything_backend import _cache_key, predict
    from scan.types import Frame, PhotoMeta

    cfg = load_config(overrides={"cache": {"dir": str(tmp_path)}})
    meta = PhotoMeta("h", (40, 30), "landscape", None, None, 26.0, "exif_f35", None, 100.0)
    frames = [Frame(Path(f"bedroom1/v{i}.jpg"), K(), room_hint="bedroom1", meta=meta) for i in range(2)]
    cache.save(cfg, "mapanything", _cache_key(frames, cfg), fake_pred())  # cached under the old room name "r"

    def no_model():
        raise AssertionError("cache hit expected")

    pred, info = predict(frames, cfg, None, no_model)
    assert info["cache"] == "hit" and list(pred["rooms"]) == ["bedroom1"] * 2  # renamed folder still matches


def test_plan_groups_auto_joint_and_fallback():
    from scan.pipeline import plan_groups
    from scan.types import Capture

    def cap(sizes):
        return Capture("c", Path("."), "photo", rooms={f"r{i}": [object()] * n for i, n in enumerate(sizes)})

    g = {"joint": "auto", "max_joint_views": 16}
    groups, joint = plan_groups(c := cap([7, 7]), g)
    assert joint and list(groups) == ["joint"] and len(groups["joint"]) == 14 and not c.warnings
    groups, joint = plan_groups(c := cap([7, 7, 6]), g)  # 20 views: too many for one run
    assert not joint and list(groups) == ["r0", "r1", "r2"] and "max_joint_views" in c.warnings[0]
    groups, joint = plan_groups(c := cap([7]), g)  # single room: nothing to join
    assert not joint and not c.warnings
    _, joint = plan_groups(cap([7, 7, 6]), {**g, "joint": True})
    assert joint


def _cached_capture_available() -> bool:
    if not (REPO / "captures/home01_photo_a/photos").is_dir():
        return False
    from scan.geometry.mapanything_backend import _cache_key
    from scan.io.ingest import ingest
    from scan.pipeline import plan_groups

    cfg = load_config()
    cap = ingest(REPO / "captures/home01_photo_a", None, cfg)
    groups, _ = plan_groups(cap, cfg["geometry"])
    return all(cache.load(cfg, "mapanything", _cache_key(fs, cfg)) is not None for fs in groups.values())


@pytest.mark.skipif(not _cached_capture_available(), reason="needs home01_photo_a + its cached model outputs")
def test_pipeline_replays_cache_deterministically(tmp_path):
    from scan.pipeline import run

    cfg = load_config()
    _, c1, s1 = run(REPO / "captures/home01_photo_a", cfg, out=tmp_path / "a", log=lambda *_: None)
    _, c2, _ = run(REPO / "captures/home01_photo_a", cfg, out=tmp_path / "b", log=lambda *_: None)
    assert set(c1) == {"bedroom1", "hall", "kitchen"}
    assert all(v["cache"] == "hit" for v in s1["geometry"]["runs"].values())
    for r in c1:
        assert np.array_equal(c1[r].points, c2[r].points)
    for r in ("bedroom1", "hall"):
        assert 100_000 < len(c1[r].points) < 300_000
        assert c1[r].views["depth"].shape[0] == 7
