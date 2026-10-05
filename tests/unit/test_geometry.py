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


def test_portrait_turn_round_trip_gives_the_same_world_points():
    from scipy.spatial.transform import Rotation

    from scan.geometry.mapanything_backend import M_TURN, turn_K, unturn_K, unturn_R

    rng = np.random.default_rng(0)
    H, W = 40, 24  # portrait
    depth = rng.uniform(1.0, 4.0, (H, W))
    K = np.array([[30.0, 0, 11.0], [0, 31.0, 21.5], [0, 0, 1]])  # off-centre, fx != fy on purpose
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [10, -20, 35], degrees=True).as_matrix()
    T[:3, 3] = [0.3, -1.2, 1.4]
    P = unproject(depth, K, T)
    # what the model would see and return for the turned view
    Kt = turn_K(K, H)
    Tt = T.copy()
    Tt[:3, :3] = T[:3, :3] @ M_TURN
    Pt = unproject(np.rot90(depth, -1), Kt, Tt)
    assert np.allclose(np.rot90(Pt, 1, axes=(0, 1)), P)  # per-pixel outputs turned back
    assert np.allclose(unturn_K(Kt, W_turned=H), K) and np.allclose(unturn_R(Tt[:3, :3]), T[:3, :3])


def test_depth_focal_fix_recovers_the_scene_a_wrong_focal_squeezed():
    from scan.geometry.cloud import depth_focal_fix

    rng = np.random.default_rng(0)
    depth = rng.uniform(1.0, 4.0, (2, 30, 40))
    T = np.stack([np.eye(4)] * 2)
    T[1, :3, 3] = [0.5, 0.0, -0.3]
    K_true, K_model = K(100.0), K(75.0)  # the model took a wider lens (75 vs 100 px)
    truth = np.stack([unproject(depth[i], K_true, T[i]) for i in range(2)])
    # same image, wrong focal: lateral/vertical extents kept, depth scaled by f'/f
    d_model = depth * 75.0 / 100.0
    pred = {"depth": d_model, "T_wc": T, "K_exif": np.stack([K_true] * 2), "K_model": np.stack([K_model] * 2),
            "pts": np.stack([unproject(d_model[i], K_model, T[i]) for i in range(2)])}
    assert np.allclose(pred["pts"][..., 1] - T[:, None, None, 1, 3], truth[..., 1] - T[:, None, None, 1, 3])  # heights kept
    fixed, r = depth_focal_fix(pred)
    assert np.allclose(r, 100.0 / 75.0) and np.allclose(fixed["pts"], truth, atol=1e-5)
    assert np.allclose(fixed["K_model"], K_true)  # rays='model' stays consistent with the fixed points


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


def test_crop_aspect_keeps_every_pixel_ray():
    from scan.geometry.mapanything_backend import crop_aspect

    img = np.arange(1024 * 576 * 3, dtype=np.uint32).reshape(1024, 576, 3).astype(np.uint8)
    Kf = np.array([[700.0, 0, 288], [0, 700.0, 512], [0, 0, 1]])
    c, Kc = crop_aspect(img, Kf, 4 / 3)
    assert c.shape[:2] == (768, 576) and np.array_equal(c[0], img[128])
    ray = lambda K, u, v: np.linalg.inv(K) @ [u, v, 1.0]
    assert np.allclose(ray(Kc, 10.0, 0.0), ray(Kf, 10.0, 128.0))
    assert crop_aspect(img, Kf, None)[0] is img


def test_pad_frame_content_box_matches_mapanything_preprocessing():
    import torch
    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    from scan.geometry.mapanything_backend import content_box, pad_frame

    rng = np.random.default_rng(0)
    img = rng.integers(40, 255, (1024, 576, 3), dtype=np.uint8)
    K = np.array([[869.0, 0, 288], [0, 869.0, 512], [0, 0, 1]])
    canvas, Kc, off = pad_frame(img, K, 1.25)
    assert canvas.shape[:2] == (1280, 720) and np.array_equal(canvas[off[1]:off[1] + 1024, off[0]:off[0] + 576], img)
    v = preprocess_inputs([{"img": Image.fromarray(canvas), "intrinsics": torch.tensor(Kc, dtype=torch.float32)}])[0]
    Ko = v["intrinsics"][0].numpy().astype(np.float64)
    raw = (v["img"][0].permute(1, 2, 0).numpy())  # normalised image; black canvas -> one constant value
    xa, ya, xb, yb = content_box(Kc, Ko, off, img.shape[:2], raw.shape[:2])
    black = raw[0, 0]
    inside = raw[ya:yb, xa:xb]
    assert not np.any(np.all(np.isclose(inside, black, atol=1e-3), axis=-1))  # no canvas pixel inside the box
    assert np.allclose(raw[ya - 6, xa:xb], black, atol=0.1) and np.allclose(raw[ya:yb, xa - 6], black, atol=0.1)
    s = Ko[0, 0] / Kc[0, 0]
    assert abs((xb - xa) - 576 * s) <= 4 and abs((yb - ya) - 1024 * s) <= 4  # 1-2 px in from each edge


def test_repose_undoes_a_squeezed_focal_and_squeezed_camera_spacing():
    """Model-like errors on a textured synthetic room: each frame's focal 0.80-0.90x the truth with depth squeezed to
    match, camera spacing squeezed 0.77x, rotations off ~1 deg. repose must give back the true cameras and points."""
    import cv2

    from scan.config import load_config
    from scan.geometry.repose import repose
    from scan.types import Frame
    from tests.unit.test_stitch_doors import cast
    from tests.unit.test_stitch_links import K as K_true
    from tests.unit.test_stitch_links import texture

    cams = [(0.8, 0.8, 1.4, 30), (1.1, 0.9, 1.5, 38), (1.4, 1.0, 1.4, 46), (1.0, 1.4, 1.5, 22), (1.4, 1.5, 1.4, 34),
            (1.8, 1.2, 1.5, 55)]
    depth, T = map(np.stack, zip(*(cast(np.array(c[:3], float), c[3]) for c in cams)))
    pts = np.stack([unproject(np.nan_to_num(d, posinf=50.0), K_true, t) for d, t in zip(depth, T)])
    rgb = texture(pts)
    rng = np.random.default_rng(0)
    rho = np.array([0.80, 0.86, 0.90, 0.83, 0.88, 0.81])
    Tm = T.copy()
    mean = T[:, :3, 3].mean(0)
    Tm[:, :3, 3] = mean + 0.77 * (T[:, :3, 3] - mean)
    for i in range(len(cams)):
        Tm[i, :3, :3] = cv2.Rodrigues(rng.normal(0, np.radians(0.7), 3))[0] @ T[i, :3, :3]
    Km = np.stack([K_true.copy() for _ in cams])
    Km[:, 0, 0] *= rho
    Km[:, 1, 1] *= rho
    pred = {"depth": (depth * rho[:, None, None]).astype(np.float32), "mask": np.isfinite(depth), "T_wc": Tm,
            "K_model": Km, "K_exif": np.stack([K_true] * len(cams)), "rgb": rgb, "conf": np.ones_like(depth),
            "pts": pts.astype(np.float32), "names": np.array([f"v{i}" for i in range(len(cams))])}
    frames = [Frame(image_path=Path(f"v{i}"), K=K_true, rgb=rgb[i], timestamp=float(i)) for i in range(len(cams))]
    out, info = repose(pred, frames, load_config())
    assert info["views_reached"] == len(cams) and info["pairs"] >= len(cams), info
    assert info["scale_sigma_log"] < 0.01  # every frame votes for the same room scale here
    assert np.abs(np.linalg.norm(out["T_wc"][:, :3, 3] - T[:, :3, 3], axis=1)).max() < 0.03
    ok = np.isfinite(depth) & (depth < 6)
    assert np.median(np.linalg.norm(out["pts"][ok] - pts[ok], axis=-1)) < 0.03


def test_similarity_ransac_never_refits_on_an_empty_set():
    """Collinear matches: the refit is unstable and may leave no inlier; it must return a model or None, not crash."""
    from scan.geometry.repose import similarity_ransac

    rng = np.random.default_rng(0)
    B = np.stack([np.linspace(0, 2, 40), np.zeros(40), np.full(40, 2.0)], 1) + rng.normal(0, 1e-4, (40, 3))
    A = B * 1.1 + [0.2, 0.0, 0.1]
    res = similarity_ransac(A, B, np.full(40, 0.01))
    assert res is None or (res[3].sum() >= 3 and 0.3 < res[0] < 3.0)


@pytest.mark.parametrize("hw", [(576, 1024), (1024, 576)])
def test_pad_to_aspect_keeps_rays_and_the_content_box_matches_preprocessing(hw):
    import torch
    from mapanything.utils.image import preprocess_inputs
    from PIL import Image

    from scan.geometry.mapanything_backend import content_box, pad_to_aspect

    img = np.random.default_rng(0).integers(40, 255, (*hw, 3), dtype=np.uint8)
    K = np.array([[904.0, 0, hw[1] / 2], [0, 904.0, hw[0] / 2], [0, 0, 1]])
    canvas, Kc, off = pad_to_aspect(img, K, 4 / 3)
    assert np.isclose(max(canvas.shape[:2]) / min(canvas.shape[:2]), 4 / 3, atol=0.01)
    assert np.allclose(np.linalg.inv(Kc) @ [10 + off[0], 20 + off[1], 1], np.linalg.inv(K) @ [10, 20, 1])  # same ray
    v = preprocess_inputs([{"img": Image.fromarray(canvas), "intrinsics": torch.tensor(Kc, dtype=torch.float32)}])[0]
    Ko = v["intrinsics"][0].numpy().astype(np.float64)
    raw = v["img"][0].permute(1, 2, 0).numpy()
    H, W = raw.shape[:2]
    xa, ya, xb, yb = content_box(Kc, Ko, off, hw, (H, W))
    assert 0 <= xa < xb <= W and 0 <= ya < yb <= H  # inside the image (a negative start would wrap the slice)
    black = raw[0, 0] if off[1] else raw[:, 0][0]
    box = raw[ya:yb, xa:xb]
    assert box.shape[:2] == (yb - ya, xb - xa)
    assert not np.any(np.all(np.isclose(box, black, atol=1e-3), axis=-1))
    s = Ko[0, 0] / Kc[0, 0]  # content size at the model's scale, less what preprocessing cropped off
    assert abs((xb - xa) - min(hw[1] * s, W)) <= 4 and abs((yb - ya) - min(hw[0] * s, H)) <= 4
