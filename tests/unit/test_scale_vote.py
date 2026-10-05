"""Photo focal scale vote (FIX_DECLARATION.md): k from the model's focal reading, applied as one similarity."""

import numpy as np
import pytest

from scan.geometry.cloud import focal_scale_vote, scale_pred, unproject


def _pred(ratios: list[float], rooms: list[str]) -> dict[str, np.ndarray]:
    """Views whose model focal is ratio x the EXIF focal, 4x6 depth maps, cameras at distinct positions."""
    n = len(ratios)
    K = np.array([[300.0, 0, 3], [0, 300.0, 2], [0, 0, 1]])
    K_exif = np.repeat(K[None], n, 0)
    K_model = K_exif.copy()
    K_model[:, 0, 0] *= ratios
    K_model[:, 1, 1] *= ratios
    T = np.repeat(np.eye(4)[None], n, 0)
    T[:, :3, 3] = np.arange(3 * n).reshape(n, 3) * 0.1
    depth = np.full((n, 4, 6), 2.5, np.float32)
    pts = np.stack([unproject(depth[i], K_exif[i], T[i]) for i in range(n)]).astype(np.float32)
    return {"K_exif": K_exif, "K_model": K_model, "T_wc": T, "depth": depth, "pts": pts, "rooms": np.array(rooms)}


def test_vote_undoes_one_focal_misreading():
    k, info = focal_scale_vote(_pred([0.96] * 4, ["a", "a", "b", "b"]))
    assert k == pytest.approx(1 / 0.96)
    assert info["room_votes"] == {"a": pytest.approx(1 / 0.96, abs=1e-4), "b": pytest.approx(1 / 0.96, abs=1e-4)}


def test_room_is_the_unit_of_evidence():
    # a room with many misread views (cropped kitchen) cannot outvote two rooms with few views
    k, _ = focal_scale_vote(_pred([1.1, 1.1, 1.2, 1.2] + [0.75] * 6, ["a"] * 2 + ["b"] * 2 + ["c"] * 6))
    assert k == pytest.approx(1 / 1.1)


def test_scale_pred_is_one_similarity():
    p = _pred([1.0, 1.0], ["a", "a"])
    q = scale_pred(p, 1.25)
    assert np.allclose(q["depth"], 1.25 * p["depth"])
    assert np.allclose(q["pts"], 1.25 * p["pts"])
    assert np.allclose(q["T_wc"][:, :3, 3], 1.25 * p["T_wc"][:, :3, 3])
    assert np.allclose(q["T_wc"][:, :3, :3], p["T_wc"][:, :3, :3])
    # points rebuilt from the scaled depth and poses equal the scaled points: the views stay consistent
    assert np.allclose(unproject(q["depth"][1], q["K_exif"][1], q["T_wc"][1]), q["pts"][1], atol=1e-5)
    assert np.allclose(p["T_wc"][:, :3, 3], np.arange(6).reshape(2, 3) * 0.1)  # input untouched
