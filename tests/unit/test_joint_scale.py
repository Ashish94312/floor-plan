"""E23b: rooms of separate model runs put on one scale by one run over all rooms."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scan.stitch.joint_scale import joint_frames, room_factors


def _frame(room, t, link=False):
    return SimpleNamespace(image_path=Path(f"{room}_t{t:06.2f}"), timestamp=t, link=link, room_hint=room)


def test_joint_frames_keep_links_and_spread_the_rest_evenly():
    cap = SimpleNamespace(rooms={
        "a": [_frame("a", t) for t in range(12)] + [_frame("a", 20.5, link=True)],
        "b": [_frame("b", t) for t in range(12)] + [_frame("b", 30.5, link=True)],
    })
    fs = joint_frames(cap, 10)
    assert len(fs) == 10 and sum(f.link for f in fs) == 2  # both links, then 4 own frames per room
    a_own = [f.timestamp for f in fs if f.room_hint == "a" and not f.link]
    assert a_own[0] == 0 and a_own[-1] == 11  # spread over the whole clip


def test_room_factor_is_the_depth_ratio_on_shared_frames():
    rng = np.random.default_rng(0)
    d = rng.uniform(1, 4, (3, 20, 30)).astype(np.float32)
    mask = np.ones_like(d, bool)
    joint = {"names": np.array(["a1", "a2", "b1"]), "depth": d, "mask": mask}
    runs = {
        "a": {"names": np.array(["a1", "a2", "a3"]), "depth": np.stack([d[0] / 1.1, d[1] / 1.1, d[0]]), "mask": np.ones((3, 20, 30), bool)},
        "b": {"names": np.array(["b1"]), "depth": d[2:] * 1.05, "mask": np.ones((1, 20, 30), bool)},
        "c": {"names": np.array(["c1"]), "depth": d[:1], "mask": np.ones((1, 20, 30), bool)},
    }
    f = room_factors(joint, runs, min_pixels=100)
    assert abs(f["a"]["c"] - 1.1) < 1e-4 and f["a"]["frames"] == 2  # room a's run is 10% small: x1.1
    assert abs(f["b"]["c"] - 1 / 1.05) < 1e-4
    assert "c" not in f  # no frame in the joint run: left as it is
