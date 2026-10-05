"""Tier 2 step 2.1: video tier detection, keyframe selection."""

from __future__ import annotations

import numpy as np
import pytest

from scan.config import load_config
from scan.errors import InputError
from scan.io.ingest import ingest, resolve_tier
from scan.io.video import keyframes

CFG = load_config()


def test_per_room_clips_detected_as_video(tmp_path):
    (tmp_path / "video" / "hall").mkdir(parents=True)
    (tmp_path / "video" / "hall" / "IMG_1.MOV").write_bytes(b"x")
    assert resolve_tier(tmp_path, None) == "video"


def test_single_walkthrough_not_supported_yet(tmp_path):
    (tmp_path / "video").mkdir()
    (tmp_path / "video" / "walkthrough.mov").write_bytes(b"x")
    with pytest.raises(InputError, match="room segmentation"):
        ingest(tmp_path, None, CFG)


def test_keyframes_one_sharpest_per_slice_and_skip_blur():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    blur = np.full((40, 40, 3), 128, np.uint8)
    frames = np.stack([sharp if i in (2, 13, 27) else blur for i in range(30)])
    picks, _s = keyframes(frames, 3, min_sharpness=30)
    assert picks == [2, 13, 27]
    picks, _ = keyframes(np.stack([blur] * 10), 3, min_sharpness=30)
    assert picks == []  # all slices blurry -> no keyframes


def test_keyframes_min_gap_avoids_near_duplicates():
    sharp = np.random.default_rng(0).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    softer = (sharp // 2 + 64).astype(np.uint8)  # same texture, lower contrast: sharp, but less so
    blur = np.full((40, 40, 3), 128, np.uint8)
    frames = np.stack([sharp if i in (9, 10) else softer if i == 15 else blur
                       for i in range(20)])  # sharpest frames 9 and 10 sit on the slice border
    assert keyframes(frames, 2, min_sharpness=30)[0] == [9, 10]
    assert keyframes(frames, 2, min_sharpness=30, min_gap=4)[0] == [9, 15]


def test_link_pairs_finds_the_shared_view_and_rejects_unrelated_frames():
    import cv2

    from scan.io.video import link_features, link_pairs

    def texture(seed):
        t = np.random.default_rng(seed).integers(0, 256, (60, 40), dtype=np.uint8)
        t = cv2.resize(t, (400, 600), interpolation=cv2.INTER_CUBIC)
        return np.dstack([t] * 3)

    scene, other = texture(1), texture(2)
    H = np.array([[0.9, 0.05, 20], [-0.04, 0.95, 15], [1e-4, 0, 1]])
    seen_from_b = cv2.warpPerspective(scene, H, (400, 600))  # the same wall seen from the other room
    blur = np.full_like(scene, 128)
    a = np.stack([blur, scene, blur])
    b = np.stack([other, blur, seen_from_b])
    sharp = lambda x: np.array([0.0 if f.std() < 1 else 100.0 for f in x])
    fa = link_features(a, sharp(a), range(3), 400, 30)
    fb = link_features(b, sharp(b), range(3), 400, 30)
    pairs = link_pairs(fa, fb, min_inliers=30)
    assert pairs and pairs[0][1:] == (1, 2)  # frame 1 of A <-> frame 2 of B
    assert not link_pairs(fa, [x for x in fb if x[0] == 0], min_inliers=30)  # unrelated texture: no link


def test_quicktime_focal_from_the_video_track_meta(tmp_path):
    """iPhone videos keep camera facts in the video track's own `meta` atom (keys + ilst), E22m."""
    import struct

    from scan.io.quicktime import focal_35mm

    def atom(t: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", 8 + len(payload)) + t + payload

    name = b"com.apple.quicktime.camera.focal_length.35mm_equivalent"
    keys = atom(b"keys", b"\0\0\0\0" + struct.pack(">I", 1) + struct.pack(">I", 8 + len(name)) + b"mdta" + name)
    ilst = atom(b"ilst", atom(struct.pack(">I", 1), atom(b"data", struct.pack(">I", 1) + b"\0\0\0\0" + b"27")))
    f = tmp_path / "clip.mov"
    f.write_bytes(atom(b"ftyp", b"qt  ") + atom(b"moov", atom(b"trak", atom(b"meta", atom(b"hdlr", b"\0" * 24) + keys + ilst))))
    assert focal_35mm(f) == 27.0
    g = tmp_path / "plain.mov"
    g.write_bytes(atom(b"ftyp", b"qt  ") + atom(b"moov", atom(b"trak", b"")))
    assert focal_35mm(g) is None
