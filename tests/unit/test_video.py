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
