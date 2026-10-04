"""Tier 1 step 1.1: ingest, validation and EXIF intrinsics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.ExifTags import IFD

from scan.config import load_config
from scan.errors import InputError
from scan.io.ingest import ingest, resolve_tier
from scan.io.photos import UnreadableImage, intrinsics, load_photo

CFG = load_config()
REPO = Path(__file__).resolve().parents[2]


def jpeg(path: Path, size=(64, 48), f35=26, orientation=1, sharp=True, seed=0, model="iPhone 15"):
    """Small JPEG with iPhone-like EXIF. Noise = sharp, flat grey = blurry."""
    rng = np.random.default_rng(seed)
    w, h = size
    px = rng.integers(0, 256, (h, w, 3), dtype=np.uint8) if sharp else np.full((h, w, 3), 128, np.uint8)
    ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "Apple", model
    if orientation != 1:
        ex[0x0112] = orientation
    if f35 is not None:
        ex.get_ifd(IFD.Exif)[0xA405] = f35
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(px).save(path, exif=ex)
    return path


def room(root: Path, name: str, n: int, **kw) -> Path:
    for i in range(n):
        jpeg(root / "photos" / name / f"IMG_{i:04d}.jpg", seed=hash((name, i)) % 2**32, **kw)
    return root / "photos" / name


# --- intrinsics ---------------------------------------------------------------------------------


def test_intrinsics_use_film_diagonal():
    K = intrinsics(26, 4032, 3024, 43.27)
    assert K[0, 0] == pytest.approx(26 * math.hypot(4032, 3024) / 43.27)  # 3028.4 px, not 26/36*4032
    assert (K[0, 2], K[1, 2]) == (2016, 1512)


def test_load_photo_applies_orientation_and_scales_K(tmp_path):
    p = jpeg(tmp_path / "a.jpg", size=(80, 60), orientation=6)  # stored landscape, shown portrait
    fr, warns = load_photo(p, "bedroom1", CFG)
    assert fr.rgb.shape == (80, 60, 3) and fr.meta.orientation == "portrait"
    assert fr.meta.full_size == (60, 80)
    assert fr.K[0, 0] == pytest.approx(26 * 100 / 43.27)
    assert fr.meta.device == "Apple iPhone 15" and fr.meta.intrinsics_source == "exif_f35" and not warns


def test_load_photo_downsizes_to_working_side(tmp_path):
    p = jpeg(tmp_path / "big.jpg", size=(2048, 1536))
    fr, _ = load_photo(p, "r", CFG)
    assert fr.rgb.shape == (768, 1024, 3)
    assert fr.K[0, 0] == pytest.approx(26 * math.hypot(2048, 1536) / 43.27 / 2)


def test_missing_focal_uses_default_with_warning(tmp_path):
    fr, warns = load_photo(jpeg(tmp_path / "a.jpg", f35=None), "r", CFG)
    assert fr.meta.intrinsics_source == "default" and fr.meta.f35_mm == 26
    assert any("no 35 mm-equivalent focal" in w for w in warns)


def test_non_main_lens_warns(tmp_path):
    _, warns = load_photo(jpeg(tmp_path / "a.jpg", f35=13), "r", CFG)  # 0.5x ultra-wide
    assert any("not the 1x lens" in w for w in warns)


def test_corrupt_file_is_unreadable(tmp_path):
    p = tmp_path / "bad.heic"
    p.write_bytes(b"not an image")
    with pytest.raises(UnreadableImage):
        load_photo(p, "r", CFG)


# --- tier detection -----------------------------------------------------------------------------


def test_tier_detection(tmp_path):
    room(tmp_path / "p", "hall", 2)
    assert resolve_tier(tmp_path / "p", None) == "photo"
    (tmp_path / "v" / "video").mkdir(parents=True)
    (tmp_path / "v" / "video" / "walkthrough.mov").write_bytes(b"x")
    assert resolve_tier(tmp_path / "v", None) == "video"
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "odometry.csv").write_text("t\n")  # bare Stray Scanner export
    assert resolve_tier(tmp_path / "s", None) == "lidar"
    (tmp_path / "l" / "lidar" / "rec1").mkdir(parents=True)
    (tmp_path / "l" / "lidar" / "rec1" / "odometry.csv").write_text("t\n")
    assert resolve_tier(tmp_path / "l", None) == "lidar"


def test_tier_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(InputError, match="No capture found"):
        resolve_tier(tmp_path / "empty", None)
    room(tmp_path / "both", "hall", 2)
    (tmp_path / "both" / "video").mkdir()
    (tmp_path / "both" / "video" / "w.mp4").write_bytes(b"x")
    with pytest.raises(InputError, match="several tiers"):
        resolve_tier(tmp_path / "both", None)
    assert resolve_tier(tmp_path / "both", "video") == "video"
    with pytest.raises(InputError, match="--tier lidar"):
        resolve_tier(tmp_path / "both", "lidar")


# --- photo validation ---------------------------------------------------------------------------


def test_valid_capture(tmp_path):
    room(tmp_path, "bedroom1", 3)
    room(tmp_path, "hall", 2)
    cap = ingest(tmp_path, None, CFG)
    assert cap.tier == "photo" and list(cap.rooms) == ["bedroom1", "hall"]
    assert [len(f) for f in cap.rooms.values()] == [3, 2] and cap.devices == ["Apple iPhone 15"]


def test_too_few_photos_names_the_folder(tmp_path):
    room(tmp_path, "hall", 2)
    room(tmp_path, "kitchen", 1)
    with pytest.raises(InputError, match=r"photos/kitchen: 1 usable photo"):
        ingest(tmp_path, None, CFG)


def test_empty_room_folder(tmp_path):
    room(tmp_path, "hall", 2)
    (tmp_path / "photos" / "kitchen").mkdir()
    with pytest.raises(InputError, match="kitchen: 0 usable"):
        ingest(tmp_path, None, CFG)


def test_no_room_folders(tmp_path):
    jpeg(tmp_path / "photos" / "IMG_1.jpg")
    with pytest.raises(InputError, match="no room folders"):
        ingest(tmp_path, None, CFG)


def test_unreadable_and_non_image_are_skipped_with_warning(tmp_path):
    d = room(tmp_path, "hall", 2)
    (d / "broken.heic").write_bytes(b"junk")
    (d / "notes.txt").write_text("x")
    (d / ".DS_Store").write_bytes(b"x")
    cap = ingest(tmp_path, None, CFG)
    assert len(cap.rooms["hall"]) == 2
    assert any("broken.heic" in w and "skipped" in w for w in cap.warnings)
    assert any("notes.txt" in w for w in cap.warnings)
    assert not any(".DS_Store" in w for w in cap.warnings)


def test_duplicate_within_room_dropped(tmp_path):
    d = room(tmp_path, "hall", 2)
    (d / "IMG_0000 2.jpg").write_bytes((d / "IMG_0000.jpg").read_bytes())  # AirDrop-style copy
    cap = ingest(tmp_path, None, CFG)
    assert len(cap.rooms["hall"]) == 2 and any("duplicate dropped" in w for w in cap.warnings)


def test_duplicate_across_rooms_is_an_error(tmp_path):
    room(tmp_path, "bedroom1", 3)
    room(tmp_path, "hall", 2)
    src = tmp_path / "photos" / "bedroom1" / "IMG_0002.jpg"
    (tmp_path / "photos" / "hall" / "IMG_0002.jpg").write_bytes(src.read_bytes())
    with pytest.raises(InputError, match="more than one room folder"):
        ingest(tmp_path, None, CFG)


def test_mixed_orientation_keeps_majority(tmp_path):
    room(tmp_path, "bedroom1", 3, size=(48, 64))  # portrait
    jpeg(tmp_path / "photos" / "bedroom1" / "IMG_9999.jpg", size=(64, 48), seed=7)  # landscape
    cap = ingest(tmp_path, None, CFG)
    names = [f.image_path.name for f in cap.rooms["bedroom1"]]
    assert "IMG_9999.jpg" not in names and len(names) == 3
    assert any("kept portrait, dropped IMG_9999.jpg" in w for w in cap.warnings)


def test_more_than_max_keeps_sharpest(tmp_path):
    room(tmp_path, "hall", 8)
    for i in range(2):
        jpeg(tmp_path / "photos" / "hall" / f"IMG_9{i}.jpg", sharp=False)  # blurry extras
    cap = ingest(tmp_path, None, CFG)
    names = [f.image_path.name for f in cap.rooms["hall"]]
    assert len(names) == 8 and not any(n.startswith("IMG_9") for n in names)
    assert names == sorted(names)  # filename order kept


def test_bad_room_name_warns(tmp_path):
    room(tmp_path, "Living Room", 2)
    cap = ingest(tmp_path, None, CFG)
    assert any("Living Room" in w and "lowercase" in w for w in cap.warnings)


# --- real capture (skipped when the data bundle isn't present) ----------------------------------


@pytest.mark.skipif(not (REPO / "captures/home01_photo_a/photos").is_dir(), reason="data bundle not present")
def test_home01_photo_a():
    cap = ingest(REPO / "captures/home01_photo_a", None, CFG)
    assert {r: len(f) for r, f in cap.rooms.items()} == {"bedroom1": 7, "hall": 7, "kitchen": 3}
    assert all(f.meta.orientation == "portrait" and f.meta.f35_mm == 26 for fs in cap.rooms.values() for f in fs)
    assert any("dropped IMG_4887.HEIC" in w for w in cap.warnings)


def test_non_main_lens_photos_excluded(tmp_path):
    room(tmp_path, "hall", 3)
    jpeg(tmp_path / "photos" / "hall" / "IMG_9000.jpg", f35=16, seed=9)  # 0.5x ultra-wide + zoom (E20)
    cap = ingest(tmp_path, None, CFG)
    assert len(cap.rooms["hall"]) == 3 and any("IMG_9000.jpg: excluded" in w for w in cap.warnings)
