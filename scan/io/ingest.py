"""C1 Ingest: detect the tier from the capture layout and validate it (ARCHITECTURE §10.1).

Layouts:
  photo  <capture>/photos/<room>/*.heic|jpg|png      one folder per room
  video  <capture>/video/<room>/*.mov|mp4            one clip per room
  lidar  <capture>/lidar/[<recording>/]odometry.csv  or a bare Stray Scanner export
         (<capture>/odometry.csv, as in the assignment samples)
"""

from __future__ import annotations

import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scan.errors import InputError
from scan.io.photos import UnreadableImage, load_photo
from scan.types import Capture, Frame, Tier

ROOM_NAME = re.compile(r"^[a-z0-9_]+$")
VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def _subdirs(p: Path) -> list[Path]:
    return sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))


def detect_tiers(root: Path) -> list[Tier]:
    found: list[Tier] = []
    photos = root / "photos"
    if photos.is_dir() and (_subdirs(photos) or any(photos.iterdir())):
        found.append("photo")
    video = root / "video"
    if video.is_dir() and (any(f.suffix.lower() in VIDEO_EXT for f in video.iterdir())
                           or any(f.suffix.lower() in VIDEO_EXT for d in _subdirs(video) for f in d.iterdir())):
        found.append("video")
    lidar = root / "lidar"
    if (root / "odometry.csv").is_file() or (lidar / "odometry.csv").is_file() or (
        lidar.is_dir() and any((d / "odometry.csv").is_file() for d in _subdirs(lidar))
    ):
        found.append("lidar")
    return found


def resolve_tier(root: Path, tier: Tier | None) -> Tier:
    if not root.is_dir():
        raise InputError(f"Capture folder not found: {root}")
    found = detect_tiers(root)
    if tier:
        if tier not in found:
            raise InputError(f"--tier {tier} given, but {root} has no {tier} input. Found: {found or 'nothing'}.")
        return tier
    if not found:
        raise InputError(
            f"No capture found in {root}. Expected photos/<room>/*.heic (photo), "
            f"video/<room>/*.mov (video) or a Stray Scanner export with odometry.csv (LiDAR)."
        )
    if len(found) > 1:
        raise InputError(f"{root} contains several tiers ({', '.join(found)}). Pass --tier to pick one.")
    return found[0]


def ingest(root: Path, tier: Tier | None, cfg: dict[str, Any]) -> Capture:
    t = resolve_tier(root, tier)
    cap = Capture(capture_id=root.name, root=root, tier=t)
    if t == "photo":
        _ingest_photos(cap, cfg)
    elif t == "video":
        from scan.io.video import ingest_video

        ingest_video(cap, cfg)
    else:
        from scan.io.lidar import ingest_lidar

        ingest_lidar(cap, cfg)
    return cap


def _ingest_photos(cap: Capture, cfg: dict[str, Any]) -> None:
    icfg = cfg["ingest"]
    exts = set(icfg["image_extensions"])
    photos = cap.root / "photos"
    room_dirs = _subdirs(photos)
    if not room_dirs:
        raise InputError(f"{photos} has no room folders. Put each room's photos in photos/<room>/, e.g. photos/kitchen/.")
    loose = [f.name for f in photos.iterdir() if f.is_file() and f.suffix.lower() in exts]
    if loose:
        cap.warnings.append(f"{photos}: {len(loose)} photo(s) outside any room folder were ignored: {', '.join(sorted(loose))}")

    files: dict[Path, list[Path]] = {}
    for d in room_dirs:
        if not ROOM_NAME.match(d.name):
            cap.warnings.append(f"Room folder '{d.name}': use lowercase letters, digits and _ only (e.g. bedroom1).")
        files[d] = []
        for f in sorted(d.iterdir()):
            if f.name.startswith(".") or f.is_dir():
                continue
            if f.suffix.lower() not in exts:
                cap.warnings.append(f"{f}: not a photo ({f.suffix}), ignored")
                continue
            files[d].append(f)

    # HEIC decode dominates (~280 ms per 12 MP photo) and releases the GIL, so decode in threads.
    # map() keeps input order, so everything downstream is deterministic.
    def load(df):
        try:
            return load_photo(df[1], df[0].name, cfg)
        except UnreadableImage as e:
            return e

    jobs = [(d, f) for d, fs in files.items() for f in fs]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        loaded = dict(zip(jobs, pool.map(load, jobs)))

    # the capture's lens = its most common focal: photos on another lens are excluded (mixing lenses skewed the
    # joint scale ~4.6%, E20). Not a fixed range: a Pro iPhone's 1x can be set to 24, 28 or 35 mm
    f35s = [r[0].meta.f35_mm for r in loaded.values() if not isinstance(r, UnreadableImage)]
    main_f35 = Counter(round(x) for x in f35s).most_common(1)[0][0] if f35s else None
    seen: dict[str, Path] = {}  # sha256 -> first file with that content
    cross_room_dupes: list[str] = []
    for d, fs in files.items():
        room = d.name
        frames: list[Frame] = []
        for f in fs:
            res = loaded[(d, f)]
            if isinstance(res, UnreadableImage):
                cap.warnings.append(f"{res}; skipped")
                continue
            frame, w = res
            cap.warnings.extend(f"{room}/{msg}" for msg in w)
            if icfg["reject_non_main_lens"] and abs(frame.meta.f35_mm / main_f35 - 1) > icfg["lens_mix_tol"]:
                cap.warnings.append(f"{f}: excluded ({frame.meta.f35_mm:g} mm, the capture's lens is {main_f35} mm: "
                                    "mixed lenses skewed the joint scale ~4.6%, E20)")
                continue
            first = seen.get(frame.meta.sha256)
            if first is not None:
                if first.parent == d:
                    cap.warnings.append(f"{f}: identical to {first.name}; duplicate dropped")
                    continue
                cross_room_dupes.append(f"{first} == {f}")
                continue
            seen[frame.meta.sha256] = f
            frames.append(frame)
        cap.rooms[room] = _select(frames, d, icfg, cap.warnings)

    if cross_room_dupes:
        raise InputError(
            "The same photo is in more than one room folder, so its room is ambiguous. "
            "Keep each photo in exactly one room:\n  " + "\n  ".join(cross_room_dupes)
        )
    cap.devices = sorted({fr.meta.device for frs in cap.rooms.values() for fr in frs if fr.meta.device})


def _select(frames: list[Frame], d: Path, icfg: dict[str, Any], warnings: list[str]) -> list[Frame]:
    """One orientation per room, at least min and at most max photos (keep the sharpest)."""
    counts = Counter(fr.meta.orientation for fr in frames)
    if len(counts) > 1 and icfg["mixed_orientation"] == "drop_minority":
        top = counts.most_common()
        keep = top[0][0] if top[0][1] != top[1][1] else frames[0].meta.orientation
        dropped = [fr.image_path.name for fr in frames if fr.meta.orientation != keep]
        frames = [fr for fr in frames if fr.meta.orientation == keep]
        warnings.append(f"{d}: mixed portrait/landscape; kept {keep}, dropped {', '.join(dropped)}")

    lo, hi = icfg["min_images_per_room"], icfg["max_images_per_room"]
    if len(frames) < lo:
        raise InputError(f"{d}: {len(frames)} usable photo(s), need at least {lo}. See docs/CAPTURE_PROTOCOL.md.")
    if len(frames) > hi:
        ranked = sorted(frames, key=lambda fr: -fr.meta.sharpness)
        keep_ids = {id(fr) for fr in ranked[:hi]}
        dropped = [fr.image_path.name for fr in frames if id(fr) not in keep_ids]
        frames = [fr for fr in frames if id(fr) in keep_ids]  # keep filename order
        warnings.append(f"{d}: {len(frames) + len(dropped)} photos, max {hi}; kept the {hi} sharpest, dropped {', '.join(dropped)}")
    return frames
