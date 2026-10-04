"""Internal types passed between stages (ARCHITECTURE §9.1). Not part of the published schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Tier = Literal["photo", "video", "lidar"]


@dataclass
class PhotoMeta:
    sha256: str
    full_size: tuple[int, int]  # (W, H) of the original, EXIF-upright
    orientation: Literal["portrait", "landscape"]
    device: str | None  # "Apple iPhone 13"
    lens: str | None
    f35_mm: float  # 35 mm-equivalent focal actually used
    intrinsics_source: Literal["exif_f35", "default"]
    captured_at: str | None  # EXIF DateTimeOriginal
    sharpness: float  # variance of the Laplacian at working resolution


@dataclass
class Frame:
    image_path: Path
    K: np.ndarray  # 3x3 intrinsics at the resolution of `rgb`
    rgb: np.ndarray | None = None  # HxWx3 uint8, EXIF-upright, working resolution
    room_hint: str | None = None  # photo tier: room folder name
    T_wc: np.ndarray | None = None  # 4x4 camera-to-world (None until geometry)
    depth: np.ndarray | None = None  # HxW metres
    depth_conf: np.ndarray | None = None
    timestamp: float | None = None  # video / LiDAR
    meta: PhotoMeta | None = None


@dataclass
class Capture:
    """What ingest hands to the pipeline."""

    capture_id: str
    root: Path
    tier: Tier
    rooms: dict[str, list[Frame]] = field(default_factory=dict)  # photo tier: room_id -> frames
    devices: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Scale:
    s: float  # multiply backbone units by s -> metres (1.0 for metric backbones)
    sigma_log: float  # std of log(s)


@dataclass
class RoomCloud:
    room_id: str
    points: np.ndarray  # N x 3, metres, gravity-aligned (z up), Manhattan-aligned (x, y)
    colors: np.ndarray  # N x 3
    frames: list[Frame]
    scale: Scale
    T_room_world: np.ndarray  # 4x4 room-local -> property frame (identity until stitched)
