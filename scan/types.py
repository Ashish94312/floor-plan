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
    intrinsics_source: Literal["exif_f35", "quicktime_f35", "vanishing_points", "default"]
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
    link: bool = False  # video: added only to tie rooms together in the model run; not used to measure (E22l)


@dataclass
class Capture:
    """What ingest hands to the pipeline."""

    capture_id: str
    root: Path
    tier: Tier
    rooms: dict[str, list[Frame]] = field(default_factory=dict)  # photo tier: room_id -> frames
    devices: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    links: list[tuple[str, str, str, str]] = field(default_factory=list)  # video: (room a, frame a, room b, frame b), E22n


@dataclass
class Scale:
    s: float  # multiply backbone units by s -> metres (1.0 for metric backbones)
    sigma_log: float  # std of log(s)


@dataclass
class RoomCloud:
    room_id: str
    points: np.ndarray  # N x 3, metres. Backbone frame after step 1.2; gravity/Manhattan-aligned after 1.3
    colors: np.ndarray  # N x 3, 0-1
    frames: list[Frame]
    scale: Scale
    T_room_world: np.ndarray  # 4x4 room-local -> property frame (identity until stitched)
    frame_id: str = ""  # which reconstruction the points live in: the room id, or "joint" for a joint run
    views: dict[str, np.ndarray] | None = None  # this room's per-view backbone outputs (depth, pose, K, rgb)
    run_views: dict[str, np.ndarray] | None = None  # every view of the room's model run, link frames included (E22n)
    alignment: object | None = None  # geometry.align.Alignment of this room's frame (step 1.3)
    planes: object | None = None  # geometry.align.RoomPlanes: floor/ceiling in the aligned frame
    layout: object | None = None  # layout.room.Layout: polygon, walls, area (step 1.4)
    view_off_axis_deg: dict[str, float] | None = None  # per photo: median wall angle off the house axes
    openings: list = field(default_factory=list)  # layout.openings.OpeningSeg (step 1.7)
    damage: list = field(default_factory=list)  # damage.detect.DamageSeg (step 1.9)
    pose: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, theta_deg of the room frame in the property frame
    placed_by: str | None = None  # joint_reconstruction | door_matching | single_room | None (unplaced)
