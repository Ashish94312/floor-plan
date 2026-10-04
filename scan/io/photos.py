"""Photo tier: one image file -> Frame (EXIF-upright pixels, intrinsics from EXIF)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import IFD
from pillow_heif import register_heif_opener

from scan.types import Frame, PhotoMeta

register_heif_opener()

# EXIF tag ids
_MAKE, _MODEL, _ORIENTATION = 0x010F, 0x0110, 0x0112
_F35, _FOCAL, _LENS, _DATETIME, _DZOOM = 0xA405, 0x920A, 0xA434, 0x9003, 0xA404


class UnreadableImage(Exception):
    pass


def _exif(img: Image.Image) -> dict[str, Any]:
    ex = img.getexif()
    sub = ex.get_ifd(IFD.Exif)

    def get(tag):
        v = sub.get(tag, ex.get(tag))
        return v.strip("\x00 ") if isinstance(v, str) else v

    return {
        "make": get(_MAKE),
        "model": get(_MODEL),
        "f35": get(_F35),
        "focal": get(_FOCAL),
        "lens": get(_LENS),
        "datetime": get(_DATETIME),
        "digital_zoom": get(_DZOOM),
    }


def intrinsics(f35_mm: float, width: int, height: int, film_diagonal_mm: float) -> np.ndarray:
    """Pinhole K for an image of `width` x `height` px. The 35 mm-equivalent focal length is
    defined on the film diagonal (43.27 mm), so f_px = f35 * image_diagonal_px / 43.27 (D18).
    Principal point at the image centre."""
    f = f35_mm * math.hypot(width, height) / film_diagonal_mm
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_photo(path: Path, room: str, cfg: dict[str, Any]) -> tuple[Frame, list[str]]:
    """Decode, apply EXIF orientation, downsize to the working size, derive K. Returns the frame
    and any warnings about this file. Raises UnreadableImage if the file can't be decoded."""
    icfg = cfg["ingest"]
    kcfg = icfg["intrinsics"]
    warnings: list[str] = []
    try:
        with Image.open(path) as im:
            ex = _exif(im)
            up = ImageOps.exif_transpose(im)
            up.load()
    except Exception as e:  # any decoder failure means the file is unusable
        raise UnreadableImage(f"{path}: cannot decode image ({type(e).__name__}: {e})") from e
    up = up.convert("RGB")
    W, H = up.size

    if ex["f35"]:
        f35, source = float(ex["f35"]), "exif_f35"
    else:
        f35, source = float(kcfg["default_f35_mm"]), "default"
        warnings.append(f"{path.name}: no 35 mm-equivalent focal length in EXIF; assuming {f35:g} mm (wider intervals)")
    lo, hi = kcfg["main_lens_f35_range_mm"]
    if not lo <= f35 <= hi:
        warnings.append(f"{path.name}: focal {f35:g} mm (35 mm eq.) is not the 1x lens; protocol says 1x only")
    if ex["digital_zoom"] and float(ex["digital_zoom"]) > 1.0:
        warnings.append(f"{path.name}: digital zoom {float(ex['digital_zoom']):g}x; protocol says no zoom")

    k = icfg["working_max_side"] / max(W, H)
    if k < 1:
        w, h = round(W * k), round(H * k)
        small = up.resize((w, h), Image.Resampling.LANCZOS)
    else:
        w, h, small = W, H, up
    rgb = np.asarray(small, dtype=np.uint8)
    K = intrinsics(f35, W, H, kcfg["film_diagonal_mm"])
    K[0] *= w / W  # fx, cx
    K[1] *= h / H  # fy, cy

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    device = " ".join(x for x in (ex["make"], ex["model"]) if x) or None
    meta = PhotoMeta(
        sha256=sha256_file(path),
        full_size=(W, H),
        orientation="portrait" if H > W else "landscape",
        device=device,
        lens=ex["lens"],
        f35_mm=f35,
        intrinsics_source=source,
        captured_at=ex["datetime"],
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )
    return Frame(image_path=path, K=K, rgb=rgb, room_hint=room, meta=meta), warnings
