"""QuickTime metadata items (`meta` -> `keys` + `ilst` atoms), for the camera facts iPhone videos carry.

ffprobe shows only the file-level items; the camera ones (lens, 35 mm-equivalent focal) sit in the video
track's own `meta` atom. iPhone 13 1x video: camera.focal_length.35mm_equivalent = 27 (E22m).
"""

from __future__ import annotations

import struct
from pathlib import Path

_CONTAINERS = ("moov", "trak", "mdia", "minf", "udta", "edts")


def _atoms(buf: bytes, off: int, end: int):
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[off:off + 8])
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", buf[off + 8:off + 16])[0], 16
        elif size == 0:
            size = end - off
        if size < hdr:
            return
        yield typ.decode("latin1"), off + hdr, off + size
        off += size


def _value(dtype: int, raw: bytes):
    if dtype == 1:
        return raw.decode("utf-8", "replace")
    if dtype == 23 and len(raw) == 4:
        return struct.unpack(">f", raw)[0]
    if dtype == 24 and len(raw) == 8:
        return struct.unpack(">d", raw)[0]
    if dtype in (21, 22) and len(raw) in (1, 2, 4, 8):
        return int.from_bytes(raw, "big", signed=dtype == 21)
    return raw.hex()


def _meta_items(buf: bytes, s: int, e: int) -> dict:
    keys, out = [], {}
    for typ, a, b in _atoms(buf, s, e):
        if typ == "keys":
            n, p = struct.unpack(">I", buf[a + 4:a + 8])[0], a + 8
            for _ in range(n):
                ks = struct.unpack(">I", buf[p:p + 4])[0]
                keys.append(buf[p + 8:p + ks].decode("latin1"))
                p += ks
        elif typ == "ilst":
            for t2, a2, b2 in _atoms(buf, a, b):
                idx = struct.unpack(">I", t2.encode("latin1"))[0]
                for t3, a3, b3 in _atoms(buf, a2, b2):
                    if t3 == "data" and 0 < idx <= len(keys):
                        out[keys[idx - 1]] = _value(struct.unpack(">I", buf[a3:a3 + 4])[0] & 0xFFFFFF, buf[a3 + 8:b3])
    return out


def metadata(path: Path) -> dict:
    """All `meta` items in the file (file level and per track), keys without the com.apple.quicktime. prefix."""
    data = Path(path).read_bytes()
    out: dict = {}

    def walk(s, e):
        for typ, a, b in _atoms(data, s, e):
            if typ == "meta":
                out.update({k.replace("com.apple.quicktime.", ""): v for k, v in _meta_items(data, a, b).items()})
            elif typ in _CONTAINERS:
                walk(a, b)

    walk(0, len(data))
    return out


def focal_35mm(path: Path) -> float | None:
    """35 mm-equivalent focal length the camera recorded for this video, if any."""
    v = metadata(path).get("camera.focal_length.35mm_equivalent")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None
