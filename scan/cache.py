"""Content-addressed cache for model outputs (ARCHITECTURE §12).

Key = SHA256 of (input image hashes, model id + revision, every setting that changes the output).
Value = an .npz of the outputs. Replaying a cached entry gives bit-identical numbers; --no-cache
forces the live model path, which must reproduce them (E11 showed bit-identical live runs).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def cache_key(**parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _path(cfg: dict, kind: str, key: str) -> Path:
    root = Path(cfg["cache"]["dir"])
    return (root if root.is_absolute() else REPO / root) / kind / f"{key}.npz"


def load(cfg: dict, kind: str, key: str) -> dict[str, np.ndarray] | None:
    p = _path(cfg, kind, key)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save(cfg: dict, kind: str, key: str, arrays: dict[str, np.ndarray]) -> Path:
    p = _path(cfg, kind, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(p)  # atomic: a crash never leaves a half-written entry
    return p
