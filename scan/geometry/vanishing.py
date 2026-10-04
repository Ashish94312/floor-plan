"""Focal length from vanishing points (E12): geometry only, no learned model.

Lines along a room's two horizontal wall directions converge to vanishing points v1, v2; the
directions are perpendicular, so for a pinhole camera with principal point c: f^2 = -(v1-c).(v2-c).
Valid on oblique (two-point perspective) views; head-on views give no answer (one VP near infinity).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

ANG_TOL = np.deg2rad(1.5)


def _segments(gray, min_len):
    lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    s = lines[:, 0, :]
    return s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= min_len]


def _line(s):
    return np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])


def _support(segs, vp):
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    d = segs[:, 2:] - segs[:, :2]
    to = vp[:2] - mid * vp[2]
    cos = np.abs((d * to).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to, axis=1) + 1e-12)
    return np.arccos(np.clip(cos, 0, 1)) < ANG_TOL


def _ransac(segs, rng, iters=2000):
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    best, best_n = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False)
        vp = np.cross(_line(segs[i]), _line(segs[j]))
        if np.linalg.norm(vp) < 1e-9:
            continue
        n = w[_support(segs, vp)].sum()
        if n > best_n:
            best, best_n = vp, n
    m = _support(segs, best)
    L = np.array([_line(s) / np.linalg.norm(_line(s)[:2]) for s in segs[m]])
    vp = np.linalg.svd(L)[2][-1]
    return vp, _support(segs, vp)


def focal_px(rgb: np.ndarray, seed: int = 0, min_lines: int = 10) -> float | None:
    """Focal length in pixels of this image, or None if the view gives no reliable answer."""
    rng = np.random.default_rng(seed)
    H, W = rgb.shape[:2]
    segs = _segments(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 0.03 * max(W, H))
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    horiz = segs[np.abs(ang - 90) > 25]
    if len(horiz) < 2 * min_lines:
        return None
    v1, m1 = _ransac(horiz, rng)
    rest = horiz[~m1]
    if len(rest) < min_lines:
        return None
    v2, m2 = _ransac(rest, rng)
    if m1.sum() < min_lines or m2.sum() < min_lines or abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None
    c = np.array([W / 2, H / 2])
    f2 = -np.dot(v1[:2] / v1[2] - c, v2[:2] / v2[2] - c)
    return math.sqrt(f2) if f2 > 0 else None
