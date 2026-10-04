"""E12. Independent focal-length check from vanishing points (no learned model involved).

In a room photo, lines along the two horizontal wall directions converge to two vanishing points
v1, v2. The directions are perpendicular, so for a pinhole camera with principal point c:
    f^2 = -(v1 - c) . (v2 - c)
This referees between the EXIF focal and MapAnything's output focal.

Usage: uv run python scripts/experiments/vanishing_focal.py <photos_dir> [ma_raw.npz] [out_dir]
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan.config import load_config  # noqa: E402
from scan.io.photos import load_photo  # noqa: E402

ANG_TOL = np.deg2rad(1.5)  # a segment supports a VP if it points at it within 1.5 degrees


def segments(gray: np.ndarray, min_len: float) -> np.ndarray:
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    s = lines[:, 0, :]
    return s[np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) >= min_len]


def homog_line(s):
    return np.cross([s[0], s[1], 1.0], [s[2], s[3], 1.0])


def support(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Boolean mask: segment direction agrees with direction from its midpoint to the VP."""
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    d = segs[:, 2:] - segs[:, :2]
    to_vp = vp[:2] - mid * vp[2]  # works for finite and (near-)infinite VPs
    cos = np.abs((d * to_vp).sum(1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(to_vp, axis=1) + 1e-12)
    return np.arccos(np.clip(cos, 0, 1)) < ANG_TOL


def ransac_vp(segs, rng, iters=3000):
    best, best_n = None, 0
    w = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    for _ in range(iters):
        i, j = rng.choice(len(segs), 2, replace=False)
        vp = np.cross(homog_line(segs[i]), homog_line(segs[j]))
        if np.linalg.norm(vp) < 1e-9:
            continue
        m = support(segs, vp)
        n = w[m].sum()  # length-weighted support
        if n > best_n:
            best, best_n = vp, n
    # least-squares refine on inliers
    m = support(segs, best)
    L = np.array([homog_line(s) / np.linalg.norm(homog_line(s)[:2]) for s in segs[m]])
    vp = np.linalg.svd(L)[2][-1]
    return vp, support(segs, vp)


def focal_from_image(rgb: np.ndarray, rng) -> tuple[float | None, dict]:
    H, W = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    segs = segments(gray, min_len=0.03 * max(W, H))
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])) % 180
    horiz = segs[np.abs(ang - 90) > 25]  # drop near-vertical lines (vertical VP)
    if len(horiz) < 8:
        return None, {"reason": "too few lines"}
    v1, m1 = ransac_vp(horiz, rng)
    rest = horiz[~m1]
    v2, m2 = ransac_vp(rest, rng)
    c = np.array([W / 2, H / 2])
    if abs(v1[2]) < 1e-9 or abs(v2[2]) < 1e-9:
        return None, {"reason": "a horizontal VP at infinity (wall facing the camera)"}
    p1, p2 = v1[:2] / v1[2], v2[:2] / v2[2]
    f2 = -np.dot(p1 - c, p2 - c)
    info = {"vp1": p1, "vp2": p2, "n1": int(m1.sum()), "n2": int(m2.sum()), "segs": horiz, "m1": m1, "m2_rest": (rest, m2)}
    if f2 <= 0:
        return None, {**info, "reason": "VPs on the same side (not two perpendicular walls)"}
    return math.sqrt(f2), info


def main():
    photos = Path(sys.argv[1])
    ma = np.load(sys.argv[2]) if len(sys.argv) > 2 else None
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    cfg = load_config()
    rng = np.random.default_rng(0)
    model_f = {}
    if ma is not None:
        Hm, Wm = ma["pts"].shape[1:3]
        for n, K in zip(ma["names"], ma["intrinsic"]):
            model_f[str(n)] = (K[0, 0], Wm)
    rows = []
    for p in sorted(photos.iterdir()):
        if p.suffix.lower() not in {".heic", ".jpg", ".jpeg", ".png"}:
            continue
        fr, _ = load_photo(p, photos.name, cfg)
        if fr.meta.orientation != "portrait":
            continue
        W = fr.rgb.shape[1]
        f_vp, info = focal_from_image(fr.rgb, rng)
        f_exif = fr.K[0, 0]
        f_model = model_f[p.name][0] * W / model_f[p.name][1] if p.name in model_f else float("nan")
        rows.append((p.name, f_exif, f_vp, f_model, info))
        if out is not None and f_vp is not None:
            out.mkdir(parents=True, exist_ok=True)
            img = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2BGR).copy()
            for s in info["segs"][info["m1"]]:
                cv2.line(img, tuple(map(int, s[:2])), tuple(map(int, s[2:])), (0, 0, 255), 2)
            rest, m2 = info["m2_rest"]
            for s in rest[m2]:
                cv2.line(img, tuple(map(int, s[:2])), tuple(map(int, s[2:])), (255, 0, 0), 2)
            cv2.imwrite(str(out / f"{p.stem}_vp.jpg"), img)
    print(f"{'photo':16s} {'EXIF f':>8s} {'VP f':>8s} {'model f':>8s}   VP/EXIF  model/EXIF  lines(v1,v2)  note   [px at {W}-wide]")
    vals = []
    for name, fe, fv, fm, info in rows:
        r = f"{fv / fe:7.3f}" if fv else "    -  "
        note = info.get("reason", "")
        print(f"{name:16s} {fe:8.1f} {fv if fv else float('nan'):8.1f} {fm:8.1f}   {r}   {fm / fe:9.3f}   {info.get('n1', '-')!s:>4s},{info.get('n2', '-')!s:<4s}  {note}")
        if fv:
            vals.append(fv / fe)
    if vals:
        print(f"median VP/EXIF = {np.median(vals):.3f} over {len(vals)} photos (spread {np.min(vals):.3f}-{np.max(vals):.3f})")


if __name__ == "__main__":
    main()
