"""C5 Room layout (ARCHITECTURE §10.7): aligned room cloud -> rectilinear floor-plan polygon.

Input: the aligned room (floor z = 0, walls along x / y, step 1.3) with its per-view depth and poses.

Free-space carving (D23): every observed pixel is a ray from its camera to a surface, and the line
of sight in between was empty. Drawing all rays from above gives the room's free space. Furniture,
ceiling beams and door gaps can't hide the room (rays pass over the bed, under the beam, and the
doorway camera's own rays carve the room). Thin necks where rays went through a doorway into the
next room are cut by a morphological opening. The outline becomes an axis-aligned polygon, and each
wall sits at the median of the wall points near its edge (consensus over views that disagree by a
few cm, as with floor/ceiling in D22).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage

from scan.geometry.align import normals


@dataclass
class WallSeg:
    wall_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    length_m: float
    support: int  # wall-band points within the refine window
    spread_m: float  # robust spread of those points across the wall (doubling / noise)


@dataclass
class Layout:
    room_id: str
    polygon: list[tuple[float, float]]  # clockwise from above, metres
    walls: list[WallSeg]
    floor_area_m2: float
    ceiling_height_m: float | None
    status: str  # ok | partial | failed
    method: str  # free_space_carving | bbox_fallback
    close_radius_m: float | None  # unused by carving (kept for the record)
    w1_rule: str = "provisional: south-most wall, west end (re-indexed by the main door in step 1.7)"
    warnings: list[str] = field(default_factory=list)


def wall_points(points: np.ndarray, N: np.ndarray, ceiling_z: float | None, cfg: dict) -> np.ndarray:
    """Points on vertical surfaces between 0.3 m and just under the ceiling (walls, plus some furniture)."""
    lc = cfg["layout"]
    top = ceiling_z if ceiling_z is not None else float(np.percentile(points[:, 2], 98))
    z = points[:, 2]
    keep = (z >= lc["wall_min_z_m"]) & (z <= top - lc["wall_below_ceiling_m"]) & (np.abs(N[:, 2]) < lc["wall_normal_max_z"])
    return points[keep]


def _grid(xy: np.ndarray, cell: float, margin: float):
    lo = xy.min(0) - margin
    hi = xy.max(0) + margin
    shape = np.ceil((hi - lo) / cell).astype(int) + 1  # (nx, ny)
    return lo, shape


def _to_ij(xy, lo, cell):
    return np.floor((xy - lo) / cell).astype(int)


def _cells(ij: np.ndarray, shape) -> np.ndarray:
    img = np.zeros((shape[1], shape[0]), np.int32)  # rows = y, cols = x
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
    np.add.at(img, (ij[ok, 1], ij[ok, 0]), 1)
    return img


def carve_free_space(rays: list[tuple[np.ndarray, np.ndarray]], lo, shape, cell, stop_short_m) -> np.ndarray:
    """Count, per 2D cell, how many camera->surface lines of sight pass through it."""
    free = np.zeros((shape[1], shape[0]), np.uint16)
    for cam_xy, P in rays:
        c = _to_ij(cam_xy[None], lo, cell)[0]
        d = P[:, :2] - cam_xy
        L = np.linalg.norm(d, axis=1, keepdims=True)
        end = cam_xy + d * np.clip((L - stop_short_m) / np.maximum(L, 1e-9), 0, 1)
        img = np.zeros_like(free, dtype=np.uint8)
        for e in _to_ij(end, lo, cell):
            cv2.line(img, (int(c[0]), int(c[1])), (int(e[0]), int(e[1])), 1, 1)
        free += img  # each view counts once per cell
    return free


def _room_region(own: np.ndarray, floor_counts, seeds_ij, cfg):
    """Room's free space -> cut doorway necks -> best component -> Manhattan regularisation.

    own: cells this room's lines of sight passed through (in a joint run: and it saw them most).
    open (rectangle, regularise_open_m): removes thin protrusions (doorway stubs).
    close (rectangle, regularise_close_m): fills small concave blocks (corner wardrobes, unseen corners)."""
    lc = cfg["layout"]
    cell = lc["cell_m"]
    mask = own.astype(np.uint8)
    # lines of sight fan out with distance and leave speckle near far walls: merge neighbours first
    kg = max(1, round(lc["free_gap_fill_m"] / cell))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kg, kg), np.uint8))
    r = round(lc["neck_cut_m"] / 2 / cell)
    neck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, neck)
    labels, n = ndimage.label(opened)
    if n == 0:
        return None, {}
    score = ndimage.sum(floor_counts, labels, index=np.arange(1, n + 1))
    for s_ in seeds_ij:
        if 0 <= s_[1] < labels.shape[0] and 0 <= s_[0] < labels.shape[1] and labels[s_[1], s_[0]] > 0:
            score[labels[s_[1], s_[0]] - 1] += 1
    best = int(np.argmax(score)) + 1
    region = ndimage.binary_fill_holes(labels == best)
    region = cv2.dilate(region.astype(np.uint8), neck) & mask  # undo the neck cut's corner rounding
    before = region.sum()
    ko = max(1, round(lc["regularise_open_m"] / cell))
    kc = max(1, round(lc["regularise_close_m"] / cell))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
    after_open = region.sum()
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
    labels, n = ndimage.label(region)
    if n > 1:
        sizes = ndimage.sum(region, labels, index=np.arange(1, n + 1))
        region = labels == (int(np.argmax(sizes)) + 1)
    region = ndimage.binary_fill_holes(region)
    stats = {"removed_protrusions_m2": round(float(before - after_open) * cell**2, 3),
             "filled_concavities_m2": round(float(region.sum() - after_open) * cell**2, 3)}
    return region, stats


def _rectilinear(contour_xy: np.ndarray, min_edge: float) -> np.ndarray:
    """Axis-aligned polygon from a raster outline: snap edges, merge runs, remove short jogs."""
    pts = contour_xy
    # 1. snap: each edge horizontal or vertical -> alternate coordinates
    segs = []
    for a, b in zip(pts, np.roll(pts, -1, 0)):
        d = b - a
        if np.hypot(*d) < 1e-9:
            continue
        horiz = abs(d[0]) >= abs(d[1])
        segs.append(["h" if horiz else "v", (a[1] + b[1]) / 2 if horiz else (a[0] + b[0]) / 2, abs(d[0]) if horiz else abs(d[1])])
    changed = True
    while changed and len(segs) > 4:
        changed = False
        # merge consecutive segments with the same orientation (length-weighted coordinate)
        merged = []
        for s in segs:
            if merged and merged[-1][0] == s[0]:
                m = merged[-1]
                w = m[2] + s[2]
                m[1] = (m[1] * m[2] + s[1] * s[2]) / w if w > 0 else m[1]
                m[2] = w
            else:
                merged.append(list(s))
        if len(merged) > 1 and merged[0][0] == merged[-1][0]:
            a, b = merged[0], merged.pop()
            w = a[2] + b[2]
            a[1] = (a[1] * a[2] + b[1] * b[2]) / w
            a[2] = w
        if len(merged) != len(segs):
            changed = True
        segs = merged
        # drop the shortest jog: it becomes zero length, its two neighbours (same orientation) merge next pass
        if len(segs) > 4:
            i = min(range(len(segs)), key=lambda k: segs[k][2])
            if segs[i][2] < min_edge:
                segs.pop(i)
                changed = True
    return segs


def _fill_small_notches(segs, max_area: float, max_len: float):
    """Remove inward corner steps (two consecutive short edges whose removal ADDS area < max_area):
    corner furniture / unseen corners. Outward bumps and large notches (real L-shapes) are kept.
    Returns (segs, filled_area_m2)."""
    filled = 0.0
    while len(segs) > 4:
        base = abs(_signed_area(_vertices(segs)))
        best = None
        for i in range(len(segs)):
            j = (i + 1) % len(segs)
            if segs[i][2] > max_len or segs[j][2] > max_len:
                continue
            trial = [s for k, s in enumerate(segs) if k not in (i, j)]
            gain = abs(_signed_area(_vertices(trial))) - base
            if 0 < gain < max_area and (best is None or gain < best[0]):
                best = (gain, trial)
        if best is None:
            break
        filled += best[0]
        segs = best[1]
    return segs, filled


def _vertices(segs) -> np.ndarray:
    """Corner points of alternating h/v segments: corner between seg i and i+1."""
    out = []
    for s, t in zip(segs, segs[1:] + segs[:1]):
        out.append((t[1], s[1]) if s[0] == "h" else (s[1], t[1]))  # (x from the vertical one, y from the horizontal)
    return np.array(out, float)


def _refine(segs, wall_xy, inside, outside):
    """Wall position = median coordinate of wall points near the outline edge, searched `inside` metres
    into the room and `outside` metres beyond it. Carved free space stops at or before a wall, never
    past it, so the true wall is at or outside the outline. Median = consensus over doubled copies."""
    V = _vertices(segs)
    ccw = _signed_area(V) > 0
    out = []
    for i, s in enumerate(segs):
        a, b = V[i - 1], V[i]  # this segment runs from corner i-1 to corner i
        d = b - a
        n_out = np.array([d[1], -d[0]]) if ccw else np.array([-d[1], d[0]])
        n_out = n_out / (np.linalg.norm(n_out) + 1e-12)
        k = 1 if s[0] == "h" else 0  # coordinate that fixes the wall line
        span = 0 if s[0] == "h" else 1
        lo, hi = sorted((a[span], b[span]))
        off = (wall_xy[:, k] - s[1]) * n_out[k]  # >0 = outside the room
        m = (off > -inside) & (off < outside) & (wall_xy[:, span] > lo) & (wall_xy[:, span] < hi)
        coord = wall_xy[m, k]
        if len(coord) >= 20:
            med = float(np.median(coord))
            spread = float(1.4826 * np.median(np.abs(coord - med)))
            out.append([s[0], med, s[2], len(coord), spread])
        else:
            out.append([s[0], s[1], s[2], len(coord), float("nan")])
    return out


def _signed_area(V: np.ndarray) -> float:
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _walls_clockwise(V: np.ndarray, support, room_id: str):
    """Clockwise from above; W1 = south-most edge (lowest y), its west end first (provisional, see Layout).
    V[i] is the corner between segment i and i+1, so edge V[i] -> V[i+1] is segment i+1."""
    n = len(V)
    edges = [(V[i], V[(i + 1) % n], support[(i + 1) % n]) for i in range(n)]
    if _signed_area(V) > 0:  # counter-clockwise -> walk the same edges backwards
        edges = [(b, a, sup) for a, b, sup in reversed(edges)]
    start = min(range(n), key=lambda i: ((edges[i][0][1] + edges[i][1][1]) / 2, min(edges[i][0][0], edges[i][1][0])))
    edges = edges[start:] + edges[:start]
    walls = []
    for k, (a, b, sup) in enumerate(edges, 1):
        walls.append(WallSeg(f"{room_id}-W{k}", (round(float(a[0]), 4), round(float(a[1]), 4)),
                             (round(float(b[0]), 4), round(float(b[1]), 4)),
                             round(float(np.hypot(*(b - a))), 4), int(sup[0]), round(float(sup[1]), 4)))
    poly = [w.start for w in walls]
    return poly, walls


def view_rays(cloud, cfg) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per view: (camera xy, its surface points in the aligned frame), subsampled."""
    from scan.geometry.cloud import unproject

    v, g, lc = cloud.views, cfg["geometry"], cfg["layout"]
    thr = np.percentile(v["conf"][v["mask"]], g["conf_percentile"])
    rng = np.random.default_rng(0)
    out = []
    for i in range(len(v["depth"])):
        keep = v["mask"][i] & (v["conf"][i] >= thr)
        if g["rays"] == "exif":
            P = unproject(v["depth"][i], v["K_exif"][i], v["T_wc"][i])[keep]  # T_wc already aligned (1.3)
        else:
            P = v["pts"][i][keep]  # already moved into the aligned frame in step 1.3
        if len(P) > lc["max_rays_per_view"]:
            P = P[rng.choice(len(P), lc["max_rays_per_view"], replace=False)]
        out.append((v["T_wc"][i][:2, 3], P))
    return out


def layout_rooms(clouds: dict, cfg: dict) -> dict[str, Layout]:
    """Lay out every room. Rooms sharing a reconstruction frame (joint run) split the carved free space
    by ownership: a cell belongs to the room whose own photos saw through it most, so lines of sight
    through a doorway into the next room don't leak into this room's outline."""
    lc = cfg["layout"]
    cell = lc["cell_m"]
    out: dict[str, Layout] = {}
    for fid in dict.fromkeys(c.frame_id for c in clouds.values()):
        rooms = {r: c for r, c in clouds.items() if c.frame_id == fid}
        data = {}
        for r, c in rooms.items():
            N = normals(c.points, cfg["alignment"]["normal_radius_m"])
            data[r] = {
                "walls": wall_points(c.points, N, c.planes.ceiling_z, cfg),
                "floor": c.points[(c.points[:, 2] < lc["floor_max_z_m"]) & (np.abs(N[:, 2]) > 0.8)],
                "rays": view_rays(c, cfg),
            }
            data[r]["cams"] = np.array([cx for cx, _ in data[r]["rays"]])
        allxy = np.vstack([np.vstack([d["walls"][:, :2], d["cams"]]) for d in data.values()])
        lo, shape = _grid(allxy, cell, margin=0.5)
        free = {r: carve_free_space(d["rays"], lo, shape, cell, lc["ray_stop_short_m"]) for r, d in data.items()}
        names = list(rooms)
        if len(names) > 1:
            stack = np.stack([free[r] for r in names])
            owner = np.argmax(stack, 0)
            own = {r: (owner == k) & (free[r] >= lc["min_views_free"]) for k, r in enumerate(names)}
        else:
            own = {names[0]: free[names[0]] >= lc["min_views_free"]}
        for r, c in rooms.items():
            out[r] = _layout_one(r, c, data[r], own[r], free[r], lo, shape, cfg)
    return out


def _layout_one(room_id, cloud, d, own, free, lo, shape, cfg) -> Layout:
    lc = cfg["layout"]
    cell = lc["cell_m"]
    ceiling_h = cloud.planes.ceiling_height_m
    walls_pts = d["walls"]
    if len(walls_pts) < lc["min_wall_points"]:
        return _fallback(room_id, cloud.points, ceiling_h, [f"only {len(walls_pts)} wall points"])
    floor_img = _cells(_to_ij(d["floor"][:, :2], lo, cell), shape).astype(float)
    region, stats = _room_region(own, floor_img, _to_ij(d["cams"], lo, cell), cfg)
    warnings: list[str] = []
    if region is None or region.sum() * cell**2 < lc["min_room_area_m2"]:
        return _fallback(room_id, cloud.points, ceiling_h, ["free-space carving found no room region"])
    if stats["filled_concavities_m2"] > 0.05:
        warnings.append(f"filled {stats['filled_concavities_m2']:.2f} m2 of small concave blocks "
                        "(corner furniture / unseen corners treated as room)")
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    c_xy = lo + (c + 0.5) * cell  # pixel centres -> metres
    segs = _rectilinear(c_xy, lc["min_edge_m"])
    if len(segs) < 4:
        return _fallback(room_id, cloud.points, ceiling_h, ["outline has fewer than 4 straight edges"])
    segs, notch = _fill_small_notches(segs, lc["notch_max_area_m2"], lc["notch_max_len_m"])
    if notch > 0:
        warnings.append(f"filled a {notch:.2f} m2 corner notch (corner furniture / unseen corner treated as room)")
    segs = _refine(segs, walls_pts[:, :2], *lc["refine_window_m"])
    V = _vertices([s_[:3] for s_ in segs])
    support = [(s_[3], s_[4]) for s_ in segs]
    poly, walls = _walls_clockwise(V, support, room_id)
    weak = [w.wall_id for w in walls if w.support < 20]
    if weak:
        warnings.append(f"walls with little point support (position from the outline only): {', '.join(weak)}")
    area = abs(_signed_area(np.array(poly)))
    lay = Layout(room_id, poly, walls, round(area, 4), ceiling_h, "ok", "free_space_carving", None, warnings=warnings)
    lay._debug = {"free": free, "region": region, "lo": lo, "cell": cell, "walls_pts": walls_pts, "cams": d["cams"]}
    return lay


def _fallback(room_id, points, ceiling_h, why) -> Layout:
    z = points[:, 2]
    P = points[(z > 1.0)] if (z > 1.0).sum() > 100 else points
    x0, y0 = np.percentile(P[:, :2], 2, axis=0)
    x1, y1 = np.percentile(P[:, :2], 98, axis=0)
    V = np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])
    poly, walls = _walls_clockwise(V, [(0, float("nan"))] * 4, room_id)
    return Layout(room_id, poly, walls, round(abs(_signed_area(np.array(poly))), 4), ceiling_h, "partial",
                  "bbox_fallback", None, warnings=why + ["layout fell back to a bounding rectangle"])
