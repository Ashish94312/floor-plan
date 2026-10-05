"""C6 Openings (ARCHITECTURE §10.8, D28): doors, windows and open doorways by visibility voting.

For every small cell on a wall (2 cm along x 5 cm up), every photo that has the cell in view casts a
vote from its raw depth map: the camera saw the wall there (depth ~ distance to the wall), saw
THROUGH it (depth beyond the wall: an opening), or saw something in front (furniture: no vote).
Raw depth includes the far, low-confidence pixels that the point-cloud filter drops, which is what a
view through a doorway looks like. Open cells form rectangles: from the floor with a lintel = door,
no lintel = opening, wall below = window. Closed doors look like wall here (need a detector).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from scan.geometry.cloud import ray_K


@dataclass
class OpeningSeg:
    opening_id: str
    type: str  # door | window | opening
    wall_id: str
    offset_m: float  # from the wall's start to the opening's near edge
    width_m: float
    height_m: float
    sill_m: float | None
    views: int  # photos that saw through it
    connects_to: str | None = None
    notes: list[str] = field(default_factory=list)
    score: float | None = None  # detector openings: median box score over its photos


def wall_votes(wall, views, ceiling_z, cfg) -> tuple[np.ndarray, np.ndarray, float]:
    """(open votes, wall votes) on the wall's u x z grid, and the wall length."""
    oc = cfg["openings"]
    du, dz, margin = oc["cell_u_m"], oc["cell_z_m"], oc["depth_margin_m"]
    a, b = np.array(wall.start, float), np.array(wall.end, float)
    L = float(np.linalg.norm(b - a))
    u_dir = (b - a) / L
    nu, nz = int(np.ceil(L / du)), int(np.ceil(ceiling_z / dz))
    uu, zz = np.meshgrid((np.arange(nu) + 0.5) * du, (np.arange(nz) + 0.5) * dz)
    P = np.stack([a[0] + uu * u_dir[0], a[1] + uu * u_dir[1], zz], -1).reshape(-1, 3)  # cell centres on the face
    open_v = np.zeros(P.shape[0])
    wall_v = np.zeros(P.shape[0])
    seen_by = np.zeros(P.shape[0])
    for i in range(len(views["depth"])):
        T = views["T_wc"][i]
        R, t = T[:3, :3], T[:3, 3]
        cam = (P - t) @ R  # world -> camera
        zc = cam[:, 2]
        K = ray_K(views, i, cfg["geometry"]["rays"])
        H, W = views["depth"][i].shape
        with np.errstate(divide="ignore", invalid="ignore"):
            px = K[0, 0] * cam[:, 0] / zc + K[0, 2]
            py = K[1, 1] * cam[:, 1] / zc + K[1, 2]
        ok = (zc > 0.3) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        ix, iy = px[ok].astype(int), py[ok].astype(int)
        valid = views["mask"][i][iy, ix]
        d_obs = views["depth"][i][iy, ix]
        d_wall = zc[ok]
        idx = np.flatnonzero(ok)[valid]
        d_obs, d_wall = d_obs[valid], d_wall[valid]
        through = d_obs > d_wall + margin
        on = np.abs(d_obs - d_wall) <= margin
        open_v[idx[through]] += 1
        wall_v[idx[on]] += 1
        seen_by[idx[through]] += 1  # count photos voting open (for `views`)
    return open_v.reshape(nz, nu), wall_v.reshape(nz, nu), L


def lintel_seen(open_v: np.ndarray, wall_v: np.ndarray, cols: slice, top_m: float, cfg: dict) -> bool:
    """Was the wall above an opening's top seen? In the band above the top (vote grid rows = cell_z_m) at least 2 rows
    hold votes and, over the band, wall votes dominate. Over the band, not per row: the reported top is a percentile of
    the open cells, so the first row above it straddles the edge and still holds some see-through votes."""
    oc, sc = cfg["openings"], cfg["stitch"]
    dz = oc["cell_z_m"]
    k0 = int(np.ceil(top_m / dz - 1e-6))
    k1 = min(open_v.shape[0], k0 + round(sc["door_lintel_band_m"] / dz))
    o, w = open_v[k0:k1, cols].sum(1), wall_v[k0:k1, cols].sum(1)
    voted = (o + w) > 0
    return bool(voted.sum() >= 2 and w.sum() >= oc["min_votes"] and w.sum() / (o.sum() + w.sum()) >= oc["min_open_ratio"])


def extract(wall, open_v, wall_v, L, ceiling_z, cfg) -> list[dict]:
    """Rectangles of seen-through cells -> openings. Width runs jamb to jamb: from the see-through
    region outward through cells with no wall evidence, stopping at the first column that votes 'wall'
    (an open door leaf hides part of the opening from oblique views, so see-through alone is too narrow)."""
    oc = cfg["openings"]
    du, dz = oc["cell_u_m"], oc["cell_z_m"]
    tot = open_v + wall_v
    ratio = np.where(tot > 0, open_v / np.maximum(tot, 1), 0.0)
    is_open = (ratio >= oc["min_open_ratio"]) & (open_v >= oc["min_votes"])
    is_wall = (tot > 0) & (ratio < 1 - oc["min_open_ratio"])
    is_open = ndimage.binary_opening(is_open, np.ones((2, 2)))
    lab, n = ndimage.label(is_open)
    out = []
    for k in range(1, n + 1):
        zs, us = np.nonzero(lab == k)
        w_m, h_m = (us.max() - us.min() + 1) * du, (zs.max() - zs.min() + 1) * dz
        if w_m < oc["min_width_m"] or h_m < oc["min_height_m"]:
            continue
        z0, z1 = np.percentile(zs, 3) * dz, (np.percentile(zs, 97) + 1) * dz
        mid = slice(int((z0 + 0.2) / dz), max(int((z0 + 0.2) / dz) + 1, int((z1 - 0.2) / dz)))
        wallness = is_wall[mid].mean(0)  # per column: share of the opening's rows voting wall
        lo, hi = int(np.percentile(us, 3)), int(np.percentile(us, 97))
        while lo > 0 and wallness[lo - 1] < 0.5:
            lo -= 1
        while hi < len(wallness) - 1 and wallness[hi + 1] < 0.5:
            hi += 1
        width = (hi - lo + 1) * du
        if width > oc["max_width_m"]:
            continue
        views = int(open_v[lab == k].max())
        if z0 <= oc["floor_gap_m"]:
            kind = "opening" if z1 >= ceiling_z - oc["lintel_min_m"] else "door"
            sill, height = None, z1
            if kind == "door" and z1 < oc["min_door_height_m"] and lintel_seen(open_v, wall_v, slice(lo, hi + 1), z1, cfg):
                continue  # its top was seen and no doorway is that low: a gap under furniture (E23a)
        else:
            kind, sill, height = "window", z0, z1 - z0
            if width < oc["min_window_m"] or height < oc["min_window_m"] or views < oc["min_window_views"]:
                continue  # small, single-photo see-through patches are noise
        out.append({"type": kind, "u0": lo * du, "width": width, "height": height, "sill": sill, "views": views})
    return out


def detect_openings(layouts: dict, clouds: dict, cfg: dict) -> dict[str, list[OpeningSeg]]:
    """Openings for every room's real walls; connects_to = the room on the other side of a shared wall."""
    from scan.layout.room import _facing

    t = cfg["stitch"]["wall_thickness_m"]
    faces = [(r, w) for r, lay in layouts.items() for w in lay.walls]
    frame = {r: clouds[r].frame_id for r in layouts}  # only rooms in the same frame can share a wall
    result: dict[str, list[OpeningSeg]] = {}
    for r, lay in layouts.items():
        c = clouds[r]
        ceil = c.planes.ceiling_z or float(np.percentile(c.points[:, 2], 98))
        ops = []
        for w in lay.walls:
            if w.kind != "wall" or w.support < cfg["layout"]["min_wall_support"]:
                continue  # an unseen wall (no points, e.g. behind every camera) would read as one wide opening
            o, wv, L = wall_votes(w, c.views, ceil, cfg)
            partners = [(pr, pw) for pr, pw in faces if pr != r and frame[pr] == frame[r] and _facing(w, pw, t + 0.15)]
            a, b = np.array(w.start), np.array(w.end)
            u_dir = (b - a) / L
            for d in extract(w, o, wv, L, ceil, cfg):
                mid = a + u_dir * (d["u0"] + d["width"] / 2)  # opening centre in the plan
                beyond = None
                if d["type"] != "window":
                    for pr, pw in partners:  # the room whose wall face spans the opening's centre
                        pa, pb = np.array(pw.start), np.array(pw.end)
                        s_ = float(np.dot(mid - pa, (pb - pa) / np.linalg.norm(pb - pa)))
                        if 0 < s_ < np.linalg.norm(pb - pa):
                            beyond = pr
                            break
                ops.append(OpeningSeg(
                    opening_id=f"{r}-O{len(ops) + 1}", type=d["type"], wall_id=w.wall_id,
                    offset_m=round(d["u0"], 3), width_m=round(d["width"], 3), height_m=round(d["height"], 3),
                    sill_m=None if d["sill"] is None else round(d["sill"], 3), views=d["views"], connects_to=beyond,
                ))
        result[r] = ops
    _share_partition_openings(layouts, result, t, cfg["layout"]["min_wall_support"])
    return result


def _share_partition_openings(layouts: dict, result: dict, t: float, min_support: int) -> None:
    """A door between two rooms is one hole in one partition. A room whose face of that partition was too poorly
    seen to vote (unsupported, skipped above) gets the opening its neighbour saw, at the same place (E23a)."""
    from scan.layout.room import _facing

    for r, ops in list(result.items()):
        for op in list(ops):
            if op.type == "window" or op.connects_to not in layouts:
                continue
            w = next(w for w in layouts[r].walls if w.wall_id == op.wall_id)
            a, b = np.array(w.start), np.array(w.end)
            u = (b - a) / np.linalg.norm(b - a)
            p0, p1 = a + u * op.offset_m, a + u * (op.offset_m + op.width_m)
            for pw in layouts[op.connects_to].walls:
                if not _facing(w, pw, t + 0.15) or pw.kind != "wall":
                    continue
                pa, pb = np.array(pw.start), np.array(pw.end)
                L2 = float(np.linalg.norm(pb - pa))
                s0, s1 = sorted((float((p0 - pa) @ (pb - pa) / L2), float((p1 - pa) @ (pb - pa) / L2)))
                s0, s1 = max(s0, 0.0), min(s1, L2)
                if s1 - s0 < 0.5 * op.width_m:
                    continue
                theirs = result[op.connects_to]
                if not any(o.wall_id == pw.wall_id and min(s1, o.offset_m + o.width_m) - max(s0, o.offset_m) > 0.5 * (s1 - s0)
                           for o in theirs) and pw.support < min_support:
                    theirs.append(OpeningSeg(f"{op.connects_to}-O{len(theirs) + 1}", op.type, pw.wall_id, round(s0, 3),
                                             round(s1 - s0, 3), op.height_m, op.sill_m, op.views, connects_to=r,
                                             notes=[f"seen from {r}: this face of the wall was barely filmed"]))
                break


def reindex_by_main_door(layout, openings: list[OpeningSeg]) -> None:
    """D11: W1 = the wall holding the main door, then clockwise. Main door = the widest door/opening
    leading out of the captured rooms (front door; connects_to None), else the widest door/opening,
    else an open boundary (open kitchen). Rooms without any keep the provisional order."""
    doors = [o for o in openings if o.type in ("door", "opening")]
    pick = None
    if doors:
        outward = [o for o in doors if o.connects_to is None]
        pick = max(outward or doors, key=lambda o: o.width_m).wall_id
    else:
        open_walls = [w for w in layout.walls if w.kind == "open"]
        if open_walls:
            pick = max(open_walls, key=lambda w: w.length_m).wall_id
    if pick is None:
        return
    k = [w.wall_id for w in layout.walls].index(pick)
    layout.walls = layout.walls[k:] + layout.walls[:k]
    room = pick.rsplit("-W", 1)[0]
    rename = {}
    for i, w in enumerate(layout.walls, 1):
        rename[w.wall_id] = f"{room}-W{i}"
        w.wall_id = f"{room}-W{i}"
    layout.polygon = [w.start for w in layout.walls]
    layout.w1_rule = "W1 = wall with the main door (D11)" if doors else "W1 = open side (no door found)"
    for o in openings:
        o.wall_id = rename.get(o.wall_id, o.wall_id)


def drop_reflections(layouts: dict, clouds: dict, openings: dict, cfg: dict) -> list[str]:
    """Remove windows that are reflections of a window on the wall next to them (see reflections). Returns log lines."""
    log = []
    du, dz = cfg["openings"]["cell_u_m"], cfg["openings"]["cell_z_m"]
    for r, ops in openings.items():
        lay, c = layouts[r], clouds[r]
        walls = {w.wall_id: w for w in lay.walls}
        ceil = c.planes.ceiling_z or float(np.percentile(c.points[:, 2], 98))
        wins = [o for o in ops if o.type == "window" and o.wall_id in walls]
        through = {}
        for o in wins:  # open cells (as extract defines them) inside the window's rectangle
            ov, wv, _L = wall_votes(walls[o.wall_id], c.views, ceil, cfg)
            sl = (slice(int(o.sill_m / dz), int((o.sill_m + o.height_m) / dz) + 1),
                  slice(int(o.offset_m / du), int((o.offset_m + o.width_m) / du) + 1))
            ratio = ov[sl] / np.maximum(ov[sl] + wv[sl], 1)
            through[o.opening_id] = int(((ratio >= cfg["openings"]["min_open_ratio"]) & (ov[sl] >= cfg["openings"]["min_votes"])).sum())
        drop = {}
        for img, real in reflections(wins, walls, through):
            drop[img.opening_id] = real.opening_id
        for img_id, real_id in drop.items():
            log.append(f"{img_id} removed: reflection of {real_id} in the glossy wall next to it "
                       "(same height, at the shared corner, no depth beyond the wall)")
        ops[:] = [o for o in ops if o.opening_id not in drop]
    return log


def reflections(wins: list, walls: dict, through: dict) -> list[tuple]:
    """A window and its reflection in a glossy wall next to it (marble tiles, glass; E23a: the kitchen's sink window
    mirrored in the tiles of the adjacent wall, found by the detector in 4 photos). A vertical plane mirror keeps
    heights and puts the image against the shared corner. So two windows on walls that meet at a corner, both
    reaching the corner (each one's near edge closer to it than the other's far edge) and at overlapping heights
    (>= half the lower one) are one window and its image. Which is which: the image lies on the wall surface (no
    open cells), a window seen through has some; both seen through = a real corner window, keep both. When depth
    can't tell (neither seen through: grilles and glass read as wall, E23a), the image is the dimmer copy (a reflection
    loses light), so the detector is less sure of it: the lower median score is the image. -> [(image, window)]"""
    out = []
    for o1, o2 in ((p, q) for i, p in enumerate(wins) for q in wins[i + 1:]):
        if o1.wall_id == o2.wall_id:
            continue
        w1, w2 = walls[o1.wall_id], walls[o2.wall_id]
        corner = next(((e1, e2) for e1 in ("start", "end") for e2 in ("start", "end")
                       if np.allclose(getattr(w1, e1), getattr(w2, e2), atol=1e-3)), None)
        if corner is None:
            continue

        def near_far(o, w, end):  # distances of the window's near and far edge from the shared corner
            a, b = o.offset_m, o.offset_m + o.width_m
            return (a, b) if end == "start" else (w.length_m - b, w.length_m - a)

        n1, f1 = near_far(o1, w1, corner[0])
        n2, f2 = near_far(o2, w2, corner[1])
        h_overlap = min(o1.sill_m + o1.height_m, o2.sill_m + o2.height_m) - max(o1.sill_m, o2.sill_m)
        if n1 > f2 or n2 > f1 or h_overlap < 0.5 * min(o1.height_m, o2.height_m):
            continue
        t1, t2 = through[o1.opening_id] > 0, through[o2.opening_id] > 0
        if t1 != t2:
            out.append((o2, o1) if t1 else (o1, o2))
        elif not t1 and o1.score is not None and o2.score is not None and o1.score != o2.score:
            out.append((o2, o1) if o1.score > o2.score else (o1, o2))
    return out


def drop_one_sided_doors(layouts: dict, openings: dict, cfg: dict) -> list[str]:
    """A door through a partition is one hole with two faces (see _share_partition_openings). A detector door on a
    partition whose other face was well seen (supported) by the neighbouring room, which found no opening there at
    all (neither see-through nor detector), has nothing behind it: a wall panel the detector took for a door (E23a:
    a white panel beside the bedroom doorway, scores 0.26 / 0.34). Doors with no filmed room behind them (a front
    door) are left alone. Returns log lines."""
    from scan.layout.room import _facing

    t = cfg["stitch"]["wall_thickness_m"]
    min_support = cfg["layout"]["min_wall_support"]
    log, drop = [], {}
    for r, ops in openings.items():
        for op in ops:
            if op.type != "door" or not any(n.startswith("detector") for n in op.notes):
                continue
            w = next(w for w in layouts[r].walls if w.wall_id == op.wall_id)
            a, b = np.array(w.start), np.array(w.end)
            u = (b - a) / np.linalg.norm(b - a)
            p0, p1 = a + u * op.offset_m, a + u * (op.offset_m + op.width_m)
            for r2, lay2 in layouts.items():
                if r2 == r:
                    continue
                for pw in lay2.walls:
                    if not _facing(w, pw, t + 0.15) or pw.kind != "wall" or pw.support < min_support:
                        continue
                    pa, pb = np.array(pw.start), np.array(pw.end)
                    L2 = float(np.linalg.norm(pb - pa))
                    s0, s1 = sorted((float((p0 - pa) @ (pb - pa) / L2), float((p1 - pa) @ (pb - pa) / L2)))
                    s0, s1 = max(s0, 0.0), min(s1, L2)
                    if s1 - s0 < 0.5 * op.width_m:
                        continue  # the neighbour's face covers too little of the door to judge
                    if not any(o.wall_id == pw.wall_id and min(s1, o.offset_m + o.width_m) - max(s0, o.offset_m) > 0
                               for o in openings.get(r2, [])):
                        drop.setdefault(r, set()).add(op.opening_id)
                        log.append(f"{op.opening_id} removed: detector door with a well-seen solid wall behind it "
                                   f"({pw.wall_id}, {pw.support} points, no opening there)")
    for r, ids in drop.items():
        openings[r][:] = [o for o in openings[r] if o.opening_id not in ids]
    return log
