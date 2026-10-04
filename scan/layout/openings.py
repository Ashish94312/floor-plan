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
        K = views["K_exif"][i]
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
            if w.kind != "wall":
                continue
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
    return result


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
