"""Room segmentation of one recording that walks through several rooms (LiDAR: one ARKit session, E25).

Rooms meet at doorways, and a doorway is a neck in the floor's free space no wider than a door. Cells farther than
half the widest interior door (segment.door_max_m) from any obstacle cannot be in a doorway: they form one core per
room (erosion, not opening: on either side of a thin wall, opened disks would touch through the door). The watershed
of the distance to the nearest obstacle, seeded with the cores, then gives every free cell to a room and puts each
boundary at the narrowest point of its neck: the doorway. An opening wider than a door (open kitchen, archway)
keeps both sides one space. Where the walk shows more than the floor, doorways also cut the free space directly:
wall above door height (a lintel, E22n) and doors the detector found (door_segments, E25g). A corridor narrower than
a door has no core of its own. The walk shows it: where the camera path leaves one room's body (every disk of half a
door's width that fits) and enters another's after more than a doorway's depth (segment.passage_min_length_m), it
crossed a passage; the floor around that stretch farther than a door's width (walking) from every room's core, if
room-sized, is seeded as its own space (E27). Each view goes to the room its camera stood in, so the rooms come out as if filmed one
by one in a shared frame, and each room's layout starts from its own segment (RoomCloud.region).
"""

from __future__ import annotations

import copy
from itertools import pairwise

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from scan.geometry.align import normals, room_planes, use_shared_floor
from scan.geometry.cloud import build_cloud
from scan.layout.room import _grid, _to_ij, carve_free_space, lintel_barrier, view_rays, wall_points
from scan.types import RoomCloud


def door_segments(views: dict, frames: list, dets: dict, cfg: dict) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Doorways from detector door boxes, in the aligned frame: (end a xy, end b xy, frames seen in).

    A box's two upright side edges are the door's jambs. In a strip straddling each (10% of the width outside to 15%
    inside: boxes often hug the opening, and inside it the depth sees through), the nearest surface is the jamb or
    the wall beside it: its median position is that end. Kept: door-sized (as detector_openings), along a house
    axis (snapped to it), and the same doorway in >= damage.min_photos frames (median of their ends)."""
    from scan.damage.detect import _frame_to_view

    sc = cfg["segment"]
    lo_w, hi_w = sc["door_box_width_m"]
    tol = np.radians(sc["door_axis_tol_deg"])
    idx = {str(n): i for i, n in enumerate(views["names"])}
    found = []  # (axis 0 = along x / 1 = along y, centre xy, width, frame name)
    for f in frames:
        i = idx.get(f.image_path.name)
        if i is None:
            continue
        H, W = views["depth"][i].shape
        s_, ox, oy = _frame_to_view(f.rgb.shape[:2], (H, W))
        horiz_x = f.upright_turns % 2 == 0  # the image axis that is horizontal in the world
        for d in dets.get(f.image_path.name, []):
            if d["cls"] != "door":
                continue
            x0, y0, x1, y1 = d["box"][0] * s_ - ox, d["box"][1] * s_ - oy, d["box"][2] * s_ - ox, d["box"][3] * s_ - oy
            (a0, a1), (c0, c1) = ((x0, x1), (y0, y1)) if horiz_x else ((y0, y1), (x0, x1))
            L, C = a1 - a0, c1 - c0
            ends = []
            for lo_a, hi_a in ((a0 - 0.10 * L, a0 + 0.15 * L), (a1 - 0.15 * L, a1 + 0.10 * L)):
                ra = slice(max(0, int(lo_a)), max(0, int(np.ceil(hi_a))))
                rc = slice(max(0, int(c0 + 0.3 * C)), max(0, int(np.ceil(c0 + 0.7 * C))))
                rows, cols = (rc, ra) if horiz_x else (ra, rc)
                m = views["mask"][i][rows, cols]
                if m.sum() < 5:
                    break
                dep = views["depth"][i][rows, cols][m]
                P = views["pts"][i][rows, cols][m][:, :2]
                near = dep <= np.percentile(dep, 20) + 0.15  # the jamb / wall, not what is seen through the door
                ends.append(np.median(P[near], 0))
            if len(ends) < 2:
                continue
            v = ends[1] - ends[0]
            w = float(np.linalg.norm(v))
            ang = np.arctan2(abs(v[1]), abs(v[0]))  # 0 = along x
            axis = 0 if ang < tol else 1 if ang > np.pi / 2 - tol else None
            if axis is None or not lo_w < w < hi_w:
                continue
            found.append((axis, (ends[0] + ends[1]) / 2, w, f.image_path.name))
    out = []
    used = [False] * len(found)
    for k, (axis, c, _, _) in enumerate(found):  # greedy consensus by centre
        if used[k]:
            continue
        grp = [j for j, g in enumerate(found) if not used[j] and g[0] == axis and np.linalg.norm(g[1] - c) < sc["door_merge_m"]]
        for j in grp:
            used[j] = True
        n = len({found[j][3] for j in grp})
        if n < cfg["damage"]["min_photos"]:
            continue
        cen = np.median([found[j][1] for j in grp], 0)
        half = np.median([found[j][2] for j in grp]) / 2
        e = np.array([1.0, 0.0]) if axis == 0 else np.array([0.0, 1.0])
        out.append((cen - half * e, cen + half * e, n))
    return out


def free_space_rooms(rays, cams_xy: np.ndarray, cfg: dict, walls_pts: np.ndarray | None = None, doors=()):
    """Label image of spaces on a grid (0 = not free), its origin and cell, each camera's label, and the labels that
    are passages (corridors narrower than a door) rather than rooms.
    walls_pts: wall points (aligned frame); those above door height (lintels) cut the free space (E22n).
    doors: doorway segments (door_segments) also cut it."""
    sc, lc = cfg["segment"], cfg["layout"]
    cell = sc["cell_m"]
    allxy = np.vstack([cams_xy] + [P[:, :2] for _, P in rays])
    lo, shape = _grid(allxy, cell, margin=0.5)
    free = carve_free_space(rays, lo, shape, cell, lc["ray_stop_short_m"]) >= lc["min_views_free"]
    kg = max(1, round(lc["free_gap_fill_m"] / cell))  # rays fan out with distance: merge neighbours
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kg, kg), np.uint8)).astype(bool)
    free_uncut = free.copy()  # before the doorway cuts: a corridor under a soffit is still one corridor
    if walls_pts is not None and sc.get("lintel_min_z_m"):
        # a door is lower than the wall it is in: wall seen above door height spans the doorway (lines of sight
        # pass under it), so it separates rooms even where the floor runs through; an open side has no wall above
        bar = lintel_barrier(walls_pts, lo, shape, {"layout": {**lc, "cell_m": cell, "lintel_min_z_m": sc["lintel_min_z_m"]}})
        free &= ~bar
    if len(doors):
        bar = np.zeros(free.shape, np.uint8)
        for a, b, _ in doors:
            e = (b - a) / max(np.linalg.norm(b - a), 1e-9) * cell  # one cell longer each end: meet the jambs
            (i0, j0), (i1, j1) = _to_ij(np.array([a - e, b + e]), lo, cell)
            cv2.line(bar, (int(i0), int(j0)), (int(i1), int(j1)), 1, 2)
        free &= ~bar.astype(bool)
    clear = ndimage.distance_transform_edt(free) * cell  # to the nearest cell no line of sight crossed
    cores, n = ndimage.label(clear > sc["door_max_m"] / 2)
    passages = []
    if n > 1 and len(cams_xy) > 1:
        passage = _passage_cells(cams_xy, free, free_uncut, cores, clear, lo, cell, sc)
        comps, m = ndimage.label(passage)
        for k in range(1, m + 1):
            comp = comps == k
            if comp.sum() * cell**2 >= lc["min_room_area_m2"]:
                cores[comp & (cores == 0)] = n + len(passages) + 1
                passages.append(n + len(passages) + 1)
        n += len(passages)
    labels = watershed(-clear, cores, mask=free) if n else cores
    for k in passages:  # a corridor under a soffit is one corridor: its seed keeps the cut cells it crosses
        labels[(cores == k) & ~free] = k
    for k in list(passages):  # each connected stretch of corridor is its own space (an outline is one piece)
        comps, m = ndimage.label(labels == k)
        sizes = ndimage.sum(np.ones_like(comps), comps, index=np.arange(1, m + 1))
        for j in np.argsort(-sizes)[1:]:
            if sizes[j] * cell**2 >= lc["min_room_area_m2"]:
                labels[comps == j + 1] = labels.max() + 1
                passages.append(int(labels.max()))
    # a 'room' smaller than a room is a recess or a lobe of the next room: give it to its neighbours
    area = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1)) * cell**2
    small = [k + 1 for k in range(n) if area[k] < lc["min_room_area_m2"]]
    if small and len(small) < n:
        seeds = np.where(np.isin(labels, small), 0, labels)
        labels = watershed(-clear, seeds, mask=free | np.isin(labels, passages))  # passages keep their cut cells
        passages = [k for k in passages if k not in small]
    ij = _to_ij(cams_xy, lo, cell).clip(0, [shape[0] - 1, shape[1] - 1])
    cam_lab = labels[ij[:, 1], ij[:, 0]]
    if (cam_lab == 0).any() and labels.max() > 0:  # camera on an unseen cell: nearest room cell
        _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        z = cam_lab == 0
        cam_lab[z] = labels[iy[ij[z, 1], ij[z, 0]], ix[ij[z, 1], ij[z, 0]]]
    return labels, lo, cell, cam_lab, set(passages)


def _passage_cells(cams_xy, free, free_uncut, cores, clear, lo, cell, sc) -> np.ndarray:
    """Floor around the walk's stretches between two different rooms' bodies that are longer than a doorway is deep."""
    r = round(sc["door_max_m"] / 2 / cell)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    body = cv2.dilate((cores > 0).astype(np.uint8), disk).astype(bool) & free
    body_lab = watershed(-clear, cores, mask=body)
    pts = np.vstack([np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / cell) + 1)) for a, b in pairwise(cams_xy)])
    ij = _to_ij(pts, lo, cell).clip(0, [free.shape[1] - 1, free.shape[0] - 1])
    lab = body_lab[ij[:, 1], ij[:, 0]]
    path = np.zeros(free.shape, np.uint8)
    runs, _ = ndimage.label(lab == 0)
    for k in range(1, runs.max() + 1):
        idx = np.flatnonzero(runs == k)
        if idx[0] == 0 or idx[-1] == len(lab) - 1:
            continue  # the walk starts or ends outside a body: no room on that side
        before, after = lab[idx[0] - 1], lab[idx[-1] + 1]
        length = np.linalg.norm(np.diff(pts[idx[0] - 1:idx[-1] + 2], axis=0), axis=1).sum()
        if before != after and length > sc["passage_min_length_m"]:
            path[ij[idx, 1], ij[idx, 0]] = 1
    # a corridor is floor more than a door's width, in walking distance, from every room's core: a cramped room (a
    # bathroom: small core, the walk near its fixtures) keeps the floor around its own core
    from skimage.graph import MCP_Geometric

    cost = np.where(free_uncut, 1.0, np.inf)
    dist, _ = MCP_Geometric(cost).find_costs(np.argwhere(cores > 0))
    return cv2.dilate(path, disk).astype(bool) & free_uncut & ~body & (dist * cell > sc["door_max_m"])


def split_recording(cloud: RoomCloud, cfg: dict, log=print, dets: dict | None = None) -> dict[str, RoomCloud]:
    """One aligned recording -> one RoomCloud per space (room1, room2, ..., passage1, ... in the order first filmed), sharing the
    recording's frame. Unchanged (one room) if the free space holds a single room. dets: detector boxes per frame."""
    g, sc = cfg["geometry"], cfg["segment"]
    rays = view_rays(cloud, cfg)
    cams = np.array([c for c, _ in rays])
    walls = wall_points(cloud.points, normals(cloud.points, cfg["alignment"]["normal_radius_m"]), cloud.planes.ceiling_z, cfg)
    doors = door_segments(cloud.views, cloud.frames, dets, cfg) if dets else []
    if doors:
        log(f"  segment {cloud.room_id}: {len(doors)} doorways from the detector "
            f"({', '.join(f'{np.linalg.norm(b - a):.2f} m x{n}' for a, b, n in doors)})")
    labels, lo, cell, cam_lab, passages = free_space_rooms(rays, cams, cfg, walls, doors)
    order = list(dict.fromkeys(int(k) for k in cam_lab if k > 0))  # first visit order
    keep = [k for k in order if (cam_lab == k).sum() >= sc["min_views_per_room"]]
    areas = {k: round(float((labels == k).sum() * cell**2), 2) for k in keep}
    log(f"  segment {cloud.room_id}: {labels.max()} free-space spaces ({len(passages)} passages), {len(keep)} filmed "
        f"from inside ({', '.join(f'{(cam_lab == k).sum()} views / {areas[k]} m2' for k in keep)})")
    if len(keep) < 2:
        return {cloud.room_id: cloud}
    # views of a room not kept (a glimpse from a doorway) go to the nearest kept room's views by time
    t = np.arange(len(cam_lab))
    kept_t = np.flatnonzero(np.isin(cam_lab, keep))
    nearest = kept_t[np.abs(t[:, None] - kept_t[None]).argmin(1)]
    room_of = np.where(np.isin(cam_lab, keep), cam_lab, cam_lab[nearest])
    out = {}
    count = {"room": 0, "passage": 0}
    for k in keep:
        idx = np.flatnonzero(room_of == k)
        kind = "passage" if k in passages else "room"
        count[kind] += 1
        rid = f"{kind}{count[kind]}"
        v = {key: val[idx] for key, val in cloud.views.items()}
        pts, cols = build_cloud(v, list(range(len(idx))), g["rays"], g["conf_percentile"], g["voxel_m"])
        names = set(map(str, v["names"]))
        frames = [copy.copy(f) for f in cloud.frames if f.image_path.name in names]
        for f in frames:
            f.room_hint = rid
        c = RoomCloud(room_id=rid, points=pts, colors=cols, frames=frames, scale=cloud.scale,
                      T_room_world=cloud.T_room_world, frame_id=cloud.frame_id, views=v, alignment=cloud.alignment,
                      up_world=cloud.up_world, region=(labels, lo, cell, k), kind=kind)
        c.planes = use_shared_floor(room_planes(pts, cfg), cfg["alignment"]["shared_floor_tol_m"])
        out[rid] = c
    return out
