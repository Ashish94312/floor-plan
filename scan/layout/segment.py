"""Room segmentation of one recording that walks through several rooms (LiDAR: one ARKit session, E25).

Rooms meet at doorways, and a doorway is a neck in the floor's free space no wider than a door. Cells farther than
half the widest interior door (segment.door_max_m) from any obstacle cannot be in a doorway: they form one core per
room (erosion, not opening: on either side of a thin wall, opened disks would touch through the door). The watershed
of the distance to the nearest obstacle, seeded with the cores, then gives every free cell to a room and puts each
boundary at the narrowest point of its neck: the doorway. An opening wider than a door (open kitchen, archway)
keeps both sides one space. Each view goes to the room its camera stood in, so the rooms come out as if filmed one
by one in a shared frame, and each room's layout starts from its own segment (RoomCloud.region).
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from scan.geometry.align import normals, room_planes, use_shared_floor
from scan.geometry.cloud import build_cloud
from scan.layout.room import _grid, _to_ij, carve_free_space, lintel_barrier, view_rays, wall_points
from scan.types import RoomCloud


def free_space_rooms(rays, cams_xy: np.ndarray, cfg: dict, walls_pts: np.ndarray | None = None):
    """Label image of rooms on a grid (0 = not free), its origin and cell, and each camera's room label.
    walls_pts: wall points (aligned frame); those above door height (lintels) cut the free space (E22n)."""
    sc, lc = cfg["segment"], cfg["layout"]
    cell = sc["cell_m"]
    allxy = np.vstack([cams_xy] + [P[:, :2] for _, P in rays])
    lo, shape = _grid(allxy, cell, margin=0.5)
    free = carve_free_space(rays, lo, shape, cell, lc["ray_stop_short_m"]) >= lc["min_views_free"]
    kg = max(1, round(lc["free_gap_fill_m"] / cell))  # rays fan out with distance: merge neighbours
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((kg, kg), np.uint8)).astype(bool)
    if walls_pts is not None and sc.get("lintel_min_z_m"):
        # a door is lower than the wall it is in: wall seen above door height spans the doorway (lines of sight
        # pass under it), so it separates rooms even where the floor runs through; an open side has no wall above
        bar = lintel_barrier(walls_pts, lo, shape, {"layout": {**lc, "cell_m": cell, "lintel_min_z_m": sc["lintel_min_z_m"]}})
        free &= ~bar
    clear = ndimage.distance_transform_edt(free) * cell  # to the nearest cell no line of sight crossed
    cores, n = ndimage.label(clear > sc["door_max_m"] / 2)
    labels = watershed(-clear, cores, mask=free) if n else cores
    # a 'room' smaller than a room is a recess or a lobe of the next room: give it to its neighbours
    area = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1)) * cell**2
    small = [k + 1 for k in range(n) if area[k] < lc["min_room_area_m2"]]
    if small and len(small) < n:
        seeds = np.where(np.isin(labels, small), 0, labels)
        labels = watershed(-clear, seeds, mask=free)
    ij = _to_ij(cams_xy, lo, cell).clip(0, [shape[0] - 1, shape[1] - 1])
    cam_lab = labels[ij[:, 1], ij[:, 0]]
    if (cam_lab == 0).any() and labels.max() > 0:  # camera on an unseen cell: nearest room cell
        _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
        z = cam_lab == 0
        cam_lab[z] = labels[iy[ij[z, 1], ij[z, 0]], ix[ij[z, 1], ij[z, 0]]]
    return labels, lo, cell, cam_lab


def split_recording(cloud: RoomCloud, cfg: dict, log=print) -> dict[str, RoomCloud]:
    """One aligned recording -> one RoomCloud per room (room1, room2, ... in the order first filmed), sharing the
    recording's frame. Unchanged (one room) if the free space holds a single room."""
    g, sc = cfg["geometry"], cfg["segment"]
    rays = view_rays(cloud, cfg)
    cams = np.array([c for c, _ in rays])
    walls = wall_points(cloud.points, normals(cloud.points, cfg["alignment"]["normal_radius_m"]), cloud.planes.ceiling_z, cfg)
    labels, lo, cell, cam_lab = free_space_rooms(rays, cams, cfg, walls)
    order = list(dict.fromkeys(int(k) for k in cam_lab if k > 0))  # first visit order
    keep = [k for k in order if (cam_lab == k).sum() >= sc["min_views_per_room"]]
    areas = {k: round(float((labels == k).sum() * cell**2), 2) for k in keep}
    log(f"  segment {cloud.room_id}: {labels.max()} free-space rooms, {len(keep)} filmed from inside "
        f"({', '.join(f'{(cam_lab == k).sum()} views / {areas[k]} m2' for k in keep)})")
    if len(keep) < 2:
        return {cloud.room_id: cloud}
    # views of a room not kept (a glimpse from a doorway) go to the nearest kept room's views by time
    t = np.arange(len(cam_lab))
    kept_t = np.flatnonzero(np.isin(cam_lab, keep))
    nearest = kept_t[np.abs(t[:, None] - kept_t[None]).argmin(1)]
    room_of = np.where(np.isin(cam_lab, keep), cam_lab, cam_lab[nearest])
    out = {}
    for n_, k in enumerate(keep, 1):
        idx = np.flatnonzero(room_of == k)
        rid = f"room{n_}"
        v = {key: val[idx] for key, val in cloud.views.items()}
        pts, cols = build_cloud(v, list(range(len(idx))), g["rays"], g["conf_percentile"], g["voxel_m"])
        names = set(map(str, v["names"]))
        frames = [copy.copy(f) for f in cloud.frames if f.image_path.name in names]
        for f in frames:
            f.room_hint = rid
        c = RoomCloud(room_id=rid, points=pts, colors=cols, frames=frames, scale=cloud.scale,
                      T_room_world=cloud.T_room_world, frame_id=cloud.frame_id, views=v, alignment=cloud.alignment,
                      up_world=cloud.up_world, region=(labels, lo, cell, k))
        c.planes = use_shared_floor(room_planes(pts, cfg), cfg["alignment"]["shared_floor_tol_m"])
        out[rid] = c
    return out
