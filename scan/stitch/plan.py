"""Stitched multi-room plan (C7). In a joint run every room is already in one property frame
(placement = the joint reconstruction itself). Door-based stitching for separate runs is step 1.8."""

from __future__ import annotations

from shapely.geometry import Polygon

from scan.schema import Adjacency, DriftCorrection, Overlap, RoomPose


def _orient(w):
    return "h" if abs(w.end[1] - w.start[1]) < abs(w.end[0] - w.start[0]) else "v"


def all_shared_walls(polys_by_room: dict, max_gap: float, min_overlap: float) -> set[str]:
    """Every wall id that runs parallel to another room's wall within max_gap, overlapping >= min_overlap
    (for drawing; a room pair can share more than one wall, e.g. around an L)."""
    out: set[str] = set()
    rooms = list(polys_by_room.items())
    for i, (_, la) in enumerate(rooms):
        for _, lb in rooms[i + 1 :]:
            for a in la.walls:
                for b in lb.walls:
                    if _pair_ok(a, b, max_gap, min_overlap) is not None:
                        out |= {a.wall_id, b.wall_id}
    return out


def _pair_ok(a, b, max_gap, min_overlap):
    o = _orient(a)
    if o != _orient(b):
        return None
    k, span = (1, 0) if o == "h" else (0, 1)
    gap = abs(a.start[k] - b.start[k])
    a0, a1 = sorted((a.start[span], a.end[span]))
    b0, b1 = sorted((b.start[span], b.end[span]))
    overlap = min(a1, b1) - max(a0, b0)
    return overlap if gap <= max_gap and overlap >= min_overlap else None


def shared_walls(lay_a, lay_b, max_gap: float, min_overlap: float) -> tuple[str, str] | None:
    """Best pair of parallel walls (one per room) within max_gap of each other, overlapping >= min_overlap."""
    best = None
    for a in lay_a.walls:
        for b in lay_b.walls:
            o = _orient(a)
            if o != _orient(b):
                continue
            k, span = (1, 0) if o == "h" else (0, 1)
            gap = abs(a.start[k] - b.start[k])
            a0, a1 = sorted((a.start[span], a.end[span]))
            b0, b1 = sorted((b.start[span], b.end[span]))
            overlap = min(a1, b1) - max(a0, b0)
            if gap <= max_gap and overlap >= min_overlap and (best is None or overlap > best[0]):
                best = (overlap, (a.wall_id, b.wall_id))
    return None if best is None else best[1]


def stitch(clouds: dict, cfg: dict, warnings: list[str]):
    """-> (placement_method, poses, adjacency, overlaps, drift)"""
    sc = cfg["stitch"]
    frames = {c.frame_id for c in clouds.values()}
    by = {c.placed_by for c in clouds.values()}
    joint = by == {"joint_reconstruction"}
    single = len(clouds) == 1
    doors = "door_matching" in by
    joint or single or (doors and len(frames) == 1)
    poses = [RoomPose(room_id=r, x=round(c.pose[0], 4), y=round(c.pose[1], 4), theta_deg=c.pose[2],
                      placed=c.placed_by is not None) for r, c in clouds.items()]
    unplaced = [r for r, c in clouds.items() if c.placed_by is None]
    if unplaced:
        warnings.append(f"not placed in the plan (no door match): {', '.join(unplaced)}")
    adjacency, overlaps = [], []
    if not single:
        names = [r for r, c in clouds.items() if c.placed_by is not None]
        names = [r for r in names if clouds[r].frame_id == clouds[names[0]].frame_id] if names else []
        for i, ra in enumerate(names):
            for rb in names[i + 1 :]:
                la, lb = clouds[ra].layout, clouds[rb].layout
                sw = shared_walls(la, lb, sc["adjacency_max_gap_m"], sc["adjacency_min_overlap_m"])
                if sw:
                    via = next((o.opening_id for o in clouds[ra].openings if o.connects_to == rb), None)
                    via = via or next((o.opening_id for o in clouds[rb].openings if o.connects_to == ra), None)
                    adjacency.append(Adjacency(room_a=ra, room_b=rb, via_opening=via, shared_wall=sw))
                inter = Polygon(la.polygon).intersection(Polygon(lb.polygon)).area
                if inter > sc["overlap_tolerance_m2"]:
                    overlaps.append(Overlap(room_a=ra, room_b=rb, area_m2=round(inter, 3)))
    method = "joint_reconstruction" if joint else ("single_room" if single else ("door_matching" if doors else "unplaced"))
    snapped = (joint or doors) and sc["snap_walls"]
    drift = DriftCorrection(
        enabled=snapped,
        method=(("joint reconstruction in one shared frame" if joint else
                 "per-room reconstructions placed by visibility-scored door matching")
                + (f" + structural wall snapping (shared partitions to one {sc['wall_thickness_m']} m wall, "
                   "collinear outer walls to one line, best-seen face sets the line)" if snapped else
                   ", wall snapping OFF (ablation)")),
    )
    return method, poses, adjacency, overlaps, drift
