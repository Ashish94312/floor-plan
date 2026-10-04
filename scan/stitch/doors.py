"""C7 photo-tier stitching of separately reconstructed rooms by door matching (ARCHITECTURE §10.9, D31).

Each room is already levelled and squared (floor z=0, walls on x/y), so placing room B in room A's
frame is a rotation by a multiple of 90 deg plus a shift. A door pair fixes both: B's door wall must
face A's, and the door centres sit one wall thickness apart. Candidates are scored by VISIBILITY:
depth seen through A's door should land inside B's outline, and through B's door inside A's (rooms
can have several similar doors; only what was seen through them tells which pair is right).
Accepted pairs form a spanning tree from the best-observed room; extra pairs (loops) are resolved by
least squares on the door constraints (drift correction; off for the ablation).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon


def _door_geom(layout, o):
    w = next(w for w in layout.walls if w.wall_id == o.wall_id)
    a, b = np.array(w.start), np.array(w.end)
    u = (b - a) / np.linalg.norm(b - a)
    n_out = np.array([-u[1], u[0]])  # clockwise polygon -> left = outside
    c = a + u * (o.offset_m + o.width_m / 2)
    return c, n_out, a + u * o.offset_m, a + u * (o.offset_m + o.width_m)


def _rot(k: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s], [s, c]], float)


def _candidate(geo_a, geo_b, t: float):
    """Rotation index + shift placing B so its door faces A's door one wall thickness away."""
    ca, na, _, _ = geo_a
    cb, nb, _, _ = geo_b
    for k in range(4):
        if np.allclose(_rot(k) @ nb, -na, atol=1e-6):
            R = _rot(k)
            return k, ca + na * t - R @ cb
    return None


def _through_points(cloud, geo, rays: str = "exif", max_per_view: int = 4000) -> np.ndarray:
    """Raw-depth 3D points (2D x,y) seen THROUGH a door from this room's photos (beyond the wall)."""
    from scan.geometry.cloud import ray_K, unproject

    c, n_out, p0, p1 = geo
    v = cloud.views
    rng = np.random.default_rng(0)
    out = []
    for i in range(len(v["depth"])):
        P = unproject(v["depth"][i], ray_K(v, i, rays), v["T_wc"][i])[v["mask"][i]][:, :2]
        cam = v["T_wc"][i][:2, 3]
        beyond = (P - c) @ n_out > 0.25
        if not beyond.any():
            continue
        P = P[beyond]
        # 2D segment camera->point must cross the door segment
        d = P - cam
        e = p1 - p0
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ = ((p0[0] - cam[0]) * e[1] - (p0[1] - cam[1]) * e[0]) / den
            s_ = ((p0[0] - cam[0]) * d[:, 1] - (p0[1] - cam[1]) * d[:, 0]) / den
        hit = (t_ > 0) & (t_ < 1) & (s_ >= 0) & (s_ <= 1)
        P = P[hit]
        if len(P) > max_per_view:
            P = P[rng.choice(len(P), max_per_view, replace=False)]
        out.append(P)
    return np.concatenate(out) if out else np.zeros((0, 2))


def _inside_share(pts: np.ndarray, poly: Polygon, margin: float) -> float | None:
    if len(pts) < 30:
        return None
    grown = poly.buffer(margin)
    return float(np.mean([grown.contains(Point(p)) for p in pts]))


def place_rooms(clouds: dict, layouts: dict, openings: dict, cfg: dict) -> dict:
    """-> {room: {"R": 2x2, "t": (2,), "theta_deg": int, "placed": bool, "via": (door_a, door_b) | None}}"""
    sc = cfg["stitch"]
    t, margin = sc["wall_thickness_m"], sc["door_match_margin_m"]
    names = list(layouts)
    polys = {r: Polygon(layouts[r].polygon) for r in names}
    doors = {r: [o for o in openings[r] if o.type in ("door", "opening")] for r in names}
    geo = {(r, o.opening_id): _door_geom(layouts[r], o) for r in names for o in doors[r]}
    through = {k: _through_points(clouds[k[0]], g, cfg["geometry"]["rays"]) for k, g in geo.items()}
    edges = []
    for i, ra in enumerate(names):
        for rb in names[i + 1 :]:
            best = None
            for oa in doors[ra]:
                for ob in doors[rb]:
                    if abs(oa.width_m - ob.width_m) > sc["door_width_tol_m"]:
                        continue
                    ga, gb = geo[(ra, oa.opening_id)], geo[(rb, ob.opening_id)]
                    cand = _candidate(ga, gb, t)
                    if cand is None:
                        continue
                    k, sh = cand
                    R = _rot(k)
                    poly_b = Polygon((np.array(layouts[rb].polygon) @ R.T) + sh)
                    if poly_b.intersection(polys[ra]).area > sc["overlap_tolerance_m2"] * 10:
                        continue
                    sa = _inside_share(through[(ra, oa.opening_id)], poly_b, margin)
                    pts_b = through[(rb, ob.opening_id)] @ R.T + sh if len(through[(rb, ob.opening_id)]) else np.zeros((0, 2))
                    sb = _inside_share(pts_b, polys[ra], margin)
                    scores = [s for s in (sa, sb) if s is not None]
                    score = float(np.mean(scores)) if scores else 0.0
                    if score >= sc["door_match_min_score"] and (best is None or score > best[0]):
                        best = (score, k, sh, oa.opening_id, ob.opening_id)
            if best:
                edges.append((ra, rb) + best)
    # spanning tree from the room with the most photos (max score first)
    root = max(names, key=lambda r: len(clouds[r].views["depth"]))
    place = {r: {"R": np.eye(2), "t": np.zeros(2), "theta_deg": 0, "placed": r == root, "via": None, "score": None} for r in names}
    for ra, rb, score, k, sh, da, db in sorted(edges, key=lambda e: -e[2]):
        for a, b, kk, ss, dd in ((ra, rb, k, sh, (da, db)), (rb, ra, -k, -(_rot(-k) @ sh), (db, da))):
            if place[a]["placed"] and not place[b]["placed"]:
                Ra, ta = place[a]["R"], place[a]["t"]
                R = Ra @ _rot(kk)
                place[b].update(R=R, t=Ra @ ss + ta, theta_deg=round(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) % 360,
                                placed=True, via=dd, score=round(score, 3))
    place["_edges"] = [(ra, rb, round(score, 3), da, db) for ra, rb, score, k, sh, da, db in edges]
    return place


def apply_placement(c, layout, R: np.ndarray, t: np.ndarray) -> None:
    """Move a room (cloud, camera poses, layout) into the property frame by a 2D rotation + shift."""
    M = np.eye(4)
    M[:2, :2], M[:2, 3] = R, t
    c.points = c.points @ M[:3, :3].T + M[:3, 3]
    c.views["T_wc"] = np.einsum("ij,sjk->sik", M, c.views["T_wc"])
    c.views["pts"] = (c.views["pts"].reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]).reshape(c.views["pts"].shape).astype(np.float32)
    for w in layout.walls:
        w.start = tuple(np.round(R @ np.array(w.start) + t, 4).tolist())
        w.end = tuple(np.round(R @ np.array(w.end) + t, 4).tolist())
    layout.polygon = [w.start for w in layout.walls]
    dbg = getattr(layout, "_debug", None)
    if dbg is not None and "walls_pts" in dbg:
        wp = dbg["walls_pts"]
        dbg["walls_pts"] = wp @ M[:3, :3].T + M[:3, 3]
        dbg.pop("free", None)  # the carving grid stays in the room's own frame: not drawn after stitching
    c.T_room_world = M @ c.T_room_world
