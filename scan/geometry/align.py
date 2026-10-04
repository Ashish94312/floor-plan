"""C3 Alignment (ARCHITECTURE §10.5): level the cloud and square it to the walls.

After this step every room cloud is in an aligned frame: floor at z = 0, z up (gravity), and the
dominant wall directions along x and y (Manhattan world, D9). In a joint run all rooms share one
aligned frame, so their relative placement is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class Alignment:
    T: np.ndarray  # 4x4: backbone world -> aligned frame
    tilt_correction_deg: float  # camera-based up vs fitted floor normal
    floor_ceiling_angle_deg: float | None  # ~0 if both planes are level
    manhattan_deg: float  # rotation applied about z
    manhattan_support: float  # share of wall points within 5 deg of an axis
    floor_rms_m: float  # flatness of the floor fit
    floor_points: int


@dataclass
class RoomPlanes:
    floor_z: float  # room's own floor height in the aligned frame (~0)
    ceiling_z: float | None
    ceiling_height_m: float | None  # ceiling_z - floor_z
    floor_rms_m: float
    ceiling_rms_m: float | None
    status: str  # "ok" | "partial"
    warnings: list[str] = field(default_factory=list)


def normals(points: np.ndarray, radius: float) -> np.ndarray:
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    return np.asarray(pc.normals)


def _peaks(h: np.ndarray, bin_m: float, rel: float) -> np.ndarray:
    """Centres of histogram peaks (heights along up) holding >= rel x the biggest peak."""
    edges = np.arange(h.min() - bin_m, h.max() + 2 * bin_m, bin_m)
    hist, _ = np.histogram(h, edges)
    smooth = gaussian_filter1d(hist.astype(float), 1.0)
    idx, _ = find_peaks(smooth, height=rel * smooth.max())
    return (edges[idx] + edges[idx + 1]) / 2


def _fit_plane(P: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Least-squares plane through P: unit normal oriented with `up`, offset d (n.x = d), RMS residual."""
    c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    n = n if n @ up > 0 else -n
    d = float(n @ c)
    return n, d, float(np.sqrt(np.mean((P @ n - d) ** 2)))


def _plane_at(P, h, up, z0, below, above, core):
    """Consensus plane of a surface whose copies (one per view) may disagree by a few cm.

    Views can disagree on a surface's height by ~5-8 cm (the floor was 'doubled' in home01, see
    WORKLOG step 1.3). Direction: least-squares normal of the dominant copy only (|h - z0| < core),
    because a partial second copy would tilt a fit over the whole layer. Height: MEDIAN over the
    whole layer z0 - below .. z0 + above, i.e. the consensus of all views, not the lowest/highest
    copy. rms reports the spread of the copies. Returns (normal, offset d with n.x = d, rms, n, layer mask)."""
    core_sel = np.abs(h - z0) < core
    layer = (h >= z0 - below) & (h <= z0 + above)
    if core_sel.sum() < 50:
        return up, float(z0), float("nan"), int(layer.sum()), layer
    n, _, _ = _fit_plane(P[core_sel], up)
    r = P[layer] @ n
    d = float(np.median(r))
    return n, d, float(np.sqrt(np.mean((r - d) ** 2))), int(layer.sum()), layer


def _angle(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))


def horizontal_planes(P, N, up, cfg):
    """Floor = lowest strong horizontal peak; ceiling = highest one >= min_ceiling_m above it."""
    a = cfg["alignment"]
    horiz = np.abs(N @ up) > np.cos(np.radians(a["horizontal_normal_deg"]))
    Ph = P[horiz]
    h = Ph @ up
    peaks = _peaks(h, a["height_bin_m"], a["peak_rel"])
    if len(peaks) == 0:
        return None, None
    band, layer, core = a["plane_band_m"], a["plane_layer_m"], a["plane_core_m"]
    floor = _plane_at(Ph, h, up, peaks[0], below=band, above=layer, core=core)  # floor layer extends upward
    ceil_peaks = [z for z in peaks if z >= floor[1] + a["min_ceiling_m"]]
    ceiling = _plane_at(Ph, h, up, ceil_peaks[-1], below=layer, above=band, core=core) if ceil_peaks else None
    return floor, ceiling


def estimate_alignment(points: np.ndarray, T_wc: np.ndarray, cfg: dict) -> Alignment:
    a = cfg["alignment"]
    N = normals(points, a["normal_radius_m"])
    # 1. rough up from how the phones were held: camera +y (image down) in world is column 1 of R_wc
    up0 = -T_wc[:, :3, 1].mean(0)
    up0 /= np.linalg.norm(up0)
    # 2. floor (+ ceiling) planes -> gravity
    floor, ceiling = horizontal_planes(points, N, up0, cfg)
    if floor is None:
        raise RuntimeError("no horizontal surface found; cannot find the floor")
    nf, df, f_rms, f_n, _ = floor
    z = nf
    fc_angle = None
    if ceiling is not None:
        fc_angle = _angle(nf, ceiling[0])
        if fc_angle < a["max_floor_ceiling_angle_deg"]:  # both level: average for a steadier up
            z = nf * f_n + ceiling[0] * ceiling[3]
            z /= np.linalg.norm(z)
    cams = T_wc[:, :3, 3]
    if np.median(cams @ z) < df:  # cameras must be above the floor
        z = -z
    # 3. Manhattan: wall normals' azimuth folded mod 90 deg
    e1 = np.cross(z, [1.0, 0, 0] if abs(z[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(z, e1)
    wall = np.abs(N @ z) < a["wall_normal_max_z"]
    phi = np.arctan2(N[wall] @ e2, N[wall] @ e1)
    fold = np.degrees(phi) % 90
    hist, edges = np.histogram(fold, bins=np.arange(0, 90.5, 0.5))
    smooth = gaussian_filter1d(np.concatenate([hist, hist, hist]).astype(float), 2)[len(hist) : 2 * len(hist)]
    peak = np.radians(edges[np.argmax(smooth)] + 0.25)
    near = np.abs(((phi - peak + np.pi / 4) % (np.pi / 2)) - np.pi / 4) < np.radians(5)
    theta = np.arctan2(np.sin(4 * phi[near]).sum(), np.cos(4 * phi[near]).sum()) / 4  # circular mean mod 90
    ex = np.cos(theta) * e1 + np.sin(theta) * e2
    ey = np.cross(z, ex)
    R = np.stack([ex, ey, z])
    dev = np.abs(((phi - theta + np.pi / 4) % (np.pi / 2)) - np.pi / 4)
    support = float((dev < np.radians(5)).mean()) if len(dev) else 0.0
    # 4. floor to z = 0
    floor_z = float(np.median((points @ R.T)[np.abs(points @ nf - df) < a["plane_band_m"]][:, 2]))
    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = -floor_z
    return Alignment(
        T=T,
        tilt_correction_deg=round(_angle(up0, z), 2),
        floor_ceiling_angle_deg=None if fc_angle is None else round(fc_angle, 2),
        manhattan_deg=round(float(np.degrees(theta)), 2),
        manhattan_support=round(support, 3),
        floor_rms_m=round(f_rms, 4),
        floor_points=f_n,
    )


def room_planes(points_aligned: np.ndarray, cfg: dict) -> RoomPlanes:
    """A room's own floor and ceiling in the aligned frame (z up). Heights are the median z of each
    surface's layer (consensus of all views); ceiling height = ceiling z - floor z."""
    a = cfg["alignment"]
    up = np.array([0.0, 0, 1])
    N = normals(points_aligned, a["normal_radius_m"])
    floor, ceiling = horizontal_planes(points_aligned, N, up, cfg)
    if floor is None:
        return RoomPlanes(0.0, None, None, float("nan"), None, "partial", ["floor not found"])
    horiz = np.abs(N @ up) > np.cos(np.radians(a["horizontal_normal_deg"]))
    z = points_aligned[horiz, 2]
    fz = float(np.median(z[floor[4]]))
    if ceiling is None:
        msg = f"ceiling not visible: no horizontal plane >= {a['min_ceiling_m']} m above the floor"
        return RoomPlanes(round(fz, 4), None, None, round(floor[2], 4), None, "partial", [msg])
    cz = float(np.median(z[ceiling[4]]))
    return RoomPlanes(round(fz, 4), round(cz, 4), round(cz - fz, 4), round(floor[2], 4), round(ceiling[2], 4), "ok")


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def view_off_axis_deg(views: dict, i: int, rays: str, cfg: dict) -> float:
    """Median angle (deg) between one view's wall normals and the nearest house axis, in the aligned
    frame. Correctly placed views sit at ~2-5 deg (home01); a rotated / misplaced view stands out (E16)."""
    from scan.geometry.cloud import view_points

    thr = np.percentile(views["conf"][views["mask"]], cfg["geometry"]["conf_percentile"])
    keep = views["mask"][i] & (views["conf"][i] >= thr)
    P = view_points(views, i, rays)[keep]
    P = P[:: max(1, len(P) // 30000)]
    N = normals(P, cfg["alignment"]["normal_radius_m"])
    wall = np.abs(N[:, 2]) < cfg["alignment"]["wall_normal_max_z"]
    if wall.sum() < 200:
        return float("nan")
    phi = np.degrees(np.arctan2(N[wall, 1], N[wall, 0])) % 90
    return float(np.median(np.minimum(phi, 90 - phi)))


def use_shared_floor(planes: RoomPlanes, tol_m: float) -> RoomPlanes:
    """Rooms in one joint frame stand on one floor: z = 0, fitted from ALL rooms' floor points (step 1.3).
    A room whose own floor estimate is further than tol_m from it did not really see its floor
    (e.g. a galley kitchen whose counters hide it: lowest surface found 0.2-0.3 m up), so its ceiling
    height is measured from the shared floor instead, with a warning."""
    if planes.floor_z is not None and abs(planes.floor_z) <= tol_m:
        return planes
    was = planes.floor_z
    planes.floor_z = 0.0
    if planes.ceiling_z is not None:
        planes.ceiling_height_m = round(planes.ceiling_z, 4)
        if planes.status == "partial" and planes.warnings == ["floor not found"]:
            planes.status = "ok"
    planes.warnings.append(
        f"own floor not reliably seen (found at z={was:+.2f} m); ceiling height measured from the shared floor "
        "of the joint reconstruction (z=0)"
    )
    return planes


def ceiling_consistency(rooms: dict, tol_m: float) -> None:
    """Warn when a room's ceiling height differs from its frame-mates' median by more than tol_m.
    Not corrected: kitchens and bathrooms can genuinely have lower ceilings."""
    hs = {r: c.planes.ceiling_height_m for r, c in rooms.items() if c.planes.ceiling_height_m is not None}
    for r, h in hs.items():
        others = [v for k, v in hs.items() if k != r]
        if len(others) >= 1 and abs(h - float(np.median(others))) > tol_m:
            rooms[r].planes.warnings.append(
                f"ceiling {h:.2f} m differs from the other rooms ({float(np.median(others)):.2f} m): "
                "real lowered ceiling, or cabinets / a beam taken as ceiling? Check."
            )
