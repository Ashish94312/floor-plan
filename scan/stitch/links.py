"""C7 video-tier stitching of separately reconstructed rooms by link frames (E22n).

Link frames (io.video.link_pairs) are frame pairs from two rooms' clips that see the same thing: the
doorway, or the room beyond it. Each frame sits in its own room's model run, so it has depth and a pose
in its own room's aligned frame (floor z=0, walls on x/y). SIFT matches between the two frames give
pixel pairs; each pixel's depth puts it at a 3D point in room A's frame and in room B's frame. They are
the same physical points, so B -> A is a rotation about z plus a shift in x/y: RANSAC over 2-point
samples, least squares on the inliers, then the rotation snaps to the nearest multiple of 90 deg (both
rooms are squared to the house axes) when it is that close.

Unlike door matching (stitch.doors) no door has to be detected in either room's outline, and a doorway
seen from one side only is enough. Diagnostics per edge: the height offset of the matched points (both
floors at z=0, so ~0) and the size ratio of the two rooms' versions of the same points (A ~ s B; 1 = they
agree), which stitch.scale uses to put every room on one scale.
"""

from __future__ import annotations

import cv2
import numpy as np

from scan.geometry.cloud import view_points


def match_pixels(rgb_a: np.ndarray, rgb_b: np.ndarray, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """SIFT matches between two images that passed a ratio test and a RANSAC fundamental matrix:
    (N x 2 pixel xy in A, N x 2 in B)."""
    sift = cv2.SIFT_create(nfeatures=4000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    good = [m for m, n in (x for x in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if len(x) == 2)
            if m.distance < ratio * n.distance]
    if len(good) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pa = np.float64([ka[m.queryIdx].pt for m in good])
    pb = np.float64([kb[m.trainIdx].pt for m in good])
    _F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    if mask is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    keep = mask.ravel().astype(bool)
    return pa[keep], pb[keep]


def lookup(views: dict, i: int, px: np.ndarray, rays: str) -> tuple[np.ndarray, np.ndarray]:
    """World points of view i at pixel positions px (N x 2, OpenCV xy: pixel centres on integers) and
    whether each has valid depth."""
    H, W = views["mask"][i].shape
    c = np.clip(np.round(px[:, 0]).astype(int), 0, W - 1)
    r = np.clip(np.round(px[:, 1]).astype(int), 0, H - 1)
    P = view_points(views, i, rays)[r, c].astype(np.float64)
    return P, views["mask"][i][r, c] & np.isfinite(P).all(1)


def _kabsch_2d(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R (2x2) and shift t minimising |A - (R B + t)|^2."""
    ma, mb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - mb).T @ (A - ma))
    D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, ma - R @ mb


def fit_rigid_2d(A: np.ndarray, B: np.ndarray, thr: float, iters: int = 500, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC rigid 2D fit A ~ R B + t over 2-point samples, refined on the inliers.
    Returns (R, t, inlier mask)."""
    rng = np.random.default_rng(seed)
    n = len(A)
    best = np.zeros(n, bool)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(B[i] - B[j]) < 0.2:  # too short a baseline to fix the rotation
            continue
        R, t = _kabsch_2d(A[[i, j]], B[[i, j]])
        inl = np.linalg.norm(A - (B @ R.T + t), axis=1) < thr
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 2:
        return np.eye(2), np.zeros(2), best
    for _ in range(2):  # refine, then re-collect inliers once
        R, t = _kabsch_2d(A[best], B[best])
        best = np.linalg.norm(A - (B @ R.T + t), axis=1) < thr
    return R, t, best


def snap_rotation(R: np.ndarray, A: np.ndarray, B: np.ndarray, tol_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Snap R to the nearest multiple of 90 deg if within tol_deg; shift re-fitted as the median offset.
    Returns (R, t, how far R was from the multiple in degrees)."""
    th = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    k = round(th / 90.0)
    off = th - 90.0 * k
    if abs(off) > tol_deg:
        return R, np.median(A - B @ R.T, axis=0), off
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    Rs = np.array([[c, -s], [s, c]], float)
    return Rs, np.median(A - B @ Rs.T, axis=0), off


def link_edge(ca, ia: int, cb, ib: int, cfg: dict) -> dict | None:
    """Placement of room B in room A's frame from one or more link frame pairs [(view in A, view in B)].
    ca/cb: RoomCloud with run_views (all views of the room's model run, aligned frame)."""
    sc, rays = cfg["stitch"], cfg["geometry"]["rays"]
    A3, B3, da, db, measured = [], [], [], [], []
    for i, j in zip(np.atleast_1d(ia), np.atleast_1d(ib)):
        va, vb = ca.run_views, cb.run_views
        pa, pb = match_pixels(va["rgb"][i], vb["rgb"][j])
        if not len(pa):
            continue
        Pa, oka = lookup(va, i, pa, rays)
        Pb, okb = lookup(vb, j, pb, rays)
        ok = oka & okb
        A3.append(Pa[ok])
        B3.append(Pb[ok])
        # size ratio only from frames whose own scale was measured (repose: tied to their room by pairs)
        measured.append(np.full(ok.sum(), bool(va.get("reposed", np.ones(len(va["depth"]), bool))[i]
                                               and vb.get("reposed", np.ones(len(vb["depth"]), bool))[j])))
        da.append(np.linalg.norm(Pa[ok] - va["T_wc"][i][:3, 3], axis=1))
        db.append(np.linalg.norm(Pb[ok] - vb["T_wc"][j][:3, 3], axis=1))
    if not A3 or sum(map(len, A3)) < sc["link_min_points"]:
        return None
    A3, B3, da, db, measured = map(np.concatenate, (A3, B3, da, db, measured))
    R, t, inl = fit_rigid_2d(A3[:, :2], B3[:, :2], sc["link_inlier_m"])
    if inl.sum() < sc["link_min_points"]:
        return None
    R, t, off = snap_rotation(R, A3[inl, :2], B3[inl, :2], sc["link_snap_deg"])
    res = np.linalg.norm(A3[inl, :2] - (B3[inl, :2] @ R.T + t), axis=1)
    # size ratio of the two rooms' versions of the same points: 3D similarity, tolerance ~ distance from the cameras
    # (the rigid inliers above would bias it to 1: a scale mismatch pushes far points out of a rigid fit)
    from scan.geometry.repose import similarity_ransac

    m = measured
    sim = (similarity_ransac(A3[m], B3[m], sc["link_scale_rel_tol"] * np.maximum(np.maximum(da[m], db[m]), 0.3))
           if m.sum() >= sc["link_min_points"] else None)
    s_ab, s_inl = (float(sim[0]), int(sim[3].sum())) if sim is not None else (float("nan"), 0)
    return {
        "R": R, "t": t, "inliers": int(inl.sum()), "matches": len(A3),
        "theta_deg": round(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))) % 360, 1),
        "snap_off_deg": round(float(off), 1), "rms_m": round(float(np.sqrt(np.mean(res**2))), 3),
        "dz_m": round(float(np.median(A3[inl, 2] - B3[inl, 2])), 3),  # both floors at z=0 -> ~0
        "scale_b_in_a": round(s_ab, 4), "scale_inliers": s_inl,  # A ~ s B: s = 1 when both rooms agree on size
        "depth_a_m": round(float(np.median(da[inl])), 2), "depth_b_m": round(float(np.median(db[inl])), 2),
    }


def link_edges(clouds: dict, links: list[tuple[str, str, str, str]], cfg: dict) -> list[dict]:
    """One edge per pair of rooms that have link frames: B's placement in A's frame (see link_edge)."""
    by_pair: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for ra, na, rb, nb in links:
        if ra not in clouds or rb not in clouds or clouds[ra].run_views is None or clouds[rb].run_views is None:
            continue
        ia = np.flatnonzero(clouds[ra].run_views["names"] == na)
        ib = np.flatnonzero(clouds[rb].run_views["names"] == nb)
        if len(ia) and len(ib):
            by_pair.setdefault((ra, rb), []).append((int(ia[0]), int(ib[0])))
    edges = []
    for (ra, rb), pairs in by_pair.items():
        e = link_edge(clouds[ra], [p[0] for p in pairs], clouds[rb], [p[1] for p in pairs], cfg)
        if e is not None:
            edges.append({"a": ra, "b": rb, "pairs": len(pairs)} | e)
    return edges
