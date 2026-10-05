"""Video: camera poses and per-frame depth scales re-solved with the true intrinsics (E22o).

MapAnything takes its own focal for video frames (~24-28 mm vs 32 true, E22b/E22n) and squeezes its world to
match; correcting each frame's depth by f_true / f_model does not work: that ratio does not predict how far
off a frame's depth is (correlation -0.45 / -0.67 against what the frames' matches say, E22o). What holds:
with the TRUE rays, a frame's model depth gives the true shape of what it sees up to one scale per frame.
So the frames' scales and poses are solved from the frames themselves:

  1. shape_i = model depth_i x true rays                       (frame i's own units)
  2. SIFT matches for every pair of frames                      (working resolution, ratio test, RANSAC F)
  3. per pair, both shapes at the matched pixels -> 3D-3D similarity (RANSAC Umeyama): relative scale,
     rotation and position. Unlike PnP it needs no baseline: a camera that only turned still counts.
  4. least squares: log scale per frame; rotations (model's, averaged with the pair rotations); positions.
     Frames no pair reaches keep their scale and the model's offset to their time neighbours (weak priors).
  5. one absolute scale per room. Geometry cannot give it (scale everything and the images do not change);
     only the model's sense of size can. What a size prior fixes is Z/f, not Z: an object h pixels tall has
     size H = Z h / f, the image gives h, the prior gives H. So frame i votes for its true depth
     Z_true = f_true Z_model / f_model,i, i.e. for its scale f_true / f_model,i; with the relative scales
     from (4) every frame votes for the room's one scale. Median over frames (frames the prior could not
     size, e.g. blank close-ups, are outliers); the votes' spread is the room's scale uncertainty.
     Ceiling heights come from the same prior: a check, not an input.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from scan.geometry.cloud import unproject


def _features(img: np.ndarray, n: int):
    k, d = cv2.SIFT_create(nfeatures=n).detectAndCompute(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
    return (np.float64([p.pt for p in k]) + 0.5 if k else np.zeros((0, 2))), d  # pixel centres at +0.5 (unproject)


def _matches(fa, fb, ratio: float, min_n: int):
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(pa) < min_n or len(pb) < min_n:
        return None
    good = [m for m, n in (x for x in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if len(x) == 2)
            if m.distance < ratio * n.distance]
    if len(good) < min_n:
        return None
    A, B = pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]]
    _F, mk = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, 1.5, 0.999)
    if mk is None or mk.sum() < min_n:
        return None
    keep = mk.ravel().astype(bool)
    return A[keep], B[keep]


def umeyama(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity A ~ s R B + t (least squares)."""
    ma, mb = A.mean(0), B.mean(0)
    A0, B0 = A - ma, B - mb
    U, D, Vt = np.linalg.svd(A0.T @ B0 / len(A))
    E = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = float(np.trace(np.diag(D) @ E) / (B0**2).sum(1).mean())
    return s, R, ma - s * R @ mb


def similarity_ransac(A, B, tol: np.ndarray, iters: int = 300, seed: int = 0):
    """RANSAC Umeyama A ~ s R B + t; match k agrees if its residual is under tol[k]. -> (s, R, t, inliers) | None"""
    rng = np.random.default_rng(seed)
    best = np.zeros(len(A), bool)
    for _ in range(iters):
        idx = rng.choice(len(A), 3, replace=False)
        if np.linalg.matrix_rank(B[idx] - B[idx].mean(0), tol=1e-3) < 2:
            continue
        s, R, t = umeyama(A[idx], B[idx])
        if not 0.3 < s < 3.0:
            continue
        inl = np.linalg.norm(A - (s * B @ R.T + t), axis=1) < tol
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < 3:
        return None
    model = umeyama(A[best], B[best])
    for _ in range(2):  # refine; a refit that loses its support (near-collinear inliers) keeps the last good model
        inl = np.linalg.norm(A - (model[0] * B @ model[1].T + model[2]), axis=1) < tol
        if inl.sum() < 3 or not 0.3 < model[0] < 3.0:
            break
        best = inl
        model = umeyama(A[best], B[best])
    s, R, t = model
    best = np.linalg.norm(A - (s * B @ R.T + t), axis=1) < tol
    if best.sum() < 3 or not 0.3 < s < 3.0:
        return None
    return s, R, t, best


def _project_rot(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def scale_votes(f_ratio: np.ndarray, rel_scale: np.ndarray, used: np.ndarray) -> tuple[float, float, int]:
    """Room scale k from per-frame votes log(f_true / f_model,i) - log(relative scale_i), frames in `used`:
    (k, robust std of the log votes, number of votes)."""
    v = np.log(f_ratio[used]) - np.log(rel_scale[used])
    if not len(v):
        return 1.0, float("nan"), 0
    med = float(np.median(v))
    return float(np.exp(med)), float(1.4826 * np.median(np.abs(v - med))), len(v)


def ceiling_height(pred: dict[str, np.ndarray], cfg: dict) -> float | None:
    """Ceiling height of a model run's own cloud (levelled with the pipeline's alignment)."""
    from scan.geometry.align import apply, estimate_alignment, room_planes
    from scan.geometry.cloud import build_cloud

    g = cfg["geometry"]
    P, _ = build_cloud(pred, list(range(len(pred["depth"]))), "model", g["conf_percentile"], g["voxel_m"])
    a = estimate_alignment(P, pred["T_wc"], cfg)
    return room_planes(apply(a.T, P), cfg).ceiling_height_m


def repose(pred: dict[str, np.ndarray], frames: list, cfg: dict) -> tuple[dict[str, np.ndarray], dict]:
    """pred: one model run (model resolution); frames: its Frames in the same order (working-resolution rgb and
    true K). Returns pred with depth / pts / T_wc / K_model replaced (K_model := true K), and diagnostics."""
    rc = cfg["geometry"]["repose"]
    S = len(pred["depth"])
    Kt = pred["K_exif"]
    z = pred["depth"].astype(np.float64)

    # pixel (working res) -> model-res pixel index: x_m = s x_w - o  (scale + centre crop)
    sc = Kt[:, 0, 0] / np.array([f.K[0, 0] for f in frames])
    ox = sc * np.array([f.K[0, 2] for f in frames]) - Kt[:, 0, 2]
    oy = sc * np.array([f.K[1, 2] for f in frames]) - Kt[:, 1, 2]
    H, W = z.shape[1:]
    feats = [_features(f.rgb, rc["sift_features"]) for f in frames]

    def shape_at(i, px):  # frame i's shape (true rays x model depth) at working-res pixels
        c = np.floor(px[:, 0] * sc[i] - ox[i]).astype(int)
        r = np.floor(px[:, 1] * sc[i] - oy[i]).astype(int)
        ok = (c >= 0) & (c < W) & (r >= 0) & (r < H)
        ok[ok] &= pred["mask"][i][r[ok], c[ok]]
        d = np.zeros(len(px))
        d[ok] = z[i][r[ok], c[ok]]
        ok &= np.isfinite(d) & (d > 0)  # the model can leave inf / nan inside its own mask
        d[~ok] = 0.0
        K = frames[i].K
        return np.stack([(px[:, 0] - K[0, 2]) / K[0, 0] * d, (px[:, 1] - K[1, 2]) / K[1, 1] * d, d], 1), ok

    edges = []  # (a, b, s_ba, R_ab, t_ab, inliers): shape_a ~ s R shape_b + t
    for a, b in itertools.combinations(range(S), 2):
        m = _matches(feats[a], feats[b], rc["ratio"], rc["min_matches"])
        if m is None:
            continue
        Ya, oka = shape_at(a, m[0])
        Yb, okb = shape_at(b, m[1])
        ok = oka & okb
        if ok.sum() < rc["min_inliers"]:
            continue
        res = similarity_ransac(Ya[ok], Yb[ok], rc["rel_tol"] * np.maximum(Ya[ok][:, 2], 0.3))  # tol ~ depth in a
        if res is not None and res[3].sum() >= rc["min_inliers"]:
            edges.append((a, b, res[0], res[1], res[2], int(res[3].sum())))

    reached = np.zeros(S, bool)
    for a, b, *_ in edges:
        reached[a] = reached[b] = True
    # 1. per-frame log scale: log sig_b - log sig_a = log s ; weak prior log sig = 0 (model depth as it is)
    A = np.zeros((len(edges) + S, S))
    bvec = np.zeros(len(edges) + S)
    for k, (a, b, s, *_r, n) in enumerate(edges):
        w = np.sqrt(n)
        A[k, b], A[k, a], bvec[k] = w, -w, w * np.log(s)
    for i in range(S):
        A[len(edges) + i, i] = 1.0 / rc["scale_prior_log"]
    lsig = np.linalg.lstsq(A, bvec, rcond=None)[0]
    sig = np.exp(lsig)

    # 2. rotations: the model's, averaged with the pair rotations (R_b = R_a R_ab)
    Rm, Cm = pred["T_wc"][:, :3, :3].copy(), pred["T_wc"][:, :3, 3].copy()
    R = Rm.copy()
    for _ in range(10):
        acc = rc["rot_prior_weight"] * Rm.copy()
        for a, b, _s, Rab, _t, n in edges:
            acc[b] += np.sqrt(n) * (R[a] @ Rab)
            acc[a] += np.sqrt(n) * (R[b] @ Rab.T)
        R = np.stack([_project_rot(M) for M in acc])
    rot_change = [float(np.degrees(np.arccos(np.clip((np.trace(Rm[i].T @ R[i]) - 1) / 2, -1, 1)))) for i in range(S)]

    # 3. positions: C_b - C_a = -R_a sig_a t_ab / ... : shape_a ~ s R shape_b + t  => centre of b in a's frame = t (a-units)
    rel = [(a, b, R[a] @ (sig[a] * t), n) for a, b, _s, _R, t, n in edges]
    moved = [(np.linalg.norm(v), np.linalg.norm(Cm[b] - Cm[a])) for a, b, v, _n in rel if np.linalg.norm(Cm[b] - Cm[a]) > rc["min_baseline_m"]]
    stretch = float(np.median([x / y for x, y in moved])) if moved else 1.0
    order = np.argsort([f.timestamp if f.timestamp is not None else k for k, f in enumerate(frames)], kind="stable")
    rows = [(b, a, v, np.sqrt(n) / rc["sigma_pnp_m"], True) for a, b, v, n in rel]
    rows += [(b, a, stretch * (Cm[b] - Cm[a]), 1.0 / rc["sigma_prior_m"], False) for a, b in itertools.pairwise(order)]
    w_now = np.array([r[3] for r in rows])
    for _ in range(6):
        M = np.zeros((len(rows) + 1, S))
        rhs = np.zeros((len(rows) + 1, 3))
        for k, (b, a, v, _w, _p) in enumerate(rows):
            M[k, b], M[k, a], rhs[k] = w_now[k], -w_now[k], w_now[k] * v
        M[-1], rhs[-1] = 1.0, Cm.sum(0)  # gauge: keep the mean camera position
        C = np.linalg.lstsq(M, rhs, rcond=None)[0]
        res = np.array([np.linalg.norm(C[b] - C[a] - v) for b, a, v, _w, _p in rows])
        hub = np.where(res > rc["huber_m"], rc["huber_m"] / np.maximum(res, 1e-9), 1.0)
        w_now = np.array([w * (h if p else 1.0) for (_b, _a, _v, w, p), h in zip(rows, hub)])
    pair_res = res[[r[4] for r in rows]] if edges else np.zeros(0)

    T = pred["T_wc"].copy()
    T[:, :3, :3], T[:, :3, 3] = R, C
    out = dict(pred)
    out["depth"] = (z * sig[:, None, None]).astype(np.float32)
    out["T_wc"] = T
    out["K_model"] = Kt.copy()
    out["pts"] = np.stack([unproject(out["depth"][i], Kt[i], T[i]) for i in range(S)]).astype(np.float32)
    out["reposed"] = reached.copy()  # per view: scale and pose measured by pairs (False: priors only)

    # 4. absolute scale: every frame votes with the model's size sense (f_true / f_model,i), see the docstring
    f_ratio = Kt[:, 0, 0] / pred["K_model"][:, 0, 0]
    k, spread, n_votes = scale_votes(f_ratio, sig, reached)
    c0 = C.mean(0)  # scale about the mean camera position (keeps the gauge)
    out["depth"] = (out["depth"] * k).astype(np.float32)
    out["pts"] = (c0 + k * (out["pts"] - c0)).astype(np.float32)
    out["T_wc"][:, :3, 3] = c0 + k * (C - c0)
    h_model, h_new = ceiling_height(pred, cfg), ceiling_height(out, cfg)  # check only
    corr = float(np.corrcoef(np.log(f_ratio[reached]), lsig[reached])[0, 1]) if reached.sum() > 2 else None
    info = {
        "pairs": len(edges), "views_reached": int(reached.sum()), "views": S,
        "unreached": [str(n) for n in pred["names"][~reached]],
        "frames": {str(pred["names"][i]): {"scale": round(float(sig[i] * k), 3), "f_ratio": round(float(f_ratio[i]), 3),
                                          "pairs": int(sum(i in (a, b) for a, b, *_ in edges))} for i in range(S)},
        "frame_scale": [round(float(x), 3) for x in np.percentile(sig * k, [10, 50, 90])],
        "corr_with_focal_ratio": None if corr is None else round(corr, 2),
        "scale_k": round(k, 3), "scale_votes": n_votes, "scale_sigma_log": round(spread, 3),
        # each reached frame's vote for this room's scale, log relative to k (flat-wide scale, stitch.scale)
        "votes_log": [round(float(x), 4) for x in (np.log(f_ratio[reached]) - np.log(sig[reached]) - np.log(k))],
        "ceiling_model_m": None if h_model is None else round(h_model, 3),
        "ceiling_reposed_m": None if h_new is None else round(h_new, 3),
        "baseline_stretch": round(stretch * k, 3), "rot_change_deg": [round(float(x), 1) for x in np.percentile(rot_change, [50, 90])],
        "pair_residual_m": [round(float(x), 3) for x in np.percentile(pair_res, [50, 90])] if len(pair_res) else None,
    }
    return out, info
