"""Per-view backbone predictions -> one room point cloud in metres (ARCHITECTURE §10.2 step 4)."""

from __future__ import annotations

import numpy as np


def unproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Pinhole back-projection of an HxW z-depth map to world points (HxWx3).
    Pixel centres at (u + 0.5, v + 0.5); T_wc is camera-to-world."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
    return cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def depth_focal_fix(pred: dict[str, np.ndarray], scale: float = 1.0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Undo the focal-depth trade-off of a view whose focal the model misjudged (E22m).

    A pinhole image fixes only depth/focal: if the model takes a focal f' instead of the true f, it can
    still reproduce the image by keeping lateral and vertical extents right and scaling depth by f'/f.
    On video frames MapAnything takes ~23-27 mm against ~30 mm, so heights come out right while
    lengths along the line of sight come out short (hall -11%). Fix per view: depth x f/f', points
    rebuilt with the true focal (K_exif x scale). Lateral/vertical extents stay as the model had them.
    Returns the corrected arrays (K_model := true K, so rays='model' and ray_K stay consistent) and f/f'."""
    K_true = pred["K_exif"].copy()
    K_true[:, 0, 0] *= scale
    K_true[:, 1, 1] *= scale
    r = K_true[:, 0, 0] / pred["K_model"][:, 0, 0]
    out = dict(pred)
    out["depth"] = (pred["depth"] * r[:, None, None]).astype(np.float32)
    out["pts"] = np.stack([unproject(out["depth"][i], K_true[i], pred["T_wc"][i]) for i in range(len(r))]).astype(np.float32)
    out["K_model"] = K_true
    out["K_exif"] = K_true
    return out, r


def ray_K(views: dict[str, np.ndarray], i: int, rays: str) -> np.ndarray:
    """Intrinsics consistent with view_points(views, i, rays): the model's for rays='model' (video,
    E22c), EXIF for rays='exif'. Anything that projects into a view or back out of it must use these."""
    return views["K_model"][i] if rays == "model" else views["K_exif"][i]


def view_points(pred: dict[str, np.ndarray], i: int, rays: str) -> np.ndarray:
    """World points of view i. rays='model': MapAnything's own (consistent with its poses, focal ~12%
    long, E12). rays='exif': depth re-projected with EXIF intrinsics (right focal, poses not
    re-fitted, E10)."""
    if rays == "model":
        return pred["pts"][i]
    if rays == "exif":
        return unproject(pred["depth"][i], pred["K_exif"][i], pred["T_wc"][i])
    raise ValueError(f"geometry.rays must be 'model' or 'exif', got {rays!r}")


def build_cloud(
    pred: dict[str, np.ndarray], views: list[int], rays: str, conf_percentile: float, voxel_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-filter, merge and voxel-thin the given views. Returns (N x 3 points, N x 3 colours 0-1)."""
    import open3d as o3d

    mask = pred["mask"][views]
    conf = pred["conf"][views]
    keep = mask & (conf >= np.percentile(conf[mask], conf_percentile))
    pts = np.concatenate([view_points(pred, i, rays)[k] for i, k in zip(views, keep)])
    cols = np.concatenate([pred["rgb"][i][k] for i, k in zip(views, keep)]).astype(np.float64) / 255.0
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(cols)
    if voxel_m > 0:
        pc = pc.voxel_down_sample(voxel_m)
    return np.asarray(pc.points), np.asarray(pc.colors)


def save_ply(path, points: np.ndarray, colors: np.ndarray) -> None:
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pc.colors = o3d.utility.Vector3dVector(colors)
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pc)
