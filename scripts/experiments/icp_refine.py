"""E14 (NEGATIVE RESULT, not used by the pipeline). Pose re-fit for EXIF rays by pairwise ICP + pose graph.

Failed on sparse corner views: most pairs overlap 5-30%, ICP slides along flat walls (corrections of
metres / tens of degrees), the graph is disconnected at fitness >= 0.3. Kept for the record (WORKLOG E14).

MapAnything's poses are consistent with its own rays, whose focal is ~12% too long (E12). Rebuilding
each view with EXIF rays gives the right shape per view but misaligns views (E10). Here every view's
EXIF-ray cloud is re-registered to its overlapping neighbours with multi-scale point-to-plane ICP,
starting from MapAnything's poses, then a pose graph spreads the pairwise results consistently.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan.geometry.cloud import unproject  # noqa: E402


REFINE_DEFAULTS = {"voxel_m": 0.02, "scales": [[0.08, 0.40, 40], [0.04, 0.15, 30], [0.02, 0.05, 30]], "min_fitness": 0.30}


def _view_cloud(pred, i, keep, voxel):
    import open3d as o3d

    P = unproject(pred["depth"][i], pred["K_exif"][i], np.eye(4))[keep]  # camera frame, EXIF rays
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    pc.orient_normals_towards_camera_location(np.zeros(3))
    return pc


def _icp(src, tgt, init, scales):
    """Coarse-to-fine point-to-plane ICP with a Tukey robust kernel. Returns (T, fitness, info)."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    T = init
    for voxel, dist, iters in scales:
        s, t = src.voxel_down_sample(voxel), tgt.voxel_down_sample(voxel)
        for pc in (s, t):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=dist / 2))
        res = reg.registration_icp(s, t, dist, T, est, reg.ICPConvergenceCriteria(max_iteration=iters))
        T = res.transformation
    fine_dist = scales[-1][1]
    info = reg.get_information_matrix_from_point_clouds(src, tgt, fine_dist, T)
    fitness = reg.evaluate_registration(src, tgt, fine_dist, T).fitness
    return T, fitness, info


def refine_poses(pred: dict[str, np.ndarray], views: list[int], cfg: dict) -> tuple[np.ndarray, dict]:
    """Refined camera-to-world poses (len(views) x 4 x 4) for EXIF-ray clouds, plus diagnostics."""
    import open3d as o3d

    r = cfg["geometry"].get("refine") or REFINE_DEFAULTS
    reg = o3d.pipelines.registration
    mask, conf = pred["mask"][views], pred["conf"][views]
    thr = np.percentile(conf[mask], cfg["geometry"]["conf_percentile"])
    clouds = [_view_cloud(pred, i, m & (c > thr), r["voxel_m"]) for i, m, c in zip(views, mask, conf)]
    T0 = pred["T_wc"][views]
    ref = T0[0]
    scales = [tuple(s) for s in r["scales"]]  # (voxel, max_correspondence, iterations)

    graph = reg.PoseGraph()
    for T in T0:
        graph.nodes.append(reg.PoseGraphNode(np.linalg.inv(ref) @ T))  # poses relative to view 0
    edges = []
    n = len(views)
    for i in range(n):
        for j in range(i + 1, n):
            init = np.linalg.inv(T0[j]) @ T0[i]  # maps view i camera frame -> view j camera frame
            T, fit, info = _icp(clouds[i], clouds[j], init, scales)
            if fit < r["min_fitness"]:
                continue
            graph.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=True))
            edges.append((i, j, round(float(fit), 3)))
    if not edges:
        return T0.copy(), {"edges": [], "status": "no overlapping pairs; kept model poses"}
    reg.global_optimization(
        graph,
        reg.GlobalOptimizationLevenbergMarquardt(),
        reg.GlobalOptimizationConvergenceCriteria(),
        reg.GlobalOptimizationOption(
            max_correspondence_distance=scales[-1][1], edge_prune_threshold=0.25, reference_node=0
        ),
    )
    refined = np.stack([ref @ nd.pose for nd in graph.nodes])
    moved = [float(np.linalg.norm(refined[k][:3, 3] - T0[k][:3, 3])) for k in range(n)]
    return refined, {"edges": edges, "status": "ok", "max_camera_shift_m": round(max(moved), 3)}
