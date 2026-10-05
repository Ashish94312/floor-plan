"""E25a: which camera convention do Stray Scanner poses use? (no learned model)

Back-project two frames' LiDAR depth with the pose as given (OpenCV camera: x right, y down, z forward)
and with the ARKit camera convention (x right, y up, z backward = OpenCV after flipping y and z).
The right convention makes the two frames' points land on the same surfaces (small nearest-neighbour gap).
Also: ARKit world is gravity-aligned with +y up, so the image-down axis of a camera held level must
point to -y under the right convention.

  uv run python scripts/experiments/e25_lidar_pose_check.py 1a8384c3f6 [gap_frames=30]
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

root = Path(sys.argv[1])
gap = int(sys.argv[2]) if len(sys.argv) > 2 else 30
odo = np.genfromtxt(root / "odometry.csv", delimiter=",", skip_header=1)
FLIP = np.diag([1.0, -1.0, -1.0])


def pose(i, arkit):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(odo[i, 5:9]).as_matrix() @ (FLIP if arkit else np.eye(3))
    T[:3, 3] = odo[i, 2:5]
    return T


def points(i, arkit):
    d = np.asarray(Image.open(root / "depth" / f"{i:06d}.png"), np.float64) / 1000.0
    c = np.asarray(Image.open(root / "confidence" / f"{i:06d}.png"))
    H, W = d.shape
    fx, fy, cx, cy = odo[i, 9:13] * (W / 1920.0)
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    cam = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], -1)[c == 2]
    T = pose(i, arkit)
    return cam @ T[:3, :3].T + T[:3, 3]


n = len(odo)
for arkit in (False, True):
    gaps, downs = [], []
    for i in range(0, n - gap, max(1, n // 12)):
        a, b = points(i, arkit), points(i + gap, arkit)
        dist, _ = cKDTree(a).query(b[:: 7])
        gaps.append(np.median(dist))
        downs.append(pose(i, arkit)[:3, 1] @ [0, 1, 0])  # world y of the image-down axis
    name = "ARKit camera (flip y,z)" if arkit else "as given (OpenCV camera)"
    print(f"{root.name} gap {gap} frames | {name:26s} | median NN gap {100 * np.median(gaps):5.1f} cm "
          f"(per pair {', '.join(f'{100 * g:.0f}' for g in gaps)}) | image-down . world-up {np.median(downs):+.2f}")
