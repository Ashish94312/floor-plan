"""E17. Per wall: face map (points on the wall plane) and see-through map (points behind it), u x z.

Usage: uv run python scripts/experiments/wall_maps.py <capture_dir> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scan.config import load_config  # noqa: E402
from scan.geometry.align import normals  # noqa: E402
from scan.pipeline import run  # noqa: E402


def maps(points, N, wall, ceil, du=0.02, dz=0.05):
    a, b = np.array(wall.start), np.array(wall.end)
    L = float(np.linalg.norm(b - a))
    u_dir = (b - a) / L
    out_dir = np.array([-u_dir[1], u_dir[0]])  # clockwise polygon -> left = outside
    rel = points[:, :2] - a
    u, off, z = rel @ u_dir, rel @ out_dir, points[:, 2]
    vertical = np.abs(N[:, 2]) < 0.5
    nu, nz = int(np.ceil(L / du)), int(np.ceil(ceil / dz))
    face = np.zeros((nz, nu))
    see = np.zeros((nz, nu))
    inside = (u > 0) & (u < L) & (z > 0) & (z < ceil)
    f = inside & vertical & (np.abs(off) < 0.12)
    s = inside & (off > 0.25) & (off < 6.0)
    np.add.at(face, ((z[f] / dz).astype(int), (u[f] / du).astype(int)), 1)
    np.add.at(see, ((z[s] / dz).astype(int), (u[s] / du).astype(int)), 1)
    return face, see, L


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cap, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    _, clouds, _ = run(cap, cfg, out=out / "_run", log=lambda *_: None)
    for r, c in clouds.items():
        N = normals(c.points, 0.10)
        ceil = c.planes.ceiling_z or 2.8
        walls = c.layout.walls
        fig, axes = plt.subplots(2, len(walls), figsize=(3.2 * len(walls), 6), squeeze=False)
        for k, w in enumerate(walls):
            face, see, L = maps(c.points, N, w, ceil)
            ext = [0, L, 0, ceil]
            axes[0, k].imshow(np.minimum(face, 5), origin="lower", extent=ext, aspect="auto", cmap="Greys")
            axes[0, k].set_title(f"{w.wall_id} ({w.kind}) face", fontsize=8)
            axes[1, k].imshow(np.minimum(see, 5), origin="lower", extent=ext, aspect="auto", cmap="Blues")
            axes[1, k].set_title("see-through (behind wall)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{r}_wall_maps.png", dpi=90)
        plt.close(fig)
        print("wrote", out / f"{r}_wall_maps.png")


if __name__ == "__main__":
    main()
