"""Diagnostic plots (not the deliverable plan render, which is step 1.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def alignment_plot(room: str, points: np.ndarray, planes, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = points[:, 2]
    top = planes.ceiling_z if planes.ceiling_z is not None else np.percentile(z, 98)
    band = points[(z > 1.0) & (z < top - 0.2)]
    band = band[:: max(1, len(band) // 60000)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].scatter(band[:, 0], band[:, 1], s=0.3, c="0.3")
    ax[0].set_aspect("equal")
    ax[0].grid(True, lw=0.3)
    ax[0].set_title(f"{room}: aligned top-down, wall band z 1.0 m to ceiling-0.2 m\nwalls should run along x / y")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[1].hist(z, bins=300, color="0.4")
    ax[1].axvline(planes.floor_z, c="r", label=f"floor {planes.floor_z:+.3f}")
    if planes.ceiling_z is not None:
        ax[1].axvline(planes.ceiling_z, c="b", label=f"ceiling {planes.ceiling_z:.3f} (h {planes.ceiling_height_m:.3f} m)")
    ax[1].legend()
    ax[1].set_title("heights (z) of all points")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def layout_plot(room: str, layout, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dbg = getattr(layout, "_debug", None)
    fig, ax = plt.subplots(figsize=(8, 8))
    if dbg is not None:
        lo, cell = dbg["lo"], dbg["cell"]
        h, w = dbg["free"].shape
        ext = [lo[0], lo[0] + w * cell, lo[1], lo[1] + h * cell]
        ax.imshow(np.minimum(dbg["free"], 4), origin="lower", extent=ext, cmap="Blues", alpha=0.5)
        ax.contour(dbg["region"].astype(float), levels=[0.5], origin="lower", extent=ext, colors="tab:green", linewidths=1)
        b = dbg["walls_pts"][:: max(1, len(dbg["walls_pts"]) // 60000)]
        ax.scatter(b[:, 0], b[:, 1], s=0.2, c="0.35", label="wall points (0.3 m - ceiling)")
        ax.scatter(dbg["cams"][:, 0], dbg["cams"][:, 1], marker="^", s=50, c="tab:orange", edgecolors="k", label="cameras")
    P = np.array(layout.polygon + layout.polygon[:1])
    ax.plot(P[:, 0], P[:, 1], "-", c="tab:red", lw=2, label=f"layout ({layout.method})")
    for wseg in layout.walls:
        mx, my = (wseg.start[0] + wseg.end[0]) / 2, (wseg.start[1] + wseg.end[1]) / 2
        ax.annotate(f"{wseg.wall_id.split('-')[-1]} {wseg.length_m:.2f}", (mx, my), color="tab:red", fontsize=9,
                    ha="center", va="center", bbox={"fc": "white", "ec": "none", "alpha": 0.8})
    ceil = f"{layout.ceiling_height_m:.3f} m" if layout.ceiling_height_m else "n/a"
    ax.set_title(f"{room}: area {layout.floor_area_m2:.2f} m2, ceiling {ceil}, {layout.status}\n"
                 "blue = free space carved by lines of sight, green = room region")
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3)
    ax.legend(loc="upper right", markerscale=3, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
