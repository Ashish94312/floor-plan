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
