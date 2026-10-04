"""E14. Compare ray/pose modes on cached MapAnything outputs: model rays, EXIF rays, EXIF rays + ICP.

Metrics per room:
  * inter-view residual: for each view's points, distance to the nearest point of all OTHER views,
    over overlapping points (< 20 cm). Doubled walls -> large residual.
  * room size vs tape (spike heuristic)
  * determinism of the ICP refinement (two runs, identical poses?)

Usage: uv run python scripts/experiments/icp_eval.py <out_dir_with_geometry.json> [out_plots_dir]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import spike_phase0 as sp  # noqa: E402

from scan import cache  # noqa: E402
from scan.config import load_config  # noqa: E402
from scan.geometry.cloud import unproject  # noqa: E402
from icp_refine import refine_poses  # noqa: E402

TAPE = {"bedroom1": (2.39, 2.915, 2.7925), "hall": (2.30, 3.70, 2.8025)}


def view_clouds(pred, poses, rays, thr):
    out = []
    for i in range(len(pred["depth"])):
        keep = pred["mask"][i] & (pred["conf"][i] > thr)
        if rays == "model":
            P = pred["pts"][i][keep]
        else:
            P = unproject(pred["depth"][i], pred["K_exif"][i], poses[i])[keep]
        out.append(P[:: max(1, len(P) // 20000)])
    return out


def residual(clouds):
    meds, within2 = [], []
    for i, P in enumerate(clouds):
        others = np.concatenate([c for j, c in enumerate(clouds) if j != i])
        d, _ = cKDTree(others).query(P, k=1)
        d = d[d < 0.20]
        if len(d):
            meds.append(np.median(d))
            within2.append((d < 0.02).mean())
    return float(np.median(meds)), float(np.mean(within2))


def main():
    out_dir = Path(sys.argv[1])
    plots = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    cfg = load_config()
    g = json.loads((out_dir / "geometry.json").read_text())
    for room, info in g["geometry"]["runs"].items():
        pred = cache.load(cfg, "mapanything", info["key"])
        mask, conf = pred["mask"], pred["conf"]
        thr = np.percentile(conf[mask], cfg["geometry"]["conf_percentile"])
        idx = list(range(len(pred["depth"])))
        t = time.perf_counter()
        refined, rinfo = refine_poses(pred, idx, cfg)
        t_icp = time.perf_counter() - t
        refined2, _ = refine_poses(pred, idx, cfg)
        det = "identical" if np.array_equal(refined, refined2) else f"max diff {np.abs(refined - refined2).max():.2e}"
        print(f"== {room}: ICP {t_icp:.1f} s, {len(rinfo['edges'])} pair edges, max camera shift {rinfo.get('max_camera_shift_m')} m, rerun {det}")
        for mode, poses in [("model", pred["T_wc"]), ("exif", pred["T_wc"]), ("exif_icp", refined)]:
            clouds = view_clouds(pred, poses, "model" if mode == "model" else "exif", thr)
            med, w2 = residual(clouds)
            P = np.concatenate(clouds)
            E = np.linalg.inv(poses)[:, :3, :]
            m = sp.measure_room(P, E, np.random.default_rng(0), down_axis=1)
            tv = TAPE[room]
            dims = "  ".join(
                f"{k.split('_')[0]} {m[k]:.3f} ({100 * (m[k] - x) / x:+.1f}%)"
                for k, x in zip(["short_side_m", "long_side_m", "ceiling_height_m"], tv)
            )
            print(f"   {mode:9s} inter-view residual median {100 * med:5.1f} cm, within 2 cm {100 * w2:4.0f}%   {dims}")
            if plots is not None:
                d = plots / room / mode
                d.mkdir(parents=True, exist_ok=True)
                sp.save_plots(d, m)


if __name__ == "__main__":
    main()
