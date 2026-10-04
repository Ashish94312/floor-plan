"""Orchestrates the stages for one capture and records timing (ARCHITECTURE §4, §12).

Tier 1 so far: ingest (1.1) -> geometry (1.2). Later steps plug in after geometry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scan.device import go_offline, seed_everything, select_device
from scan.geometry.cloud import build_cloud, save_ply
from scan.io.ingest import ingest
from scan.types import Capture, RoomCloud, Scale, Tier


def run(
    capture_dir: Path,
    cfg: dict[str, Any],
    tier: Tier | None = None,
    out: Path | None = None,
    use_cache: bool = True,
    device: str | None = None,
    log=print,
) -> tuple[Capture, dict[str, RoomCloud], dict]:
    timing: dict[str, float] = {}
    t = time.perf_counter()
    cap = ingest(capture_dir, tier, cfg)
    timing["ingest_s"] = round(time.perf_counter() - t, 2)
    if cap.tier != "photo":
        raise NotImplementedError(f"{cap.tier} tier")

    go_offline()
    seed_everything()
    dev = select_device(device)
    out = out or capture_dir / "out"
    clouds, info = geometry_photo(cap, cfg, dev, use_cache, timing, log)

    summary = {
        "capture_id": cap.capture_id,
        "tier": cap.tier,
        "device": str(dev),
        "geometry": info,
        "rooms": {
            r: {
                "points": len(c.points),
                "frame_id": c.frame_id,
                "extent_m": np.round(c.points.max(0) - c.points.min(0), 3).tolist(),
                "ply": str(Path("rooms") / r / "cloud.ply"),
            }
            for r, c in clouds.items()
        },
        "timing_s": timing,
        "warnings": cap.warnings,
    }
    t = time.perf_counter()
    for r, c in clouds.items():
        save_ply(out / "rooms" / r / "cloud.ply", c.points, c.colors)
    timing["write_s"] = round(time.perf_counter() - t, 2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "geometry.json").write_text(json.dumps(summary, indent=2))
    return cap, clouds, summary


def plan_groups(cap: Capture, g: dict) -> tuple[dict[str, list], bool]:
    """Which frames go into one backbone run. Joint (all rooms together) shares one metric scale and
    ties rooms together (E15); it is used when it fits in memory (geometry.max_joint_views)."""
    n_views = sum(len(fs) for fs in cap.rooms.values())
    joint = g["joint"]
    if joint == "auto":
        joint = len(cap.rooms) > 1 and n_views <= g["max_joint_views"]
        if len(cap.rooms) > 1 and not joint:
            cap.warnings.append(
                f"{n_views} photos > geometry.max_joint_views={g['max_joint_views']}: rooms reconstructed separately "
                "(no shared scale; wider intervals). Use --joint to force one run if memory allows."
            )
    joint = bool(joint)
    return ({"joint": [f for fs in cap.rooms.values() for f in fs]} if joint else dict(cap.rooms)), joint


def geometry_photo(cap: Capture, cfg, dev, use_cache: bool, timing: dict, log) -> tuple[dict[str, RoomCloud], dict]:
    """C2 photo tier: MapAnything per room (default) or once over all rooms (geometry.joint)."""
    from scan.geometry.mapanything_backend import load_model, predict

    g = cfg["geometry"]
    model = None

    def get_model():
        nonlocal model
        if model is None:
            t = time.perf_counter()
            log("  loading MapAnything ...")
            model = load_model(dev)
            timing["model_load_s"] = round(time.perf_counter() - t, 2)
        return model

    groups, joint = plan_groups(cap, g)
    clouds: dict[str, RoomCloud] = {}
    runs = {}
    t_all = time.perf_counter()
    for gid, frames in groups.items():
        log(f"  geometry: {gid} ({len(frames)} photos)")
        pred, pinfo = predict(frames, cfg, dev, get_model, use_cache)
        runs[gid] = pinfo
        for room in dict.fromkeys(f.room_hint for f in frames):
            idx = [i for i, r in enumerate(pred["rooms"]) if r == room]
            pts, cols = build_cloud(pred, idx, g["rays"], g["conf_percentile"], g["voxel_m"])
            clouds[room] = RoomCloud(
                room_id=room,
                points=pts,
                colors=cols,
                frames=cap.rooms[room],
                scale=Scale(s=1.0, sigma_log=g["sigma_log_floor_photo"]),
                T_room_world=np.eye(4),
                frame_id=gid,
                views={k: v[idx] for k, v in pred.items()},
            )
    timing["geometry_s"] = round(time.perf_counter() - t_all, 2)
    info = {"backbone": g["backbone"], "joint": bool(joint), "rays": g["rays"], "f_scale": g["f_scale"], "runs": runs}
    return clouds, info
