"""Orchestrates the stages for one capture and records timing (ARCHITECTURE §4, §12).

Tier 1 so far: ingest (1.1) -> geometry (1.2). Later steps plug in after geometry.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scan.device import go_offline, seed_everything, select_device
from scan.geometry.align import Alignment, apply, estimate_alignment, room_planes, view_off_axis_deg
from scan.geometry.cloud import build_cloud, save_ply
from scan.io.ingest import ingest
from scan.layout.room import classify_open_walls, layout_rooms
from scan.output import assemble, write
from scan.render.debug import alignment_plot, layout_plot
from scan.stitch.snap import merge_open_boundaries, snap_walls
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
    t = time.perf_counter()
    alignments = align_rooms(clouds, cfg, log)
    timing["alignment_s"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    layouts = layout_rooms(clouds, cfg)
    for r, lay in layouts.items():
        clouds[r].layout = lay
        log(f"  layout {r}: {len(lay.walls)} walls, area {lay.floor_area_m2:.2f} m2, {lay.status} ({lay.method})")
    timing["layout_s"] = round(time.perf_counter() - t, 2)
    snaps = []
    shared_frame = len({c.frame_id for c in clouds.values()}) == 1 and len(clouds) > 1
    if shared_frame and cfg["stitch"]["snap_walls"]:
        snaps = snap_walls(layouts, cfg)
        for m in snaps:
            log(f"  snap {m['wall_id']}: {100 * m['shift_m']:+.1f} cm")
    opened, open_pairs = classify_open_walls(layouts, cfg)
    if opened:
        log(f"  open boundaries (no wall): {', '.join(opened)}")
        if shared_frame:
            snaps += merge_open_boundaries(layouts, open_pairs)  # no wall -> rooms meet at one line

    t = time.perf_counter()
    result = assemble(cap, clouds, cfg, timing)
    files = write(result, out)
    timing["output_s"] = round(time.perf_counter() - t, 2)

    # diagnostics (not part of the contract)
    dbg = out / "debug"
    summary = {
        "capture_id": cap.capture_id,
        "tier": cap.tier,
        "device": str(dev),
        "geometry": info,
        "alignment": {fid: {k: v for k, v in vars(a).items() if k != "T"} | {"T": np.round(a.T, 6).tolist()}
                      for fid, a in alignments.items()},
        "rooms": {
            r: {
                "points": len(c.points),
                "frame_id": c.frame_id,
                "extent_m": np.round(c.points.max(0) - c.points.min(0), 3).tolist(),
                "planes": vars(c.planes),
                "view_off_axis_deg": c.view_off_axis_deg,
                "layout": dict(asdict(c.layout).items()),
                "ply": str(Path("debug") / r / "cloud.ply"),
            }
            for r, c in clouds.items()
        },
        "timing_s": timing,
        "warnings": cap.warnings,
        "wall_snaps": snaps,
        "files": {k: str(v.relative_to(out)) for k, v in files.items()},
    }
    t = time.perf_counter()
    for r, c in clouds.items():
        save_ply(dbg / r / "cloud.ply", c.points, c.colors)
        alignment_plot(r, c.points, c.planes, dbg / r / "align.png")
        layout_plot(r, c.layout, dbg / r / "layout.png")
    timing["debug_s"] = round(time.perf_counter() - t, 2)
    (dbg / "geometry.json").write_text(json.dumps(summary, indent=2))
    summary["result"] = result
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


def align_rooms(clouds: dict[str, RoomCloud], cfg, log) -> dict[str, Alignment]:
    """C3: one alignment per reconstruction frame (all rooms in a joint run share it), then each
    room's own floor/ceiling planes. Points and camera poses move into the aligned frame."""
    alignments: dict[str, Alignment] = {}
    for fid in dict.fromkeys(c.frame_id for c in clouds.values()):
        rooms = [c for c in clouds.values() if c.frame_id == fid]
        a = estimate_alignment(
            np.concatenate([c.points for c in rooms]), np.concatenate([c.views["T_wc"] for c in rooms]), cfg
        )
        alignments[fid] = a
        log(f"  alignment {fid}: tilt corrected {a.tilt_correction_deg} deg, floor-ceiling angle "
            f"{a.floor_ceiling_angle_deg} deg, walls rotated {a.manhattan_deg} deg "
            f"(support {a.manhattan_support:.0%}), floor rms {100 * a.floor_rms_m:.1f} cm")
        for c in rooms:
            c.points = apply(a.T, c.points)
            c.views["T_wc"] = np.einsum("ij,sjk->sik", a.T, c.views["T_wc"])  # poses now in the aligned frame
            c.views["pts"] = apply(a.T, c.views["pts"].reshape(-1, 3)).reshape(c.views["pts"].shape).astype(np.float32)
            c.alignment = a
            c.planes = room_planes(c.points, cfg)
            check_views(c, cfg, log)
    return alignments


def check_views(c: RoomCloud, cfg, log) -> None:
    """Drop views whose walls are rotated away from the house axes (misplaced by the backbone, E16),
    rebuild the room cloud and planes without them. Never below 2 views: then keep all, mark partial."""
    g = cfg["geometry"]
    limit = cfg["alignment"]["max_view_off_axis_deg"]
    dev = [view_off_axis_deg(c.views, i, g["rays"], cfg) for i in range(len(c.views["depth"]))]
    c.view_off_axis_deg = {str(n): round(d, 1) for n, d in zip(c.views["names"], dev)}
    bad = [i for i, d in enumerate(dev) if not math.isnan(d) and d > limit]
    if not bad:
        return
    names = [str(c.views["names"][i]) for i in bad]
    good = [i for i in range(len(dev)) if i not in bad]
    if len(good) < 2:
        c.planes.status = "partial"
        c.planes.warnings.append(
            f"photos disagree: {', '.join(f'{n} ({dev[i]:.0f} deg off the house axes)' for n, i in zip(names, bad))}; "
            f"too few consistent photos to drop them. Room shape unreliable: retake {c.room_id} (4-8 corner photos)."
        )
        log(f"  views {c.room_id}: {len(bad)} inconsistent of {len(dev)}, kept (too few left), room marked partial")
        return
    c.views = {k: v[good] for k, v in c.views.items()}
    c.frames = [f for f in c.frames if f.image_path.name in set(map(str, c.views["names"]))]
    c.points, c.colors = build_cloud(c.views, list(range(len(good))), g["rays"], g["conf_percentile"], g["voxel_m"])
    c.planes = room_planes(c.points, cfg)
    c.planes.warnings.append(f"dropped misplaced photos (> {limit} deg off the house axes): {', '.join(names)}")
    log(f"  views {c.room_id}: dropped {', '.join(names)} (misplaced), rebuilt from {len(good)} photos")
