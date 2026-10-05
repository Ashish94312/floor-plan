"""Orchestrates the stages for one capture and records timing (ARCHITECTURE §4, §12).

Tier 1 so far: ingest (1.1) -> geometry (1.2). Later steps plug in after geometry.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scan.damage.detect import damage_regions, detector_openings, surface_boxes
from scan.device import go_offline, seed_everything, select_device
from scan.geometry.align import (
    Alignment,
    apply,
    ceiling_consistency,
    estimate_alignment,
    room_planes,
    use_shared_floor,
    view_off_axis_deg,
)
from scan.geometry.cloud import build_cloud, depth_focal_fix, focal_scale_vote, save_ply, scale_pred
from scan.io.ingest import ingest
from scan.io.lidar import UNSPLIT
from scan.layout.openings import (
    detect_openings,
    drop_one_sided_doors,
    drop_reflections,
    reindex_by_main_door,
)
from scan.layout.room import classify_open_walls, layout_rooms, repair_outlines
from scan.output import assemble, scale_vote_on, write
from scan.render.debug import alignment_plot, layout_plot
from scan.stitch.doors import apply_placement, place_rooms
from scan.stitch.links import link_edges
from scan.stitch.scale import flat_scale, scale_room
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
    if cap.tier == "video":  # video-specific geometry settings (E22c, E22k)
        cfg = copy.deepcopy(cfg)
        for k in ("rays", "max_joint_views", "depth_focal_fix", "max_aspect", "pad_scale", "pad_to_aspect"):
            if cfg["video"].get(k):
                cfg["geometry"][k] = cfg["video"][k]
        if cfg["video"].get("door_height_prior_m"):  # video: room size anchored on seen door tops (E22z)
            cfg["stitch"]["door_height_prior_m"] = cfg["video"]["door_height_prior_m"]
        if cfg["video"].get("repose"):
            cfg["geometry"]["repose"]["enabled"] = True

    go_offline()
    seed_everything()
    dev = select_device(device)
    out = out or capture_dir / "out"
    if cap.tier == "lidar":
        clouds, info = geometry_lidar(cap, cfg, timing, log)
    else:
        clouds, info = geometry_photo(cap, cfg, dev, use_cache, timing, log)
    t = time.perf_counter()
    alignments = align_rooms(clouds, cfg, log, check=cap.tier != "lidar")
    timing["alignment_s"] = round(time.perf_counter() - t, 2)
    if cap.tier == "lidar" and UNSPLIT in clouds:  # one walk through the home: rooms split at doorways (E25)
        from scan.damage.detect import detect
        from scan.layout.segment import split_recording

        t = time.perf_counter()
        whole = clouds.pop(UNSPLIT)
        info["detections"] = detect(whole.frames, cfg, dev, use_cache)  # doorways for the split; reused below
        rooms = split_recording(whole, cfg, log, info["detections"])
        clouds.update(rooms)
        cap.rooms.pop(UNSPLIT)
        cap.rooms.update({r: c.frames for r, c in rooms.items()})
        timing["segment_s"] = round(time.perf_counter() - t, 2)  # includes the detector pass
    runs = info["runs"]
    if (len({c.frame_id for c in clouds.values()}) > 1 and cap.links and cfg["stitch"]["flat_scale"]
            and all("repose" in runs.get(r, {}) for r in clouds)):
        # one scale for the flat: link frames measure the rooms' size ratios, all frames vote for the level (E22p)
        pre = link_edges(clouds, cap.links, cfg)
        fs = flat_scale(list(clouds), {r: np.array(runs[r]["repose"]["votes_log"]) for r in clouds}, pre)
        info["flat_scale_links"] = [{k: v for k, v in e.items() if k not in ("R", "t")} for e in pre]
        for r, f in fs.items():
            c = clouds[r]
            scale_room(c, f["c"])
            c.planes = room_planes(c.points, cfg)
            c.scale = Scale(s=1.0, sigma_log=max(cfg["geometry"]["sigma_log_floor_photo"], np.nan_to_num(f["sigma_log"])))
            log(f"  flat scale {r}: x{f['c']:.3f} (group {f['group']}, {f['votes']} votes, spread {f['sigma_log']:.3f})")
        info["flat_scale"] = fs
    if cfg["geometry"]["repose"]["enabled"] and cfg["uncertainty"]["apply_scale_bias"]:
        # the model's systematic size error on this kind of footage, learnt from taped captures (E22q)
        from scan.uncertainty.calibrate import load as load_calibration

        mode = f"{cap.tier}_{'joint' if info['joint'] else 'per_room'}_repose"
        b = load_calibration().get(mode, {}).get("scale_bias")
        if b:
            for c in clouds.values():
                scale_room(c, b["factor"])
                c.planes = room_planes(c.points, cfg)
                c.scale = Scale(s=1.0, sigma_log=max(c.scale.sigma_log, b["sigma_log"]))
            info["scale_bias"] = {"mode": mode, "factor": b["factor"], "sigma_log": b["sigma_log"], "rooms": b["rooms"]}
            log(f"  size bias ({mode}): x{b['factor']:.4f}, spread {b['sigma_log']:.3f} from {b['rooms']} taped rooms")
        else:
            cap.warnings.append(f"no size calibration for {mode}: video sizes rest on the model's size sense alone")
    if cfg["stitch"].get("joint_scale") and len({c.frame_id for c in clouds.values()}) > 1:
        # every room on one scale: one model run over frames of all rooms, compared frame by frame (E23b)
        from scan.geometry.mapanything_backend import load_model, predict
        from scan.stitch.joint_scale import joint_frames, room_factors

        held: dict = {}

        def joint_model():
            if "m" not in held:
                held["m"] = load_model(dev)
            return held["m"]

        jf = joint_frames(cap, cfg["geometry"]["max_joint_views"])
        jpred, jinfo = predict(jf, cfg, dev, joint_model, use_cache)
        held.clear()
        fac = room_factors(jpred, {r: c.run_views for r, c in clouds.items() if c.run_views is not None})
        for r, f in fac.items():
            c = clouds[r]
            scale_room(c, f["c"])
            c.planes = room_planes(c.points, cfg)
            c.scale = Scale(s=1.0, sigma_log=float(np.hypot(np.nan_to_num(c.scale.sigma_log), np.nan_to_num(f["sigma_log"]))))
            log(f"  joint scale {r}: x{f['c']:.3f} from {f['frames']} shared frames (per frame {f['per_frame']})")
        info["joint_scale"] = {"views": len(jf), "cache": jinfo.get("cache"), "rooms": fac}
    t = time.perf_counter()
    layouts = layout_rooms(clouds, cfg)
    for r, lay in layouts.items():
        clouds[r].layout = lay
        log(f"  layout {r}: {len(lay.walls)} walls, area {lay.floor_area_m2:.2f} m2, {lay.status} ({lay.method})")
    if cfg["stitch"].get("door_height_prior_m") and len({c.frame_id for c in clouds.values()}) > 1:
        # each room's size from its doors whose top was seen: a standard door height against the model's (E22z)
        from scan.stitch.door_scale import door_scale, measured_door_heights

        info["door_scale"] = {}
        heights = {r: measured_door_heights(c, layouts[r], cfg) for r, c in clouds.items()}
        floor = cfg["geometry"]["sigma_log_floor_photo"]
        if "joint_scale" in info:  # rooms already share one scale: all their doors set one size for the flat
            pooled = [h for hs in heights.values() for h in hs]
            sig = float(np.median([max(float(np.nan_to_num(c.scale.sigma_log)), floor) for c in clouds.values()]))
            ds_flat = door_scale(pooled, sig, cfg)
        for r, c in clouds.items():
            hs = heights[r]
            ds = ds_flat if "joint_scale" in info else door_scale(hs, max(float(np.nan_to_num(c.scale.sigma_log)), floor), cfg)
            info["door_scale"][r] = {"heights_m": [round(h, 3) for h in hs], "factor": ds and round(ds[0], 4)}
            if ds:
                scale_room(c, ds[0])
                c.planes = room_planes(c.points, cfg)
                c.scale = Scale(s=1.0, sigma_log=ds[1])
                log(f"  door scale {r}: doors {', '.join(f'{h:.2f}' for h in hs) or 'none'} m -> x{ds[0]:.3f} (spread {ds[1]:.3f})")
        if any(v["factor"] for v in info["door_scale"].values()):
            layouts = layout_rooms(clouds, cfg)
            for r, lay in layouts.items():
                clouds[r].layout = lay
    timing["layout_s"] = round(time.perf_counter() - t, 2)
    snaps, links = [], []
    if len({c.frame_id for c in clouds.values()}) > 1:  # separate reconstructions -> place by door matching
        pre = detect_openings(layouts, clouds, cfg)
        links = link_edges(clouds, cap.links, cfg) if cap.links and cfg["stitch"]["link_placement"] else []
        for e in links:
            log(f"  link {e['a']} <-> {e['b']}: {e['inliers']}/{e['matches']} points, theta {e['theta_deg']} deg "
                f"(snapped {e['snap_off_deg']:+.1f}), rms {100 * e['rms_m']:.0f} cm, dz {100 * e['dz_m']:+.0f} cm, "
                f"scale {e['b']}/{e['a']} {e['scale_b_in_a']}, depth {e['depth_a_m']} / {e['depth_b_m']} m")
        place = place_rooms(clouds, layouts, pre, cfg, link_edges=links)
        for ra, rb, score, da, db in place.pop("_edges"):
            log(f"  door match {da} <-> {db}: visibility score {score}")
        used = [pl["method"] for pl in place.values() if pl["method"]]
        for r, pl in place.items():
            c = clouds[r]
            if pl["placed"]:
                apply_placement(c, layouts[r], pl["R"], pl["t"])
                c.frame_id = "stitched"
                c.pose = (float(pl["t"][0]), float(pl["t"][1]), float(pl["theta_deg"]))
                c.placed_by = pl["method"] or (used[0] if used else "door_matching")  # the root: how the others joined it
                log(f"  placed {r}: theta {pl['theta_deg']} deg, shift ({pl['t'][0]:+.2f}, {pl['t'][1]:+.2f}) m"
                    + (f" via {pl['via'][0]} <-> {pl['via'][1]}" if pl["via"] else " (root)"))
            else:
                log(f"  {r}: no link or door match, left unplaced")
        stitched = {r: c for r, c in clouds.items() if c.frame_id == "stitched"}
        if len(stitched) > 1 and cfg["stitch"]["relayout_after_placement"]:
            # placed rooms now share a frame: lay them out again so free space seen through a doorway goes to the
            # room whose own frames saw through it most (ownership, as in a joint run), E22n
            for r, lay in layout_rooms(stitched, cfg).items():
                layouts[r] = clouds[r].layout = lay
                log(f"  relayout {r}: {len(lay.walls)} walls, area {lay.floor_area_m2:.2f} m2 (ownership in the stitched frame)")
    else:
        for c in clouds.values():
            c.placed_by = "joint_reconstruction" if len(clouds) > 1 else "single_room"
    # steps that compare rooms run on the rooms sharing the main frame (unplaced rooms are left alone)
    frames = [c.frame_id for c in clouds.values()]
    main = max(set(frames), key=frames.count)
    group = {r: layouts[r] for r, c in clouds.items() if c.frame_id == main}
    shared_frame = len(group) > 1
    if shared_frame and cfg["stitch"]["snap_walls"]:
        snaps = snap_walls(group, cfg)
        for m in snaps:
            log(f"  snap {m['wall_id']}: {100 * m['shift_m']:+.1f} cm")
    opened, open_pairs = classify_open_walls(group, cfg)
    if opened:
        log(f"  open boundaries (no wall): {', '.join(opened)}")
        if shared_frame:
            snaps += merge_open_boundaries(group, open_pairs)  # no wall -> rooms meet at one line
    for r in repair_outlines(layouts):  # walls moved by snapping can make an outline cross itself
        log(f"  {r}: outline crossed itself; kept its largest valid piece")
    t = time.perf_counter()
    openings = detect_openings(layouts, clouds, cfg)
    for r, ops in openings.items():
        reindex_by_main_door(layouts[r], ops)  # D11: W1 = main-door wall
        clouds[r].openings = ops
        for o in ops:
            log(f"  {o.opening_id}: {o.type} on {o.wall_id}, {o.width_m:.2f} x {o.height_m:.2f} m"
                + (f", sill {o.sill_m:.2f}" if o.sill_m is not None else "") + (f" -> {o.connects_to}" if o.connects_to else ""))
    timing["openings_s"] = round(time.perf_counter() - t, 2)

    # detector: closed doors / covered windows / mirrors + damage (after W1 is fixed: no renumbering)
    t = time.perf_counter()
    boxes = surface_boxes(cap, clouds, cfg, dev, use_cache, info.pop("detections", None))
    for line in detector_openings(boxes, {r: c.openings for r, c in clouds.items()}, cfg):
        log(f"  {line}")
    for line in drop_reflections(layouts, clouds, {r: c.openings for r, c in clouds.items()}, cfg):
        log(f"  {line}")
    for line in drop_one_sided_doors(layouts, {r: c.openings for r, c in clouds.items()}, cfg):
        log(f"  {line}")
    for r, regs in damage_regions(boxes, cfg).items():
        clouds[r].damage = regs
        for d in regs:
            log(f"  {d.damage_id}: {d.cls} on {d.surface_id}, {d.width_m:.2f} x {d.height_m:.2f} m ({d.photos} photos)")
    timing["detector_damage_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    hints_file = capture_dir / "hints.yaml"  # layout facts the user knows (not measurements), E23c
    hints = (yaml.safe_load(hints_file.read_text()) or {}) if hints_file.exists() else {}
    if hints.get("same_line") and shared_frame:
        # last geometric step: openings / damage were measured on walls where the rooms' points are; carry them over
        from scan.stitch.snap import join_declared, reanchor, surface_anchors

        anchors = surface_anchors(group, clouds)
        joined = join_declared(group, cfg, hints["same_line"])
        reanchor(group, clouds, anchors)
        snaps += joined
        for m in joined:
            log(f"  declared line {m['wall_id']}: {100 * m['shift_m']:+.1f} cm")
        cap.warnings.append(f"hints.yaml: same wall line declared for {hints['same_line']}")
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
        "links": [{k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()} for e in links],
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


def vote_level(cap: Capture, cfg, joint: bool) -> dict | None:
    """The photo vote's level: the model's absolute size sense, calibrated on taped captures (FIX_DECLARATION.md).
    None (with a warning) until scan-calibrate --bias-only has fitted it for this mode."""
    from scan.uncertainty.calibrate import load as load_calibration

    mode = f"photo_{'joint' if joint else 'per_room'}_vote"
    b = load_calibration().get(mode, {}).get("scale_bias") if cfg["uncertainty"]["apply_scale_bias"] else None
    if not b:
        cap.warnings.append(f"no size calibration for {mode}: sizes rest on the model's size sense alone")
    return b


def geometry_photo(cap: Capture, cfg, dev, use_cache: bool, timing: dict, log) -> tuple[dict[str, RoomCloud], dict]:
    """C2 photo tier: MapAnything per room (default) or once over all rooms (geometry.joint)."""
    from scan.geometry.mapanything_backend import load_model, predict, refit_with_depth
    from scan.geometry.repose import repose

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
    vote = scale_vote_on(cfg, cap.tier)
    level = vote_level(cap, cfg, joint) if vote else None
    clouds: dict[str, RoomCloud] = {}
    runs = {}
    t_all = time.perf_counter()
    for gid, frames in groups.items():
        log(f"  geometry: {gid} ({len(frames)} photos)")
        pred, pinfo = predict(frames, cfg, dev, get_model, use_cache)
        fix = g.get("depth_focal_fix")
        if g["repose"]["enabled"]:  # video: poses from PnP with the true focal on focal-corrected depth (E22o)
            pred, rinfo = repose(pred, frames, cfg)
            pinfo = pinfo | {"repose": rinfo}
            log(f"  repose {gid}: {rinfo}")
        elif fix == "refit":  # video: second pass, poses re-solved for focal-corrected depth (E22n)
            pred, rinfo = refit_with_depth(pred, cfg, dev, get_model, g.get("depth_focal_scale", 1.0), use_cache)
            pinfo = pinfo | {"refit": rinfo}
        elif fix:  # video: model focal ~23-27 mm vs ~32 true squeezes depth (E22m)
            pred, r = depth_focal_fix(pred, g.get("depth_focal_scale", 1.0))
            pinfo = pinfo | {"depth_focal_ratio": [round(float(x), 3) for x in np.percentile(r, [10, 50, 90])]}
        if vote:  # photo: the model's depth follows its orientation-dependent focal reading (FIX_DECLARATION.md)
            k, vinfo = focal_scale_vote(pred)
            c = level["factor"] if level else 1.0
            pred = scale_pred(pred, k * c)  # once, on the raw run: alignment and layout see metric geometry
            pinfo = pinfo | {"scale_vote": vinfo | {"level": c}}
            log(f"  scale vote {gid}: x{k:.3f} (room votes {vinfo['room_votes']}), level x{c:.4f}")
        runs[gid] = pinfo
        for room in dict.fromkeys(f.room_hint for f in frames):
            # link frames (video) only tie rooms together in the model run; they look into the other room (E22l)
            idx = [i for i, r in enumerate(pred["rooms"]) if r == room and not frames[i].link]
            pts, cols = build_cloud(pred, idx, g["rays"], g["conf_percentile"], g["voxel_m"])
            run = [i for i, r in enumerate(pred["rooms"]) if r == room]
            clouds[room] = RoomCloud(
                room_id=room,
                points=pts,
                colors=cols,
                frames=[f for f in cap.rooms[room] if not f.link],
                # reposed video: the spread of the frames' scale votes is the room's scale uncertainty (E22o);
                # photo vote: the level's own uncertainty adds to the per-room scatter (independent sources)
                scale=Scale(s=1.0, sigma_log=math.hypot(g["sigma_log_floor_photo"], level["sigma_log"]) if level else
                            max(g["sigma_log_floor_photo"],
                                np.nan_to_num(pinfo.get("repose", {}).get("scale_sigma_log", 0.0)))),
                T_room_world=np.eye(4),
                frame_id=gid,
                views={k: v[idx] for k, v in pred.items()},
                run_views={k: v[run] for k, v in pred.items()},
            )
    timing["geometry_s"] = round(time.perf_counter() - t_all, 2)
    info = {"backbone": g["backbone"], "joint": bool(joint), "rays": g["rays"], "f_scale": g["f_scale"], "runs": runs,
            "scale_vote": vote}
    return clouds, info


def geometry_lidar(cap: Capture, cfg, timing: dict, log) -> tuple[dict[str, RoomCloud], dict]:
    """C2 LiDAR tier: the phone's depth through the phone's poses (no learned model). Each recording is its own
    ARKit session (own frame), metric to the sensor's accuracy."""
    from scan.io.lidar import WORLD_UP, lidar_views

    g = cfg["geometry"]
    t = time.perf_counter()
    clouds: dict[str, RoomCloud] = {}
    for room, frames in cap.rooms.items():
        v = lidar_views(frames, cfg)
        # K_exif == K_model and pts == depth through them: any rays setting reads the same points
        pts, cols = build_cloud(v, list(range(len(frames))), g["rays"], g["conf_percentile"], g["voxel_m"])
        log(f"  geometry: {room} ({len(frames)} LiDAR frames, {v['mask'].mean():.0%} of depth pixels kept) -> {len(pts)} points")
        clouds[room] = RoomCloud(room_id=room, points=pts, colors=cols, frames=frames,
                                 scale=Scale(s=1.0, sigma_log=cfg["lidar"]["sigma_log"]), T_room_world=np.eye(4),
                                 frame_id=room, views=v, up_world=WORLD_UP)
    timing["geometry_s"] = round(time.perf_counter() - t, 2)
    return clouds, {"backbone": "arkit_lidar", "joint": False, "rays": g["rays"], "runs": {}, "scale_vote": False}


def align_rooms(clouds: dict[str, RoomCloud], cfg, log, check: bool = True) -> dict[str, Alignment]:
    """C3: one alignment per reconstruction frame (all rooms in a joint run share it), then each
    room's own floor/ceiling planes. Points and camera poses move into the aligned frame.
    check: drop views the backbone misplaced (check_views); off for measured (LiDAR) poses."""
    alignments: dict[str, Alignment] = {}
    for fid in dict.fromkeys(c.frame_id for c in clouds.values()):
        rooms = [c for c in clouds.values() if c.frame_id == fid]
        a = estimate_alignment(
            np.concatenate([c.points for c in rooms]), np.concatenate([c.views["T_wc"] for c in rooms]), cfg,
            up=rooms[0].up_world,
        )
        alignments[fid] = a
        log(f"  alignment {fid}: tilt corrected {a.tilt_correction_deg} deg, floor-ceiling angle "
            f"{a.floor_ceiling_angle_deg} deg, walls rotated {a.manhattan_deg} deg "
            f"(support {a.manhattan_support:.0%}), floor rms {100 * a.floor_rms_m:.1f} cm")
        shared = len(rooms) > 1
        for c in rooms:
            c.points = apply(a.T, c.points)
            for v in (x for x in (c.views, c.run_views) if x is not None):
                v["T_wc"] = np.einsum("ij,sjk->sik", a.T, v["T_wc"])  # poses now in the aligned frame
                v["pts"] = apply(a.T, v["pts"].reshape(-1, 3)).reshape(v["pts"].shape).astype(np.float32)
            c.alignment = a
            c.planes = room_planes(c.points, cfg)
            if check:
                check_views(c, cfg, log)
            if shared:
                c.planes = use_shared_floor(c.planes, cfg["alignment"]["shared_floor_tol_m"])
        if shared:
            ceiling_consistency({c.room_id: c for c in rooms}, cfg["alignment"]["ceiling_mismatch_warn_m"])
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
