"""Assemble the ScanResult (the published schema) from the pipeline state and write the deliverables:
out/result.json, out/plan.png, out/plan.svg, out/rooms/<room>.png. Diagnostics live in out/debug/."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from scan.config import config_hash
from scan.render.plan import render_room, render_stitched
from scan.schema import (
    ConcealedFlag,
    DamageRegion,
    Measurement,
    Opening,
    Room,
    ScanResult,
    ScopeItem,
    StitchedPlan,
    Wall,
)
from scan.stitch.plan import stitch
from scan.uncertainty.intervals import area_sigma_parts, ceiling_sigma, measurement, wall_sigmas

REPO = Path(__file__).resolve().parents[1]


def software_info(cfg: dict) -> dict[str, str]:
    from scan.geometry.mapanything_backend import MAPANYTHING_COMMIT
    from scan.weights import CODE, MODELS

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=5, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):  # no git (e.g. unpacked bundle)
            return ""

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    m = MODELS["mapanything-apache"]
    return {
        "git_commit": commit + ("-dirty" if dirty else ""),
        "backbone": f"{m.repo_id}@{m.revision[:8]} (code {MAPANYTHING_COMMIT[:8]}, dinov2 {CODE['dinov2-code'].commit[:8]})",
        "config_hash": config_hash(cfg),
        "geometry": f"joint={cfg['geometry']['joint']} rays={cfg['geometry']['rays']}",
    }


def _room(r: str, c, cfg: dict, tier: str, u: dict) -> Room:
    lay, pl = c.layout, c.planes
    s_log = c.scale.sigma_log
    sig = wall_sigmas(lay, s_log, u, tier)
    walls = [Wall(wall_id=w.wall_id, start=w.start, end=w.end, length=measurement(w.length_m, s, "m", u, tier, mtype="wall"),
                  kind=w.kind) for w, s in zip(lay.walls, sig)]
    cs = ceiling_sigma(pl, s_log, u, tier)
    ceiling = None if cs is None else measurement(pl.ceiling_height_m, cs, "m", u, tier, mtype="ceiling")
    a_scale, a_edge = area_sigma_parts(lay, s_log, u, tier)
    area = measurement(lay.floor_area_m2, math.hypot(a_scale, a_edge), "m2", u, tier, mtype="area")
    status = "ok" if (lay.status == "ok" and pl.status == "ok") else "partial"
    ow = cfg["openings"]["sigma_width_m"]
    g = u["sigma_geom_m"][tier]
    ops = []
    for o in c.openings:
        single = 0.03 if o.views < 2 else 0.0  # one photo only: wider interval (ARCHITECTURE §10.8)

        def sig(v, single=single):
            return math.sqrt((v * s_log) ** 2 + ow**2 + single**2)

        ops.append(Opening(
            opening_id=o.opening_id, type=o.type, wall_id=o.wall_id,
            offset_along_wall=measurement(o.offset_m, math.hypot(g, ow), "m", u, tier),
            width=measurement(o.width_m, sig(o.width_m), "m", u, tier, mtype="opening"),
            height=measurement(o.height_m, sig(o.height_m), "m", u, tier, mtype="opening"),
            sill_height=None if o.sill_m is None else measurement(o.sill_m, sig(o.sill_m), "m", u, tier),
            connects_to=o.connects_to, views=o.views,
        ))
    return Room(room_id=r, label=None, status=status, polygon=lay.polygon, walls=walls, ceiling_height=ceiling,
                floor_area=area, openings=ops, layout_method=lay.method, warnings=list(pl.warnings) + list(lay.warnings))


def assemble(cap, clouds: dict, cfg: dict, timing: dict) -> ScanResult:
    from scan.uncertainty.calibrate import load as load_calibration
    from scan.uncertainty.intervals import k_for

    tier = cap.tier
    mode = "joint" if all(c.placed_by == "joint_reconstruction" for c in clouds.values()) else "per_room"
    u = {**cfg["uncertainty"], "mode": mode, "calibration": load_calibration()}
    warnings = list(cap.warnings)
    rooms = [_room(r, c, cfg, tier, u) for r, c in clouds.items()]
    method, poses, adjacency, overlaps, drift = stitch(clouds, cfg, warnings)
    # footprint = sum of room areas (D10). Shared scale (joint frame) -> scale terms add linearly.
    parts = [area_sigma_parts(c.layout, c.scale.sigma_log, u, tier) for c in clouds.values()]
    edge = math.sqrt(sum(e**2 for _, e in parts))
    scale = sum(s for s, _ in parts) if method == "joint_reconstruction" else math.sqrt(sum(s**2 for s, _ in parts))
    footprint = measurement(sum(c.layout.floor_area_m2 for c in clouds.values()), math.hypot(scale, edge), "m2", u, tier)
    plan = StitchedPlan(placement_method=method, room_poses=poses, adjacency=adjacency, footprint_area=footprint,
                        overlaps=overlaps, drift_correction=drift)
    damage, flags, scope = _damage(clouds, {**cfg, "uncertainty": u}, tier)
    software = software_info(cfg) | {
        "interval_mode": mode,
        "interval_k": json.dumps({t: round(k_for(u, tier, t), 4) for t in ("default", "wall", "ceiling", "area", "opening")}),
    }
    return ScanResult(capture_id=cap.capture_id, tier=tier, devices=cap.devices, interval_level=u["interval_level"],
                      rooms=rooms, stitched_plan=plan, damage=damage, concealed_damage_flags=flags, scope_items=scope,
                      timing_s=timing, warnings=warnings, software=software)


def _damage(clouds: dict, cfg: dict, tier: str):
    """Damage regions, concealed flags and scope items with intervals. Box extents overestimate, so the
    extent sigma is generous (ARCHITECTURE §10.10)."""
    from scan.damage.rules import apply_rules

    u = cfg["uncertainty"]
    box = cfg["damage"].get("sigma_extent_m", 0.05)
    damage, flags, scope = [], [], []
    for r, c in clouds.items():
        s_log = c.scale.sigma_log
        for d in c.damage:
            def m(v, unit="m", s_log=s_log):
                return measurement(v, math.hypot(v * s_log, box), unit, u, tier)

            damage.append(DamageRegion(damage_id=d.damage_id, surface_id=d.surface_id, cls=d.cls, width=m(d.width_m),
                                       height=m(d.height_m), area=measurement(d.area_m2, 0.5 * d.area_m2 + 0.01, "m2", u, tier),
                                       position_on_surface=(d.from_left_m, d.from_floor_m), detection_score=d.score))
        if not c.damage:
            continue
        h = c.planes.ceiling_height_m or 2.5
        areas = {}
        for w in c.layout.walls:
            holes = sum(o.width_m * o.height_m for o in c.openings if o.wall_id == w.wall_id)
            areas[w.wall_id] = max(0.0, w.length_m * h - holes)
        areas[f"{r}-C"] = areas[f"{r}-F"] = c.layout.floor_area_m2
        f_, s_ = apply_rules(c.damage, areas, c.planes.ceiling_z, prefix=f"{r}-")
        flags += [ConcealedFlag(**f) for f in f_]
        for it in s_:
            q = it.pop("quantity")
            unit = it.pop("unit")
            sig = 0.0 if unit == "count" else 0.1 * q + 0.02
            scope.append(ScopeItem(**it, quantity=measurement(q, sig, unit if unit != "count" else "count", u, tier)))
    return damage, flags, scope


def write(result: ScanResult, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    files = {"result": out / "result.json", "plan_png": out / "plan.png", "plan_svg": out / "plan.svg"}
    files["result"].write_text(result.model_dump_json(by_alias=True, indent=2) + "\n")
    ScanResult.model_validate_json(files["result"].read_text())  # round-trip check against the schema
    for room in result.rooms:
        files[f"room_{room.room_id}"] = out / "rooms" / f"{room.room_id}.png"
        render_room(room, files[f"room_{room.room_id}"], result.damage)
    render_stitched(result, files["plan_png"], files["plan_svg"])
    return files


__all__ = ["Measurement", "assemble", "software_info", "write"]
