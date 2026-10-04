"""Score a ScanResult against ground truth: per-wall errors, gates, calibration (ARCHITECTURE §13)."""

from __future__ import annotations

from scan.eval.gt import GTRoom, gt_adjacency
from scan.eval.match import match_walls


def _m(meas, gt, tol_rel=None):
    err = meas.value - gt
    out = {"pred": meas.value, "lo": meas.lo, "hi": meas.hi, "gt": round(gt, 4), "err_m": round(err, 4),
           "err_pct": round(100 * err / gt, 2), "covered": meas.lo <= gt <= meas.hi,
           "half_width_pct": round(100 * (meas.hi - meas.lo) / 2 / gt, 1)}
    if tol_rel is not None:
        out["within_tol"] = abs(err) <= tol_rel * gt
    return out


def evaluate(result, gt: dict[str, GTRoom], cfg: dict) -> dict:
    e = cfg["eval"]
    tol = e["wall_tolerance"][result.tier]
    rooms_out, scored = {}, []
    pred_rooms = {r.room_id: r for r in result.rooms}
    for rid, g in gt.items():
        if not g.captured:
            continue
        p = pred_rooms.get(rid)
        if p is None:
            rooms_out[rid] = {"status": "missing from result"}
            continue
        ro: dict = {"status": p.status}
        m = match_walls(p.walls, g.walls)
        if m is None:
            ro["walls"] = f"wall count differs: plan {len(p.walls)}, ground truth {len(g.walls)}"
        else:
            ro["shift"] = m["shift"]
            walls = []
            for pw, gw in m["pairs"]:
                row = {"plan": pw.wall_id, "gt_id": gw.wall_id, "kind_plan": pw.kind, "kind_gt": gw.kind}
                if gw.length_m is not None:
                    row |= _m(pw.length, gw.length_m, tol)
                    scored.append(row)
                walls.append(row)
            ro["walls"] = walls
            by_gt = {gw.wall_id: pw for pw, gw in m["pairs"]}
            wall_map = {gw.wall_id: pw.wall_id for pw, gw in m["pairs"]}
            ro["openings"] = _match_openings(p, g, wall_map, e)
            ro["damage"] = _match_damage(result, rid, g, wall_map)
            ro["checks"] = []
            for ids, total in g.checks:
                v = sum(by_gt[i].length.value for i in ids)
                ro["checks"].append({"sum": ids, "pred": round(v, 4), "gt": total,
                                     "err_pct": round(100 * (v - total) / total, 2)})
        if g.ceiling_m is not None and p.ceiling_height is not None:
            c = _m(p.ceiling_height, g.ceiling_m)
            c["err_cm"] = round(100 * c["err_m"], 2)
            c["g2_pass"] = abs(c["err_m"]) <= e["g2_ceiling_m"]
            ro["ceiling"] = c
            scored.append(c)
        if g.area_m2() is not None:
            ro["area"] = _m(p.floor_area, g.area_m2())
            scored.append(ro["area"])
        rooms_out[rid] = ro

    sp = result.stitched_plan
    pred_adj = {frozenset((a.room_a, a.room_b)) for a in sp.adjacency}
    true_adj = gt_adjacency(gt)
    areas = [g.area_m2() for g in gt.values() if g.captured]
    footprint = None if any(a is None for a in areas) or not areas else _m(sp.footprint_area, sum(areas))
    wall_rows = [w for r in rooms_out.values() for w in (r.get("walls") or []) if isinstance(w, dict) and "err_pct" in w]
    ceilings = {r: v["ceiling"] for r, v in rooms_out.items() if "ceiling" in v}
    coverage = sum(x["covered"] for x in scored) / len(scored) if scored else None

    op_rows = [o for r in rooms_out.values() for o in r.get("openings", [])]
    n_gt = sum(1 for o in op_rows if o["kind"] in ("matched", "missed"))
    n_phantom = sum(1 for o in op_rows if o["kind"] == "phantom")
    n_ok = sum(1 for o in op_rows if o.get("width_ok"))
    g1 = n_ok / (n_gt + n_phantom) if (n_gt + n_phantom) else None
    gates = {
        "G1 openings (width <= 2 cm, >= 85%)": "n/a (no measured openings)" if g1 is None else
        f"{'pass' if g1 >= e['g1_pass_rate'] else 'fail'}: {n_ok}/{n_gt + n_phantom} = {g1:.0%} "
        f"({n_gt} measured, {n_phantom} phantom)",
        "G2 ceiling <= 1.5 cm": ("pass" if all(c["g2_pass"] for c in ceilings.values()) else "fail") if ceilings else "n/a",
        "G3 repeatability": "n/a (needs a repeat capture, e.g. home01_photo_b)",
        "G4 drift handling": f"{'on' if sp.drift_correction.enabled else 'off'}: {sp.drift_correction.method}",
        "G5 stitch": ("pass" if (pred_adj == true_adj and not sp.overlaps
                                  and (footprint is None or abs(footprint["err_pct"]) <= 100 * e["g5_footprint_rel"]))
                      else "fail") + ("" if footprint else " (footprint n/a: not all walls measured)"),
        "damage": _damage_gate(rooms_out),
        f"walls within ±{100 * tol:.0f}%": (f"{sum(w['within_tol'] for w in wall_rows)}/{len(wall_rows)}" if wall_rows else "n/a"),
        f"calibration (target {result.interval_level:.0%})": (f"{coverage:.0%} of {len(scored)} intervals contain the truth"
                                                              if coverage is not None else "n/a"),
    }
    import json

    return {
        "capture_id": result.capture_id,
        "tier": result.tier,
        "mode": f"{result.tier}_{result.software.get('interval_mode', 'per_room')}",
        "interval_k": json.loads(result.software.get("interval_k", "{}")) or {"default": 1.5},
        "rooms": rooms_out,
        "adjacency": {"plan": sorted(map(sorted, pred_adj)), "gt": sorted(map(sorted, true_adj)), "equal": pred_adj == true_adj},
        "overlaps": len(sp.overlaps),
        "footprint": footprint,
        "summary": {
            "walls_scored": len(wall_rows),
            "mean_abs_wall_err_pct": round(sum(abs(w["err_pct"]) for w in wall_rows) / len(wall_rows), 2) if wall_rows else None,
            "max_abs_wall_err_pct": max((abs(w["err_pct"]) for w in wall_rows), default=None),
            "coverage": None if coverage is None else round(coverage, 3),
            "mean_half_width_pct": round(sum(x["half_width_pct"] for x in scored) / len(scored), 1) if scored else None,
        },
        "gates": gates,
    }


def _match_openings(p, g, wall_map: dict, e: dict) -> list[dict]:
    """GT openings with a measured width vs predicted openings on the matched wall (doors/openings vs
    windows), nearest by width. Unmatched GT = missed; unmatched prediction = phantom (G1)."""
    pred = list(p.openings)
    used, rows = set(), []
    for go in g.openings:
        if go.width_m is None:
            continue
        cls = "window" if go.type == "window" else "door"
        cands = [o for o in pred if o.wall_id == wall_map.get(go.wall_id) and o.opening_id not in used
                 and ("window" if o.type == "window" else "door") == cls]
        if not cands:
            rows.append({"kind": "missed", "gt": go.opening_id, "gt_width": go.width_m, "type": go.type})
            continue
        o = min(cands, key=lambda o: abs(o.width.value - go.width_m))
        used.add(o.opening_id)
        err = o.width.value - go.width_m
        rows.append({"kind": "matched", "gt": go.opening_id, "plan": o.opening_id, "type": o.type,
                     "gt_width": go.width_m, "width": o.width.value, "err_cm": round(100 * err, 1),
                     "width_ok": abs(err) <= e["g1_width_m"], "covered": o.width.lo <= go.width_m <= o.width.hi})
    gt_walls = set(wall_map.values())
    for o in pred:
        if o.opening_id not in used and o.wall_id in gt_walls:
            rows.append({"kind": "phantom", "plan": o.opening_id, "type": o.type, "width": o.width.value})
    return rows


def _match_damage(result, rid: str, g, wall_map: dict) -> list[dict]:
    pred = [d for d in result.damage if d.surface_id.startswith(rid + "-")]
    used, rows = set(), []
    for gd in g.damage:
        cands = [d for d in pred if d.surface_id == wall_map.get(gd.wall_id) and d.cls == gd.cls and d.damage_id not in used]
        if not cands:
            rows.append({"kind": "missed", "gt": gd.damage_id, "class": gd.cls})
            continue
        d = min(cands, key=lambda d: abs(d.position_on_surface[0] - (gd.from_left_m or 0)))
        used.add(d.damage_id)
        rows.append({"kind": "matched", "gt": gd.damage_id, "plan": d.damage_id, "class": d.cls,
                     "width_err_cm": None if gd.width_m is None else round(100 * (d.width.value - gd.width_m), 1),
                     "height_err_cm": None if gd.height_m is None else round(100 * (d.height.value - gd.height_m), 1),
                     "from_left_err_cm": None if gd.from_left_m is None else round(100 * (d.position_on_surface[0] - gd.from_left_m), 1)})
    rows += [{"kind": "phantom", "plan": d.damage_id, "class": d.cls} for d in pred if d.damage_id not in used]
    return rows


def _damage_gate(rooms_out: dict) -> str:
    rows = [d for r in rooms_out.values() for d in r.get("damage", [])]
    if not rows:
        return "n/a (no damage in the ground truth or the result)"
    n = {k: sum(1 for d in rows if d["kind"] == k) for k in ("matched", "missed", "phantom")}
    return f"found {n['matched']}/{n['matched'] + n['missed']}, phantom {n['phantom']}"
