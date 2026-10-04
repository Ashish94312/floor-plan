"""Eval report as markdown (out/eval.md) — the same text is printed by scan-eval."""

from __future__ import annotations


def markdown(ev: dict) -> str:
    L = [f"# Evaluation: {ev['capture_id']} ({ev['tier']} tier)", ""]
    for rid, r in ev["rooms"].items():
        L.append(f"## {rid} (status {r.get('status')}, wall rotation shift {r.get('shift', '—')})")
        if isinstance(r.get("walls"), str):
            L.append(f"- {r['walls']}")
        elif r.get("walls"):
            L += ["", "| plan | tape | kind | plan [lo, hi] m | tape m | error | in ±tol | interval covers |", "|---|---|---|---|---|---|---|---|"]
            for w in r["walls"]:
                if "err_pct" in w:
                    L.append(f"| {w['plan']} | {w['gt_id']} | {w['kind_plan']} | {w['pred']:.3f} [{w['lo']:.2f}, {w['hi']:.2f}] | "
                             f"{w['gt']:.3f} | {100 * w['err_m']:+.1f} cm ({w['err_pct']:+.1f}%) | "
                             f"{'✅' if w['within_tol'] else '❌'} | {'✅' if w['covered'] else '❌'} |")
                else:
                    L.append(f"| {w['plan']} | {w['gt_id']} | {w['kind_plan']} | — | not measured | — | — | — |")
        for c in r.get("checks", []):
            L.append(f"- check {' + '.join(c['sum'])}: plan {c['pred']:.3f} m, tape {c['gt']:.3f} m ({c['err_pct']:+.1f}%)")
        for o in r.get("openings", []):
            if o["kind"] == "matched":
                L.append(f"- opening {o['gt']} ↔ {o['plan']} ({o['type']}): width {o['width']:.2f} m vs tape {o['gt_width']:.2f} m "
                         f"→ {o['err_cm']:+.1f} cm {'✅' if o['width_ok'] else '❌'} (interval {'covers' if o['covered'] else 'misses'})")
            elif o["kind"] == "missed":
                L.append(f"- opening {o['gt']} ({o['type']}, {o['gt_width']:.2f} m): **missed**")
            else:
                L.append(f"- opening {o['plan']} ({o['type']}, {o['width']:.2f} m): not in the tape file (phantom, or not measured yet)")
        for d in r.get("damage", []):
            if d["kind"] == "matched":
                L.append(f"- damage {d['gt']} ↔ {d['plan']} ({d['class']}): width {d['width_err_cm']:+} cm, height {d['height_err_cm']:+} cm, "
                         f"position {d['from_left_err_cm']:+} cm")
            elif d["kind"] == "missed":
                L.append(f"- damage {d['gt']} ({d['class']}): **missed**")
            else:
                L.append(f"- damage {d['plan']} ({d['class']}): phantom (not in the tape file)")
        if "ceiling" in r:
            c = r["ceiling"]
            L.append(f"- ceiling: plan {c['pred']:.3f} [{c['lo']:.2f}, {c['hi']:.2f}] m, tape {c['gt']:.4f} m → "
                     f"{c['err_cm']:+.1f} cm, G2 {'✅' if c['g2_pass'] else '❌'}")
        if "area" in r:
            a = r["area"]
            L.append(f"- area: plan {a['pred']:.2f} m², tape {a['gt']:.2f} m² ({a['err_pct']:+.1f}%)")
        L.append("")
    adj = ev["adjacency"]
    L += ["## Whole property", f"- adjacency: plan {adj['plan']} vs tape {adj['gt']} → {'✅ equal' if adj['equal'] else '❌ differ'}",
          f"- overlaps: {ev['overlaps']}",
          "- footprint: " + ("n/a (not all walls measured)" if not ev["footprint"] else f"{ev['footprint']['err_pct']}%"), "",
          "## Gates", "", "| gate | result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in ev["gates"].items()]
    s = ev["summary"]
    L += ["", (f"Walls scored {s['walls_scored']}, mean |error| {s['mean_abs_wall_err_pct']}%, max {s['max_abs_wall_err_pct']}%. "
               f"Interval coverage {s['coverage']}, mean half-width {s['mean_half_width_pct']}%.")]
    return "\n".join(L) + "\n"
