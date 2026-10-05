"""scan-bench: run every capture of a suite (config/bench.yaml), score it against the tape, write reports/benchmark.md.

Each capture runs as its own process with the default config (`scan <capture>`, the one command a user runs), then
again with --no-drift-correction where the suite asks for the G4 ablation. The report covers the gates per capture
and tier, repeatability (same rooms twice at one tier), the drift ablation, the head-to-head against a consumer app,
and timing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from scan.config import load_config
from scan.eval.gates import evaluate
from scan.eval.gt import load_gt
from scan.eval.report import markdown
from scan.schema import ScanResult

RUN = "from scan.cli import scan_main; scan_main()"


def run_capture(cap: Path, out: Path, extra: list[str], log) -> None:
    log(f"  scan {cap} -> {out} {' '.join(extra)}")
    p = subprocess.run([sys.executable, "-c", RUN, str(cap), "--out", str(out), *extra], capture_output=True, text=True,
                       check=False)
    (out.parent / f"{out.name}.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"scan {cap} failed (exit {p.returncode}); see {out.parent / (out.name + '.log')}")


def score(out: Path, gt: dict, cfg: dict) -> dict:
    res = json.loads((out / "result.json").read_text())
    ev = evaluate(ScanResult.model_validate(res), gt, cfg)
    (out / "eval.json").write_text(json.dumps(ev, indent=2, default=str))
    (out / "eval.md").write_text(markdown(ev))
    return {"eval": ev, "timing": res["timing_s"], "tier": res["tier"], "warnings": res["warnings"],
            "commit": res["software"]["git_commit"]}


def _walls(ev: dict, room: str) -> dict[str, dict]:
    return {w["gt_id"]: w for w in (ev["rooms"].get(room, {}).get("walls") or []) if isinstance(w, dict) and "pred" in w}


def ours(ev: dict, d: dict) -> float | None:
    """Our value for one head-to-head dimension, None if our plan does not give it (counts as a loss)."""
    r = ev["rooms"].get(d["room"], {})
    if d["kind"] in ("wall", "wall_mean"):
        w = _walls(ev, d["room"])
        if not all(k in w for k in d["walls"]):
            return None
        return sum(w[k]["pred"] for k in d["walls"]) / len(d["walls"])
    m = r.get("ceiling" if d["kind"] == "ceiling" else "area")
    return m.get("pred") if isinstance(m, dict) else None


def truth(gt: dict, d: dict) -> float:
    room = gt[d["room"]]
    if d["kind"] in ("wall", "wall_mean"):
        L = {w.wall_id: w.length_m for w in room.walls}
        return sum(L[k] for k in d["walls"]) / len(d["walls"])
    return room.ceiling_m if d["kind"] == "ceiling" else room.area_m2()


def head_to_head(app: dict, ev: dict, gt: dict) -> list[dict]:
    """Win / tie / loss per shared dimension. Tie: the absolute errors differ by at most 1 cm (0.05 m2 for areas)."""
    rows = []
    for d in app["dimensions"]:
        t, o = truth(gt, d), ours(ev, d)
        ea, eo = abs(d["app"] - t), (abs(o - t) if o is not None else None)
        tie = 0.05 if d["kind"] == "area" else 0.01
        res = "loss" if eo is None else ("tie" if abs(eo - ea) <= tie else ("win" if eo < ea else "loss"))
        rows.append({**d, "truth": t, "ours": o, "err_ours": eo, "err_app": ea, "result": res})
    return rows


def repeat(ev_a: dict, ev_b: dict, gt: dict) -> list[dict]:
    """Per room: matched tape walls scored in both captures (G3: |L1 - L2| <= max(1 cm, 0.5% x L_tape)) and the
    ceiling spread (G2: <= 1 cm)."""
    rows = []
    for room in gt:
        wa, wb = _walls(ev_a, room), _walls(ev_b, room)
        for k in sorted(set(wa) & set(wb)):
            d, tol = abs(wa[k]["pred"] - wb[k]["pred"]), max(0.01, 0.005 * wa[k]["gt"])
            rows.append({"room": room, "item": f"wall {k}", "a": wa[k]["pred"], "b": wb[k]["pred"], "diff": d,
                         "tol": tol, "pass": d <= tol})
        for k in sorted(set(wa) ^ set(wb)):
            rows.append({"room": room, "item": f"wall {k}", "a": wa.get(k, {}).get("pred"),
                         "b": wb.get(k, {}).get("pred"), "diff": None, "tol": None, "pass": False})
        ca, cb = (ev.get("rooms", {}).get(room, {}).get("ceiling") for ev in (ev_a, ev_b))
        if isinstance(ca, dict) and isinstance(cb, dict) and "pred" in ca and "pred" in cb:
            d = abs(ca["pred"] - cb["pred"])
            rows.append({"room": room, "item": "ceiling", "a": ca["pred"], "b": cb["pred"], "diff": d, "tol": 0.01,
                         "pass": d <= 0.01})
    return rows


def run_suite(suite_path: Path, skip_run: bool = False, log=print) -> Path:
    suite = yaml.safe_load(suite_path.read_text())
    gt, cfg = load_gt(Path(suite["ground_truth"])), load_config()
    caps = [Path(c["path"]) for c in suite["captures"]]
    res, nodrift = {}, {}
    for c in caps:
        if not skip_run:
            run_capture(c, c / "out", [], log)
        res[c] = score(c / "out", gt, cfg)
    for c in map(Path, suite.get("drift_ablation", [])):
        if not skip_run:
            run_capture(c, c / "out_nodrift", ["--no-drift-correction"], log)
        nodrift[c] = score(c / "out_nodrift", gt, cfg)
    h2h = suite.get("head_to_head")
    app = yaml.safe_load(Path(h2h["app"]).read_text()) if h2h else None
    out = Path("reports")
    out.mkdir(exist_ok=True)
    md = report(suite, res, nodrift, app, gt)
    (out / "benchmark.md").write_text(md)
    (out / "benchmark.json").write_text(json.dumps(
        {"captures": {str(c): r["eval"] for c, r in res.items()}, "timing": {str(c): r["timing"] for c, r in res.items()},
         "nodrift": {str(c): r["eval"]["footprint"] for c, r in nodrift.items()},
         "head_to_head": {str(c): head_to_head(app, res[Path(c)]["eval"], gt) for c in h2h["ours"]} if app else {}},
        indent=2, default=str))
    return out / "benchmark.md"


def _f(x, nd=3, unit=""):
    return "—" if x is None else f"{x:.{nd}f}{unit}"


def report(suite: dict, res: dict, nodrift: dict, app: dict | None, gt: dict) -> str:
    L = ["# Benchmark", "",
         f"Ground truth: `{suite['ground_truth']}` (laser). Every number below is regenerated by `uv run scan-bench`.",
         (f"Code: `{next(iter(res.values()))['commit']}`. LiDAR tier: **no capture** (no Pro iPhone available); "
          "photo and video only."), ""]
    # --- gates per capture
    L += ["## Gates per capture", "",
          ("| capture | tier | walls within tier tolerance | mean / max wall error | interval coverage (target 90%) "
           "| G1 openings | G2 ceilings ≤ 1.5 cm | G5 stitch | footprint vs tape | damage |"),
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in res.items():
        ev, g, s = r["eval"], r["eval"]["gates"], r["eval"]["summary"]
        tol_key = next(k for k in g if k.startswith("walls within"))
        cov_key = next(k for k in g if k.startswith("calibration"))
        ceil = [f"{v['ceiling']['err_cm']:+.1f}" for v in ev["rooms"].values()
                if isinstance(v.get("ceiling"), dict) and "err_cm" in v["ceiling"]]
        fp = ev["footprint"]
        L.append(f"| {c.name} | {r['tier']} | {g[tol_key]} ({tol_key.split()[-1]}) | "
                 f"{_f(s['mean_abs_wall_err_pct'], 1, '%')} / {_f(s['max_abs_wall_err_pct'], 1, '%')} | {g[cov_key]} | "
                 f"{g[next(k for k in g if k.startswith('G1'))]} | {g[next(k for k in g if k.startswith('G2'))]} "
                 f"({', '.join(ceil)} cm) | {g['G5 stitch']} | "
                 f"{_f(fp and fp['pred'], 2)} vs {_f(fp and fp['gt'], 2)} m² ({_f(fp and fp['err_pct'], 1, '%')}) | "
                 f"{g['damage']} |")
    # --- repeatability
    L += ["", "## Repeatability (same rooms, same tier)", "",
          ("G3: |L1 − L2| ≤ max(1 cm, 0.5% × tape). G2 spread: ceilings ≤ 1 cm apart. "
           "A wall scored in only one capture counts as a fail."), ""]
    evs = {c: r["eval"] for c, r in res.items()}
    for a, b in suite.get("repeat_pairs", []):
        a, b = Path(a), Path(b)
        rows = repeat(evs[a], evs[b], gt)
        n = sum(x["pass"] for x in rows)
        L += [f"### {a.name} vs {b.name}: {n}/{len(rows)} pass", "",
              "| room | item | first (m) | second (m) | difference | allowed | |", "|---|---|---|---|---|---|---|"]
        L += [f"| {x['room']} | {x['item']} | {_f(x['a'])} | {_f(x['b'])} | "
              f"{_f(x['diff'] and 100 * x['diff'], 1, ' cm')} | {_f(x['tol'] and 100 * x['tol'], 1, ' cm')} | "
              f"{'✅' if x['pass'] else '❌'} |" for x in rows]
        L.append("")
    # --- drift ablation
    if nodrift:
        L += ["## G4: drift correction on vs off (stitched footprint)", "",
              "| capture | tape (m²) | on | off |", "|---|---|---|---|"]
        for c, r in nodrift.items():
            on, off = evs[c]["footprint"], r["eval"]["footprint"]
            L.append(f"| {c.name} | {_f(on and on['gt'], 2)} | {_f(on and on['pred'], 2)} ({_f(on and on['err_pct'], 1, '%')}) "
                     f"| {_f(off and off['pred'], 2)} ({_f(off and off['err_pct'], 1, '%')}) |")
        L.append("")
    # --- head-to-head
    if app:
        h = suite["head_to_head"]
        L += ["## Head-to-head", "",
              (f"{app['app']} (version: {app.get('app_version') or 'TO FILL'}), {app['device']}. "
               "Required tier is LiDAR; with no LiDAR capture our photo and video tiers stand in. "
               "Tie: absolute errors within 1 cm (0.05 m² for areas). A dimension our plan does not give is a loss. "
               f"Headline capture chosen before the run: **{Path(h['headline']).name}**."), ""]
        for c in map(Path, h["ours"]):
            rows = head_to_head(app, evs[c], gt)
            wins = sum(x["result"] in ("win", "tie") for x in rows)
            L += [f"### {c.name}: beat or tie on {wins}/{len(rows)} ({100 * wins / len(rows):.0f}%, pass ≥ 70%)", "",
                  "| room | dimension | tape | ours | our error | app | app error | result |",
                  "|---|---|---|---|---|---|---|---|"]
            L += [f"| {x['room']} | {x['name']} | {x['truth']:.3f} | {_f(x['ours'])} | {_f(x['err_ours'])} | "
                  f"{x['app']:.3f} | {x['err_app']:.3f} | {x['result']} |" for x in rows]
            L.append("")
    # --- timing
    L += ["## Timing (seconds, M4 MacBook, 16 GB)", "",
          "| capture | ingest | model load | geometry | alignment | layout | openings | detector | output | total |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c, r in res.items():
        t = r["timing"]
        keys = ["ingest_s", "model_load_s", "geometry_s", "alignment_s", "layout_s", "openings_s", "detector_damage_s",
                "output_s"]
        L.append(f"| {c.name} | " + " | ".join(_f(t.get(k), 1) for k in keys) + f" | {sum(t.values()):.1f} |")
    L += ["", ("Model outputs are cached by input hash (deterministic replay, E19); "
               "a cold run (`scan --no-cache`) adds the model time shown in the worklog."), ""]
    return "\n".join(L)
