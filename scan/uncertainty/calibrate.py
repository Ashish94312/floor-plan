"""Interval calibration (ARCHITECTURE §11, D30): split-conformal k per tier/mode and measurement type.

For each scored measurement, r = |error| / half-width-used. New k = k_used x the ceil((n+1)q)-th smallest
r (split-conformal: >= q coverage if future rooms are exchangeable with these). Small-sample guards:
n too small for that order statistic -> max(r) x small_n_inflate; and k never below k_min.
Leave-one-room-out coverage is reported so no room grades its own interval.
"""

from __future__ import annotations

import math
from pathlib import Path

import yaml

CAL_PATH = Path(__file__).resolve().parents[2] / "config" / "calibration.yaml"
FIT_TYPES = ("wall", "ceiling", "area")  # opening widths excluded while their tape definition is in doubt (D28)


def residuals(ev: dict) -> list[dict]:
    """(mode, type, room, ratio, k_used) for every scored measurement in one eval.json."""
    mode, kmap = ev["mode"], ev["interval_k"]
    rows = []
    for room, r in ev["rooms"].items():
        items = [("wall", w) for w in (r.get("walls") or []) if isinstance(w, dict) and "err_m" in w]
        items += [(t, r[t]) for t in ("ceiling", "area") if t in r]
        for t, m in items:
            half = m["hi"] - m["pred"]
            if half > 0:
                rows.append({"mode": mode, "type": t, "room": f"{ev['capture_id']}/{room}",
                             "ratio": abs(m["err_m"]) / half, "k_used": kmap.get(t, kmap.get("default"))})
    return rows


def conformal_k(rows: list[dict], q: float, k_min: float, inflate: float) -> tuple[float, str]:
    n = len(rows)
    if n == 0:
        return float("nan"), "no data"
    scaled = sorted(r["ratio"] * r["k_used"] for r in rows)  # r in units of k: |err| / (z sigma)
    idx = math.ceil((n + 1) * q)
    if idx <= n:
        k, how = scaled[idx - 1], f"conformal order statistic {idx}/{n}"
    else:
        k, how = scaled[-1] * inflate, f"n={n} too small for q={q}: max x {inflate}"
    if k < k_min:
        return k_min, how + f", floored at k_min={k_min}"
    return k, how


def fit(evals: list[dict], cfg: dict) -> dict:
    c = cfg["calibration_fit"]
    q = cfg["uncertainty"]["interval_level"]
    rows = [r for ev in evals for r in residuals(ev)]
    out: dict = {}
    for mode in sorted({r["mode"] for r in rows}):
        out[mode] = {}
        for t in FIT_TYPES:
            sub = [r for r in rows if r["mode"] == mode and r["type"] == t]
            if not sub:
                continue
            k, how = conformal_k(sub, q, c["k_min"], c["small_n_inflate"])
            loo = []
            for room in sorted({r["room"] for r in sub}):
                train = [r for r in sub if r["room"] != room]
                test = [r for r in sub if r["room"] == room]
                if not train:
                    continue
                k_r, _ = conformal_k(train, q, c["k_min"], c["small_n_inflate"])
                loo += [r["ratio"] * r["k_used"] <= k_r for r in test]
            out[mode][t] = {"k": round(k, 4), "n": len(sub), "method": how,
                            "max_ratio_at_k_used": round(max(r["ratio"] for r in sub), 4),
                            "loo_coverage": None if not loo else round(sum(loo) / len(loo), 3), "loo_n": len(loo)}
    return out


def write(cal: dict, path: Path = CAL_PATH) -> Path:
    header = "# Written by scan-calibrate (D30). Per tier_mode and measurement type: interval inflation k.\n"
    path.write_text(header + yaml.safe_dump(cal, sort_keys=True))
    return path


def load(path: Path = CAL_PATH) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}
