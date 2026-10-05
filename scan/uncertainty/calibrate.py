"""Interval calibration (ARCHITECTURE §11, D30): split-conformal k per tier/mode and measurement type.

For each scored measurement, r = |error| / half-width-used. New k = k_used x the ceil((n+1)q)-th smallest
r (split-conformal: >= q coverage if future rooms are exchangeable with these). Small-sample guards:
n too small for that order statistic -> max(r) x small_n_inflate; and k never below k_min.
Leave-one-room-out coverage is reported so no room grades its own interval.

Size bias (E22q): the systematic part of the size error, b = log(tape / plan), per tier/mode. After the video
camera re-solve a room's error is one isotropic scale, so taped walls and ceilings measure the same b. Model:
b = mu + e_room (e: content-dependent, spread tau). Measurements of one room share its e, so the room is the
unit: each reconstructed room gives its median b; captures of one PHYSICAL room are not independent either, so
mu = median over physical rooms (of the median over their captures). tau = RMS of the reconstructed rooms
around mu: the spread a new room keeps after the correction. Tested leave-one-physical-room-out.
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


def size_rows(ev: dict) -> list[dict]:
    """log(tape / plan) of every scored wall and ceiling in one eval.json, with its room and physical room."""
    rows = []
    prop = ev["capture_id"].split("_")[0]
    for room, r in ev["rooms"].items():
        items = [w for w in (r.get("walls") or []) if isinstance(w, dict) and "err_m" in w]
        items += [r["ceiling"]] if isinstance(r.get("ceiling"), dict) and "err_m" in r["ceiling"] else []
        for m in items:
            if m.get("pred", 0) > 0 and m.get("gt", 0) > 0:
                rows.append({"mode": ev["mode"], "room": f"{ev['capture_id']}/{room}", "phys": f"{prop}/{room}",
                             "log": math.log(m["gt"] / m["pred"])})
    return rows


def _median(x):
    x = sorted(x)
    n = len(x)
    return (x[n // 2] + x[(n - 1) // 2]) / 2


def _bias(rows: list[dict], min_meas: int = 1) -> tuple[float, float, dict, dict]:
    rooms: dict[str, list] = {}
    for r in rows:
        rooms.setdefault(r["room"], []).append(r)
    # a room's median is robust only with several measurements (n tolerates (n-1)//2 bad ones: a lost wall, a
    # spurious ceiling layer); a room scored by one measurement would carry its failure straight into the fit
    rooms = {k: v for k, v in rooms.items() if len(v) >= min_meas}
    room_b = {k: _median([r["log"] for r in v]) for k, v in rooms.items()}
    phys: dict[str, list] = {}
    for k, v in rooms.items():
        phys.setdefault(v[0]["phys"], []).append(room_b[k])
    phys_b = {p: _median(v) for p, v in phys.items()}
    mu = _median(list(phys_b.values()))
    tau = math.sqrt(sum((b - mu) ** 2 for b in room_b.values()) / len(room_b))
    return mu, tau, room_b, {k: rooms[k][0]["phys"] for k in rooms}


def fit_size_bias(rows: list[dict], min_meas: int = 3) -> dict:
    """Per mode: {log_bias, factor, sigma_log, rooms, physical_rooms, loo}. loo: each physical room corrected with
    the bias learnt from the OTHER physical rooms (none of its own captures). Rooms with fewer than min_meas
    scored measurements are left out."""
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        sub = [r for r in rows if r["mode"] == mode]
        counts: dict[str, int] = {}
        for r in sub:
            counts[r["room"]] = counts.get(r["room"], 0) + 1
        sub = [r for r in sub if counts[r["room"]] >= min_meas]
        if not sub:
            continue
        mu, tau, room_b, room_phys = _bias(sub)
        loo = []
        for p in sorted(set(room_phys.values())):
            train = [r for r in sub if r["phys"] != p]
            if not train:
                continue
            mu_p = _bias(train)[0]
            for k, b in room_b.items():
                if room_phys[k] == p:
                    loo.append({"room": k, "raw_pct": round(100 * (math.exp(-b) - 1), 2),
                                "corrected_pct": round(100 * (math.exp(mu_p - b) - 1), 2)})
        out[mode] = {"log_bias": round(mu, 4), "factor": round(math.exp(mu), 4), "sigma_log": round(tau, 4),
                     "rooms": len(room_b), "physical_rooms": len(set(room_phys.values())), "loo": loo}
    return out


def write(cal: dict, path: Path = CAL_PATH) -> Path:
    header = ("# Written by scan-calibrate (D30). Per tier_mode and measurement type: interval inflation k;\n"
              "# scale_bias: systematic size error, plan x factor (E22q).\n")
    path.write_text(header + yaml.safe_dump(cal, sort_keys=True))
    return path


def load(path: Path = CAL_PATH) -> dict:
    return yaml.safe_load(path.read_text()) if path.exists() else {}
