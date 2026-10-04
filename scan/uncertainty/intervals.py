"""Intervals on every measurement (ARCHITECTURE §11). First version; calibrated in step 1.10.

For a length-like measurement m:
    sigma^2 = (m * sigma_log_s)^2 + sigma_geom^2 + sigma_fit^2 + sigma_extra^2
    interval = m +- k_tier * z * sigma          (z = 1.645 for a 90% interval)
sigma_fit uses each wall's measured spread (robust std of its supporting points): with views that
disagree by a few cm it is genuinely unclear which copy is right, so the spread is the honest
position uncertainty, not the (tiny) standard error of the median.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from scan.schema import Measurement


def z(level: float) -> float:
    return float(norm.ppf(0.5 + level / 2))


def k_for(ucfg: dict, tier: str, mtype: str) -> float:
    """Calibrated k for (tier_mode, measurement type) if config/calibration.yaml has it, else the default."""
    cal = (ucfg.get("calibration") or {}).get(f"{tier}_{ucfg.get('mode', 'per_room')}", {})
    return float(cal[mtype]["k"]) if mtype in cal else float(ucfg["k_tier"][tier])


def measurement(value: float, sigma: float, unit: str, ucfg: dict, tier: str, nonneg: bool = True,
                mtype: str = "default") -> Measurement:
    half = k_for(ucfg, tier, mtype) * z(ucfg["interval_level"]) * sigma
    lo = value - half
    return Measurement(value=round(value, 4), lo=round(max(lo, 0.0) if nonneg else lo, 4),
                       hi=round(value + half, 4), unit=unit)


def _pos_sigma(spread: float, support: int, ucfg: dict) -> float:
    """Uncertainty of one wall line's position."""
    s = spread if (not math.isnan(spread) and support >= 20) else 0.0
    extra = ucfg["sigma_extra_m"]["weak_wall"] if support < 20 else 0.0
    return math.hypot(s, extra)


def wall_sigmas(layout, sigma_log: float, ucfg: dict, tier: str) -> list[float]:
    """Length uncertainty per wall: scale + local noise + the two bounding walls' positions."""
    walls = layout.walls
    n = len(walls)
    pos = [_pos_sigma(w.spread_m, w.support, ucfg) for w in walls]
    g = ucfg["sigma_geom_m"][tier]
    extra = ucfg["sigma_extra_m"]["fallback_layout"] if layout.method == "bbox_fallback" else 0.0
    out = []
    for i, w in enumerate(walls):
        fit = math.hypot(pos[i - 1], pos[(i + 1) % n])  # the neighbours fix this wall's end points
        out.append(math.sqrt((w.length_m * sigma_log) ** 2 + g**2 + fit**2 + extra**2))
    return out


def ceiling_sigma(planes, sigma_log: float, ucfg: dict, tier: str) -> float | None:
    if planes.ceiling_height_m is None:
        return None
    g = ucfg["sigma_geom_m"][tier]
    fit = math.hypot(planes.floor_rms_m or 0.0, planes.ceiling_rms_m or 0.0)
    return math.sqrt((planes.ceiling_height_m * sigma_log) ** 2 + g**2 + fit**2)


def area_sigma_parts(layout, sigma_log: float, ucfg: dict, tier: str) -> tuple[float, float]:
    """(scale part, independent edge part) of the floor-area sigma; kept apart for the footprint sum."""
    A = layout.floor_area_m2
    g = ucfg["sigma_geom_m"][tier]
    edge = math.sqrt(sum((w.length_m * math.hypot(_pos_sigma(w.spread_m, w.support, ucfg), g)) ** 2 for w in layout.walls))
    if layout.method == "bbox_fallback":
        edge = math.hypot(edge, ucfg["sigma_extra_m"]["fallback_layout"] * math.sqrt(A) * 2)
    return 2 * A * sigma_log, edge
