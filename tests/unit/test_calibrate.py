"""Step 1.10: split-conformal interval calibration and its small-sample guards."""

from __future__ import annotations

import pytest

from scan.config import load_config
from scan.uncertainty.calibrate import conformal_k, fit
from scan.uncertainty.intervals import k_for

CFG = load_config()


def rows(ratios, k_used=1.0, room="a"):
    return [{"mode": "photo_joint", "type": "wall", "room": room, "ratio": r, "k_used": k_used} for r in ratios]


def test_conformal_order_statistic_with_enough_data():
    rs = [0.1 * i for i in range(1, 20)]  # n = 19 -> ceil(20 * 0.9) = 18th smallest
    k, how = conformal_k(rows(rs), 0.9, k_min=0.0, inflate=2.0)
    assert k == pytest.approx(1.8) and "order statistic 18/19" in how


def test_small_sample_inflated_and_floored():
    k, how = conformal_k(rows([0.2, 0.3]), 0.9, k_min=0.0, inflate=2.0)
    assert k == pytest.approx(0.6) and "too small" in how
    k, how = conformal_k(rows([0.01, 0.02]), 0.9, k_min=0.5, inflate=2.0)
    assert k == 0.5 and "floored" in how


def test_ratio_scales_with_k_used():
    k, _ = conformal_k(rows([0.1] * 19, k_used=1.5), 0.9, k_min=0.0, inflate=2.0)
    assert k == pytest.approx(0.15)  # residuals 0.1 of a k=1.5 interval = 0.15 in units of z*sigma


def test_k_for_uses_calibration_by_mode_and_type():
    u = {**CFG["uncertainty"], "mode": "joint", "calibration": {"photo_joint": {"wall": {"k": 0.7}}}}
    assert k_for(u, "photo", "wall") == 0.7
    assert k_for(u, "photo", "opening") == CFG["uncertainty"]["k_tier"]["photo"]  # not calibrated -> default
    assert k_for({**u, "mode": "per_room"}, "photo", "wall") == CFG["uncertainty"]["k_tier"]["photo"]


def test_fit_reports_leave_one_room_out():
    ev = lambda room, err: {"mode": "photo_joint", "interval_k": {"wall": 1.5}, "capture_id": "c",
                            "rooms": {room: {"walls": [{"err_m": err, "hi": 3.5, "pred": 3.0}]}}}
    cal = fit([ev("a", 0.02), ev("b", 0.03)], CFG)
    w = cal["photo_joint"]["wall"]
    assert w["n"] == 2 and w["loo_n"] == 2 and w["loo_coverage"] == 1.0 and w["k"] >= CFG["calibration_fit"]["k_min"]
