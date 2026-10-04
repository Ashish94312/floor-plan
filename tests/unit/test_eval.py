"""Tier 1 step 1.6: ground truth loading, wall matching by rotation, gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scan.config import load_config
from scan.eval.gates import evaluate
from scan.eval.gt import GTRoom, GTWall, load_gt
from scan.eval.match import match_walls
from scan.schema import Measurement

CFG = load_config()
REPO = Path(__file__).resolve().parents[2]


def pw(wid, v, kind="wall", rel=0.1):
    return SimpleNamespace(wall_id=wid, kind=kind, length=Measurement(value=v, lo=v * (1 - rel), hi=v * (1 + rel), unit="m"))


def test_match_finds_rotation_and_uses_open_walls():
    pred = [pw("p1", 3.7), pw("p2", 2.3), pw("p3", 2.7), pw("p4", 1.3, "open"), pw("p5", 1.0), pw("p6", 3.6)]
    gt = [GTWall("W1", 3.61), GTWall("W2", 3.70), GTWall("W3", 2.30), GTWall("W4", None),
          GTWall("W5", None, "open"), GTWall("W6", None)]
    m = match_walls(pred, gt)
    assert m["shift"] == 5 and [p.wall_id for p, _ in m["pairs"]][:2] == ["p6", "p1"]
    assert match_walls(pred[:4], gt) is None  # different wall counts are reported, not forced


def test_gates_on_synthetic_result():
    room = SimpleNamespace(room_id="r", status="ok", walls=[pw("r-W1", 3.0), pw("r-W2", 2.0), pw("r-W3", 3.0), pw("r-W4", 2.0)],
                           ceiling_height=Measurement(value=2.79, lo=2.6, hi=3.0, unit="m"),
                           floor_area=Measurement(value=6.0, lo=5.0, hi=7.0, unit="m2"))
    sp = SimpleNamespace(adjacency=[], overlaps=[], footprint_area=Measurement(value=6.0, lo=5, hi=7, unit="m2"),
                         drift_correction=SimpleNamespace(enabled=False, method="none"))
    res = SimpleNamespace(capture_id="c", tier="photo", rooms=[room], stitched_plan=sp, interval_level=0.9)
    gt = {"r": GTRoom("r", True, 2.80, [GTWall("W1", 2.95), GTWall("W2", 2.0), GTWall("W3", 2.95), GTWall("W4", 2.0)])}
    ev = evaluate(res, gt, CFG)
    walls = ev["rooms"]["r"]["walls"]
    assert [round(w["err_pct"], 2) for w in walls] == [pytest.approx(1.69, abs=0.01), 0.0, pytest.approx(1.69, abs=0.01), 0.0]
    assert ev["rooms"]["r"]["ceiling"]["g2_pass"] is True  # -1.0 cm
    assert ev["gates"]["G5 stitch"].startswith("pass") and ev["summary"]["coverage"] == 1.0


def test_load_home01_gt():
    gt = load_gt(REPO / "data/ground_truth/home01.yaml")
    assert gt["bedroom1"].area_m2() == pytest.approx(2.39 * 2.915)
    assert gt["hall"].walls[4].kind == "open" and "kitchen" in gt["hall"].connects_to
    assert gt["hall"].ceiling_m == pytest.approx(2.8025)
