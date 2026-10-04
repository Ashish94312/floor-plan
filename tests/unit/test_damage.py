"""Tier 1 step 1.9: rules engine, multi-photo damage filter, detector openings."""

from __future__ import annotations

import pytest

from scan.config import load_config
from scan.damage.detect import DamageSeg, SurfaceBox, damage_regions, detector_openings
from scan.damage.rules import apply_rules

CFG = load_config()


def stain(u0, v0, w=0.5, h=0.35, photo="a", score=0.4, cls="water_stain", sid="r-W2"):
    return SurfaceBox(cls, sid, u0, u0 + w, v0, v0 + h, score, photo)


def test_damage_needs_two_photos_and_is_merged():
    boxes = {"r": [stain(1.0, 0.5, photo="a"), stain(1.05, 0.52, photo="b"),  # same stain, two photos
                   stain(2.5, 1.0, photo="a"),  # one photo only: dropped
                   stain(0.2, 0.2, photo="a", sid="r-F"), stain(0.2, 0.2, photo="b", sid="r-F")]}  # floor: not used
    regs = damage_regions(boxes, CFG)["r"]
    assert len(regs) == 1 and regs[0].surface_id == "r-W2" and regs[0].photos == 2
    assert regs[0].from_left_m == pytest.approx(1.025) and regs[0].area_m2 == pytest.approx(0.5 * 0.35 * 0.6, rel=0.02)


def test_rules_fire_with_rule_ids():
    d = DamageSeg("r-D1", "r-W2", "water_stain", 0.5, 0.3, 0.09, 1.0, 0.05, 0.4, 2)  # bottom 5 cm off the floor
    flags, scope = apply_rules([d], {"r-W2": 7.5}, ceiling_z=2.8, prefix="r-")
    assert [f["rule_id"] for f in flags] == ["STAIN_NEAR_FLOOR"]
    assert {s["action"] for s in scope} == {"Moisture-meter check behind surface", "Stain-block prime and repaint full surface"}
    assert next(s for s in scope if s["unit"] == "m2")["quantity"] == 7.5
    crack = DamageSeg("r-D2", "r-W1", "crack", 0.05, 0.6, 0.003, 1.0, 1.0, 0.3, 2)
    flags, scope = apply_rules([crack], {"r-W1": 6.0}, ceiling_z=2.8)
    assert [f["rule_id"] for f in flags] == ["LONG_CRACK"] and any(s["quantity"] == 0.6 for s in scope)


def test_detector_door_added_only_where_no_opening():
    from scan.layout.openings import OpeningSeg

    existing = OpeningSeg("r-O1", "door", "r-W1", 1.0, 0.8, 2.0, None, 4)
    door = lambda u0, photo, sid="r-W1": SurfaceBox("door", sid, u0, u0 + 0.85, 0.02, 2.0, 0.5, photo)
    boxes = {"r": [door(1.0, "a"), door(1.02, "b"),  # same as the see-through door: skipped
                   door(0.2, "a", "r-W3"), door(0.22, "b", "r-W3")]}  # closed door elsewhere: added
    ops = {"r": [existing]}
    detector_openings(boxes, ops, CFG)
    assert [(o.wall_id, o.type) for o in ops["r"]] == [("r-W1", "door"), ("r-W3", "door")]
