"""Tier 1 step 1.5: published schema, interval model, stitched plan, output files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from scan.config import load_config
from scan.layout.room import layout_rooms
from scan.output import assemble, write
from scan.schema import Measurement, ScanResult, json_schema
from scan.types import Capture
from tests.unit.test_layout import corner_cams, synthetic_room

CFG = load_config()
REPO = Path(__file__).resolve().parents[2]


def test_measurement_must_contain_value_and_forbids_extra_keys():
    Measurement(value=1.0, lo=0.9, hi=1.1, unit="m")
    with pytest.raises(ValidationError):
        Measurement(value=1.0, lo=1.05, hi=1.1, unit="m")
    with pytest.raises(ValidationError):
        Measurement(value=1.0, lo=0.9, hi=1.1, unit="m", extra=1)


def test_committed_schema_file_matches_the_models():
    committed = json.loads((REPO / "schema" / "scan_output.schema.json").read_text())
    assert committed == json.loads(json.dumps(json_schema())), "run: uv run scan-schema"


def _two_rooms_joint():
    """Bedroom 3 x 2.4 next to a hall 3.7 x 2.3 across a 10 cm wall, in one shared frame."""
    bed = [(0.0, 0.0), (3.0, 0.0), (3.0, 2.4), (0.0, 2.4)]
    hall = [(-3.8, 0.0), (-0.1, 0.0), (-0.1, 2.3), (-3.8, 2.3)]
    rooms = {"bedroom1": synthetic_room("bedroom1", bed, corner_cams(bed)),
             "hall": synthetic_room("hall", hall, corner_cams(hall))}
    for c in rooms.values():
        c.frame_id = "joint"
        c.placed_by = "joint_reconstruction"
    for r, lay in layout_rooms(rooms, CFG).items():
        rooms[r].layout = lay
    return rooms


def _all_measurements(obj):
    if isinstance(obj, dict):
        if set(obj) >= {"value", "lo", "hi", "unit"}:
            yield obj
        for v in obj.values():
            yield from _all_measurements(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_measurements(v)


def test_assemble_joint_rooms_contract(tmp_path):
    rooms = _two_rooms_joint()
    cap = Capture("synthetic", tmp_path, "photo", devices=["Test Phone"])
    res = assemble(cap, rooms, CFG, {"total_s": 1.0})
    data = json.loads(res.model_dump_json(by_alias=True))
    for key in ("rooms", "stitched_plan", "damage", "concealed_damage_flags", "scope_items", "interval_level", "tier"):
        assert key in data
    ms = list(_all_measurements(data))
    assert len(ms) >= 2 * (4 + 2) + 1  # walls + ceiling + area per room, plus footprint
    assert all(m["lo"] <= m["value"] <= m["hi"] for m in ms)
    sp = res.stitched_plan
    assert sp.placement_method == "joint_reconstruction" and all(p.placed for p in sp.room_poses)
    assert [(a.room_a, a.room_b) for a in sp.adjacency] == [("bedroom1", "hall")] and sp.overlaps == []
    assert sp.footprint_area.value == pytest.approx(3.0 * 2.4 + 3.7 * 2.3, rel=0.03)
    # shared scale -> footprint interval at least as wide as the per-room scale terms added linearly
    files = write(res, tmp_path / "out")
    for k in ("result", "plan_png", "plan_svg", "room_bedroom1", "room_hall"):
        assert files[k].exists() and files[k].stat().st_size > 0
    ScanResult.model_validate_json(files["result"].read_text())


def test_overlapping_rooms_are_reported(tmp_path):
    rooms = _two_rooms_joint()
    lay = rooms["hall"].layout
    lay.polygon = [(x + 1.0, y) for x, y in lay.polygon]  # push the hall 1 m into the bedroom
    res = assemble(Capture("s", tmp_path, "photo"), rooms, CFG, {})
    assert res.stitched_plan.overlaps and res.stitched_plan.overlaps[0].area_m2 > 1.0


def test_separate_frames_are_unplaced_with_warning(tmp_path):
    rooms = _two_rooms_joint()
    rooms["hall"].frame_id = "hall"
    rooms["bedroom1"].frame_id = "bedroom1"
    for c in rooms.values():
        c.placed_by = None  # no door match
    res = assemble(Capture("s", tmp_path, "photo"), rooms, CFG, {})
    assert res.stitched_plan.placement_method == "unplaced"
    assert not any(p.placed for p in res.stitched_plan.room_poses)
    assert any("not placed" in w for w in res.warnings)


@pytest.mark.skipif(not (REPO / "captures/home01_photo_a/out/result.json").exists(), reason="run the pipeline first")
def test_home01_result_json_validates():
    res = ScanResult.model_validate_json((REPO / "captures/home01_photo_a/out/result.json").read_text())
    assert {r.room_id for r in res.rooms} == {"bedroom1", "hall", "kitchen"}
    assert np.all([w.length.lo <= w.length.value <= w.length.hi for r in res.rooms for w in r.walls])
