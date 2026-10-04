"""Same capture in -> same output out (ARCHITECTURE §16). The live-vs-cached check is
scripts/check_determinism.py (about 3 min; E19: bit-identical, max |diff| 0.0 over 210 numbers)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scan.config import load_config
from tests.unit.test_geometry import _cached_capture_available

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not _cached_capture_available(), reason="needs home01_photo_a + its cached model outputs")
def test_two_runs_identical(tmp_path):
    from scan.pipeline import run

    cfg = load_config()
    outs = []
    for i in (1, 2):
        out = tmp_path / str(i)
        run(REPO / "captures/home01_photo_a", cfg, out=out, log=lambda *_: None)
        outs.append(out)
    a, b = (json.loads((o / "result.json").read_text()) for o in outs)
    a.pop("timing_s"), b.pop("timing_s")
    assert a == b
    for f in ("plan.png", "plan.svg", "rooms/hall.png"):
        h = [hashlib.sha256((o / f).read_bytes()).hexdigest() for o in outs]
        assert h[0] == h[1], f
