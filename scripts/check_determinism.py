"""Determinism check (ARCHITECTURE §12/§16, FR-RUN-05, NFR-07).

1. cached run twice  -> result.json identical (except timing) and every output file byte-identical
2. live run (--no-cache: model load + MapAnything + OWLv2) vs cached -> max |difference| over all numbers

Usage: uv run python scripts/check_determinism.py <capture_dir> <report_dir>
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan.config import load_config
from scan.pipeline import run

FILES = ["result.json", "plan.png", "plan.svg"]


def numbers(x, path=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from numbers(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from numbers(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield path, float(x)


def load(out):
    d = json.loads((out / "result.json").read_text())
    d.pop("timing_s")
    return d


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    cap, rep = Path(sys.argv[1]), Path(sys.argv[2])
    rep.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    outs, times = {}, {}
    for name, use_cache in (("cached_1", True), ("cached_2", True), ("live", False)):
        t = time.perf_counter()
        run(cap, cfg, out=rep / name, use_cache=use_cache, log=lambda *_: None)
        times[name] = round(time.perf_counter() - t, 1)
        outs[name] = rep / name
    a, b, live = load(outs["cached_1"]), load(outs["cached_2"]), load(outs["live"])
    files_same = {f: sha(outs["cached_1"] / f) == sha(outs["cached_2"] / f) for f in FILES}
    na, nl = dict(numbers(a)), dict(numbers(live))
    same_keys = set(na) == set(nl)
    diffs = sorted(((abs(na[k] - nl[k]), k) for k in na if k in nl), reverse=True)
    report = {
        "cached_twice_identical": a == b,
        "cached_twice_files_identical": files_same,
        "live_vs_cached_same_structure": same_keys,
        "live_vs_cached_numbers": len(diffs),
        "live_vs_cached_max_abs_diff": diffs[0][0] if diffs else 0.0,
        "live_vs_cached_largest": [{"path": k, "abs_diff": d} for d, k in diffs[:5]],
        "live_vs_cached_identical": a == live,
        "seconds": times,
    }
    (rep / "determinism.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
