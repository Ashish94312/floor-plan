#!/usr/bin/env bash
# Calibrate the photo focal vote (FIX_DECLARATION.md) from the taped photo captures, from scratch:
#   1. runs with the vote only (no level)        -> scan-calibrate --bias-only fits the level
#   2. runs with the level                       -> scan-calibrate fits the interval widths
#   3. final runs and evals
set -euo pipefail
cd "$(dirname "$0")/.."
GT=data/ground_truth/home01.yaml
CAPS=(captures/home01_photo_a captures/home01_photo_b)

uv run python -c "
from scan.uncertainty.calibrate import load, write
cal = load(); cal.pop('photo_joint_vote', None); write(cal)"

run_all() { for c in "${CAPS[@]}"; do uv run scan "$c" > "$c/out.log" 2>&1; uv run scan-eval "$c/out" --gt "$GT" > /dev/null; done; }
evals() { for c in "${CAPS[@]}"; do echo "$c/out/eval.json"; done; }

run_all
uv run scan-calibrate --bias-only $(evals)
run_all
uv run scan-calibrate $(evals)
run_all
for c in "${CAPS[@]}"; do echo "== $c"; tail -1 "$c/out/eval.md"; done
